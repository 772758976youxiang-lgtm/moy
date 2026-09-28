"""SQLite 存身份和候选；复杂字段用 JSON，进化结果一次事务提交。"""
import json
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from uuid import uuid4


class StoreError(RuntimeError):
    """可向 CLI 用户直接展示的存储错误。"""


@dataclass
class Meme:
    name: str
    semantic_core: str
    meme_id: str = field(default_factory=lambda: uuid4().hex)
    style: list[str] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)
    templates: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    parents: list[str] = field(default_factory=list)
    children: list[str] = field(default_factory=list)
    lifecycle: dict[str, int | float | str] = field(default_factory=lambda: {
        'generation': 0, 'fitness': 0.0, 'usage': 0, 'age': 0, 'status': 'active'})
    embedding: list[float] | None = None


@dataclass
class Candidate:
    meme_id: str
    text: str
    context: str
    candidate_id: str = field(default_factory=lambda: uuid4().hex)
    parent_id: str | None = None
    generation: int = 0
    fitness: float | None = None
    usage: int = 0
    age: int = 0
    scores: dict[str, float] = field(default_factory=dict)
    selected: bool = False


class MemeStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def _connect(self, initialize: bool = False) -> sqlite3.Connection:
        if not initialize and not self.path.exists():
            raise StoreError('数据库未初始化，请先运行 python main.py init。')
        conn = sqlite3.connect(self.path)
        if not initialize:
            names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not {'memes', 'candidates'} <= names:
                conn.close()
                raise StoreError('数据库结构未初始化，请先运行 python main.py init。')
        return conn

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect(initialize=True)
        try:
            with conn:
                conn.execute('CREATE TABLE IF NOT EXISTS memes (meme_id TEXT PRIMARY KEY, name TEXT UNIQUE NOT NULL, payload TEXT NOT NULL)')
                conn.execute('CREATE TABLE IF NOT EXISTS candidates (candidate_id TEXT PRIMARY KEY, meme_id TEXT NOT NULL, payload TEXT NOT NULL)')
                conn.execute('CREATE INDEX IF NOT EXISTS candidates_meme ON candidates(meme_id)')
        finally:
            conn.close()

    def save_meme(self, meme: Meme) -> None:
        conn = self._connect()
        try:
            with conn:
                conn.execute('INSERT INTO memes VALUES (?, ?, ?) ON CONFLICT(meme_id) DO UPDATE SET name=excluded.name, payload=excluded.payload',
                             (meme.meme_id, meme.name, json.dumps(asdict(meme), ensure_ascii=False)))
        finally:
            conn.close()

    def list_memes(self) -> list[Meme]:
        conn = self._connect()
        try:
            return [Meme(**json.loads(r[0])) for r in conn.execute('SELECT payload FROM memes ORDER BY name')]
        finally:
            conn.close()

    def get_meme(self, name: str) -> Meme:
        for meme in self.list_memes():
            if name in (meme.name, meme.meme_id):
                return meme
        raise StoreError(f'找不到模因：{name}。请用 list 查看。')

    def list_candidates(self, meme_id: str) -> list[Candidate]:
        conn = self._connect()
        try:
            return [Candidate(**json.loads(r[0])) for r in conn.execute(
                'SELECT payload FROM candidates WHERE meme_id=? ORDER BY rowid', (meme_id,))]
        finally:
            conn.close()

    def save_run(self, meme: Meme, new: list[Candidate], winners: list[Candidate], evolution: bool) -> None:
        if not new:
            raise StoreError('候选池为空，无法保存运行结果。')
        conn = self._connect()
        try:
            with conn:
                row = conn.execute('SELECT payload FROM memes WHERE meme_id=?', (meme.meme_id,)).fetchone()
                if row is None:
                    raise StoreError('模因不存在，无法保存运行结果。')
                current = Meme(**json.loads(row[0]))
                rows = conn.execute('SELECT candidate_id, payload FROM candidates WHERE meme_id=?', (meme.meme_id,)).fetchall()
                existing = {cid: json.loads(payload) for cid, payload in rows}
                for cid, old in existing.items():
                    if old['context'] == new[0].context:
                        old['selected'] = False
                    if evolution:
                        old['age'] += 1
                    conn.execute('UPDATE candidates SET payload=? WHERE candidate_id=?', (json.dumps(old, ensure_ascii=False), cid))
                if evolution:
                    current.lifecycle['age'] = int(current.lifecycle.get('age', 0)) + 1
                selected = {c.candidate_id for c in winners}
                for candidate in new:
                    if candidate.candidate_id in existing:
                        candidate.age = existing[candidate.candidate_id]['age']
                    candidate.usage += int(candidate.candidate_id in selected)
                    candidate.selected = candidate.candidate_id in selected
                    conn.execute('INSERT INTO candidates VALUES (?, ?, ?) ON CONFLICT(candidate_id) DO UPDATE SET payload=excluded.payload',
                                 (candidate.candidate_id, candidate.meme_id, json.dumps(asdict(candidate), ensure_ascii=False)))
                current.lifecycle['usage'] = int(current.lifecycle.get('usage', 0)) + 1
                current.lifecycle['generation'] = max(int(current.lifecycle.get('generation', 0)), max((c.generation for c in new), default=0))
                current.lifecycle['fitness'] = max((c.fitness or 0 for c in winners), default=0)
                conn.execute('UPDATE memes SET payload=? WHERE meme_id=?', (json.dumps(asdict(current), ensure_ascii=False), meme.meme_id))
        finally:
            conn.close()
