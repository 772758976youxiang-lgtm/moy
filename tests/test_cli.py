import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(ROOT / 'main.py'), *args],
        capture_output=True, encoding='utf-8', env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})


def test_full_cli_and_idempotent_seed(tmp_path):
    db = str(tmp_path / '实验.db')
    for _ in range(2):
        assert run_cli('init', '--db', db).returncode == 0
    for command in ('generate', 'evolve'):
        result = run_cli(command, '--db', db, '--meme', '打工人', '--context', '老板要求加班', '--mock')
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)['candidates']
    result = run_cli('list', '--db', db)
    assert result.returncode == 0
    assert len(json.loads(result.stdout)) == 2


def test_cli_errors(tmp_path):
    db = str(tmp_path / 'missing.db')
    missing = run_cli('list', '--db', db)
    assert missing.returncode == 1 and 'init' in missing.stderr
    assert run_cli('init', '--db', db).returncode == 0
    for meme, context, count in [('不存在', '测试', '2'), ('打工人', ' ', '2'), ('打工人', '测试', '0')]:
        result = run_cli('generate', '--db', db, '--meme', meme, '--context', context, '--num-candidates', count, '--mock')
        assert result.returncode == 1
        assert 'Traceback' not in result.stderr
