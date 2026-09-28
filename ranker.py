"""可解释启发式评分，不是传播概率或真实用户偏好预测。"""
import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Protocol

from config import Config
from meme_store import Candidate, Meme


def normalized(text: str) -> str:
    return ''.join(re.findall(r'[\w]', text.lower()))


def grams(text: str) -> set[str]:
    clean = normalized(text)
    return {clean[i:i + 2] for i in range(len(clean) - 1)} or set(clean)


@dataclass
class Scores:
    context_fit: float
    novelty: float
    replicability: float
    duplication: float
    risk: float
    total: float


class Ranker(Protocol):
    """LLM judge / learned ranker 实现同一接口即可替换规则评分器。"""
    def score(self, text: str, context: str, meme: Meme, references: list[str]) -> Scores: ...


class RuleRanker:
    def __init__(self, config: Config) -> None:
        self.config = config

    def score(self, text: str, context: str, meme: Meme, references: list[str]) -> Scores:
        cfg = self.config
        clean = normalized(text)
        target = grams(context)
        fit = len(grams(text) & target) / max(1, len(target))
        reference_texts = [normalized(r) for r in references + meme.examples if normalized(r)]
        duplicate = max((SequenceMatcher(None, clean, r).ratio() for r in reference_texts), default=0.0)
        novelty = 1.0 - duplicate
        brevity = min(1.0, cfg.short_text_length / max(1, len(text)))
        structure = float(any(marker in text for marker in cfg.reusable_markers))
        reusable = (brevity + structure) / 2 if clean else 0.0
        risk = float(any(word in text for word in cfg.risk_terms))
        total = max(0.0, min(1.0, cfg.context_weight * fit + cfg.novelty_weight * novelty
                    + cfg.replicability_weight * reusable - cfg.duplication_penalty * duplicate - cfg.risk_penalty * risk)) if clean else 0.0
        return Scores(fit, novelty, reusable, duplicate, risk, total)


def rank_candidates(candidates: list[Candidate], context: str, meme: Meme,
                    ranker: Ranker, references: list[str]) -> list[Candidate]:
    # 每个候选只比较历史、例句和自身之外的候选，不会把自己当作重复。
    for candidate in candidates:
        peers = [c.text for c in candidates if c.candidate_id != candidate.candidate_id]
        scores = ranker.score(candidate.text, context, meme, references + peers)
        candidate.scores = asdict(scores)
        candidate.fitness = scores.total
    return sorted(candidates, key=lambda c: c.fitness or 0, reverse=True)
