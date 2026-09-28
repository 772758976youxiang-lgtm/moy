"""候选选择 → 变异 → 联合选择；完整成功后才写入数据库。"""
from config import Config
from generator import Generator
from meme_store import Candidate, Meme, MemeStore
from ranker import Ranker, rank_candidates


def generate_ranked(store: MemeStore, meme: Meme, context: str, generator: Generator,
                    ranker: Ranker, config: Config) -> list[Candidate]:
    history = store.list_candidates(meme.meme_id)
    generation = int(store.get_meme(meme.meme_id).lifecycle.get('generation', 0))
    pool = [Candidate(meme.meme_id, text, context, generation=generation)
            for text in generator.generate(meme, context, config.strategy, config.num_candidates)]
    ranked = rank_candidates(pool, context, meme, ranker, [c.text for c in history])
    store.save_run(meme, pool, ranked[:config.top_k], evolution=False)
    return ranked


def evolve(store: MemeStore, meme: Meme, context: str, generator: Generator,
           ranker: Ranker, config: Config) -> list[Candidate]:
    current = store.get_meme(meme.meme_id)
    history = store.list_candidates(meme.meme_id)
    pool = [c for c in history if c.context == context and c.selected]
    if not pool:
        pool = [Candidate(meme.meme_id, text, context)
                for text in generator.generate(current, context, config.strategy, config.num_candidates)]
    pool_ids = {c.candidate_id for c in pool}
    references = [c.text for c in history if c.candidate_id not in pool_ids]
    parents = rank_candidates(pool, context, current, ranker, references)[:config.top_k]
    children: list[Candidate] = []
    for parent in parents:
        texts = generator.generate(current, context, config.strategy, config.mutations_per_parent, parent.text)
        children.extend(Candidate(meme.meme_id, text, context, parent_id=parent.candidate_id,
                                  generation=parent.generation + 1) for text in texts)
    combined = pool + children
    winners = rank_candidates(combined, context, current, ranker, references)[:config.top_k]
    store.save_run(current, combined, winners, evolution=True)
    return winners
