from dataclasses import replace

import pytest

from config import Config
from meme_store import Meme, MemeStore, StoreError
from generator import MockGenerator
from ranker import RuleRanker
from evolution import evolve


@pytest.fixture
def store(tmp_path):
    result = MemeStore(tmp_path / 'memes.db')
    result.initialize()
    return result


def test_store_roundtrip_and_idempotent_init(store):
    meme = Meme(name='测试', semantic_core='测试语义', contexts=['办公'], embedding=[0.1])
    store.save_meme(meme)
    store.initialize()
    assert store.get_meme('测试') == meme
    assert len(store.list_memes()) == 1
    with pytest.raises(StoreError, match='找不到'):
        store.get_meme('不存在')


def test_missing_database(tmp_path):
    with pytest.raises(StoreError, match='init'):
        MemeStore(tmp_path / 'missing.db').list_memes()


def test_ranker_penalties_and_context():
    ranker = RuleRanker(Config())
    meme = Meme(name='打工人', semantic_core='工作', contexts=['加班'])
    original = '老板说自愿加班，我选择自愿下班。'
    good = ranker.score(original, '老板要求加班', meme, [])
    duplicate = ranker.score(original, '老板要求加班', meme, [original])
    risky = ranker.score('我要杀了你', '老板要求加班', meme, [])
    assert duplicate.duplication == 1
    assert duplicate.novelty == 0
    assert duplicate.total < good.total
    assert risky.risk == 1
    assert good.context_fit > ranker.score('晴天去游泳', '老板要求加班', meme, []).context_fit


def test_evolution_persists_lineage_and_counters(store):
    config = replace(Config(), num_candidates=4, top_k=2, mutations_per_parent=2)
    meme = Meme(name='打工人', semantic_core='职场自嘲', contexts=['加班'])
    store.save_meme(meme)
    generator = MockGenerator(config)
    first = evolve(store, meme, '老板说自愿加班', generator, RuleRanker(config), config)
    assert len(first) == config.top_k
    pool = store.list_candidates(meme.meme_id)
    assert len(pool) == 8
    by_id = {c.candidate_id: c for c in pool}
    children = [c for c in pool if c.parent_id]
    assert children
    for child in children:
        assert child.generation == by_id[child.parent_id].generation + 1
    assert all(c.fitness is not None for c in pool)
    assert store.get_meme(meme.name).lifecycle['usage'] == 1
    evolve(store, meme, '老板说自愿加班', generator, RuleRanker(config), config)
    second_pool = store.list_candidates(meme.meme_id)
    assert any(c.candidate_id not in by_id and c.parent_id in {w.candidate_id for w in first} for c in second_pool)
    assert store.get_meme(meme.name).lifecycle['usage'] == 2
    assert any(c.age > 0 for c in store.list_candidates(meme.meme_id))
    assert len([c for c in second_pool if c.selected]) == config.top_k
    assert all(c.generation == next(p.generation for p in second_pool if p.candidate_id == c.parent_id) + 1 for c in second_pool if c.parent_id)


def test_invalid_configuration():
    with pytest.raises(ValueError):
        Config(num_candidates=0)


def test_failed_evolution_does_not_write(store):
    class Broken(MockGenerator):
        def generate(self, *args, **kwargs):
            raise RuntimeError('模拟失败')
    meme = Meme(name='测试', semantic_core='测试')
    store.save_meme(meme)
    with pytest.raises(RuntimeError):
        evolve(store, meme, '测试', Broken(Config()), RuleRanker(Config()), Config())
    assert store.list_candidates(meme.meme_id) == []
