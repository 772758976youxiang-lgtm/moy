"""所有默认参数集中在此；CLI 可覆盖常用生成参数。"""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class WebConfig:
    """本地交互页面的端口、输入上限与历史展示数量。"""
    host: str = '127.0.0.1'
    port: int = 8765
    max_body_bytes: int = 16384
    max_context_chars: int = 2000
    max_strategy_chars: int = 500
    history_limit: int = 60
    export_limit: int = 20
    max_candidates: int = 12
    max_tokens: int = 1024
    max_selected: int = 6
    max_mutations: int = 4
    max_temperature: float = 2.0


@dataclass(frozen=True)
class Config:
    db_path: Path = ROOT / 'data' / 'memes.db'
    model_name: str = 'Qwen/Qwen3-8B'
    device: str = 'auto'
    local_files_only: bool = True
    quantization: str = 'none'
    temperature: float = 0.7
    top_p: float = 0.8
    top_k_sampling: int = 20
    num_candidates: int = 4
    max_new_tokens: int = 160
    max_input_tokens: int = 4096
    top_k: int = 2
    mutations_per_parent: int = 2
    strategy: str = '保持语义核心，用简短反转和自嘲表达'
    context_weight: float = 0.45
    novelty_weight: float = 0.25
    replicability_weight: float = 0.30
    duplication_penalty: float = 0.35
    risk_penalty: float = 1.0
    short_text_length: int = 80
    risk_terms: tuple[str, ...] = ('杀了你', '去死', '人肉搜索', '身份证号')
    reusable_markers: tuple[str, ...] = ('：', ':', '《', '不是', '以为', '结果', '选择')

    def __post_init__(self) -> None:
        if not (self.temperature > 0 and 0 < self.top_p <= 1):
            raise ValueError('temperature 必须大于 0，top_p 必须在 (0, 1]。')
        if min(self.num_candidates, self.max_new_tokens, self.top_k,
               self.mutations_per_parent, self.max_input_tokens) < 1:
            raise ValueError('候选数、token 数和 top-k 必须为正数。')
        if self.device not in ('auto', 'cuda', 'cpu'):
            raise ValueError('device 只能为 auto/cuda/cpu。')
        if self.quantization not in ('none', '4bit', '8bit'):
            raise ValueError('quantization 只能为 none/4bit/8bit。')
