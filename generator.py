"""mock 不导入 torch；真实模型首次生成时才加载。"""
import warnings
from typing import Any, Protocol

from config import Config
from meme_store import Meme
from prompts import build_messages


class GenerationError(RuntimeError):
    """模型依赖、缓存、显存或生成失败。"""


class Generator(Protocol):
    def generate(self, meme: Meme, context: str, strategy: str, count: int,
                 parent: str | None = None) -> list[str]: ...


class MockGenerator:
    """确定性模板只用于验证流程，不代表 Qwen 的生成质量。"""
    def __init__(self, config: Config) -> None:
        self.config = config

    def generate(self, meme: Meme, context: str, strategy: str, count: int,
                 parent: str | None = None) -> list[str]:
        if count < 1 or not context.strip():
            raise ValueError('候选数必须为正，场景不能为空。')
        if parent:
            forms = [f'《{meme.name}续集》以为故事结束了，结果：{parent}',
                     f'换个视角：{parent}——这次我选择给自己放个假。']
        else:
            forms = [f'{context}。{meme.name}：我选择精神下班。',
                     f'以为是生活，结果是《{context}》体验服。',
                     f'{meme.name}今日成就：听完“{context}”，笑容还在加载。',
                     f'不是我不努力，是“{context}”给人生加了难度。']
        return [forms[i % len(forms)] + (f'（第{i // len(forms) + 1}次循环）' if i >= len(forms) else '') for i in range(count)]


class QwenGenerator:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.tokenizer: Any = None
        self.model: Any = None
        self.torch: Any = None

    def _load(self) -> None:
        if self.model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise GenerationError('缺少真实模型依赖。请安装 requirements.txt，或使用 --mock。') from exc
        cfg = self.config
        cuda = torch.cuda.is_available()
        if cfg.device == 'cuda' and not cuda:
            raise GenerationError('CUDA 不可用。请检查 NVIDIA 驱动及 CUDA 版 PyTorch，或用 --device cpu / --mock。')
        if cfg.quantization != 'none' and (not cuda or cfg.device == 'cpu'):
            raise GenerationError('本 PoC 的可选量化仅支持 CUDA；请关闭量化或切换 CUDA。')
        if not cuda and cfg.device == 'auto':
            warnings.warn('CUDA 不可用，将用 CPU float32；8B 模型需要较多内存且运行较慢。')
        try:
            kwargs: dict[str, Any] = {
                'local_files_only': cfg.local_files_only,
                'torch_dtype': torch.float32 if cfg.device == 'cpu' or not cuda else 'auto',
                'device_map': cfg.device if cuda and cfg.device != 'cpu' else 'cpu',
            }
            if cfg.quantization != 'none':
                kwargs['quantization_config'] = BitsAndBytesConfig(
                    load_in_4bit=cfg.quantization == '4bit', load_in_8bit=cfg.quantization == '8bit')
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, local_files_only=cfg.local_files_only)
            self.model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **kwargs)
            self.model.eval()
            self.torch = torch
        except torch.OutOfMemoryError as exc:
            raise GenerationError('模型加载内存不足。尝试 --device auto、可选量化，或 --mock。') from exc
        except (OSError, ValueError, ImportError, RuntimeError) as exc:
            raise GenerationError(f'模型加载失败：{exc}\n未下载时可使用 --allow-download；已有本地权重可用 --model 路径。也可先用 --mock。') from exc

    def generate(self, meme: Meme, context: str, strategy: str, count: int,
                 parent: str | None = None) -> list[str]:
        if count < 1 or not context.strip():
            raise ValueError('候选数必须为正，场景不能为空。')
        self._load()
        cfg = self.config
        try:
            prompt = self.tokenizer.apply_chat_template(build_messages(meme, context, strategy, parent),
                        tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inputs = self.tokenizer(prompt, return_tensors='pt')
            if inputs['input_ids'].shape[-1] > cfg.max_input_tokens:
                raise GenerationError(f'输入超过 {cfg.max_input_tokens} token，请缩短场景或模因素材。')
            inputs = inputs.to(self.model.device)
            result: list[str] = []
            # 顺序生成，避免候选批量占用额外显存。
            with self.torch.inference_mode():
                for _ in range(count):
                    output = self.model.generate(**inputs, do_sample=True, temperature=cfg.temperature,
                        top_p=cfg.top_p, top_k=cfg.top_k_sampling, max_new_tokens=cfg.max_new_tokens,
                        pad_token_id=self.tokenizer.eos_token_id)
                    text = self.tokenizer.decode(output[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
                    if '</think>' in text:
                        text = text.split('</think>', 1)[1].strip()
                    if not text:
                        raise GenerationError('模型生成了空内容，请增加 max-new-tokens 或调整策略。')
                    result.append(text)
            return result
        except self.torch.OutOfMemoryError as exc:
            raise GenerationError('生成时内存不足，请减少 max-new-tokens、缩短输入或启用量化。') from exc
        except (RuntimeError, ValueError) as exc:
            if isinstance(exc, GenerationError):
                raise
            raise GenerationError(f'生成失败：{exc}') from exc
