"""最小命令行入口。所有路径默认相对于项目目录。"""
import argparse
import json
import sqlite3
import sys
from dataclasses import asdict, replace
from pathlib import Path

from config import Config, ROOT
from evolution import evolve, generate_ranked
from generator import GenerationError, MockGenerator, QwenGenerator
from meme_store import Meme, MemeStore, StoreError
from ranker import RuleRanker


def make_parser() -> argparse.ArgumentParser:
    defaults = Config()
    parser = argparse.ArgumentParser(description='Qwen3-8B 模因实验 PoC')
    subs = parser.add_subparsers(dest='command', required=True)
    for command in ('init', 'list', 'generate', 'evolve'):
        sub = subs.add_parser(command)
        sub.add_argument('--db', type=Path, default=defaults.db_path, help='数据库路径')
        if command not in ('generate', 'evolve'):
            continue
        sub.add_argument('--meme', required=True)
        sub.add_argument('--context', required=True)
        sub.add_argument('--mock', action='store_true')
        sub.add_argument('--model', default=defaults.model_name)
        sub.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default=defaults.device)
        sub.add_argument('--allow-download', action='store_true', help='允许从 Hugging Face 下载权重')
        sub.add_argument('--quantization', choices=('none', '4bit', '8bit'), default=defaults.quantization)
        sub.add_argument('--strategy', default=defaults.strategy)
        for name in ('num_candidates', 'max_new_tokens', 'top_k', 'mutations_per_parent'):
            sub.add_argument('--' + name.replace('_', '-'), type=int, default=getattr(defaults, name))
        for name in ('temperature', 'top_p'):
            sub.add_argument('--' + name.replace('_', '-'), type=float, default=getattr(defaults, name))
    return parser


def main() -> int:
    args = make_parser().parse_args()
    try:
        store = MemeStore(args.db)
        if args.command == 'init':
            store.initialize()
            existing = {m.name for m in store.list_memes()}
            for entry in json.loads((ROOT / 'data' / 'seed_memes.json').read_text(encoding='utf-8')):
                if entry['name'] not in existing:
                    store.save_meme(Meme(**entry))
                    existing.add(entry['name'])
            print(f'初始化完成：{args.db}；已有模因不会被覆盖。')
        elif args.command == 'list':
            print(json.dumps([asdict(m) for m in store.list_memes()], ensure_ascii=False, indent=2))
        else:
            if not args.context.strip():
                raise ValueError('用户场景不能为空。')
            config = replace(Config(), db_path=args.db, model_name=args.model, device=args.device,
                local_files_only=not args.allow_download, quantization=args.quantization,
                strategy=args.strategy, temperature=args.temperature, top_p=args.top_p,
                num_candidates=args.num_candidates, max_new_tokens=args.max_new_tokens,
                top_k=args.top_k, mutations_per_parent=args.mutations_per_parent)
            meme = store.get_meme(args.meme)
            generator = MockGenerator(config) if args.mock else QwenGenerator(config)
            operation = evolve if args.command == 'evolve' else generate_ranked
            result = operation(store, meme, args.context, generator, RuleRanker(config), config)
            print(json.dumps({'mode': 'mock' if args.mock else 'Qwen3', 'command': args.command,
                              'candidates': [asdict(c) for c in result]}, ensure_ascii=False, indent=2))
        return 0
    except (StoreError, GenerationError, ValueError, OSError, sqlite3.Error) as exc:
        print(f'错误：{exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
