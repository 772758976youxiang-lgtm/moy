"""SFT / DPO 接入骨架：仅校验 JSONL，绝不启动训练。"""
import argparse
import json
from pathlib import Path
from config import ROOT


def validate_messages(messages: object, label: str) -> None:
    if not isinstance(messages, list) or not messages:
        raise ValueError(f'{label} 必须为非空消息列表')
    for message in messages:
        if not isinstance(message, dict) or message.get('role') not in ('system', 'user', 'assistant') or not isinstance(message.get('content'), str) or not message['content'].strip():
            raise ValueError(f'{label} 消息必须包含合法 role 和非空 content')


def validate_data(path: Path, kind: str) -> int:
    count = 0
    for line_no, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            for key in (('messages',) if kind == 'sft' else ('prompt', 'chosen', 'rejected')):
                validate_messages(row[key], key)
            if kind == 'sft' and row['messages'][-1]['role'] != 'assistant':
                raise ValueError('SFT 最后一条应为 assistant')
            if kind == 'dpo' and (row['chosen'] == row['rejected'] or any(row[key][-1]['role'] != 'assistant' for key in ('chosen', 'rejected'))):
                raise ValueError('DPO 两种回答必须不同，并以 assistant 结束')
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'第 {line_no} 行格式错误：{exc}') from exc
        count += 1
    if count == 0:
        raise ValueError('数据文件为空')
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description='训练接入骨架：仅校验数据')
    parser.add_argument('--kind', required=True, choices=('sft', 'dpo'))
    parser.add_argument('--data', type=Path)
    args = parser.parse_args()
    path = args.data or ROOT / 'data' / ('meme_sft.jsonl' if args.kind == 'sft' else 'preference_pairs.jsonl')
    try:
        count = validate_data(path, args.kind)
        print(f'{args.kind.upper()} 格式校验通过：{count} 条；未进行训练。')
        # 后续在这里接入数据集划分、chat template 和 SFTTrainer / DPOTrainer。
        return 0
    except (OSError, ValueError) as exc:
        print(f'错误：{exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
