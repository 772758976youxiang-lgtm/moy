"""本地交互页面：标准库 HTTP 服务，直接复用四个实验模块。"""
import argparse
import json
import math
import secrets
import sqlite3
import threading
import time
import traceback
import webbrowser
from dataclasses import asdict, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from config import Config, ROOT, WebConfig
from evolution import evolve, generate_ranked
from generator import GenerationError, MockGenerator, QwenGenerator
from meme_store import Meme, MemeStore, StoreError
from ranker import RuleRanker


class WebApp:
    def __init__(self, config: Config, settings: WebConfig) -> None:
        self.config = config
        self.settings = settings
        self.store = MemeStore(config.db_path)
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.Lock()
        self.model: QwenGenerator | None = None
        self.model_key: tuple[str, str, str] | None = None
        self.exports: dict[str, dict[str, Any]] = {}

    def state(self) -> dict[str, Any]:
        defaults = {key: getattr(self.config, key) for key in (
            'temperature', 'top_p', 'num_candidates', 'max_new_tokens',
            'top_k', 'mutations_per_parent', 'strategy', 'device', 'quantization')}
        base: dict[str, Any] = {'defaults': defaults, 'limits': asdict(self.settings),
                               'busy': self.lock.locked(), 'model': self.config.model_name}
        try:
            memes = self.store.list_memes()
        except StoreError:
            return {**base, 'initialized': False, 'memes': [], 'history': {}, 'candidate_count': 0}
        history: dict[str, Any] = {}
        count = 0
        for meme in memes:
            candidates = self.store.list_candidates(meme.meme_id)
            count += len(candidates)
            history[meme.meme_id] = [asdict(c) for c in candidates[-self.settings.history_limit:][::-1]]
        return {**base, 'initialized': True, 'memes': [asdict(m) for m in memes],
                'history': history, 'candidate_count': count}

    def initialize(self) -> dict[str, Any]:
        self.store.initialize()
        existing = {m.name for m in self.store.list_memes()}
        seeds = json.loads((ROOT / 'data' / 'seed_memes.json').read_text(encoding='utf-8'))
        for entry in seeds:
            if entry['name'] not in existing:
                self.store.save_meme(Meme(**entry))
                existing.add(entry['name'])
        return {'message': '模因库已就绪，已有数据已保留。'}

    def run(self, data: dict[str, Any]) -> dict[str, Any]:
        action = data.get('action')
        if action not in ('generate', 'evolve'):
            raise ValueError('请选择生成候选或进化一轮。')
        context = data.get('context', '')
        meme_name = data.get('meme', '')
        strategy = data.get('strategy', self.config.strategy)
        for value, label, limit in ((context, '场景', self.settings.max_context_chars),
                                    (strategy, '策略', self.settings.max_strategy_chars),
                                    (meme_name, '模因', self.settings.max_context_chars)):
            if not isinstance(value, str) or not value.strip() or len(value) > limit:
                raise ValueError(f'{label}不能为空，且长度不能超过 {limit} 字。')
        mock = data.get('mock', True)
        download = data.get('allow_download', False)
        if type(mock) is not bool or type(download) is not bool:
            raise ValueError('运行模式和下载设置必须为布尔值。')
        updates: dict[str, Any] = {'strategy': strategy, 'local_files_only': not download}
        integer_limits = {'num_candidates': self.settings.max_candidates,
                          'max_new_tokens': self.settings.max_tokens,
                          'top_k': self.settings.max_selected,
                          'mutations_per_parent': self.settings.max_mutations}
        for key, upper in integer_limits.items():
            value = data.get(key, getattr(self.config, key))
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError(f'{key} 必须为 1～{upper} 的整数。')
            updates[key] = value
        for key, upper in (('temperature', self.settings.max_temperature), ('top_p', 1.0)):
            value = data.get(key, getattr(self.config, key))
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= upper:
                raise ValueError(f'{key} 必须大于 0 且不超过 {upper}。')
            updates[key] = value
        updates['device'] = data.get('device', self.config.device)
        updates['quantization'] = data.get('quantization', self.config.quantization)
        config = replace(self.config, **updates)
        meme = self.store.get_meme(meme_name)
        if mock:
            generator = MockGenerator(config)
        else:
            # 复用同一模型，调整温度和候选数不会再次加载 8B 权重。
            key = (config.model_name, config.device, config.quantization)
            if self.model is not None and self.model_key != key:
                raise ValueError('真实模型已加载。修改设备或量化方式前，请重启页面服务以释放旧模型。')
            if self.model is None:
                self.model = QwenGenerator(config)
                self.model_key = key
            self.model.config = config
            generator = self.model
        started = time.monotonic()
        operation = evolve if action == 'evolve' else generate_ranked
        try:
            candidates = operation(self.store, meme, context.strip(), generator, RuleRanker(config), config)
        except GenerationError:
            if self.model is not None and self.model.model is None:
                self.model = None
                self.model_key = None
            raise
        run_id = secrets.token_hex(16)
        result = {'run_id': run_id, 'action': action, 'mode': 'mock' if mock else 'Qwen3', 'meme': meme.name,
                'meme_id': meme.meme_id, 'context': context.strip(),
                'elapsed': round(time.monotonic() - started, 2),
                'candidates': [asdict(c) for c in candidates]}
        self.exports[run_id] = result
        if len(self.exports) > self.settings.export_limit:
            self.exports.pop(next(iter(self.exports)))
        return result


class LocalServer(ThreadingHTTPServer):
    app: WebApp
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    server: LocalServer

    def log_message(self, format: str, *args: Any) -> None:
        # 页面轮询无需刷屏；异常由错误响应或服务终端呈现。
        return

    def respond(self, status: int, content: bytes, mime: str, filename: str | None = None) -> None:
        self.send_response(status)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(content)))
        self.send_header('Cache-Control', 'no-store')
        if filename:
            self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'")
        self.end_headers()
        try:
            self.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def json_response(self, status: int, data: dict[str, Any]) -> None:
        self.respond(status, json.dumps(data, ensure_ascii=False).encode('utf-8'), 'application/json; charset=utf-8')

    def local_request(self) -> bool:
        allowed = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
        if self.headers.get('Host') not in allowed:
            return False
        origin = self.headers.get('Origin')
        return origin is None or origin in {f'http://{host}' for host in allowed}

    def do_GET(self) -> None:
        if not self.local_request():
            self.json_response(403, {'error': '仅允许本机页面访问。'})
            return
        path = urlsplit(self.path).path
        if path.startswith('/api/export/'):
            run_id = path.removeprefix('/api/export/')
            result = self.server.app.exports.get(run_id)
            if result is None:
                self.json_response(404, {'error': '本次导出已过期，请重新运行实验。'})
            else:
                self.respond(200, json.dumps(result, ensure_ascii=False, indent=2).encode('utf-8'),
                             'application/json; charset=utf-8', f'meme-{run_id}.json')
            return
        if path == '/api/state':
            try:
                self.json_response(200, self.server.app.state())
            except (OSError, sqlite3.Error, ValueError) as exc:
                self.json_response(500, {'error': f'读取模因库失败：{exc}'})
            return
        assets = {'/': ('index.html', 'text/html; charset=utf-8'),
                  '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                  '/style.css': ('style.css', 'text/css; charset=utf-8')}
        if path not in assets:
            self.json_response(404, {'error': '页面不存在。'})
            return
        file_name, mime = assets[path]
        content = (ROOT / 'web' / file_name).read_text(encoding='utf-8')
        if path == '/':
            content = content.replace('__REQUEST_TOKEN__', self.server.app.token)
        self.respond(200, content.encode('utf-8'), mime)

    def do_POST(self) -> None:
        app = self.server.app
        if not self.local_request() or not secrets.compare_digest(self.headers.get('X-Meme-Token', ''), app.token):
            self.json_response(403, {'error': '页面验证失效，请刷新后重试。'})
            return
        path = urlsplit(self.path).path
        if path not in ('/api/init', '/api/run'):
            self.json_response(404, {'error': '接口不存在。'})
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= app.settings.max_body_bytes:
                raise ValueError('请求为空或超过长度限制。')
            if self.headers.get_content_type() != 'application/json':
                raise ValueError('请求格式必须为 JSON。')
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError('请求内容必须为对象。')
        except (ValueError, UnicodeDecodeError) as exc:
            self.json_response(400, {'error': str(exc)})
            return
        if not app.lock.acquire(blocking=False):
            self.json_response(409, {'error': '已有实验正在运行，请稍候再试。'})
            return
        try:
            self.json_response(200, app.initialize() if path == '/api/init' else app.run(data))
        except (ValueError, StoreError) as exc:
            self.json_response(400, {'error': str(exc)})
        except (GenerationError, OSError, sqlite3.Error) as exc:
            self.json_response(503, {'error': str(exc)})
        except Exception:
            traceback.print_exc()
            self.json_response(500, {'error': '运行失败，请查看启动窗口中的错误信息。'})
        finally:
            app.lock.release()


def create_server(config: Config | None = None, port: int = WebConfig().port) -> LocalServer:
    settings = WebConfig(port=port)
    server = LocalServer((settings.host, port), Handler)
    server.app = WebApp(config or Config(), settings)
    return server


def main() -> int:
    parser = argparse.ArgumentParser(description='启动本地模因实验台')
    parser.add_argument('--port', type=int, default=WebConfig().port)
    parser.add_argument('--db', type=Path, default=Config().db_path)
    parser.add_argument('--open', action='store_true', help='启动后打开默认浏览器')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('端口必须在 1～65535 之间。')
    try:
        server = create_server(replace(Config(), db_path=args.db), args.port)
    except OSError as exc:
        print(f'启动失败：{exc}。可尝试 --port 8766。')
        return 1
    url = f'http://127.0.0.1:{server.server_port}'
    print(f'模因实验台：{url}\n仅供本机使用；按 Ctrl+C 停止。', flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
