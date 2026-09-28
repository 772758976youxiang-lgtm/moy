"""本地页面的真实 HTTP 流程测试，不加载模型。"""
import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from config import Config
from web_app import create_server


@pytest.fixture
def server(tmp_path):
    instance = create_server(Config(db_path=tmp_path / 'web.db'), port=0)
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    yield instance
    instance.shutdown()
    instance.server_close()
    thread.join()


def request(server, path, payload=None, token=True):
    headers = {'Content-Type': 'application/json'}
    if token:
        headers['X-Meme-Token'] = server.app.token
    req = Request(f'http://127.0.0.1:{server.server_port}{path}',
        data=None if payload is None else json.dumps(payload).encode(), headers=headers)
    try:
        response = urlopen(req, timeout=10)
    except HTTPError as exc:
        response = exc
    with response:
        body = response.read().decode()
        return response.status, json.loads(body) if response.headers.get_content_type() == 'application/json' else body


def test_web_full_flow(server):
    status, html = request(server, '/')
    assert status == 200 and '模因实验台' in html
    assert request(server, '/api/state')[1]['initialized'] is False
    assert request(server, '/api/init', {})[0] == 200
    state = request(server, '/api/state')[1]
    assert len(state['memes']) == 2
    payload = {'action': 'generate', 'meme': '打工人', 'context': '老板说自愿加班', 'mock': True}
    status, result = request(server, '/api/run', payload)
    assert status == 200 and len(result['candidates']) == 4
    export_status, exported = request(server, '/api/export/' + result['run_id'])
    assert export_status == 200 and exported == result
    payload['action'] = 'evolve'
    status, result = request(server, '/api/run', payload)
    assert status == 200 and len(result['candidates']) == 2
    assert result['mode'] == 'mock'
    state = request(server, '/api/state')[1]
    assert state['candidate_count'] >= 8
    assert state['history']


@pytest.mark.parametrize('change', [
    {'context': ' '}, {'num_candidates': 0}, {'num_candidates': 500},
    {'temperature': 'bad'}, {'temperature': float('nan')},
    {'action': 'delete'}, {'mock': 'false'}, {'meme': '不存在'},
])
def test_web_invalid_input(server, change):
    request(server, '/api/init', {})
    payload = {'action': 'generate', 'meme': '打工人', 'context': '测试', 'mock': True, **change}
    status, result = request(server, '/api/run', payload)
    assert status == 400 and result['error']
    assert request(server, '/api/state')[1]['candidate_count'] == 0


def test_web_token_and_missing_database(server):
    assert request(server, '/api/init', {}, token=False)[0] == 403
    status, result = request(server, '/api/run', {'action': 'generate', 'meme': '打工人', 'context': '测试', 'mock': True})
    assert status == 400 and 'init' in result['error']
    assert request(server, '/unknown')[0] == 404


def test_web_busy(server):
    server.app.lock.acquire()
    try:
        status, result = request(server, '/api/init', {})
        assert status == 409 and '运行' in result['error']
    finally:
        server.app.lock.release()
