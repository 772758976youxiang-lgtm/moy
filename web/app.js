/* 页面只负责交互与展示，生成和评分仍由 Python 核心模块完成。 */
'use strict';
const $ = (id) => document.getElementById(id);
const token = document.querySelector('meta[name="meme-token"]').content;
let state = null;
let result = null;
let running = false;
let toastTimer;
const numericFields = ['num_candidates', 'top_k', 'temperature', 'top_p', 'max_new_tokens', 'mutations_per_parent'];
const metrics = [['context_fit', '匹配'], ['novelty', '新颖'], ['replicability', '复用'], ['duplication', '重复'], ['risk', '风险']];

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
function toast(message) {
  $('toast').textContent = message;
  $('toast').hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { $('toast').hidden = true; }, 2600);
}
function notice(message, error = false) {
  $('notice').replaceChildren();
  $('notice').textContent = message;
  $('notice').className = 'notice' + (error ? ' error' : '');
  $('notice').hidden = !message;
}
async function api(path, payload) {
  const options = { headers: { 'X-Meme-Token': token } };
  if (payload !== undefined) {
    options.method = 'POST';
    options.headers['Content-Type'] = 'application/json';
    options.body = JSON.stringify(payload);
  }
  let response;
  try { response = await fetch(path, options); }
  catch { throw new Error('无法连接本地服务。请确认启动窗口仍在运行，然后重试。'); }
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || '请求失败，请重试。');
  return data;
}
function switchTab(history) {
  $('results-view').hidden = history;
  $('history-view').hidden = !history;
  for (const [id, active] of [['results-tab', !history], ['history-tab', history]]) {
    $(id).classList.toggle('active', active);
    $(id).setAttribute('aria-selected', String(active));
    $(id).tabIndex = active ? 0 : -1;
  }
}
function selectedMeme() { return state?.memes.find(m => m.meme_id === $('meme').value); }
function updateMeme() {
  const meme = selectedMeme();
  $('meme-info').replaceChildren();
  $('generation-count').textContent = meme?.lifecycle.generation ?? 0;
  if (meme) {
    $('meme-info').append(element('p', '', meme.semantic_core));
    const tags = element('div', 'tags');
    for (const style of meme.style) tags.append(element('span', 'tag', style));
    $('meme-info').append(tags);
  } else $('meme-info').append(element('p', '', '初始化后即可选择示例模因。'));
  renderHistory();
}
function updateCount() { $('char-count').textContent = `${$('context').value.length} / ${$('context').maxLength}`; }
function updateMode() {
  const mock = $('mode').value === 'mock';
  $('real-settings').hidden = mock;
  $('mode-hint').textContent = mock ? '无需下载模型，先体验完整流程。' : '使用本机 Qwen3-8B 生成；首次加载需要等待。';
}
function syncButtons() {
  const unavailable = running || !state?.initialized || !state?.memes.length || state?.busy;
  $('controls').disabled = running;
  $('generate').disabled = unavailable;
  $('evolve').disabled = unavailable;
  $('export').disabled = running || !result;
}
async function loadState(first = false) {
  state = await api('/api/state');
  const previous = $('meme').value;
  $('meme').replaceChildren();
  for (const meme of state.memes) $('meme').append(new Option(meme.name, meme.meme_id));
  if (state.memes.some(m => m.meme_id === previous)) $('meme').value = previous;
  else if (state.memes.some(m => m.name === '打工人')) $('meme').value = state.memes.find(m => m.name === '打工人').meme_id;
  $('meme-count').textContent = state.memes.length;
  $('candidate-count').textContent = state.candidate_count;
  $('history-limit').textContent = state.limits.history_limit;
  if (first) {
    for (const field of [...numericFields, 'strategy', 'device', 'quantization']) $(field).value = state.defaults[field];
    const limits = { num_candidates: 'max_candidates', top_k: 'max_selected', max_new_tokens: 'max_tokens', mutations_per_parent: 'max_mutations', temperature: 'max_temperature' };
    for (const [field, key] of Object.entries(limits)) $(field).max = state.limits[key];
    $('context').maxLength = state.limits.max_context_chars;
    $('strategy').maxLength = state.limits.max_strategy_chars;
    updateCount();
  }
  if (!state.initialized || state.memes.length === 0) {
    notice('模因库还未准备好，先添加内置示例即可开始。');
    const button = element('button', '', '初始化示例模因');
    button.type = 'button';
    button.addEventListener('click', async () => {
      button.disabled = true;
      try { await api('/api/init', {}); notice(''); await loadState(); toast('模因库已就绪'); }
      catch (error) { notice(error.message, true); }
      finally { button.disabled = false; }
    });
    $('notice').append(button);
  } else if (state.busy && !running) notice('另一个页面正在运行实验。完成后点击“候选记录 → 刷新”查看结果。');
  updateMeme();
  syncButtons();
}
function candidateCard(candidate, index, history = false) {
  const card = element('article', 'candidate-card' + (candidate.selected ? ' winner' : ''));
  const top = element('div', 'card-top');
  const badges = element('div', 'card-badges');
  badges.append(element('span', 'rank-number', String(index + 1).padStart(2, '0')));
  badges.append(element('span', 'badge neutral', `第 ${candidate.generation} 代`));
  if (candidate.selected) badges.append(element('span', 'badge', '已入选'));
  if (candidate.parent_id) badges.append(element('span', 'badge green', '变体'));
  const score = element('span', 'score', ((candidate.fitness ?? 0) * 100).toFixed(1));
  score.append(element('small', '', '/ 100'));
  top.append(badges, score);
  card.append(top, element('p', 'candidate-text', candidate.text));
  const scores = element('div', 'scores');
  for (const [key, label] of metrics) {
    const value = candidate.scores[key] ?? 0;
    const metric = element('div');
    const title = element('div', 'metric-label', label);
    title.append(element('b', '', `${Math.round(value * 100)}`));
    const progress = element('progress', 'metric-track' + (key === 'duplication' ? ' penalty' : key === 'risk' ? ' risk' : ''));
    progress.max = 1;
    progress.value = value;
    progress.setAttribute('aria-label', `${label} ${Math.round(value * 100)} 分`);
    metric.append(title, progress);
    scores.append(metric);
  }
  card.append(scores);
  if (history) card.append(element('p', 'history-context', `场景：${candidate.context}`));
  const bottom = element('div', 'card-bottom');
  const lineage = candidate.parent_id ? `父代 ${candidate.parent_id.slice(0, 8)} · ` : '初始表达 · ';
  const detail = element('span', 'lineage', `${lineage}入选 ${candidate.usage} 次 · 年龄 ${candidate.age}`);
  detail.title = `候选 ID：${candidate.candidate_id}\n父代 ID：${candidate.parent_id || '无'}`;
  const copy = element('button', 'copy-button', '复制表达 ⧉');
  copy.type = 'button';
  copy.addEventListener('click', async () => {
    try { await navigator.clipboard.writeText(candidate.text); toast('已复制表达'); }
    catch { toast('无法访问剪贴板，请选择正文手动复制'); }
  });
  bottom.append(detail, copy);
  card.append(bottom);
  return card;
}
function renderHistory() {
  const candidates = state?.history[$('meme').value] || [];
  $('history-cards').replaceChildren();
  if (!candidates.length) $('history-cards').append(element('p', 'history-empty', '还没有记录。完成第一次生成后，这里会保留所有候选。'));
  candidates.forEach((c, i) => $('history-cards').append(candidateCard(c, i, true)));
}
function renderResult() {
  $('cards').replaceChildren();
  $('empty').hidden = Boolean(result);
  $('run-meta').hidden = !result;
  $('result-count').textContent = result?.candidates.length ?? 0;
  if (!result) return;
  $('run-meta').replaceChildren(element('strong', '', `${result.mode === 'mock' ? '模拟实验' : 'Qwen3-8B'} · ${result.action === 'evolve' ? '进化完成' : '生成完成'}`),
    document.createTextNode(`　${result.meme} / ${result.elapsed.toFixed(2)} 秒`), element('br'), document.createTextNode(`场景：${result.context}`));
  result.candidates.forEach((c, i) => $('cards').append(candidateCard(c, i)));
  syncButtons();
}
async function run(action) {
  if (running || !$('experiment-form').reportValidity()) return;
  if (!$('context').value.trim()) { $('context').focus(); notice('先写一个你想表达的场景。', true); return; }
  const payload = { action, meme: $('meme').value, context: $('context').value.trim(), strategy: $('strategy').value,
    mock: $('mode').value === 'mock', device: $('device').value, quantization: $('quantization').value,
    allow_download: $('allow-download').checked };
  for (const field of numericFields) payload[field] = Number($(field).value);
  running = true;
  notice('');
  switchTab(false);
  syncButtons();
  $('empty').hidden = true;
  $('cards').hidden = true;
  $('run-meta').hidden = true;
  $('loading').hidden = false;
  $('loading-title').textContent = action === 'evolve' ? '正在选择、变异，再选择' : '正在生成你的候选表达';
  $('loading-detail').textContent = payload.mock ? '正在运行模拟生成与规则评分…' : '正在加载或调用 Qwen3-8B，首次加载可能需要几分钟。';
  const started = Date.now();
  $('elapsed').textContent = '已等待 0 秒';
  const timer = setInterval(() => { $('elapsed').textContent = `已等待 ${Math.floor((Date.now() - started) / 1000)} 秒`; }, 1000);
  try {
    result = await api('/api/run', payload);
    try { await loadState(); } catch (error) { notice(`实验已保存，但统计刷新失败：${error.message}`, true); }
    toast(action === 'evolve' ? '进化完成，候选已保存' : '生成完成，候选已保存');
  } catch (error) { notice(error.message, true); }
  finally {
    running = false;
    clearInterval(timer);
    $('loading').hidden = true;
    $('cards').hidden = false;
    renderResult();
    syncButtons();
  }
}
$('experiment-form').addEventListener('submit', event => { event.preventDefault(); run('generate'); });
$('evolve').addEventListener('click', () => run('evolve'));
$('mode').addEventListener('change', updateMode);
$('meme').addEventListener('change', updateMeme);
$('context').addEventListener('input', updateCount);
document.querySelectorAll('[data-scene]').forEach(button => button.addEventListener('click', () => { $('context').value = button.dataset.scene; updateCount(); $('context').focus(); }));
$('results-tab').addEventListener('click', () => switchTab(false));
$('history-tab').addEventListener('click', () => switchTab(true));
document.querySelector('.tabs').addEventListener('keydown', event => {
  if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) {
    event.preventDefault();
    const history = event.key === 'End' || (event.key !== 'Home' && $('results-tab').getAttribute('aria-selected') === 'true');
    switchTab(history);
    $(history ? 'history-tab' : 'results-tab').focus();
  }
});
$('refresh-history').addEventListener('click', async () => {
  try { notice(''); await loadState(); toast('记录已刷新'); } catch (error) { notice(error.message, true); }
});
$('export').addEventListener('click', () => {
  if (!result) return;
  const link = element('a');
  link.href = `/api/export/${encodeURIComponent(result.run_id)}`;
  link.download = `meme-${result.action}-${result.run_id}.json`;
  document.body.append(link);
  link.click();
  link.remove();
});
window.addEventListener('beforeunload', event => { if (running) { event.preventDefault(); event.returnValue = ''; } });
switchTab(false);
syncButtons();
loadState(true).catch(error => { notice(error.message, true); syncButtons(); });
