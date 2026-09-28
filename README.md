# Qwen3-8B 模因实验 PoC

这是一个本地 Python 小实验：给定模因和场景，生成多条表达，按规则评分，再对入选表达做变异并重新选择。先用 mock 检查流程，再用真实模型观察质量。支持命令行与本地浏览器交互页面。

## 本地交互页面（新增）

双击项目中的 `run_web.cmd`，或在 PowerShell 执行：

```powershell
cd D:\模因
python web_app.py --open
```

打开 [模因实验台](http://127.0.0.1:8765)。启动窗口需要保持运行；关闭窗口或按 Ctrl+C 即可停止。页面只在本机提供服务，不需要 Node.js 或新增 Python 依赖。端口占用时可用 `python web_app.py --port 8766 --open`。

页面使用原来的 `data/memes.db`，已有数据会保留。也可以用 `python web_app.py --db data/web-experiment.db --open` 创建独立实验空间，在页面点击“初始化示例模因”。

使用步骤：

1. 默认选择“模拟实验”，选一个模因并输入场景，也可点场景快捷按钮。
2. 点击“生成候选”，查看按总分排序的表达与五项分数。
3. 点击“进化一轮”，对同场景幸存者继续变异；“候选记录”可查看原候选和变体。
4. 可复制表达，或将本次结果导出为 JSON；记录按所选模因展示，最多显示最近 60 条。本地服务临时保留最近 20 次运行的 JSON 下载，重启服务后下载链接失效，数据库中的候选仍然保留。
5. 在运行模式中选择“真实模型”，按本文安装依赖，按需开启“允许首次下载模型权重”。

“调整更多参数”可设置温度、Top P、输出长度和变体数量。真实模型会留在内存中复用，改变温度无需重复加载；加载成功后若要切换设备或量化方式，请重启服务。mock 与真实模式共用当前数据库，要做独立对照实验请使用不同 `--db`。历史记录为原有候选数据，不推断每条历史表达的生成模式；本次结果明确标注模式。

生成期间会显示等待时间并禁用重复提交。关闭页面不会取消已经开始的模型任务；若意外刷新，可以在任务完成后查看候选记录。本地页面适用于单人实验，不是公网部署服务。

新增文件：`web_app.py`（本地接口）、`web/`（页面、样式和交互）、`run_web.cmd`（双击启动）、`tests/test_web.py`（HTTP 流程测试）。修改前的 README 和配置已保存在 `backups/before-web/`。

## 1. 环境与安装

建议 Windows 10/11、Python 3.11（本次实际测试版本 3.11.9）。mock 不需要显卡、模型或联网。打开 PowerShell：

```powershell
cd D:\模因
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-mock.txt
```

可以激活虚拟环境后使用下文的 `python`；如果 PowerShell 阻止激活，直接把 `python` 换成 `.\.venv\Scripts\python.exe` 即可，无需修改执行策略。

```powershell
.\.venv\Scripts\Activate.ps1
```

## 2. 先跑 mock 完整流程

```powershell
python main.py init
python main.py generate --meme "打工人" --context "老板说今天大家自愿加班" --mock
python main.py evolve --meme "打工人" --context "老板说今天大家自愿加班" --mock
python main.py list
python -m pytest -q
```

`init` 创建 `data/memes.db` 并添加两个示例；重复执行不会覆盖同名已有模因。`generate` 输出全部候选及评分；`evolve` 输出最终 top-k。两者都会保存完整候选池，失败时不会写入半轮进化结果。mock 使用确定性模板，重复运行相同输入会产生重复惩罚，这是预期行为。

可用独立数据库做新实验：每条命令均支持在子命令后加 `--db data/experiment.db`。路径默认基于项目目录；显式传入相对路径则相对于当前工作目录。

## 3. 真实 Qwen3-8B

```powershell
python -m pip install -r requirements.txt
python main.py generate --meme "打工人" --context "老板说今天大家自愿加班" --allow-download
python main.py evolve --meme "打工人" --context "老板说今天大家自愿加班"
```

官方模型为 [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)。项目按官方 chat template 调用，使用 `enable_thinking=False`；依赖要求 Transformers ≥4.51.0，此 PoC 限定 4.x API。

首次 `--allow-download` 会从 Hugging Face 下载较大的模型文件，需要可访问该站点及足够磁盘空间。默认不联网下载，仅使用缓存；缓存缺失时会提示使用 mock、允许下载或指定本地模型目录。

```powershell
python main.py generate --meme "打工人" --context "周一开会" --model "D:\models\Qwen3-8B" --device auto
```

真实模型内存需求远高于 mock：仅 8.2B 参数的 16 位权重约 16.4GB，实际还需要运行缓存和额外空间，不能把权重大小当作最低显存。CPU float32 权重本身约 32.8GB，实际需要更多内存且速度较慢。默认 `--device auto` 使用 Accelerate 分配，优先可用 GPU，放不下时可分配到 CPU；这不保证任意机器都能运行。

若没有 CUDA，auto 会提示并使用 CPU。`--device cuda` 强制显卡，缺少 CUDA 时明确报错。PyTorch 的 CUDA 版本需与机器情况匹配，安装说明见 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/)。可检查：

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

可选量化（需要兼容的 CUDA 与 bitsandbytes 环境，未在本次验收中运行）：

```powershell
python -m pip install bitsandbytes
python main.py generate --meme "打工人" --context "周一开会" --quantization 4bit
```

## 4. 常用参数

```powershell
python main.py generate --meme "打工人" --context "工资到账又交房租" --mock --num-candidates 6 --temperature 0.7 --top-p 0.8 --max-new-tokens 160 --strategy "使用自嘲和反转"
python main.py evolve --meme "打工人" --context "开会到下班" --mock --top-k 2 --mutations-per-parent 2
python main.py generate --help
```

所有默认值和评分权重集中在 `config.py`。mock 接收同样的接口参数，但模板不模拟采样温度、策略或模型质量。真实生成逐条执行以节省显存；超长输入会提示缩短，不会静默截掉关键素材。

## 5. 四个模块及数据含义

| 文件 | 作用 |
| --- | --- |
| meme_store.py | Meme / Candidate 数据对象，SQLite 表和 JSON 字段 |
| generator.py、prompts.py | 官方 Qwen3、mock 及提示词 |
| ranker.py | 规则评分与可替换 Ranker 接口 |
| evolution.py | 同场景幸存者（首次则新生成）→ top-k → 生成变体 → 原候选和变体联合选 top-k |
| main.py、config.py | 命令行入口与集中配置 |
| train_stub.py、data/ | 训练数据格式、示例、格式校验骨架 |
| tests/ | 自动测试，使用临时数据库，不改真实实验数据 |

Meme 包含 `meme_id/name/semantic_core/style/contexts/templates/examples/parents/children/lifecycle/embedding`。embedding 默认空。首版变异的是“表达候选”，不是新建模因身份，所以 Meme 的 parents/children 预留为空；实际谱系存于 Candidate 的 parent_id。

候选 `generation` 在变异时为父代 +1；首次 evolve 根候选从 0 开始，单独 generate 的根候选采用模因当前最高代次，parent_id 为空。`fitness` 是最近一次参与选择时的分数，`usage` 为最终入选次数，`age` 为此候选保存后经历的后续进化轮数，`selected` 表示该场景最近一次运行是否入选。相同场景再次 evolve 会复用这些幸存者，其他历史候选用于重复检测。再次 generate 会为该场景重新生成候选池。模因 lifecycle 的 generation 为已生成候选最高代次、fitness 为最近运行最佳入选分、usage 为成功 generate/evolve 次数、age 为成功 evolve 次数。

评分均为 0～1：

- context_fit：正文与用户场景的字符二元组覆盖率。
- novelty：1 减去与例句、历史及其他候选的最高文本相似度。
- replicability：简短程度和可复用句式标记的平均值。
- duplication：上述最高文本相似度，分数越高惩罚越大。
- risk：是否命中配置中的风险词，命中为 1。

总分 = `0.45*context_fit + 0.25*novelty + 0.30*replicability - 0.35*duplication - 1.0*risk`，限制在 0～1。风险检测只是少量词匹配；没有硬过滤保证，也不具备完整语义识别能力。重复场景原文可得到较高匹配分，简短句不等于有传播性，因此这些分数只用作可解释的实验代理指标，不能预测真实传播效果。

## 6. 以后接入 SFT / DPO

```powershell
python train_stub.py --kind sft
python train_stub.py --kind dpo
python train_stub.py --kind sft --data data/meme_sft.jsonl
```

`meme_sft.jsonl` 每行包含 `meme_id` 和 `messages`（system/user/assistant）。`preference_pairs.jsonl` 每行包含 `prompt/chosen/rejected` 消息列表及来源。这些只是格式示例，不是可用于训练的充分数据。

骨架只做格式校验，不导入训练框架、不更新模型。以后先收集并人工审核场景和优质回答，按模因/时间拆分训练与验证集；复用 Qwen chat template 接入 SFTTrainer。再收集相同 prompt 的真实偏好对接入 DPOTrainer。可以训练一个通用模因适配器，不需要每个模因一个 LoRA。替换规则 Ranker 时，实现 `ranker.Ranker.score()` 并在 main 中注入即可。

## 7. 已做与未做

已做：结构化模因库、官方模型加载接口、可解释评分、候选谱系持久化、最小进化、CLI、mock、数据校验和测试。

未做：真实传播反馈、奖励模型训练、SFT/DPO 训练、PPO、复制子方程、多模态、向量检索、自动新模因发现、复杂种群管理、Docker 或分布式服务。

本次验收使用 mock；真实 Qwen3-8B 权重未下载，真实推理及量化未验证。真实模式可能因网络、驱动、依赖版本或内存不足需要调整环境。

常见错误：未初始化先执行 `init`；模因不存在用 `list` 查看；权重缺失加 `--allow-download`；显存不足用 auto、减少 token 或尝试可选量化。输出中文乱码时，可先在 PowerShell 执行 `$env:PYTHONIOENCODING='utf-8'`。
