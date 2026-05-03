# claude-autoresearch 通用批量跑操作手册

用 claude-autoresearch 自带的 batch 脚本对一批 `(ref.py, kernel.py?)` 任务跑 `/autoresearch`，**全自动模式**。
脚本在 [claude-autoresearch/.autoresearch/scripts/batch/](../claude-autoresearch/.autoresearch/scripts/batch/)。

> 跟 [triton-cuda-to-ascend-via-autoresearch.md](./triton-cuda-to-ascend-via-autoresearch.md) 的关系：那一份是针对 sglang→ascend 迁移的特化驱动器（带 .pt cache、PyTorch fallback adapter、verify_seed），写死在 akg-hitl。
> 这一份是**通用**的：任意 DSL、任意 op 列表，只要你能给一对 ref/kernel 文件就能跑。

---

## 一键跑全部

> **约定**：
> 1. 已经 SSH 到目标机器（远端长跑场景请用 tmux 或 screen，理由见后文「历史踩坑」#15）
> 2. 已经 `cd` 进 claude-autoresearch repo 根目录（所有 `.autoresearch/scripts/*` 走相对路径）
> 3. 下文示例硬件参数：`--backend ascend --arch ascend910b3 --devices 0`、worker `--port 9111`。**这只是某一台 NPU 机器的示例值**，请按你的目标机型替换 `worker --start` 命令里的 `--backend / --arch / --devices / --port`（CUDA 机器换 `--backend cuda --arch <sm>`，端口同理）
> 4. 用 `<repo>/workspace/` 里现成的所有验证过的 sample 作为输入。想跑自己的 op：把符合 `<op_name>_ref.py` / `<op_name>_kernel.py` 命名约定的文件 cp 进 `$WS/refs` 和 `$WS/kernels`，`discover.py` 自动找出来填进 manifest。

下面的脚本可以从头到尾直接复制粘贴跑：

```bash
# ─── 一次性约定 ──────────────────────────────────────────────────────────
WS=<batch_dir>                       # 替换为你想用的路径（例如 /tmp/batch_001），脚本会自动建子目录
# 前提（见上文说明）：cwd 在 claude-autoresearch repo 根目录、Python / claude CLI / NPU 环境已激活

# ─── 1. 准备 workspace ──────────────────────────────────────────────────
#       这里把 <repo>/workspace/ 里 sample 整套拷过去；
#       想跑自己的，把 ref/kernel 文件按命名约定拷进 $WS/refs / $WS/kernels 即可。
mkdir -p $WS/refs $WS/kernels
cp workspace/*_ref.py    $WS/refs/
cp workspace/*_kernel.py $WS/kernels/

# ─── 2. 自动发现 op，bootstrap manifest ──────────────────────────────────
#       扫 $WS/refs/<op>_ref.py + $WS/kernels/<op>_kernel.py，
#       配对成功的 op 自动写进 $WS/manifest.yaml。
python .autoresearch/scripts/batch/discover.py $WS \
    --mode ref-kernel --dsl triton_ascend --write-manifest

# ─── 3. 预检 Tier 1（秒级，不需要 worker daemon） ─────────────────────────
python .autoresearch/scripts/batch/verify.py $WS
#  全 PASS 才进下一步；FAIL/ERROR 就修了再 verify

# ─── 4. 起 worker daemon（持久跑，整批 op 共用） ──────────────────────────
python .autoresearch/scripts/ar_cli.py worker --start \
    --backend ascend --arch ascend910b3 --devices 0 \
    --host 127.0.0.1 --port 9111 --bg
curl -s --noproxy '*' http://127.0.0.1:9111/api/v1/status   # 应返回 {"status":"ready",...}

# ─── 5. (可选) 预检 Tier 2：用 worker 实跑 ref vs kernel ──────────────────
python .autoresearch/scripts/batch/verify.py $WS --full

# ─── 6. 后台跑批量（tmux daemon，理由见踩坑 #15） ────────────────────────
tmux new -d -s ar_batch \
    "python -u .autoresearch/scripts/batch/run.py $WS --mode ref-kernel --dsl triton_ascend"

# ─── 7. 另开 SSH 终端实时监控 ────────────────────────────────────────────
python .autoresearch/scripts/batch/monitor.py $WS --watch -n 10

# ─── 8. 跑完离线汇总 ────────────────────────────────────────────────────
python .autoresearch/scripts/batch/summarize.py $WS
```

完事。所有发现的 op 顺序执行，每个 30-60 分钟（`--max-rounds 30` 默认值）。

> 想跑别的 op：往 `$WS/refs/` 和 `$WS/kernels/` 里 cp 自己的文件（严格遵守 `<op>_ref.py` / `<op>_kernel.py` 命名），然后**重跑 step 2** 的 `discover.py --write-manifest` 就把新增的 op merge 进 manifest 了（已存在的 op 列表整体被新扫描结果替换）。
>
> 想筛选：`discover.py $WS --filter '*norm' --exclude 'foo'` 先看输出，确认无误再加 `--write-manifest`。

**为什么用 worker daemon 不用 `--devices`：** local 模式下每个 baseline.py 起一个 eval 子进程，多个 op 在同一卡上抢资源 → 互相 hang。worker daemon 持久占设备，所有 op 串行提交到它，干净不抢。

> ⚠️ 不传 `--devices` 也不传 `--worker-url` 时，runner 默认用 `--worker-url 127.0.0.1:9111` 并**启动时强制 health check**。daemon 没起来会立即报错并打印怎么起，不会埋在第一个 op 几千行日志里才炸。

---

## 两种模式

| Mode | 用法 | `/autoresearch` 行为 |
|---|---|---|
| **`--mode ref-kernel`** | 每个 op 已有**验证过**的 seed kernel | scaffold `--run-baseline` 跑 seed → baseline PASS → phase 直接 `PLAN` → 30 轮全花在性能优化 |
| **`--mode ref`** | 每个 op 只有 reference 实现 | 走完整 pipeline，先 `GENERATE_KERNEL` 从 0 生成 kernel，过 baseline 后再优化 |

整批必须**统一一种模式**（脚本级 `--mode` flag）。混用会让 summary 里的 speedup 分布失真（`ref-kernel` 模式专攻性能优化，`ref` 模式还包含从无到有的 kernel 生成时间）。

---

## Workspace 约定

```
<workspace_dir>/                  ← 传给 run.py 和 monitor.py 的位置参数
  manifest.yaml                   # 用户手写（也支持 manifest.json）
  batch_progress.json             # runner 自动写：每个 op 的 status / task_dir / metrics
  batch.log                       # runner 自动写：tee 的 claude --print 全部 stdout
  refs/                           # manifest 里的 ref_dir
    <op_name>_ref.py              # ⚠️ 文件名必须严格遵守
  kernels/                        # manifest 里的 kernel_dir（仅 ref-kernel 模式需要）
    <op_name>_kernel.py           # ⚠️ 文件名必须严格遵守
```

`ref_dir` / `kernel_dir` 在 manifest 里是**相对 workspace 目录**的路径。也支持绝对路径，但建议保持子目录习惯，方便整体打包/迁移。

**文件名约定是强制的**。`run.py` 启动时会做 pre-flight 校验：

| 校验项 | 报错示例 |
|---|---|
| workspace 目录不存在 | `workspace dir not found: <path>` |
| manifest 缺失 | `no manifest.yaml or manifest.json in <workspace_dir>` |
| `ref_dir` / `kernel_dir` 缺失 | `kernel_dir required when mode=ref-kernel` |
| op 文件按约定拼路径找不到 | `refs\<op_name>_ref.py not found` |
| op_name 重复 | `duplicate op_name: <op_name>` |
| `--devices` 和 `--worker-url` 都传了 | `--devices and --worker-url are mutually exclusive` |
| worker daemon 不通 | `worker daemon at 127.0.0.1:9111 is unreachable... start it first: ...` |

任何一项失败 → 立即退出，**不会**进队列开始跑。

### 不想手写 ops 列表：`discover.py`

把 ref / kernel 文件按命名约定 cp 进 `$WS/refs` 和 `$WS/kernels` 后，让 `discover.py` 扫一遍把配对成功的 op 写进 manifest：

```bash
# 第一次创建 manifest（必须传 --mode --dsl，写到 $WS/manifest.yaml；
# 没装 pyyaml 时退化到 manifest.json）：
python .autoresearch/scripts/batch/discover.py $WS \
    --mode ref-kernel --dsl triton_ascend --write-manifest

# 加 / 删了 ref/kernel 文件后，重新同步 ops 列表（沿用 manifest 里已有的 mode/dsl/dirs）：
python .autoresearch/scripts/batch/discover.py $WS --write-manifest

# 不写文件，只看会发现哪些 op：
python .autoresearch/scripts/batch/discover.py $WS                       # 一行一个
python .autoresearch/scripts/batch/discover.py $WS --json                # JSON 数组
python .autoresearch/scripts/batch/discover.py $WS --filter '*norm'      # 子集（保留）
python .autoresearch/scripts/batch/discover.py $WS --exclude 'foo*'      # 子集（排除，可重复）
```

`ref-kernel` 模式下，配对失败的 op（只有 ref 没 kernel、或反过来）会以 warning 打到 stderr，不会进 manifest。

---

## 预检（verify.py）

`run.py` 的 pre-flight 只检查**文件存在**。但文件存在不等于文件能跑 —— kernel 可能 import 缺包、可能没 `class ModelNew`、可能 ref 的 `get_inputs()` 抛异常。这种错误在 `claude --print` 里发现 = 浪费 30 分钟。

`verify.py` 在调 `run.py` 之前把这些筛掉。两档：

### Tier 1（默认，**不需要硬件**，秒级）

每个 op 在独立 subprocess 里：
1. ref/kernel 文件的 Python 语法编译过
2. import 模块能成功（缺依赖、import 错误立即暴露）
3. 模块有期望的 export：
   - `ref.py` 必须有 `Model`、`get_inputs`、`get_init_inputs`
   - `kernel.py` 必须有 `ModelNew`

```bash
python .autoresearch/scripts/batch/verify.py <batch_dir>
```

输出（subset，假设跑 10 个 op）：

```
verify  workspace=<batch_dir>  mode=ref-kernel  tier=1  ops=10

  [  1/10] batchnorm  ... P
  [  2/10] groupnorm  ... P
  [  3/10] layernorm  ... E
  [  4/10] rmsnorm    ... P
  ...

  op         t1_ref  t1_kern  t2      ok   note
  ---------  ------  -------  ------  ---  --------------------------------
  batchnorm  PASS    PASS     -       P
  groupnorm  PASS    PASS     -       P
  layernorm  PASS    FAIL     -       E    ModuleNotFoundError: No module named 'triton'
  rmsnorm    PASS    PASS     -       P
  ...

  total=10  pass=8  fail=0  error=2  elapsed=12.3s
  results: <batch_dir>/verify_results.json
```

`P/F/E` 总览列：
- **P** = pass，所有 tier 都过
- **F** = fail，结构性错误（语法挂、缺 `class ModelNew` 等）
- **E** = error，环境/运行时错误（缺 `triton`、import 时抛异常等）

退出码 0=全过，1=有任何 fail/error。CI 友好。

### Tier 2（`--full`，需要硬件）

仅在 `--mode ref-kernel` 时有意义（`--mode ref` 没 kernel 可比）。每个 op 在独立 subprocess 里：
1. 加载 ref + kernel
2. 跑 `ref(*get_inputs())` 和 `kernel(*get_inputs())`
3. `torch.allclose` 比对（atol/rtol 默认 1e-2，与 autoresearch 实跑同款；可覆盖见下文「精度容差」）

```bash
python .autoresearch/scripts/batch/verify.py <batch_dir> --full
```

t1 全过的才会跑 t2；t1 任何一项 FAIL/ERROR 时跳过 t2。t2 列额外可能值：
- **PASS** = 数值等价
- **FAIL** = 数值不等（note 里给 `max_abs_diff` + 越界元素数）
- **ERROR** = 跑挂了（构造异常、forward 异常、超时等）

为什么 Tier 2 仍然有用：scaffold `--run-baseline` 也会做一遍同样的事。区别在于
1. verify.py 跑完整批 5 分钟，scaffold 一个 op 走完整 claude --print 30 分钟
2. verify.py 失败时**只**告诉你哪个 op、哪一步挂；scaffold 失败 → claude 会试着自己修，多浪费几轮

### `--only` 子集

Debug 单个 op：

```bash
python .autoresearch/scripts/batch/verify.py <batch_dir> --only layernorm
python .autoresearch/scripts/batch/verify.py <batch_dir> --only layernorm --full
```

### 输出文件

每次跑都覆盖写 `<workspace>/verify_results.json`，包含每个 op 的所有 tier 结果（含 traceback 末段、`max_abs_diff`、`elapsed_s` 等）。CI 可解析这个 JSON。

### 精度容差（与 autoresearch 实跑对齐）

verify.py Tier 2 和 autoresearch 跑 `verify_<op>.py` 用的是同一个比较函数 [`.autoresearch/scripts/correctness.py`](.autoresearch/scripts/correctness.py)（同 dtype 处理、`equal_nan=False`），所以**当 atol/rtol 相同时**（默认两侧都是 `1e-2 / 1e-2`），verify Tier 2 PASS = autoresearch 实跑也会 PASS。

⚠️ **当 atol/rtol 不同时不成立**：当前 batch [run.py](.autoresearch/scripts/batch/run.py) **不会**把 manifest 里的 `correctness_atol/rtol` 透传到 `/autoresearch` 命令（详见下文「容差解析顺序」末尾的注）。所以 manifest 里写了 `1e-3` 等更严容差只影响 verify.py 自己；autoresearch 实跑仍按默认 `1e-2`。如果你既要 verify 严，也要 autoresearch 严，得手动 `claude` 进交互模式 + 给 `/autoresearch` 单独传同一对 `--correctness-atol / --correctness-rtol`。

容差解析顺序：

| 来源 | 字段 | 优先级 |
|---|---|---|
| `verify.py --correctness-atol / --correctness-rtol` | CLI flag | 最高 |
| `<workspace>/manifest.{yaml,json}` 顶层 `correctness_atol / correctness_rtol` | manifest | 中 |
| 默认 `1e-2 / 1e-2` | hard-coded | 兜底（与 autoresearch loader 默认一致） |

`/autoresearch` 这一侧也新增了 `--correctness-atol / --correctness-rtol`（默认 1e-2），scaffold 会把值写进 `task.yaml.metric.correctness_atol/rtol`，eval 包里的 `verify_<op>.py` 通过同一个 [`correctness.py`](.autoresearch/scripts/correctness.py) 消费。

整批想跑严一点的最干净写法是写进 manifest（verify 和后续 batch run 都会读到）：

```yaml
# <workspace>/manifest.yaml
mode: ref-kernel
dsl: triton_ascend
ref_dir: refs
kernel_dir: kernels
correctness_atol: 1.0e-3
correctness_rtol: 1.0e-3
ops:
  - op1
  - op2
```

> 注：当前 batch [run.py](.autoresearch/scripts/batch/run.py) 还没把 manifest 里的 atol/rtol 透传到 `/autoresearch` 命令里 —— 想让 autoresearch 实跑也用更严的容差，目前只能手动 `claude` 进交互模式 + 单独传 `--correctness-atol/--correctness-rtol` 给 `/autoresearch`。这是后续可以补的一步透传，不影响 verify 自身的对齐。

仅临时调试 verify 时，CLI 覆盖更方便：

```bash
python .autoresearch/scripts/batch/verify.py <batch_dir> --full \
    --correctness-atol 1e-3 --correctness-rtol 1e-3
```

verify.py 启动时会打印 `tols: atol=… rtol=…` 一行，把实际生效的值告诉你；同样写进 `verify_results.json` 顶层。

---

## 自动化边界 —— 哪些步骤被自动化了？

```
你做的事：                              脚本做的事：
────────────────────────────────────────────────────────────────────
mkdir <workspace>/refs /kernels
cp ref/kernel 文件按命名约定放进去

discover.py <workspace>                 ┌─ 扫 ref_dir / kernel_dir
   --write-manifest                     ├─ 配对 <op>_ref.py + <op>_kernel.py
                                        └─ 写 / 更新 manifest.yaml 的 ops 列表

verify.py <workspace>                   ┌─ 每个 op subprocess 隔离
（推荐 Tier 1）                         ├─ Tier 1: compile / import / 必备 export 检查
                                        └─ 输出表格 + verify_results.json

（推荐）起 worker daemon

verify.py <workspace> --full            ┌─ Tier 2: 加载 ref + kernel
（可选 Tier 2）                         ├─ ref(*inputs) vs kernel(*inputs)
                                        └─ torch.allclose（atol/rtol 同 task.yaml；调 .autoresearch/scripts/correctness.py 公共模块）

run.py <workspace_dir>                  ┌─ load + validate manifest
   --mode ref-kernel                    ├─ pre-flight 检查所有 ref/kernel 文件
                                        ├─ health check worker
                                        ├─ merge 到 batch_progress.json (新 op = pending)
                                        │
                                        │  for each pending op:
                                        │    ┌─ 起 headless `claude --print`
                                        │    │  在 claude-autoresearch repo cwd 下
                                        │    │
                                        │    │  prompt 内容（自动生成）：
                                        │    │  - /autoresearch --ref ... [--kernel ...]
                                        │    │    --op-name ... --dsl ... --worker-url ...
                                        │    │  - Non-interactive contract（A/B/C/D 四节）
                                        │    │  - 强调："scaffold 后立刻 export AR_TASK_DIR"
                                        │    │  - 强调："follow hooks，不停问"
                                        │    │  - 强调："最后打 AUTORESEARCH_RESULT 行"
                                        │    │
                                        │    │  Claude 在那个 session 里自动：
                                        │    │    - scaffold task_dir
                                        │    │    - export AR_TASK_DIR
                                        │    │    - 模式 1：BASELINE PASS → 直接 PLAN
                                        │    │      模式 2：GENERATE_KERNEL 先
                                        │    │    - PLAN → EDIT → VERIFY 循环 ≤max-rounds
                                        │    │    - FINISH
                                        │    │    - 打 AUTORESEARCH_RESULT 标记
                                        │    │
                                        │    ├─ stdout 实时 stream 到 batch.log
                                        │    ├─ 解析 marker 拿 task_dir + phase
                                        │    │  （拿不到就 fallback 扫 ar_tasks/）
                                        │    └─ 自动更新 batch_progress.json
                                        │       从 task_dir 抽 baseline_metric / best_metric
                                        │
                                        │    rc != 0 不会停批量，下一个继续
                                        │
                                        └─ 打总结（done/error/skip 计数 + speedup 分布）

monitor.py <workspace_dir>             ┌─ 读 batch_progress.json + 当前 active task
   --watch                             └─ 队列 / phase / metrics / speedup / errored 列表

monitor.py <workspace_dir>             ┌─ exec autoresearch 自带的 dashboard.py
   --dashboard                         └─ 全 TUI 看 active task 的 plan / history / phase

summarize.py <workspace_dir>           ┌─ 静态读 batch_progress.json
                                        └─ 状态计数 / speedup 分布 / regressions / 错误列表
                                          （不看 ar_tasks/，跑完后离线 review 用）
```

**人需要做的就三件事：**
1. 准备 workspace（mkdir + cp + 写 manifest）。一次性。
2. 起 worker daemon。一次性。
3. 起 `run.py`。一次性。

之后纯看戏。

---

## 监控

跑批量时另开终端看进度。**互不干扰，纯只读**。

### 主推：`monitor.py --watch`

```bash
cd /path/to/claude-autoresearch
python .autoresearch/scripts/batch/monitor.py <batch_dir> --watch -n 10
```

输出：

```
━━━ batch monitor  2026-04-30 22:42:13 ━━━
workspace  <batch_dir>
mode=ref-kernel  dsl=triton_ascend

queue   total= 10  done=  4  error=  1  skip=  0  pending=  4  running=  1
        [████▶▒    ]

active  groupnorm_1714485678_a8f3c2
        phase=EDIT  rounds=12/30  failures=1  plan_v=2  status=in_progress
        baseline=18.421  best=14.012  speedup=1.31x
        heartbeat: 4s ago

        history (last 3 rounds):
          R10 keep    latency_us=1023  correct=true  vectorize block_n
          R11 discard latency_us=1156  correct=true  reorder loops
          R12 keep    latency_us=892   correct=true  fuse epilogue

        plan.md head:
          ## P-001  block size sweep  (status: done)
          ## P-002  vectorize over block_n  (status: done)
          ## P-003  fuse normalization epilogue  (status: in-progress)

batch.log (last 6 lines):
  [pipeline] round 12 keep
  [phase] PLAN -> EDIT
  ...

done speedup  median=1.42x  best=2.18x  worst=0.93x  (n=4)
              improved=3  on-par=0  regress=1

errored ops (1):
  - foo_kernel: phase=GENERATE_KERNEL status=stuck rc=0

(refresh every 10s; Ctrl-C to stop  |  full TUI: monitor.py --dashboard)
```

### 钻进当前 op 看细节：`--dashboard`

```bash
python .autoresearch/scripts/batch/monitor.py <batch_dir> --dashboard
```

`execvp` 进 claude-autoresearch 自带的 [`dashboard.py`](../claude-autoresearch/.autoresearch/scripts/dashboard.py) —— 完整 TUI、方向键导航、看 plan.md 全文、history.jsonl 全部记录、phase machine 状态。

可显式指定 task：
```bash
python .autoresearch/scripts/batch/monitor.py <batch_dir> --dashboard \
    --task-dir /path/to/claude-autoresearch/ar_tasks/<op>_<ts>_<uuid>
```

### 看 Claude 实时输出（最详细）

```bash
tail -f <batch_dir>/batch.log
```

看到每个 op：Claude 跑的 bash / Edit / Write、hook 输出（`[AR Phase: ...]`、`[AR] kernel.py invalid`）、run.py 的 `[run] result: op=... task_dir=... phase=FINISH` 总结。

### 看某一个具体 op 的内部状态

```bash
TASK=$(ls -td /path/to/claude-autoresearch/ar_tasks/*/ | head -1)
cat $TASK/.ar_state/.phase                 # 当前 phase
cat $TASK/.ar_state/progress.json          # rounds、metrics、failures
ls  $TASK/.ar_state/                       # plan.md、history.jsonl 等
cat $TASK/kernel.py                        # 当前最佳 kernel
```

### 看 batch 主进程是否还活着

```bash
tmux ls | grep -q ar_batch && echo ALIVE || echo DEAD
tmux attach -t ar_batch                    # 进 tmux 看实时屏幕（Ctrl-b d 脱离）
```

### 汇总报告（跑完后 / 离线 review）：`summarize.py`

跟 `monitor.py` 互补：

| | `monitor.py` | `summarize.py` |
|---|---|---|
| 数据源 | progress JSON **+** ar_tasks/ 实时状态 | progress JSON **only**（静态） |
| 看的是 | 此刻在跑什么 | batch 跑完后回顾 |
| 包含 active task / heartbeat / log tail？ | 是 | 否 |
| 复制粘贴友好（chat / ticket）？ | 一般 | 是 |

```bash
python .autoresearch/scripts/batch/summarize.py <batch_dir>
```

输出：

```
batch summary  (2026-04-30T23:10:11)
workspace  <batch_dir>
mode=ref-kernel  dsl=triton_ascend
────────────────────────────────────────────────────────────
  total:    10
  done    : 7
  error   : 2
  pending : 1

speedup (baseline / best, higher better):
  ops with metric: 7
  median:          1.42x
  best:            2.18x
  worst:           0.93x
  improved:        6  (>1.05x)
  on-par:          0    (0.95-1.05x)
  regress:         1     (<0.95x)

regressions (1 ops slower than baseline):
  - groupnorm: baseline 5.234 -> best 5.567  (0.94x)

errored ops (2):
  - softmax: phase=GENERATE_KERNEL  phase=GENERATE_KERNEL status=stuck rc=1
  - foo_op:  phase=EDIT             phase=EDIT max-rounds exhausted

still pending: 1
  - layernorm
```

跑完整批后用它出"今天 batch 跑了什么"的报告：贴 chat 给同事、贴 ticket、写日报。比 `monitor.py` 干净，不会带 `[refresh every Ns; Ctrl-C to stop]` 这种边角文字。

---

## 断点续跑

**先记住一条总规律：**

- 已 `done` 的不会重跑（完成时 metric 已写 `batch_progress.json`）
- 已 `error` 的默认跳过；想重试 → `--retry-errored`
- `pending` 的会被 `run.py` 自动续上
- `running` 的会在下次 `run.py` 启动时**自动**降级为 `error`（note 标 `stale running, demoted on batch restart`），随后当 error 处理 —— 想重跑加 `--retry-errored` 即可，不需要手工改 JSON

按"断到什么程度"分四档：

### 档 A — 终端断了，但 tmux 里的 batch 进程还活着（最常见）

**啥都不用做。** 重连，`tmux ls` 看到 `ar_batch` → `tmux attach -t ar_batch` 看实时屏幕，或 `tail -f <batch_dir>/batch.log`。

### 档 B — batch 主进程被杀了（机器重启 / `tmux kill-session` / 用裸 nohup 没用 tmux）

```bash
# 看现在啥状态
python .autoresearch/scripts/batch/monitor.py <batch_dir>

# 把 ar_tasks 里的孤儿清一下（可选；被杀那瞬间在跑的 op 留下了半成品 task_dir）
rm -rf /path/to/claude-autoresearch/ar_tasks/*

# 重启 batch；done 自动跳过、pending 续上
tmux new -d -s ar_batch \
    'python -u .autoresearch/scripts/batch/run.py <batch_dir> --mode ref-kernel --dsl triton_ascend'
```

被杀那瞬间在跑的 op 在 progress 里大概率仍是 `running`（因为没机会更新到 done/error）。`run.py` 启动时持有 workspace 锁的同时会扫一遍 progress，把所有 `running` 一律降级为 `error`（note 写 `stale running, demoted on batch restart`），所以**直接重起 batch + `--retry-errored` 就能把它捞回来**：

```bash
tmux new -d -s ar_batch \
    'python -u .autoresearch/scripts/batch/run.py <batch_dir> --mode ref-kernel --dsl triton_ascend --retry-errored'
```

> ⚠️ 同一 workspace 不能同时跑两个 `run.py`：第二个会因 `<workspace>/.batch.lock` 被前者占住而立即 `sys.exit`。死进程留下的 stale lock 会在下次启动时被自动判活并清理。

### 档 C — 某个 op 跑挂了 / Claude 进程崩了 / 单 op wall-clock 超时

**自动处理。** `run.py` 探测到 `claude` rc != 0、phase != FINISH、或者 `--timeout-min` 超时时，自动把那个 op 标 `error` 并写错误 note，下一个继续。

跑完整批之后想把 errored 的捞回来重试一遍：

```bash
tmux new -d -s ar_batch \
    'python -u .autoresearch/scripts/batch/run.py <batch_dir> --mode ref-kernel --dsl triton_ascend --retry-errored'
```

### 档 D — 想从 autoresearch 自己的 round 进度续上（不重跑已经做过的轮）

`run.py` 当前每次都让 Claude **新建** task_dir。"省下已跑的轮数"目前不支持，重试 = 整个 op 从 0 重新来。要用 autoresearch 自带的 round-level resume，手动接管最简单：

```bash
# 找到那个 op 最新的 task_dir
ls -td /path/to/claude-autoresearch/ar_tasks/<op>_*/ | head -1

# 在另一个终端手动跑
cd /path/to/claude-autoresearch
claude
# /autoresearch --resume <task_dir>
# 跑完后手动改 batch_progress.json 把那个 op status 改 done + 填 task_dir
```

但 30 轮一般也就 30-60 分钟，重跑成本通常比维护 resume 机制低 —— 直接 `--retry-errored`。

---

## 手动逐个跑（不用 batch）

适用场景：想盯着每个 op 的实际过程、想中途介入 Claude 的决策、跑 batch 时某个 op 有奇怪问题想单独 debug。

> 这个 batch 脚本**没有** akg-hitl 那一套的 `next_op.py`（手动模式专用 helper）。直接用 `--only` + `--limit` 单跑某个 op 就够了；想要更深的介入就直接进 `claude` 交互手动粘 `/autoresearch`。

### 方案 1：用 batch 脚本跑单个 op

```bash
# 只跑某一个 op（无视其他）
python .autoresearch/scripts/batch/run.py <batch_dir> \
    --mode ref-kernel --dsl triton_ascend \
    --only <op_name>
```

仍然走 headless `claude --print`、自动 record。如果跑挂了：

```bash
# 重试那一个
python .autoresearch/scripts/batch/run.py <batch_dir> \
    --mode ref-kernel --dsl triton_ascend \
    --only <op_name> --retry-errored
```

### 方案 2：完全手动进交互 claude

适合调试 / 想看每一步 / 想中途打断让 Claude 试某个具体改动。

**注意 cwd 必须是 claude-autoresearch repo**，否则 hooks、settings.json、`.autoresearch/scripts/*` 都找不到。

```bash
cd /path/to/claude-autoresearch
claude
```

进 Claude 之后：

1. 粘 `/autoresearch --ref $WS/refs/<op>_ref.py --kernel $WS/kernels/<op>_kernel.py --op-name <op> --dsl triton_ascend --worker-url 127.0.0.1:9111 --max-rounds 30 --eval-timeout 120`（把 `<op>` 替换成你要跑的 op 名）
2. scaffold 末尾会打 `Task directory created: /path/to/claude-autoresearch/ar_tasks/<op>_<ts>_<uuid>` —— **立刻**让 Claude 跑：
   ```bash
   export AR_TASK_DIR=/path/to/claude-autoresearch/ar_tasks/<op>_<ts>_<uuid>
   ```
   ⚠️ 没这步 → `.autoresearch/.active_task` 没写 → PostToolUse Edit hook 永远 gated → phase 永远卡在 GENERATE_KERNEL → 整个 op 报废。**这是手动模式最容易翻车的地方**。
3. scaffold 自动跑 baseline。`--mode ref-kernel` 时 seed 应当 PASS → phase 直接 PLAN。
4. Claude 在 hook 引导下自己跑 PLAN → EDIT → VERIFY 循环。看 stderr 里的 `[AR Phase: ...]` 即可。
5. 跑到 FINISH 或 max-rounds 用完，Claude 停。

跑完后**不会自动写进 `batch_progress.json`**（你没走 batch 脚本）。要把这次手动跑的结果纳入 batch 状态，手动编辑 `batch_progress.json` 把那个 op 改成：

```json
"<op_name>": {
  "status": "done",
  "task_dir": "/path/to/claude-autoresearch/ar_tasks/<op_name>_<ts>_<uuid>",
  "final_phase": "FINISH",
  ...
}
```

或者用 `python -c "..."` 小脚本批量修。

### 跟自动模式的区别

| | `run.py` | 手动 `claude` 交互 |
|---|---|---|
| Claude 怎么起 | headless `claude --print`，无人值守 | 你自己 `claude` 进交互 |
| `AR_TASK_DIR` export | prompt 强调，模型按指令做 | **你必须自己手动跑这条 export** |
| 出错处理 | 自动标 error，下一个继续 | 你自己判断 + 手动改 progress |
| 中途介入 | 不行（除非 kill batch） | 想说啥就说啥 |
| 速度 | 一个接一个 | 取决于你 |

---

## 中途介入

| 场景 | 怎么办 |
|---|---|
| 想暂停整个批量 | `tmux kill-session -t ar_batch` —— 当前 op 的 claude 也会被杀，标 error |
| 想跳过某个特别难的 op | 编辑 `batch_progress.json`，把它 status 改 `skip`，run.py 下次扫到时跳过 |
| 想重试某个 errored op | `python .autoresearch/scripts/batch/run.py <ws> --mode ... --dsl ... --only <op> --retry-errored` |
| 想清掉所有陈旧 ar_tasks | `rm -rf /path/to/claude-autoresearch/ar_tasks/*`（**只在 run.py 没跑时做**） |
| 想换设备 | `tmux kill-session -t ar_batch`，改 worker daemon 的 `--devices`，再起 run.py |

---

## 最终交付物

跑完后两类产物，**都要保留**：

### A. workspace 里你写的输入

```
<workspace_dir>/
├── manifest.yaml
├── refs/<op>_ref.py
└── kernels/<op>_kernel.py     # 仅 ref-kernel 模式
```

### B. autoresearch 输出的 task dir（每个 done op 一个）

```
/path/to/claude-autoresearch/ar_tasks/<op>_<ts>_<uuid>/
├── kernel.py                  ← 性能优化后的 kernel
├── reference.py               ← scaffold 拷过来的 ref
├── task.yaml                  ← arch / dsl / metric 配置
└── .ar_state/
    ├── .phase                 ← FINISH
    ├── progress.json          ← baseline_metric / best_metric / rounds
    ├── plan.md                ← agent 优化历史
    ├── history.jsonl          ← 每轮 keep/discard 决策
    └── ranking.md             ← 最终排名
```

每个 op 在 `batch_progress.json` 里 `cases.<op>.task_dir` 字段记录了这个绝对路径。一句话收齐所有优化后的 kernel：

```bash
mkdir -p /tmp/optimized_kernels
python -c "
import json, shutil
from pathlib import Path
prog = json.load(open('<batch_dir>/batch_progress.json'))
for k, v in prog['cases'].items():
    if v.get('status') == 'done' and v.get('task_dir'):
        src = Path(v['task_dir']) / 'kernel.py'
        if src.exists():
            shutil.copy(src, f'/tmp/optimized_kernels/{k}.py')
            print('copied', k)
"
```

---

## 命令速查

```bash
# 自动发现 op + 写 manifest（cp 文件后跑）
python .autoresearch/scripts/batch/discover.py <batch_dir> --mode ref-kernel --dsl triton_ascend --write-manifest
python .autoresearch/scripts/batch/discover.py <batch_dir> --write-manifest                  # 沿用已有 mode/dsl
python .autoresearch/scripts/batch/discover.py <batch_dir>                                   # 只列出，不写
python .autoresearch/scripts/batch/discover.py <batch_dir> --filter '*norm' --exclude 'foo*' # 筛

# 预检（推荐 batch 前跑一次）
python .autoresearch/scripts/batch/verify.py <batch_dir>                # Tier 1 默认
python .autoresearch/scripts/batch/verify.py <batch_dir> --full         # 也跑 Tier 2
python .autoresearch/scripts/batch/verify.py <batch_dir> --only opA,opB # 子集

# 全自动批量（tmux daemon；不用裸 nohup 见踩坑 #15）
tmux new -d -s ar_batch \
    'python -u .autoresearch/scripts/batch/run.py <batch_dir> --mode ref-kernel --dsl triton_ascend'

# 限定子集 / 重试错的 / 限量 / 改设备 / 改超时
python .autoresearch/scripts/batch/run.py <batch_dir> --mode ref-kernel --dsl triton_ascend --only opA,opB
python .autoresearch/scripts/batch/run.py <batch_dir> --mode ref-kernel --dsl triton_ascend --retry-errored
python .autoresearch/scripts/batch/run.py <batch_dir> --mode ref-kernel --dsl triton_ascend --limit 5
python .autoresearch/scripts/batch/run.py <batch_dir> --mode ref-kernel --dsl triton_ascend --max-rounds 50
python .autoresearch/scripts/batch/run.py <batch_dir> --mode ref-kernel --dsl triton_ascend --devices 0    # local eval
python .autoresearch/scripts/batch/run.py <batch_dir> --mode ref-kernel --dsl triton_ascend --timeout-min 300

# 监控（另开终端）
python .autoresearch/scripts/batch/monitor.py <batch_dir>                      # 一次性快照（含 active task）
python .autoresearch/scripts/batch/monitor.py <batch_dir> --watch -n 10        # 自动刷新
python .autoresearch/scripts/batch/monitor.py <batch_dir> --dashboard          # 钻进 active task 的 TUI
python .autoresearch/scripts/batch/summarize.py <batch_dir>                    # 静态汇总（跑完后 review 用）
tail -f <batch_dir>/batch.log                                                  # claude 实时输出
tmux attach -t ar_batch                                                                # 进 tmux 看屏幕（Ctrl-b d 脱离）
tmux ls | grep ar_batch                                                                # 主进程是否还活着

# Worker daemon 管理
python .autoresearch/scripts/ar_cli.py worker --start --backend ascend --arch ascend910b3 --devices 0 --host 127.0.0.1 --port 9111 --bg
python .autoresearch/scripts/ar_cli.py worker --status --port 9111
python .autoresearch/scripts/ar_cli.py worker --stop --port 9111
```

---

## 环境

| 角色 | 路径 |
|---|---|
| Batch 脚本 | `<repo>/.autoresearch/scripts/batch/{run.py,monitor.py,verify.py,summarize.py,discover.py,manifest.py}` |
| Workspace | 用户自选，参考 `<batch_dir>/` |
| Workspace 内自动文件 | `batch_progress.json`（runner 写）、`batch.log`（runner 写）、`verify_results.json`（verify.py 写） |
| Autoresearch 任务输出 | `<repo>/ar_tasks/<op>_<ts>_<uuid>/` |
| Worker daemon log | `/tmp/ar_worker_<port>.log` |
| `claude` CLI | 必须在 `PATH`，或用 `--claude-bin` 指定 |
| pyyaml | 可选；不装的话 manifest 必须用 JSON 格式 |
| torch / torch_npu | Tier 2 verify 需要；Tier 1 verify 不需要 |

---

## `run.py` 参数

```
位置参数：
  workspace_dir           workspace 目录路径，目录下需有 manifest.yaml/json

必填：
  --mode {ref-kernel,ref}  整批的模式（也接受 manifest.mode；CLI 优先）
  --dsl <name>             DSL 名（同上；如 triton_ascend / triton_cuda / ascendc）

硬件选择（默认 worker-url=127.0.0.1:9111，会启动 health check）：
  --devices N              NPU 设备 id（in-process eval，不需要 daemon）
  --worker-url host:port   worker daemon URL（mutually exclusive 与 --devices）

per-op 透传给 /autoresearch：
  --max-rounds 30          每个 op 最多多少轮
  --eval-timeout 120       单次 eval 超时（秒）

batch 自己的兜底：
  --timeout-min 180        单 op 整体 wall-clock 上限（分钟）

队列筛选：
  --only A,B,C             只跑指定 op
  --limit N                只跑前 N 个（0=不限）
  --retry-errored          也把 status=error 的算入队列

调度：
  --cooldown-sec 5         op 之间 sleep（设 0 关闭）

claude CLI 透传：
  --claude-bin claude      claude 可执行文件
  --model ""               指定 model（空=默认）
  --extra-claude-arg ...   额外参数（可重复多次）
```

---

## 故障排查

### Worker daemon 不通
- 现象：`worker daemon at 127.0.0.1:9111 is unreachable`
- 修：`python .autoresearch/scripts/ar_cli.py worker --status --port 9111`，没起就 `--start`；或者改用 `--devices 0` 走 in-process eval。

### Claude 没按 prompt 跑 `export AR_TASK_DIR`，phase 卡 GENERATE_KERNEL
- 现象：`monitor.py` 看到 `phase=GENERATE_KERNEL` 长时间不动；run.py 最终标 `error: phase=GENERATE_KERNEL status=stuck`。
- 大前提：模型偶尔会跳过这条 prompt。批量会自动跳到下一个。
- 修：跑完后 `--retry-errored` 重试一遍。仍然卡的就手动接管。

### `claude --print` 启动失败
- `which claude` 看在不在 PATH。否则加 `--claude-bin /full/path` 显式指定。

### 单 op wall-clock 超时
- 默认 180 min/op。复杂 op 可能不够 → `--timeout-min 300`。

### Pre-flight 报 `<file> not found` 但文件明明在
- 检查文件名是否严格遵守 `<op_name>_ref.py` / `<op_name>_kernel.py`。`layernorm.py` ❌、`layernorm_ref.py` ✅。
- 检查 manifest.yaml 里的 `ref_dir` / `kernel_dir` 是否相对 workspace 目录正确。

### 装了 pyyaml 但说 `manifest.yaml is YAML but pyyaml is not installed`
- 多半是装到了别的 conda env / venv。`python -c "import yaml; print(yaml.__file__)"` 确认。

### `verify.py` 在 Windows 报 `OMP: Error #15: Initializing libiomp5md.dll`
- PyTorch + NumPy MKL 双初始化冲突（Windows 装 PyTorch 的常见问题）。verify.py 已经默认设了 `KMP_DUPLICATE_LIB_OK=TRUE` 解决，无须手工处理。如果仍然出现：检查环境变量是不是被 `KMP_DUPLICATE_LIB_OK=FALSE` 显式覆盖了。Linux/NPU 环境无此问题。

### `verify.py --full` 全部报 `kernel.py` ModuleNotFoundError: triton
- Tier 2 需要 worker daemon 同款的 Python 环境。在错的 conda env 里跑就会这样。`source` 进对的 env 再跑。Tier 1 单独的 import 失败也是一样原因。

### Hooks 把 task 重定向到陈旧 task_dir
- 之前残留的 ar_tasks 干扰。批量大跑前清一次：`rm -rf <repo>/ar_tasks/*`。

### `--worker-url` 模式 baseline 报 `proxy connection refused`
- `ALL_PROXY` 劫持了 127.0.0.1。run.py 启动 claude 时已经强制 `NO_PROXY="127.0.0.1,localhost"` 透传，但你自己起 worker daemon 那个终端可能没 set。`export NO_PROXY=127.0.0.1,localhost` 后重起 daemon。

### Batch 跑了一半 SIGHUP 干掉一片
- 用了裸 `nohup` 不是 `tmux`。见踩坑 #15。

---

## Autoresearch 内部机制速记（debug 用）

```
hook_post_edit (Write/Edit kernel.py 后)
  └→ gate: [ -f .autoresearch/.active_task ] || exit 0  ⚠️
  └→ phase_machine.validate_kernel(task_dir)
       ├→ 1. is_placeholder_file? 是 → reject
       └→ 2. quick_check._check_editable_files → CodeChecker
              (syntax → compile → imports → stray-text → DSL → autotune)
       └→ 都过 → write_phase(task_dir, BASELINE)
       └→ 任何一步挂 → emit "[AR] kernel.py invalid" + 原因；phase 不动

hook_post_bash (任何 bash 之后)
  └→ 检测 "AR_TASK_DIR=" → _handle_activation → set_task_dir 写 .active_task → _fresh_start 设 phase
  └→ 检测 baseline.py / pipeline.py / create_plan.py → 推进对应 phase

hook_guard_bash (bash 之前)
  └→ 没有 .active_task gate（关键差异！）
  └→ 直接读 .ar_state/.phase 决定允/禁；GENERATE_KERNEL 时禁所有 user bash
```

**关键不对称：guard_bash 不依赖 `.active_task`，但 post_edit 依赖。** 所以"忘 export AR_TASK_DIR" 的症状就是 phase 永远 GENERATE_KERNEL + bash 永远被拦。`run.py` 的 prompt 反复强调 export 就是为了避免这个坑。

---

## 历史踩坑（继承自 akg-hitl，对 generic 版仍然适用）

1. **未跑 `export AR_TASK_DIR`** → `.active_task` 没写、PostToolUse Edit hook 被 gate 拦、phase 卡住。修：headless prompt 里反复强调（见 [run.py PROMPT_TEMPLATE](../claude-autoresearch/.autoresearch/scripts/batch/run.py)）。

2. **`--permission-mode bypassPermissions` 在 root 下被 Claude CLI 拒** → 改用 `acceptEdits`（auto-allow Edit/Write，bash 走 settings.json allow list）。Prompt 里**不**列出 allow list 全模式（无意义 —— claude 真要看自己 cat 一下 settings.json 即可），仅靠 `acceptEdits` + settings.json 的运行时 deny 来兜底。

3. **`run.py` print() 被块缓冲**（nohup + stdout 重定向到文件）→ 启动用 `python -u`，且 run.py 内部 `sys.stdout.reconfigure(line_buffering=True)`。

4. **`--devices` local 模式让多个 baseline.py 在同一卡抢资源 hang** → 推荐 worker daemon。`run.py` 默认 `--worker-url 127.0.0.1:9111`，启动时 health check 强制要求 daemon 在线（除非显式传 `--devices N`）。

5. **`ALL_PROXY=http://127.0.0.1:17890` hijack 了 baseline 内部对 worker 的 HTTP 调用** → run.py 启动 claude 时强制 `NO_PROXY="127.0.0.1,localhost"`，subprocess env 透传到 claude → baseline.py → eval_wrapper.py。

6. **裸 `nohup python run.py &` SSH 一关 claude 集体 `exit(129)`**。`nohup` 只让直接子进程 `python` 忽略 SIGHUP，孙子 `claude`（Node）启动时重置 signal handler，仍可被 SIGHUP 干掉。SSH session 关闭 → 内核给 controlling terminal 的整个 process group 发 SIGHUP → run.py 自己活，但每个 claude 进来一个砍一个，标 error。**修法**：用 `tmux new -d -s ar_batch '...'`。tmux server 是常驻 daemon，整棵 batch 树挂它下面，跟 ssh session 完全无关，SIGHUP 永远到不了。等价的纯命令行方案是 `setsid nohup python -u ... < /dev/null > batch.log 2>&1 &`。

---

## 与 akg-hitl 那一份的关系（一句话）

- **akg-hitl driver** = 针对 sglang→ascend 迁移的 specialized 驱动（自动发现 `.pt` cache、生成 PyTorch fallback adapter、verify_seed.py 精度门控、46 个 op 写死）
- **本 batch 脚本** = 通用骨架，**只**接 `(ref.py, kernel.py?)` 文件 + manifest，不管你是 sglang 迁移、纯新写、还是别的 DSL 实验

如果你做的事就是 sglang→ascend 迁移：用 akg-hitl 的（已经打包好流水线）。
如果你做别的：用这个。
