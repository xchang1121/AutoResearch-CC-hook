# AutoResearch

基于 Claude Code 的算子迭代优化框架。Python + PyYAML。

职责划分：Claude 负责读代码、写 plan、改 kernel、诊断失败；Hook 负责阶段
转移、plan 校验、eval 调度、KEEP/DISCARD、回滚。

## 快速开始

```bash
cd autoresearch
claude
```

启动后，`scaffold` + 首轮 `baseline` 原子执行，进入 PLAN：

```
/autoresearch --ref workspace/sinkhorn_ref.py --kernel workspace/sinkhorn_kernel.py \
  --op-name sinkhorn --dsl triton_ascend --arch ascend910b3 \
  --worker-url 127.0.0.1:9002 --max-rounds 200
```

候选 ref / kernel 源文件统一放 [workspace/](workspace/)，命名
`<op_name>_ref.py` / `<op_name>_kernel.py`。`/autoresearch` 的
`--ref` / `--kernel` 直接指向这两个文件。

`--dsl` 是主键，决定 verify/profile 脚本按哪个 adapter 生成；`--backend` /
`--arch` / `--framework` 可选，不给走 DSL 预设。完整 DSL 列表和预设见
[DSL 分派层](#dsl-分派层)。

长跑使用 `/loop` 自驱模式，失败和上下文溢出会自动恢复：

```
/loop /autoresearch --resume
```

实时监控另开终端运行：

```bash
python .autoresearch/scripts/dashboard.py
```

`--resume` 不带参数时取最近活跃 task。

## 启动模式

输入来源 × 起步阶段：

| 参数 | 用例 | 起步阶段 |
|------|------|----------|
| `--ref X.py --kernel Y.py` | 已有 PyTorch ref 和种子 kernel | PLAN |
| `--ref X.py` | 只有 ref，需要生成 kernel | GENERATE_KERNEL |
| `--desc "..."` | 自然语言描述 | GENERATE_REF → GENERATE_KERNEL |
| `--desc "..." --kernel Y.py` | 自然语言 + 种子 kernel | GENERATE_REF |

DSL / backend / arch / framework 四个维度独立：

| flag | 取值 | 是否必填 |
|------|------|----------|
| `--dsl` | `triton_ascend` / `triton_cuda` / `ascendc` / `cuda_c` / `cpp` / `tilelang_cuda` / `tilelang_npuir` / `pypto` / `swft` / `torch` | 否（默认 `config.yaml:default_dsl`） |
| `--backend` | `ascend` / `cuda` / `cpu` | 否（默认 DSL 预设） |
| `--arch` | 如 `ascend910b3` / `a100` / `x86_64` | 否（默认 DSL 预设） |
| `--framework` | `torch` / `mindspore` / `numpy` | 否（默认 `torch`） |

`--dsl` 是主键。`--backend` / `--arch` / `--framework` 给了就必须和
`--dsl` 的预设一致，不一致直接报错，不做任何隐式推导。规则由 vendored
factory ([ar_vendored/op/verifier/adapters/factory.py](.autoresearch/scripts/ar_vendored/op/verifier/adapters/factory.py))
校验。

可选开关：

- `--no-code-checker` — 关掉静态 [CodeChecker](#codechecker-静态分析)（DSL 合规、autotune
  规则、import 检查等）。占位 kernel 拒绝逻辑仍然生效，只跳过 CodeChecker
  pipeline。task.yaml 写入 `code_checker: {enabled: false}`，事后改 yaml
  也能切换。

Resume 形式：`/autoresearch --resume [task_dir]`。

## 单一入口：`/autoresearch`

项目唯一的 slash command，入参含义：

- 以 `--` 开头：新建任务（scaffold + 首次 baseline 原子完成）
- 为已存在目录：resume 该目录
- `--resume`：resume 最近活跃 task
- 无参数：交互式询问

连续 3 次 FAIL 后 Hook 切换到 `DIAGNOSE`。phase_machine 的 guidance 引导
Claude spawn subagent 做 root-cause 分析。

## 两阶段精度检查

scaffold 时本地 CPU 运行 PyTorch `Model`，把输入 / 输出序列化为
`.ar_state/reference.pt`。后续每轮 verify 流程：

1. Worker 解包得到 `reference.pt`
2. `torch.load` 读取 ref inputs / outputs
3. 运行 `ModelNew.forward`，与 ref 比对
4. 输出 `max_abs` / `max_rel` / `bad_elems(%)`

缺少 `.pt` 时降级为 inline 对比。容差在 `task.yaml` 中配置：

```yaml
metric:
  primary: latency_us
  correctness_atol: 1.0e-2
  correctness_rtol: 1.0e-2
```

verify 失败时 ref 时延仍由 `/api/v1/profile` 单独测得，与 verify 解耦，
dashboard 顶栏始终显示 PyTorch baseline。

## DSL 分派层

verify / profile 脚本按 DSL **独立生成**。
[task_config._gen_verify_script](.autoresearch/scripts/task_config.py) /
[_gen_profile_script](.autoresearch/scripts/task_config.py) 不再硬编码
triton 模板，转而驱动 vendored 的 adapter：

```python
adapter = get_dsl_adapter(config.dsl)           # 10 个 DSL × 专属 adapter
adapter.get_import_statements(framework)         # triton autotune patch / tilelang compile patch / ...
adapter.benchmark_impl(warmup=..., runs=..., ...)  # profiler_npu / do_bench / nsys / msprof 的
                                                   # 代码片段
adapter.get_special_setup_code()                 # 一次性初始化
```

Adapter 集群和 HTTP worker / DevicePool / msprof / nsys 整套都在
[ar_vendored/](.autoresearch/scripts/ar_vendored/)（**~5500 行**，纯 Python
模块，零运行期外部依赖）。目录分四块：

| 子目录 | 作用 |
|--------|------|
| `op/verifier/` | profiler / roofline / DSL adapter factory |
| `op/utils/` | triton autotune patch、tilelang compile patch |
| `core/worker/`, `core/async_pool/`, `worker/` | `LocalWorker` 类、`DevicePool`、FastAPI HTTP server |
| `utils/` | `process_utils.run_command` |

生成出的 `verify_<op>.py` / `profile_<op>_<mode>.py` 把 `ar_vendored/` 通
过 `sys.path.insert(0, script_dir)` 加进 path，本地 / 远端两条路径完全共
享，不需要装任何额外的 pip 包。ar_vendored 也 bundle 到每个 task 的
tarball（`_build_package`），worker 端解包就可 `import ar_vendored`。

`--dsl` / `task.yaml:dsl` 的合法值和预设见
[config.yaml:dsls](.autoresearch/config.yaml)。工厂是权威来源，config 只
填默认 backend / arch / framework / device_type。

## 执行后端

verify / profile 脚本按 DSL 生成后，两个 transport 共用，对 `EvalResult`
的结构 / 字段 / metric 名一视同仁：

- **本地后端（默认）** — 没配 `--worker-url` 时自动启用。
  [local_worker.py](.autoresearch/scripts/local_worker.py) 解 tarball 到
  `tempfile`，按 DSL 分流：
  - `triton_*` / `tilelang_*` / `pypto` / `torch` / `cpp` →
    `_profile_via_subprocess`（脚本 `adapter.benchmark_impl` 自测，
    profiler_npu / do_bench）
  - `ascendc` on `ascend` → `_profile_via_msprof`（`msprof
    --application=` 包 script，`analyze_prof_data` 读 op_summary CSV）
  - `cuda_c` on `cuda` → `_profile_via_nsys`（`nsys profile`，
    `analyze_nsys_data` 读 rep）
  - CLI 工具不在 PATH 时自动降级到 `_profile_via_subprocess`
  
  开机自检：`torch.cuda` / `torch_npu` / cpu 三选一，缺哪个报哪个。
- **远端 Worker** — 通过 `--worker-url` 显式指定。框架打 tarball POST
  到 [ar_vendored/worker/server.py](.autoresearch/scripts/ar_vendored/worker/server.py)
  的 `/api/v1/{verify,profile}`，worker 端解包跑同一份脚本 + 同一套
  adapter。适合多卡 / DevicePool / roofline 整套场景。**worker 端不需要
  装任何额外的 pip 包**，tarball 自带 ar_vendored。

两条腿的路由决策由 `config.dsl` + `config.backend` 独立驱动；本地的
msprof / nsys 分支和远端 `LocalWorker.profile` 走同一份 DSL 判断逻辑，
所以同一个 task 在本地和远端走出来的 metric 是可比的。

### 远程 Worker

远端 NPU / CUDA 硬件通过 SSH tunnel 接入。HTTP server
([ar_vendored/worker/server.py](.autoresearch/scripts/ar_vendored/worker/server.py)
+ [core/worker/local_worker.py](.autoresearch/scripts/ar_vendored/core/worker/local_worker.py)
+ [core/async_pool/device_pool.py](.autoresearch/scripts/ar_vendored/core/async_pool/device_pool.py)）
是 autoresearch 自带的，worker 端不需要装任何额外 pip 包。

### 启动远端 worker

项目唯一入口：[ar_cli.py](.autoresearch/scripts/ar_cli.py)。所有 worker
生命周期操作都走这一条命令。**在 remote shell 里**（SSH 进去或物理登录）
激活好 python 环境后：

```bash
# 起（daemon；detach + log → /tmp/ar_worker_9111.log）
python .autoresearch/scripts/ar_cli.py worker --start \
    --backend ascend --arch ascend910b3 --devices 2,5 \
    --host 127.0.0.1 --port 9111 --bg

# 查
python .autoresearch/scripts/ar_cli.py worker --status --port 9111

# 停
python .autoresearch/scripts/ar_cli.py worker --stop --port 9111
```

不加 `--bg` 就是 foreground（Ctrl-C 退出），适合调试看 uvicorn 日志。

Daemon 模式会轮询端口到就绪（最多 30s），worker 起不来会 dump log 尾部，
不会留僵尸 PID。`--stop` 先用 `ss` / `lsof` 定位 PID，确认 cmdline 含
`ar_vendored.worker.server` 才 kill — 撞到别人占同端口的 service 不会
误伤。跨平台（Windows / Linux / macOS）。

CLI 不帮你激活 python 环境 — 用户自己 `conda activate` / `source env.sh`
/ 用 venv 都行，只要进 `ar_cli.py` 时 `python -c "import fastapi,
uvicorn, torch, torch_npu"` 能跑通。

worker 进程的运行期依赖（用哪个 DSL 才需要装哪些）：

- `fastapi` + `uvicorn`（HTTP server 本体）
- `torch` + `torch_npu`（ascend） / CUDA runtime（cuda）
- `triton`（triton_* DSL）
- `pandas`（msprof / nsys CSV 解析）
- CANN toolkit 的 `msprof` CLI（走 ascendc 时） / Nsight Systems 的
  `nsys` CLI（走 cuda_c 时）

框架侧（客户端）完全不需要这些 — task_config.py 只用 stdlib + pyyaml 和
远端通信。tarball 内带 `ar_vendored/`，worker 解包即可 `import ar_vendored`。

### 建立本地 tunnel

```bash
ssh -f -N -L 127.0.0.1:9002:127.0.0.1:9002 \
  -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 npu

curl http://127.0.0.1:9002/api/v1/status
# {"status":"ready","backend":"ascend","arch":"ascend910b3","devices":[4]}
```

任务启动加 `--worker-url 127.0.0.1:9002`。多 URL 逗号分隔，框架按可达性选择。

## Dashboard

```bash
# 自动取当前任务，默认 5 秒刷新
python .autoresearch/scripts/dashboard.py

# 指定任务目录和刷新间隔
python .autoresearch/scripts/dashboard.py ar_tasks/my_task --watch 2
```

键位：`↑` / `↓` / `PgUp` / `PgDn` / `Home` / `End` 滚动 history，`q` / `Esc`
退出。

顶栏显示 task 名、阶段、plan 版本、budget、Baseline（PyTorch ref 时延）、
Seed（种子 kernel 时延）、Best、改进比。下栏为 history 表（每条带 `pN:`
前缀）和当前 plan。

## 主循环

单轮流程：**PLAN → EDIT → quick_check → eval → KEEP/DISCARD → settle**。
连续失败进入 DIAGNOSE，plan 全部 settle 进入 REPLAN，预算用尽进入
FINISH。执行体基于 Claude Code 的 Edit / Bash 工具 + Python 脚本 + Hook
守护。

```
INIT
  ├─ (--desc?)            GENERATE_REF ─→ GENERATE_KERNEL
  ├─ (--ref only?)                       GENERATE_KERNEL
  └─ (--ref + --kernel?)                            ─────→ BASELINE
                                                            │
                                          ┌─ scaffold --run-baseline 原子完成
                                          ▼
   ┌────────────────────────  PLAN  ◀────────────────────────┐
   │   create_plan.py 校验 (≥3 项 / 多样性 / rationale 长度) │
   ▼                                                          │
  EDIT  ──→  pipeline.py:                                     │
            quick_check → eval_wrapper → keep_or_discard      │
            → settle ──→ history.jsonl + plan.md + .phase     │
            │                                                 │
            ├─ KEEP    : git commit (kernel.py)，best 更新   │
            ├─ DISCARD : no_improvement++，回滚              │
            └─ FAIL    : consecutive_failures++              │
            │                                                 │
            ├─ consecutive_failures ≥ 3 ─→ DIAGNOSE ─────────┤
            ├─ plan 全部 settle          ─→ REPLAN ──────────┘
            └─ 预算用完                  ─→ FINISH
```

每个 `pN` 必有 KEEP / DISCARD / FAIL 终态。REPLAN 时旧版 pending 项被
`create_plan.py` 写为 `DISCARD (superseded by replan vN)`。

各阶段产物：

| 阶段 | Claude 操作 | 产物 |
|------|-------------|------|
| GENERATE_REF | Edit `reference.py` | reference.py |
| GENERATE_KERNEL | Edit `kernel.py` | kernel.py (种子) |
| BASELINE | `baseline.py` | seed_metric → progress.json |
| PLAN / DIAGNOSE / REPLAN | `create_plan.py @<xml_path>` | plan.md（含 (ACTIVE) 标记）+ 全局 pN |
| EDIT | Edit `kernel.py` → `pipeline.py` | history.jsonl 记录 + 可选 git commit + 下一 .phase |
| FINISH | Write `ranking.md` | ranking.md |

## CodeChecker 静态分析

每轮 EDIT 完跑 [pipeline.py](.autoresearch/scripts/pipeline.py) 时，
[quick_check.py](.autoresearch/scripts/quick_check.py) 会先对 editable
files 跑一遍 [code_checker.py](.autoresearch/scripts/code_checker.py)
（AST → py_compile → import 解析 → 散落中文 → DSL 合规 → @triton.autotune
合规）。同一个 pipeline 也被 [phase_machine.validate_kernel](.autoresearch/scripts/phase_machine.py)
在 GENERATE_KERNEL → BASELINE 推进前调一次，两边共享一份规则。

**当前的 DSL 合规规则专为 `triton_ascend` / `triton_cuda` 设计**
（`class ModelNew + @triton.jit` 模板、`@triton.autotune` 强制携带
`restore_value`）。`ascendc` / `cuda_c` / `tilelang_*` / `pypto` task 跑
到会误报，建议 scaffold 时加 `--no-code-checker`。

何时关掉：

- DSL 不在 `triton_*` 系（CodeChecker 的 triton kernel / autotune 规则
  对你不适用）
- 用 ad-hoc kernel 风格，故意不走 `class ModelNew + @triton.jit` 模板
- 静态规则误报多到掩盖真实问题

关掉的方式（任选一个）：

```bash
# scaffold 时
/autoresearch --ref X.py --kernel Y.py --op-name foo --no-code-checker

# 已有 task：直接改 task.yaml
code_checker:
  enabled: false
```

关掉后 `quick_check` / `validate_kernel` 跳过 CodeChecker pipeline；
**占位 kernel（scaffold 写的 TODO）仍会被拒**，所以 GENERATE_KERNEL
阶段不会被绕过。

## Hooks 与状态机

[phase_machine.py](.autoresearch/scripts/phase_machine.py) 提供 phase 常量
和规则查询。`<task_dir>/.ar_state/.phase` 记录当前阶段。Hook 脚本在
Claude Code 的 PreToolUse / PostToolUse 事件中调用这些规则决定允许或
阻断工具调用。

### 1. phase_machine.py

导出内容：

**phase 常量**（[:30-41](.autoresearch/scripts/phase_machine.py#L30-L41)）：
`INIT` / `GENERATE_REF` / `GENERATE_KERNEL` / `BASELINE` / `PLAN` / `EDIT` /
`DIAGNOSE` / `REPLAN` / `FINISH`。

**规则表**（[:153-176](.autoresearch/scripts/phase_machine.py#L153-L176)）：

```python
_BASH_RULES = {
    INIT:            _BashPolicy("strict",     required={"export AR_TASK_DIR="}),
    BASELINE:        _BashPolicy("strict",     required={"baseline.py"}),
    GENERATE_REF:    _BashPolicy("strict",     required=set()),
    GENERATE_KERNEL: _BashPolicy("strict",     required=set()),
    PLAN:            _BashPolicy("permissive", banned=set()),
    DIAGNOSE:        _BashPolicy("permissive", banned=set()),
    REPLAN:          _BashPolicy("permissive", banned=set()),
    EDIT:            _BashPolicy("permissive", banned={"create_plan.py"}),
    FINISH:          _BashPolicy("permissive", banned=set()),
}

_EDIT_RULES = {
    GENERATE_REF:    {"ref"},        # 仅允许写 reference.py
    GENERATE_KERNEL: {"editable"},   # 仅允许写 task.yaml.editable_files
    EDIT:            {"editable"},
    # 其他 phase：无用户文件可写
}
```

`strict` 为白名单子串匹配，`permissive` 为黑名单子串匹配。PLAN / EDIT /
DIAGNOSE / REPLAN 需要 `git log`、读文件等 ad-hoc 操作，使用 permissive；
BASELINE / INIT 只允许单一命令，使用 strict。

**查询函数**（[:212-273](.autoresearch/scripts/phase_machine.py#L212-L273)）：
`check_bash` 和 `check_edit`，输入 phase 名 + 命令 / 文件名，返回
`(allowed, reason)`。纯函数，不读写任何状态。

跨 phase 全局黑名单也在此定义：`quick_check.py` / `eval_wrapper.py` /
`keep_or_discard.py` / `settle.py` 在任何 phase 均禁止手动调用（只能由
`pipeline.py` 子进程执行）；`git commit` 仅允许 `keep_or_discard.py` 在
KEEP 时调用。读类命令（`ls` / `cat` / `grep` / `git log|diff|status` /
`dashboard.py` / `echo` / `pwd`）跨 phase 放行。

### 2. 状态文件

`<task_dir>/.ar_state/.phase` 存当前 phase，内容为一行文本。

写入该文件的主体：

- `scaffold.py --run-baseline` 成功后写 `PLAN`
- `create_plan.py` 校验通过后写 `EDIT`
- `pipeline.py` 收尾时由 `compute_next_phase()` 计算并写入
- `hook_post_edit.py` / `hook_post_bash.py` 在对应事件后按情况写入

Claude 不直接写 `.phase`。`hook_guard_edit.py` 对 `.ar_state/*` 放行是
出于脚本自身写状态的需要，guidance 不引导 Claude 修改该文件。约束生效
点是下一次 Hook 进程启动时的规则查询。

### 3. Hook 实现

[hook_guard_bash.py:59-76](.autoresearch/scripts/hook_guard_bash.py#L59-L76)：

```python
def main():
    hook_input = read_hook_input()              # 从 stdin 读 Claude Code 传入的 JSON
    if hook_input.get("tool_name") != "Bash":
        sys.exit(0)

    task_dir = get_task_dir()                   # 从 .autoresearch/.active_task 读
    command  = hook_input["tool_input"]["command"]
    phase    = read_phase(task_dir)             # 读 <task_dir>/.ar_state/.phase

    ok, reason = check_bash(phase, command)
    if not ok:
        print(json.dumps({"decision": "block", "reason": f"[AR] {reason}. …"}))
        sys.exit(2)                             # 退出码 2 = Claude Code 拒绝该工具调用
    sys.exit(0)
```

Hook 遵循 Claude Code 的 PreToolUse 协议：stdout 输出
`{"decision":"block","reason":"..."}` 并 `sys.exit(2)`，Claude Code 会
拒绝工具调用并把 reason 作为工具错误反馈给 LLM。`hook_guard_edit.py`
结构相同，调用 `check_edit`。

### 4. 端到端示例

`.phase` 为 `BASELINE`，Claude 尝试执行 `create_plan.py`：

```
1. Claude 工具调用: Bash(command="python .autoresearch/scripts/create_plan.py …")
2. Claude Code 按 .claude/settings.json 的 PreToolUse/Bash 匹配 → hook_guard_bash.py
3. Claude Code 通过 stdin 传入 {tool_name:"Bash", tool_input:{command:"…"}}
4. hook_guard_bash.py 执行:
     task_dir = 读 .autoresearch/.active_task      → "/…/ar_tasks/xxx"
     phase    = 读 <task_dir>/.ar_state/.phase     → "BASELINE"
     ok, why  = check_bash("BASELINE", command)
         _BASH_RULES["BASELINE"] = ("strict", required={"baseline.py"})
         command 不含 "baseline.py" → (False, "phase BASELINE: …")
5. Hook 输出 {"decision":"block","reason":"[AR] phase BASELINE: allowed commands = ['baseline.py']. [AR Phase: BASELINE] …"}
6. sys.exit(2)
7. Claude Code 拒绝执行；LLM 接收 block reason，按 guidance 改用 baseline.py
```

链路：

```
  Claude 工具调用
        │
        ▼
  Claude Code PreToolUse ──配置源── .claude/settings.json
        │
        ▼
  hook_guard_bash.py (独立 Python 进程，每次调用 fork 一次)
        │
        ├── 读 .ar_state/.phase                     ── 当前 phase
        ├── check_bash(phase, cmd) 查 _BASH_RULES   ── 规则查询
        └── {"decision":"block"} + exit 2           ── 阻断
```

### 5. Hook 接线（`.claude/settings.json`）

| 事件 | 匹配工具 | Hook 脚本 | 职责 |
|------|----------|-----------|------|
| PreToolUse | Edit / Write | `hook_guard_edit.py` | 调 `check_edit`，按 phase 拦截非法写入 |
| PreToolUse | Bash | `hook_guard_bash.py` | 调 `check_bash`，按 phase 拦截非法命令，检测幻觉脚本名 |
| PostToolUse | Edit / Write | `hook_post_edit.py` | Edit 完成后更新 `.phase` |
| PostToolUse | Bash | `hook_post_bash.py` | 脚本退出后切 phase；处理 `export AR_TASK_DIR=` 激活 |
| Stop | — | `hook_stop_save.py` | 写入 stop reason 和时间戳到 progress.json，供 resume 使用 |

`hook_guard_edit.py` 在 phase 规则之外还有全局约束：

- `plan.md` 一律禁止写入：由 `create_plan.py` / `settle.py` / `pipeline.py`
  输出，手工修改会破坏审计记录
- `.ar_state/*` 一律放行：脚本和 Hook 自身需要写状态文件
- EDIT 阶段额外的 git gate：上一轮 kernel.py 未经过 `pipeline.py` 收尾
  就再次 Edit 会被拦截并提示先运行 `pipeline.py`，防止单轮内累积多个未
  结算改动

### 6. Mealy 状态机

- **状态**：`.phase` 中的 phase 名，共 9 个
- **输入**：Claude 的工具调用（Bash / Edit）、脚本退出码、`progress.json`
  中的计数（`consecutive_failures` / `eval_rounds` / 剩余 budget 等）
- **输出**：`check_bash` / `check_edit` 返回值 → Hook 决定 block 或 pass
- **转移**：`hook_post_bash.py` / `hook_post_edit.py` /
  `pipeline.py.compute_next_phase()` 在 PostToolUse 或子流程结束时写
  `.phase`。`consecutive_failures ≥ 3` 转入 DIAGNOSE，plan 全部 settle
  转入 REPLAN，预算用尽转入 FINISH

Claude 无法绕过约束的四个依据：

1. 规则集中在 phase_machine，两个 PreToolUse Hook 共享同一份
   `_BASH_RULES` / `_EDIT_RULES`，修改一处，两端同步生效。
2. 全局黑名单覆盖所有 phase：`quick_check.py` / `eval_wrapper.py` /
   `keep_or_discard.py` / `settle.py` / `git commit` 在任何 phase 均禁止
   直接调用。Claude 无法手动运行 pipeline 子步骤，也无法跳过
   KEEP / DISCARD 直接 commit。
3. `pipeline.py` 作为单轮原子操作：EDIT 阶段结束必须调用
   `python .autoresearch/scripts/pipeline.py "$AR_TASK_DIR"`，内部串行
   执行 quick_check → eval_wrapper → keep_or_discard → settle →
   `compute_next_phase()` 写 `.phase`。pipeline 未完成时
   `hook_guard_edit` 的 git gate 阻止 Claude 离开 EDIT。`keep_or_discard`
   三态：KEEP 触发 `git commit` + 重置失败计数 + 更新 best；DISCARD
   回滚到上一 KEEP commit；FAIL 失败计数 +1 + 回滚。
4. `create_plan.py` 校验阻塞：全局单调 pid（`progress.json.next_pid`
   顺序分配，pN 不复用不跳号）；至少 3 项；最多 1 项纯参数调优；
   rationale 长度 30–400 字符。不通过即非零退出，`hook_post_bash` 不
   推进 phase，LLM 只能按 stderr 修改 JSON 重试。REPLAN 时旧版 pending
   项被批量 settle 为 `DISCARD (superseded by replan vN)`。

### 7. Guidance 与 Resume

每次 phase 切换，Hook 调用 `phase_machine.get_guidance(task_dir)` 生成
phase-specific 提示（包含 editable_files、当前 active item、最近三条
history、剩余 budget 等），通过 `additionalContext` 回注给 LLM。

`/autoresearch --resume` 由 `resume.py` 定位最新 task 并 `export
AR_TASK_DIR=…`，PostToolUse 触发 `_handle_activation()`：存在 `.phase`
时直接恢复；仅存 `progress.json` 时调用 `compute_resume_phase()`，按
seed_metric / plan 状态路由；reference.py / kernel.py 的存在性决定
GENERATE_REF / GENERATE_KERNEL / BASELINE 入口。DIAGNOSE 阶段的
guidance 要求 spawn subagent 输出 Root cause / Fix direction / What to
avoid，使下一轮 plan 换方向，避免只调整超参。

## Knowledge 库

`knowledge/` 按 DSL / backend 组织，88 份优化知识文档：

```
knowledge/triton-ascend/   Triton on Ascend NPU (guides + cases)
knowledge/triton-cuda/     Triton on CUDA GPU
knowledge/cuda-c/          CUDA C
knowledge/cpp/             CPU C++
knowledge/tilelang-cuda/   TileLang DSL
knowledge/pypto/           PyTorch operator patterns
```

PLAN 阶段 Claude 用 `Glob("knowledge/<dsl>/**/*.md")` 检索，Read 对应
SKILL.md（YAML frontmatter 含 id / category / description / keywords），
把 id 写入 plan item 的 rationale。

命名采用 `knowledge/` 而非 `skills/`，以区别于仓库顶层 `skills/`
（Triton / AscendC agent 的行为型 skill，安装时 `mv skills/<backend>/*
.claude/skills/` 进入 Claude Code 原生 skill 系统）。AutoResearch 的
knowledge 文档仅通过 Glob + Read 访问，不进入 `.claude/`。

## 配置与状态

| 路径 | 用途 | Git |
|------|------|-----|
| `workspace/<op>_ref.py` / `workspace/<op>_kernel.py` | 候选 ref / kernel 源文件，`/autoresearch --ref/--kernel` 的输入 | ✔ |
| `.autoresearch/config.yaml` | DSL → backend/arch/framework/device_type 预设表；worker_only_modules；hallucinated_scripts | ✔ |
| `.autoresearch/code_checker.yaml` | CodeChecker 规则表（triton 模板 / autotune 合规） | ✔ |
| `.autoresearch/scripts/ar_vendored/` | DSL adapter + profiler + msprof/nsys runner + HTTP worker server | ✔ |
| `.autoresearch/scripts/ar_cli.py` | 统一 CLI：`ar_cli worker --start/--stop/--status`，支持 `--bg` daemon | ✔ |
| `task.yaml` | 任务配置（每个 task 目录一份，含 dsl/backend/arch/framework 四字段） | 随 task 分发到 worker |
| `.ar_state/progress.json` | 运行时状态 | — |
| `.ar_state/plan.md` | 规划 + 结算历史（权威态） | — |
| `.ar_state/history.jsonl` | 每轮 decision / metrics / commit | — |
| `.ar_state/reference.pt` | 缓存的 PyTorch ref 输出 | — |
| `.ar_state/.phase` | 当前阶段 | — |
| `.claude/settings.json` | Hook + 权限配置 | ✔ |
| `.claude/settings.local.json` | API key、model 覆盖 | ✗ |
| `.claude/scheduled_tasks.lock` | Session lock | ✗ |

## 依赖

- Python ≥ 3.10
- `pip install pyyaml torch`
- Claude Code CLI 或 VS Code 扩展
- 按 DSL 追加的可选运行期依赖（autoresearch 自己不依赖；scaffold 时选了对
  应 DSL 才会被 adapter 拉入）：
  - `triton_ascend` / `tilelang_npuir`：`torch_npu`、`triton`、CANN（为了
    `profiler_npu` 的 `torch_npu.profiler`）
  - `triton_cuda` / `tilelang_cuda` / `pypto`：`triton`、CUDA runtime
  - `ascendc`：CANN toolkit（`msprof` CLI 在 PATH）
  - `cuda_c`：Nsight Systems（`nsys` CLI 在 PATH）
  - 所有 DSL 走 local `_profile_via_msprof` / `_profile_via_nsys` 时需要
    `pandas`（读 op_summary / nsys rep）
- 远端 NPU / CUDA 机器（可选），通过 SSH tunnel 暴露 worker HTTP 端口。
  worker 端不需要装 autoresearch — tarball 自带 `ar_vendored/`。
