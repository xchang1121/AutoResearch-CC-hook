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
  --op-name sinkhorn --backend ascend --arch ascend910b2 \
  --worker-url 127.0.0.1:9002 --max-rounds 200
```

候选 ref / kernel 源文件统一放 [workspace/](workspace/)，命名
`<op_name>_ref.py` / `<op_name>_kernel.py`。`/autoresearch` 的
`--ref` / `--kernel` 直接指向这两个文件。

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

| 参数 | 用例 | 起步阶段 |
|------|------|----------|
| `--ref X.py --kernel Y.py` | 已有 PyTorch ref 和种子 kernel | PLAN |
| `--ref X.py` | 只有 ref，需要生成 kernel | GENERATE_KERNEL |
| `--desc "..."` | 自然语言描述 | GENERATE_REF → GENERATE_KERNEL |
| `--desc "..." --kernel Y.py` | 自然语言 + 种子 kernel | GENERATE_REF |

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

## 执行后端

verify / profile 由 `task_config._gen_verify_script` 和 `_gen_profile_script`
当场生成自包含 Python 脚本，两个 transport 共用：

- **本地后端（默认）** — 没配 `--worker-url` 时自动启用。在
  [.autoresearch/scripts/local_worker.py](.autoresearch/scripts/local_worker.py)
  里把生成脚本解到 `tempfile` 跑 `subprocess`。开机自检：`torch.cuda` /
  `torch_npu` / cpu 三选一，缺哪个报哪个。适合单机开发和调试。
- **远端 Worker** — 通过 `--worker-url` 显式指定。框架打 tarball POST
  到 worker，worker 端解包跑同一份脚本。适合 NPU / 多卡 / msprof 精
  确计时场景。

两条腿出来的 `EvalResult` 结构、字段、metric 名都一致，下游不需要
区分。

### 远程 Worker

远端 NPU / CUDA 硬件通过 SSH tunnel 接入。评测路径复用 akg agent 的 worker
实现：框架负责打包 + 调用 `/api/v1/verify` 和 `/api/v1/profile`，实际
verify / profile 执行器来自 akg agent。

### 启动远端 worker

```bash
ssh npu 'bash -lc "source /path/to/conda/etc/profile.d/conda.sh && conda activate <env> && \
  cd /path/to/akg_agents && \
  nohup bash scripts/server_related/start_worker_service.sh ascend ascend910b2 4 9002 \
  > /tmp/worker_9002.log 2>&1 < /dev/null &"'
```

位置参数：`backend arch device_id port`。

### 建立本地 tunnel

```bash
ssh -f -N -L 127.0.0.1:9002:127.0.0.1:9002 \
  -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 npu

curl http://127.0.0.1:9002/api/v1/status
# {"status":"ready","backend":"ascend","arch":"ascend910b2","devices":[4]}
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
| PLAN / DIAGNOSE / REPLAN | `create_plan.py '<JSON>'` | plan.md（含 (ACTIVE) 标记）+ 全局 pN |
| EDIT | Edit `kernel.py` → `pipeline.py` | history.jsonl 记录 + 可选 git commit + 下一 .phase |
| FINISH | Write `ranking.md` | ranking.md |

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
| `task.yaml` | 任务配置（每个 task 目录一份） | 随 task 分发到 worker |
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
- 远端 NPU / CUDA 机器（可选），通过 SSH tunnel 暴露 worker HTTP 端口
