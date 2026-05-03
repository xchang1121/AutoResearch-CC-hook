# AutoResearch

基于 Claude Code 的算子迭代优化框架。Python + PyYAML。

职责划分：Claude 负责读代码、写 plan、改 kernel、诊断失败；Hook 负责阶段
转移、plan 校验、eval 调度、KEEP/DISCARD、回滚。

## 快速开始

```bash
cd claude-autoresearch
claude
```

启动后（scaffold + 首轮 baseline 原子执行，进入 PLAN）：

```
/autoresearch --ref workspace/sinkhorn_ref.py --kernel workspace/sinkhorn_kernel.py \
  --op-name sinkhorn --dsl triton_ascend --devices 5 --max-rounds 200
```

候选源文件放 [workspace/](workspace/)，命名 `<op_name>_ref.py` /
`<op_name>_kernel.py`。CLI flag 详见下面 [启动模式](#启动模式)。

长跑自驱：`/loop /autoresearch --resume`（失败 / 上下文溢出自动恢复；
不带参数取最近活跃 task）。实时监控另开终端
`python .autoresearch/scripts/dashboard.py`。

## 启动模式

输入来源 × 起步阶段：

| 参数 | 用例 | 起步阶段 |
|------|------|----------|
| `--ref X.py --kernel Y.py` | 已有 PyTorch ref 和种子 kernel | PLAN |
| `--ref X.py` | 只有 ref，需要生成 kernel | GENERATE_KERNEL |
| `--desc "..."` | 自然语言描述 | GENERATE_REF → GENERATE_KERNEL |
| `--desc "..." --kernel Y.py` | 自然语言 + 种子 kernel | GENERATE_REF |

CLI 只暴露三个维度（DSL / hardware / framework），其余全部派生：

| flag | 取值 | 是否必填 |
|------|------|----------|
| `--dsl` | `triton_ascend` / `triton_cuda` / `ascendc` / `cuda_c` / `cpp` / `tilelang_cuda` / `tilelang_npuir` / `pypto` / `swft` / `torch` | **必填** —— `/autoresearch` 入口（[parse_args.py](.autoresearch/scripts/parse_args.py)）强制要求显式传值。`scaffold.py` 内部对从 task.yaml 恢复的旧 task 仍有 `config.yaml:default_dsl` 兜底，但新建 task 时不再走该兜底。 |
| `--devices` | 本地 NPU/GPU 下标，逗号分隔：`5` 或 `0,1,2,3` | **XOR** — 和 `--worker-url` 二选一必填 |
| `--worker-url` | 远端 worker URL：`127.0.0.1:9070` | **XOR** — 和 `--devices` 二选一必填 |
| `--framework` | `torch` / `mindspore` / `numpy` | 否（默认 `torch`） |

**自动派生（用户不手写）：**

- `backend` ← DSL 名字（`triton_ascend` → `ascend`；`cuda_c` → `cuda`；`cpp` → `cpu` …；公共入口 [hw_detect.backend_for_dsl](.autoresearch/scripts/hw_detect.py)，内部表在同文件 `_DSL_BACKEND`）
- `arch`：
  - 给了 `--devices` → `npu-smi info` / `nvidia-smi` / `uname -m` 派生（如 `ascend910b3` / `a100` / `x86_64`）
  - 给了 `--worker-url` → `GET /api/v1/status` 拿 worker 自报的 arch
- `device_type`（torch.device 前缀）← `backend` 映射：`ascend → npu`、`cuda → cuda`、`cpu → cpu`

硬件必须由 `--devices` 或 `--worker-url` 之一指定，两者互斥。两个都不给
或都给 → 硬错误。

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

Reference 输出由 worker 端（本地或远端）按需计算并缓存，scaffold 不再
做本地 CPU 预跑，也不写 `.ar_state/reference.pt`（旧行为，已移除）。
每轮 verify 流程：

1. 首轮 verify：worker 在 sandbox 里 import `reference.py`，跑
   `Model(*get_init_inputs())(*get_inputs())`，结果缓存到
   `/tmp/ar_cache/<op>_<sha(reference.py)>/reference.pt`
2. 后续轮：命中缓存直接 `torch.load` ref outputs，省掉一次多 GiB 上传
3. 运行 `ModelNew.forward`，与 ref 比对
4. 输出 `max_abs` / `max_rel` / `bad_elems(%)`

`reference.py` 内容变了 → sha 变了 → 缓存自动失效。容差在 `task.yaml`
中配置：

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
[task_config.package_builder._gen_verify_script](.autoresearch/scripts/task_config/package_builder.py) /
[_gen_profile_script](.autoresearch/scripts/task_config/package_builder.py) 不再硬编码
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

生成的 `verify_<op>.py` / `profile_<op>_<mode>.py` 通过
`sys.path.insert(0, script_dir)` 把 `ar_vendored/` 加进 path，本地 / 远端
共用同一份脚本。ar_vendored 也 bundle 进每个 task 的 tarball
（`_build_package`），worker 端解包即可 `import ar_vendored`。

`--dsl` / `task.yaml:dsl` 的合法值由 [factory.py](.autoresearch/scripts/ar_vendored/op/verifier/adapters/factory.py)
决定（10 个 DSL adapter）。DSL → backend / device_type 的映射是纯函数，
硬编码在 [hw_detect.py](.autoresearch/scripts/hw_detect.py)。arch 在 scaffold
时根据 `--devices` / `--worker-url` 自动填到 task.yaml。

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
  adapter。适合多卡 / DevicePool / roofline 整套场景。worker 端需要
  rsync 项目过去来启动 `ar_cli.py worker --start`。

两条腿的路由决策由 `config.dsl` + `config.backend` 独立驱动；本地的
msprof / nsys 分支和远端 `LocalWorker.profile` 走同一份 DSL 判断逻辑，
所以同一个 task 在本地和远端走出来的 metric 是可比的。

### 远程 Worker

远端 NPU / CUDA 硬件通过 SSH tunnel 接入，HTTP server 由 autoresearch
自带（[ar_vendored/worker/](.autoresearch/scripts/ar_vendored/worker/) +
[core/worker/](.autoresearch/scripts/ar_vendored/core/worker/) +
[core/async_pool/](.autoresearch/scripts/ar_vendored/core/async_pool/)）。
客户端只用 stdlib + pyyaml，依赖只装 worker 侧：`fastapi` + `uvicorn`、
`torch`（+ `torch_npu` / CUDA runtime）、`triton`（triton_* DSL）、
`pandas`（msprof / nsys CSV）、CANN `msprof` CLI（ascendc）或 Nsight
`nsys` CLI（cuda_c）。

流程：rsync 项目到 worker 机器 → 在 worker shell 里起 server → 本地建
SSH tunnel → 启动任务加 `--worker-url`：

```bash
# worker 机器上（自行 conda activate / source env.sh）
python .autoresearch/scripts/ar_cli.py worker --start \
    --backend ascend --arch ascend910b3 --devices 2,5 \
    --host 127.0.0.1 --port 9111 --bg

# 本地机器上
ssh -f -N -L 127.0.0.1:9002:127.0.0.1:9002 \
  -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 npu
curl http://127.0.0.1:9002/api/v1/status
# {"status":"ready","backend":"ascend","arch":"ascend910b3","devices":[4]}
```

`ar_cli.py worker` 子命令还有 `--stop` / `--status` / foreground 模式
（去掉 `--bg`）；端口就绪探测、PID cmdline 校验避免撞车，详见
`ar_cli.py worker --help`。多 worker URL 逗号分隔，框架按可达性挑选。

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
                                         PLAN
                                          │ create_plan.py 校验 (≥3 项 /
                                          │ 多样性 / rationale 长度)
                                          ▼
   ┌─────────────────────────────────── EDIT ◀──────────────┐
   │  pipeline.py:                                          │
   │    quick_check → eval_wrapper → keep_or_discard        │
   │    → settle ──→ history.jsonl + plan.md + .phase       │
   │   ├─ KEEP    : git commit (editable_files)，best 更新   │
   │   ├─ DISCARD : 回滚 editable_files                      │
   │   └─ FAIL    : consecutive_failures++，回滚            │
   │                                                        │
   │   ├─ consecutive_failures ≥ 3 ─→ DIAGNOSE ─→ create_plan ─┤
   │   ├─ plan 全部 settle          ─→ REPLAN  ─→ create_plan ─┤
   │   └─ eval_rounds == max_rounds ─→ FINISH
   └─────────────────────────────────────────────────────────┘
```

DIAGNOSE / REPLAN 不绕回 PLAN——`create_plan.py` 校验通过后 hook 直接写
phase = EDIT。

每个 `pN` 要么在 `history.jsonl` 里有 KEEP / DISCARD / FAIL 终态，要么在
REPLAN/DIAGNOSE 边界被静默丢弃（不写假 DISCARD 行，不写 history.jsonl）。
pid 计数器单调推进、不复用；审计链是 `plan_version` + `history.jsonl`
缺记录——某 pid 只在 plan vN 出现而 history.jsonl 找不到，就是 N→N+1 时
被丢的。

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

[quick_check.py](.autoresearch/scripts/quick_check.py) 和
[phase_machine.validators.validate_kernel](.autoresearch/scripts/phase_machine/validators.py) 共享
一份 [code_checker.py](.autoresearch/scripts/code_checker.py) pipeline（AST
→ py_compile → import 解析 → 散落中文 → DSL 合规 → `@triton.autotune`
合规），分别在每轮 EDIT 收尾和 GENERATE_KERNEL → BASELINE 推进前触发。

当前 DSL 规则专为 `triton_ascend` / `triton_cuda` 设计（`class ModelNew +
@triton.jit`，autotune 强制携带 `restore_value`）。其他 DSL 会误报，
scaffold 时加 `--no-code-checker`，或改 `task.yaml: code_checker.enabled:
false`。关掉后占位 kernel（scaffold TODO）仍会被拒，GENERATE_KERNEL 不会
被绕过。

## Hooks 与状态机

[phase_machine/](.autoresearch/scripts/phase_machine/) 是一个包，按职责
拆成 `state_store` / `validators` / `phase_policy` / `guidance`，由
[__init__.py](.autoresearch/scripts/phase_machine/__init__.py) re-export
保持外部导入兼容。`<task_dir>/.ar_state/.phase` 记录当前阶段。Hook 脚本
在 Claude Code 的 PreToolUse / PostToolUse 事件中调用这些规则决定允许
或阻断工具调用。

### 1. phase_machine 包

子模块导出内容：

**phase 常量**（[state_store.py:34-44](.autoresearch/scripts/phase_machine/state_store.py#L34-L44)）：
`INIT` / `GENERATE_REF` / `GENERATE_KERNEL` / `BASELINE` / `PLAN` / `EDIT` /
`DIAGNOSE` / `REPLAN` / `FINISH`。

**规则表**（[phase_policy.py](.autoresearch/scripts/phase_machine/phase_policy.py)）：

```python
_BASH_RULES = {
    INIT:            _BashPolicy("strict",     required={"export AR_TASK_DIR="}),
    BASELINE:        _BashPolicy("strict",     required={"baseline.py"}),
    GENERATE_REF:    _BashPolicy("strict",     required=set()),
    GENERATE_KERNEL: _BashPolicy("strict",     required=set()),
    PLAN:            _BashPolicy("permissive", banned=set()),
    # DIAGNOSE: only create_plan.py is a legal AR-script invocation;
    # the phase exists to produce a new plan via Task -> artifact ->
    # create_plan (or manual-planning fallback after the cap).
    DIAGNOSE:        _BashPolicy("strict",     required={"create_plan.py"}),
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

`strict` 为白名单匹配，`permissive` 为黑名单子串匹配。strict 模式下
`.py` required 项通过 `parse_invoked_ar_script()` 校验真实 python 调用
（不是子串扫描）—— 避免 `python -c "print('create_plan.py')"` 这种把脚本
名混进字面量绕过 gate。非 `.py` required 项（如 INIT 的
`export AR_TASK_DIR=`）仍走子串匹配。PLAN / EDIT / REPLAN 需要 `git log`、
读文件等 ad-hoc 操作，使用 permissive；INIT / BASELINE / GENERATE_*
/ DIAGNOSE 收紧到 strict。DIAGNOSE 在 strict 下只放行 `create_plan.py`
（hook_guard_bash 还会按 artifact 状态再加一道 gate；详见 CLAUDE.md
不变量 #9）。

**查询函数**（[phase_policy.py](.autoresearch/scripts/phase_machine/phase_policy.py)）：
`check_bash` 和 `check_edit`，输入 phase 名 + 命令 / 文件名，返回
`(allowed, reason)`。纯函数，不读写任何状态。`check_bash` 按 bash 链
分隔符（`&&` `||` `;` `|`）切片后**逐段**评估——避免链式命令把单个
strict-required 子串带飞整个链路。

跨 phase 全局黑名单也在此定义：`quick_check.py` / `eval_wrapper.py` /
`keep_or_discard.py` / `settle.py` 在任何 phase 均禁止手动调用（只能由
`pipeline.py` 子进程执行）；`git commit` 仅允许 `keep_or_discard.py` 在
KEEP 时调用。读类命令（`ls` / `cat` / `grep` / `git log|diff|status` /
`dashboard.py` / `echo` / `pwd`）跨 phase 放行。

### 2. 状态文件

`<task_dir>/.ar_state/.phase` 存当前 phase，内容为一行文本。

写入该文件的主体：

- `_baseline_init.py` 在 baseline correctness 通过且 seed_metric 非空时写
  `PLAN`（由 `baseline.py` 调起；correctness 失败时 hard-exit，让
  `hook_post_bash.py` 走降级路径）
- `create_plan.py` 校验通过后写 `EDIT`
- `pipeline.py` 收尾时由 `compute_next_phase()` 计算并写入
- `hook_post_edit.py` 在 GENERATE_REF / GENERATE_KERNEL 验证通过后推进
  到 BASELINE（同一步里把 reference.py / kernel.py 提交成 git 基线，
  后续 rollback 才能回到种子而非 scaffold 占位）
- `hook_post_bash.py` 在 `export AR_TASK_DIR=` 激活、`baseline.py` 完成
  （成功 → PLAN；correctness 或 seed_metric 失败 → 降级到
  GENERATE_KERNEL 重试）、`create_plan.py` 通过（→ EDIT）后按情况写入

Claude 不直接写 `.phase`。`hook_guard_edit.py` 对 `.ar_state/` 的写入
走精确白名单：放行 `.ar_state/plan_items.xml`（`/autoresearch` 写给
`create_plan.py` 的 XML 入参）、`.ar_state/ranking.md`（仅 FINISH 阶段）、
以及 `.ar_state/diagnose_v<N>.md`（仅 DIAGNOSE 阶段，由 ar-diagnosis 子代理或
主 agent 在 fallback 路径写入；hook payload 不区分主/子代理，只校验内容）；
`.phase` / `progress.json` / `history.jsonl` / `plan.md` / 心跳和 marker
都由 Hook 和脚本机控，Claude 的 Edit/Write 拦在 PreToolUse 就会被驳回。
约束生效点是下一次 Hook 进程启动时的规则查询。

### 3. Hook 执行协议

Hook 是独立 Python 进程，stdin 读 Claude Code 传入的工具调用 JSON，
stdout 输出 `{"decision":"block","reason":"..."}` + `sys.exit(2)` 即阻断
（reason 作为工具错误反馈给 LLM）。
[hook_guard_bash.py](.autoresearch/scripts/hook_guard_bash.py) 和
[hook_guard_edit.py](.autoresearch/scripts/hook_guard_edit.py) 都只做三步：
读 `.phase` → 调 `check_bash` / `check_edit` → 过就 exit 0，不过就 block。

### 4. Hook 接线（`.claude/settings.json`）

| 事件 | 匹配工具 | Hook 脚本 | 职责 |
|------|----------|-----------|------|
| PreToolUse | Edit / Write | `hook_guard_edit.py` | 调 `check_edit`，按 phase 拦截非法写入 |
| PreToolUse | Bash | `hook_guard_bash.py` | 调 `check_bash`，按 phase 拦截非法命令，检测幻觉脚本名 |
| PostToolUse | Edit / Write | `hook_post_edit.py` | Edit 完成后更新 `.phase` |
| PostToolUse | Bash | `hook_post_bash.py` | 脚本退出后切 phase；处理 `export AR_TASK_DIR=` 激活 |
| Stop | — | `hook_stop_save.py` | 把 `last_stop_reason` 写入 progress.json（`save_progress` 顺带刷新 `last_updated`），供 resume 使用 |

`hook_guard_edit.py` 在 phase 规则之外还有全局约束：

- `.ar_state/` 精确白名单：`plan_items.xml`（`create_plan.py` 的 XML
  入参）、`ranking.md`（FINISH 阶段）、`diagnose_v<N>.md`（DIAGNOSE 阶段，
  内容契约见 CLAUDE.md invariant #9）可写。`plan.md` / `.phase` /
  `progress.json` / `history.jsonl` / 心跳和 marker 一律拦住——这些由
  `create_plan.py` / `settle.py` / `pipeline.py` 写入，手工修改会破坏
  审计记录或让状态机脱轨
- EDIT 阶段额外的 git gate：上一轮 kernel.py 未经过 `pipeline.py` 收尾
  就再次 Edit 会被拦截并提示先运行 `pipeline.py`，防止单轮内累积多个未
  结算改动

### 5. 为什么 Claude 绕不过去

- **规则单点**：`_BASH_RULES` / `_EDIT_RULES` 只在 phase_machine 定义一次，
  两个 PreToolUse hook 共享；改一处两端同步。
- **pipeline 子步骤禁止手调**：`quick_check.py` / `eval_wrapper.py` /
  `keep_or_discard.py` / `settle.py` 跨 phase 黑名单；`git commit` 全部走
  `git_utils.commit_in_task`，三个调用方分别是 `scaffold._git_init`（初始
  baseline）、`hook_post_edit`（GENERATE_REF / GENERATE_KERNEL 的 seed
  commit）、`keep_or_discard`（轮末 KEEP commit）。Claude 直接 `git commit`
  被全局 ban，也无法跑任一 pipeline 分片。
- **EDIT 的 git gate**：新轮首次 Edit 前若 editable_files 仍有未提交 diff
  且没有 `.edit_started` 标记，说明上一轮没走 `pipeline.py` 收尾——
  gate block 并提示先跑 pipeline，防止单轮累积多个未结算的 diff。
- **plan 校验阻塞**：`create_plan.py` 强制 ≥3 项、rationale 30-400 字符、
  最多 1 项纯调参，pid 由 `progress.json.next_pid` 单调分配（不复用不
  跳号）。不通过即非零退出，`hook_post_bash` 不推进 phase。REPLAN 时
  旧版 pending 项静默丢弃（`dropped` 字段报告，不写 history）。

### 6. Guidance 与 Resume

每次 phase 切换，Hook 内部生成 phase-specific 提示（包含 editable_files、
当前 active item、最近三条 history、剩余 budget 等），通过 `[AR Phase: ...]`
消息和 `additionalContext` 回注给 LLM。**LLM 只消费这条消息，不要自己
调取**——`phase_machine` 是库不是 CLI，`hook_guard_bash` 会拒绝把它当
脚本调用。如果某一步看不到新的 `[AR Phase: ...]`，下一条合法命令跑出来
hook 自然会再发一条；不要尝试手动"刷新"。

`/autoresearch --resume` 由 `resume.py` 定位最新 task 并 `export
AR_TASK_DIR=…`，PostToolUse 触发 `_handle_activation()`：存在 `.phase`
时直接恢复；仅存 `progress.json` 时调用 `compute_resume_phase()`，按
`seed_metric` / `baseline_correctness` / plan 状态路由 —— **seed 没
产出 latency 或 correctness 失败都会回落 GENERATE_KERNEL**，与
`hook_post_bash` 的在线降级语义一致，避免中断后 resume 把一个没过
verify 的 seed 直接带进 PLAN。reference.py / kernel.py 的存在性决定
GENERATE_REF / GENERATE_KERNEL / BASELINE 入口。DIAGNOSE 阶段的
guidance 要求 spawn subagent 输出 Root cause / Fix direction / What to
avoid，使下一轮 plan 换方向，避免只调整超参。

## Skills 库

仓库顶层 `skills/` 提供 DSL 优化素材（guides / cases / fundamentals /
examples 子目录），按 DSL 名字组织：

```
skills/triton-ascend/   Triton on Ascend NPU
skills/triton-cuda/     Triton on CUDA GPU
skills/cuda-c/          CUDA C
skills/cpp/             CPU C++
skills/tilelang-cuda/   TileLang DSL
skills/pypto/           PyTorch operator patterns
```

PLAN 阶段 Claude 根据 hook 输出的 `[AR Phase: PLAN]` 提示（由 hook
内部生成，agent 仅消费消息，不要也不能直接调 phase_machine 模块）执行
`Glob("skills/<dsl>/**/*.md")` 检索，Read 命中的 SKILL.md（YAML
frontmatter 含 id / category / description / keywords），把 id 写进
plan item 的 rationale。

`skills/` 下还有若干跨 DSL 的工作流 skill（如 `kernel-agent/`、
`kernel-workflow/`、`performance-summary/`、`task-constructor/`、
`designer/`），用于 agent 自身的行为指导；这些不需要额外安装，
Claude Code 会直接按 frontmatter 匹配到合适场景。所有 skill 文档
均通过 Glob + Read 访问，不会被拷贝或移动到 `.claude/` 下。

## 配置与状态

| 路径 | 用途 |
|------|------|
| `workspace/<op>_ref.py` / `workspace/<op>_kernel.py` | 候选 ref / kernel 源文件，`/autoresearch --ref/--kernel` 的输入 |
| `.autoresearch/config.yaml` | `default_dsl` / `worker_only_modules` / `hallucinated_scripts` |
| `.autoresearch/scripts/hw_detect.py` | DSL → backend 硬编码表；npu-smi / nvidia-smi / worker-status 派生 arch |
| `.autoresearch/code_checker.yaml` | CodeChecker 规则表（triton 模板 / autotune 合规） |
| `.autoresearch/scripts/ar_vendored/` | DSL adapter + profiler + msprof/nsys runner + HTTP worker server |
| `.autoresearch/scripts/ar_cli.py` | 统一 CLI：`ar_cli worker --start/--stop/--status`，支持 `--bg` daemon |
| `task.yaml` | 任务配置（每个 task 目录一份，含 dsl/backend/arch/framework 四字段；打包进 tarball 发 worker） |
| `.ar_state/progress.json` | 运行时状态 |
| `.ar_state/plan.md` | 规划 + 结算历史（权威态） |
| `.ar_state/history.jsonl` | 每轮 decision / metrics / commit |
| `.ar_state/plan_items.xml` | PLAN / DIAGNOSE / REPLAN 阶段 Claude 写给 `create_plan.py` 的 XML 入参 |
| `.ar_state/diagnose_v<N>.md` | DIAGNOSE 阶段产出的结构化诊断报告（marker + 三个 section + R\<n\> 引用），见 CLAUDE.md 不变量 #9 |
| `.ar_state/.phase` | 当前阶段 |
| `/tmp/ar_cache/<op>_<sha>/reference.pt` | Worker 缓存的 PyTorch ref 输出（首轮 verify 计算后复用；sha 随 `reference.py` 变化自动失效） |
| `.claude/settings.json` | Hook + 权限配置 |
| `.claude/settings.local.json` | API key、model 覆盖（不进 git） |
| `.claude/scheduled_tasks.lock` | Session lock（不进 git） |

## 依赖

- Python ≥ 3.10
- `pip install pyyaml torch`
- Claude Code CLI 或 VS Code 扩展
- 按 DSL 追加的运行期依赖（scaffold 时选了对应 DSL 才会被 adapter 拉入）：
  - `triton_ascend` / `tilelang_npuir`：`torch_npu`、`triton`、CANN（为了
    `profiler_npu` 的 `torch_npu.profiler`）
  - `triton_cuda` / `tilelang_cuda` / `pypto`：`triton`、CUDA runtime
  - `ascendc`：CANN toolkit（`msprof` CLI 在 PATH）
  - `cuda_c`：Nsight Systems（`nsys` CLI 在 PATH）
  - 所有 DSL 走 local `_profile_via_msprof` / `_profile_via_nsys` 时需要
    `pandas`（读 op_summary / nsys rep）
- 远端 NPU / CUDA 机器（可选），通过 SSH tunnel 暴露 worker HTTP 端口。
  worker 端 rsync 项目过去，`ar_cli.py worker --start` 即可。
