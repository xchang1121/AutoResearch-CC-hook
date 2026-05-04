# AutoResearch

基于 Claude Code 的算子迭代优化框架。Claude 负责读代码、写 plan、改
kernel、诊断失败；Hook 负责阶段转移、plan 校验、eval 调度、KEEP/DISCARD、
回滚。Python + PyYAML，零运行期外部依赖。

## Quick Start

```bash
cd claude-autoresearch
claude
```

候选源文件放 [workspace/](workspace/)，命名 `<op_name>_ref.py` /
`<op_name>_kernel.py`。启动后（scaffold + 首轮 baseline 原子执行，进入
PLAN）：

```
/autoresearch --ref workspace/sinkhorn_ref.py --kernel workspace/sinkhorn_kernel.py \
  --op-name sinkhorn --dsl triton_ascend --devices 5 --max-rounds 200
```

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

`/autoresearch` 入参语义：

- `--`-prefixed flag → 新建任务（scaffold + 首次 baseline 原子完成）
- 已存在的目录路径 → resume 该目录
- `--resume` → resume 最近活跃 task
- 无参数 → 交互式询问

CLI 只暴露三个维度，其余全部派生：

| flag | 取值 | 说明 |
|------|------|------|
| `--dsl` | `triton_ascend` / `triton_cuda` / `ascendc` / `cuda_c` / `cpp` / `tilelang_cuda` / `tilelang_npuir` / `pypto` / `swft` / `torch` | **必填**，新建 task 时必须显式传值 |
| `--devices` | 本地 NPU/GPU 下标，逗号分隔：`5` 或 `0,1,2,3` | **XOR**：和 `--worker-url` 二选一 |
| `--worker-url` | 远端 worker URL：`127.0.0.1:9070` | **XOR**：和 `--devices` 二选一 |
| `--framework` | `torch` / `mindspore` / `numpy` | 默认 `torch` |
| `--no-code-checker` | 关闭 CodeChecker 静态分析（仅 triton_* 默认规则；其他 DSL 会误报，scaffold 时关掉，事后改 `task.yaml: code_checker.enabled: false` 也能切换）| 可选 |

派生项（用户不写）：`backend` 由 DSL 决定（`triton_ascend → ascend`、
`cuda_c → cuda`、…，见
[hw_detect.py](.autoresearch/scripts/hw_detect.py)）；`arch` 由 `--devices`
本地探测（`npu-smi info` / `nvidia-smi`）或 `--worker-url` 上 `GET
/api/v1/status` 自报。

## 主循环

单轮：**PLAN → EDIT → quick_check → eval → KEEP/DISCARD → settle**。
连续 3 次 FAIL 切到 DIAGNOSE，plan 全部 settle 切到 REPLAN，预算耗尽切到
FINISH。

```
INIT
  ├─ (--desc?)             GENERATE_REF ─→ GENERATE_KERNEL
  ├─ (--ref only?)                        GENERATE_KERNEL
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
`phase = EDIT`。每个 `pN` 要么在 `history.jsonl` 里有 KEEP / DISCARD / FAIL
终态，要么在 REPLAN/DIAGNOSE 边界被静默丢弃；pid 单调推进、不复用。

阶段产物：

| 阶段 | Claude 操作 | 产物 |
|------|-------------|------|
| GENERATE_REF | Edit `reference.py` | reference.py |
| GENERATE_KERNEL | Edit `kernel.py` | kernel.py (种子) |
| BASELINE | `baseline.py` | seed_metric → progress.json |
| PLAN / DIAGNOSE / REPLAN | `create_plan.py` | plan.md（含 (ACTIVE) 标记）+ 全局 pN |
| EDIT | Edit `kernel.py` → `pipeline.py` | history.jsonl + 可选 git commit + 下一 .phase |
| FINISH | Write `ranking.md` | ranking.md |

## Dashboard

```bash
python .autoresearch/scripts/dashboard.py             # 当前任务，5 秒刷新
python .autoresearch/scripts/dashboard.py <task_dir> --watch 2
```

键位：`↑` / `↓` / `PgUp` / `PgDn` / `Home` / `End` 滚动 history，`q` /
`Esc` 退出。顶栏：task 名 / 阶段 / plan 版本 / budget / Baseline / Seed
/ Best / 改进比；下栏：history 表 + 当前 plan。

## 远程 Worker

远端 NPU / CUDA 通过 SSH tunnel 接入。HTTP server 自带
（[ar_vendored/worker/](.autoresearch/scripts/ar_vendored/worker/) +
`core/worker/` + `core/async_pool/`），worker 端依赖：`fastapi` +
`uvicorn`、`torch`（+ `torch_npu` / CUDA runtime）、按 DSL 追加
`triton` / `pandas` / `msprof` / `nsys`。

```bash
# worker 机器（先 conda activate / source env.sh）
python .autoresearch/scripts/ar_cli.py worker --start \
    --backend ascend --arch ascend910b3 --devices 2,5 \
    --host 127.0.0.1 --port 9111 --bg

# 本地
ssh -f -N -L 127.0.0.1:9002:127.0.0.1:9002 \
  -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 npu
curl http://127.0.0.1:9002/api/v1/status
# {"status":"ready","backend":"ascend","arch":"ascend910b3","devices":[4]}
```

`ar_cli.py worker` 还有 `--stop` / `--status` / 前台模式，详见
`--help`。多 URL 逗号分隔，框架按可达性挑选。

## 精度与 metric

Reference 输出由 worker 端按需计算并缓存：

1. 首轮：worker 在 sandbox 跑 `Model(*get_init_inputs())(*get_inputs())`，
   缓存到 `/tmp/ar_cache/<op>_<sha(reference.py)>/reference.pt`
2. 后续：命中缓存直接 `torch.load`
3. 跑 `ModelNew.forward`、与 ref 比对，输出 `max_abs` / `max_rel` /
   `bad_elems(%)`

`reference.py` 改了 → sha 变 → 缓存自动失效。容差在 `task.yaml` 配置：

```yaml
metric:
  primary: latency_us
  correctness_atol: 1.0e-2
  correctness_rtol: 1.0e-2
```

verify 失败时 ref 时延仍由 `/api/v1/profile` 单独测得，与 verify 解耦，
dashboard 顶栏始终显示 PyTorch baseline。

## Skills 库

`skills/` 提供 DSL 优化素材，按 DSL 名字组织（`skills/triton-ascend/` /
`skills/triton-cuda/` / `skills/cuda-c/` / `skills/cpp/` /
`skills/tilelang-cuda/` / `skills/pypto/`）。PLAN 阶段 hook 提示 Claude
`Glob("skills/<dsl>/**/*.md")` 检索，把命中的 SKILL id 写进 plan item
rationale。还有跨 DSL 的工作流 skill（`kernel-agent/` / `kernel-workflow/`
/ `designer/` / …），Claude Code 按 frontmatter 自动匹配，无需安装。

## 配置与状态

| 路径 | 用途 |
|------|------|
| `workspace/<op>_ref.py` / `<op>_kernel.py` | 候选 ref / kernel 输入 |
| `task.yaml` | 任务配置（dsl / backend / arch / framework / metric / editable_files） |
| `.ar_state/.phase` | 当前阶段 |
| `.ar_state/plan.md` | 规划 + 结算历史（权威态） |
| `.ar_state/history.jsonl` | 每轮 decision / metrics / commit |
| `.ar_state/progress.json` | 运行时状态 |
| `.ar_state/plan_items.xml` | PLAN/DIAGNOSE/REPLAN 写给 `create_plan.py` 的 XML |
| `.ar_state/diagnose_v<N>.md` | DIAGNOSE 结构化诊断报告（见 CLAUDE.md 不变量 #10） |
| `.autoresearch/config.yaml` | `default_dsl` / `worker_only_modules` / `hallucinated_scripts` |
| `.autoresearch/code_checker.yaml` | CodeChecker 规则（triton 模板 / autotune 合规） |
| `.claude/settings.json` | Hook + 权限配置 |
| `.claude/settings.local.json` | API key、model 覆盖（不进 git） |

`.ar_state/` 内除 `plan_items.xml` / `ranking.md`(FINISH) /
`diagnose_v<N>.md`(DIAGNOSE) 外都由 hook 和脚本机控，Claude 不能手写。

## 内部机制（按需阅读）

外部接口稳定（slash 命令、`task.yaml`、`.ar_state/` 路径），下面是想动
内部时的入口：

| 想了解 | 看哪里 |
|--------|--------|
| Bash gate（哪条命令在哪个 phase 合法） | [phase_policy.py](.autoresearch/scripts/phase_machine/phase_policy.py) 头部注释——三层架构：`classify` → 静态 phase 表 → `check_bash`。AR-script / lifecycle / readonly 分类 + canonical-form grammar。 |
| Hook 接线 | [.claude/settings.json](.claude/settings.json) 注册 5 个 hook（PreToolUse Edit/Bash/Task + PostToolUse Edit/Bash + Stop）；脚本一律 `hook_*.py`，每个文件首段 docstring 说明职责。 |
| phase 转移 | [phase_machine/state_store.py](.autoresearch/scripts/phase_machine/state_store.py) 定义阶段常量；`compute_next_phase` / `compute_resume_phase` 在 [phase_policy.py](.autoresearch/scripts/phase_machine/phase_policy.py) 末尾。 |
| DSL adapter（profiler / autotune / 编译选项） | [ar_vendored/op/verifier/adapters/factory.py](.autoresearch/scripts/ar_vendored/op/verifier/adapters/factory.py) 注册 10 个 adapter；模板生成在 [task_config/package_builder.py](.autoresearch/scripts/task_config/package_builder.py)。 |
| 本地 vs 远端执行路由 | [local_worker.py](.autoresearch/scripts/local_worker.py) 按 DSL 分流到 `_profile_via_subprocess` / `_profile_via_msprof` / `_profile_via_nsys`；远端走 [ar_vendored/worker/server.py](.autoresearch/scripts/ar_vendored/worker/server.py)。 |
| CodeChecker 规则 | [code_checker.py](.autoresearch/scripts/code_checker.py) + `.autoresearch/code_checker.yaml`。当前覆盖 `triton_ascend` / `triton_cuda`，其他 DSL 关掉。 |
| DIAGNOSE 契约 | [CLAUDE.md](CLAUDE.md) 不变量 #9（canonical-form bash）和 #10（DIAGNOSE artifact）。 |
| 子代理（ar-diagnosis） | [.claude/agents/ar-diagnosis.md](.claude/agents/ar-diagnosis.md) prompt + 工具白名单。 |

## 依赖

- Python ≥ 3.10
- `pip install pyyaml torch`
- Claude Code CLI 或 VS Code 扩展
- 按 DSL 追加（scaffold 选了对应 DSL 才会用到）：
  - `triton_ascend` / `tilelang_npuir`：`torch_npu` + `triton` + CANN
  - `triton_cuda` / `tilelang_cuda` / `pypto`：`triton` + CUDA runtime
  - `ascendc`：CANN toolkit（`msprof` 在 PATH）
  - `cuda_c`：Nsight Systems（`nsys` 在 PATH）
  - 走 `_profile_via_msprof` / `_profile_via_nsys` 时还需 `pandas`
- 远端机器（可选）：SSH tunnel 暴露 worker 端口；rsync 项目过去
  `ar_cli.py worker --start`。
