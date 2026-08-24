---
name: ascendc-cube-matmul-mix
description: Ascend C Cube（Matmul/卷积/Attention）算子 MIX 启动与落地实战。覆盖：如何让 Cube 核真正在 NPU 上执行并被 profiler 捕获（而非只跑 AIV/vector）、MultiCoreMatmulTiling 的 baseK/SetSingleShape/多核切分陷阱、L0A/L0B/L0C 容量约束、workspace 装配、MTE2→V 同步、Release/Debug 编译差异。触发关键词：matmul、cube、AIC、混合核、MIX、GetTensorC、Iterate、REGIST_MATMUL_OBJ、ml0/nl0/kl0=0、blockDim 死锁、L0B 溢出、结果全 0/NaN、no_npu、量化矩阵乘、grouped_matmul、attention。
---

# Ascend C Cube 算子 MIX 启动与落地

Cube（AIC）算子若用「向量核启动方式」拉起，AIC 相位根本不执行 → 结果全 0/未初始化，profiler 也抓不到 NPU kernel。以下是让 Cube 算子真正跑起来、算对、且被捕获的 8 个关键点，任何一个错都会静默产生「全 0 / NaN / 死锁 / 精度错」。经实测（int8 quant_matmul，Ascend910_9362）逐一验证。

## 1. 用 MIX 启动，而非 vector-only 启动
- 裸 chevron `<<<blockDim,l2Ctrl,stream>>>` 是 **AIV-only**，Cube 相位不跑。
- 正确方式：`ascendc_library()`（`$ASCEND_HOME/tools/tikcpp/ascendc_kernel_cmake/ascendc.cmake`）编译 kernel，生成 `aclrtlaunch_<kernel>.h`，用 `ACLRT_LAUNCH_KERNEL(<kernel>)(blockDim, stream, args...)` 拉起（MIX）。或走标准 op 框架（op_host+op_kernel+ops-info 注册）。
- 纯 CXX 工程即可：`include(ascendc.cmake); ascendc_library(k STATIC kernel.cpp)`。**不要**和 ASC-language(`.asc`/`find_package(ASC)`)混用——共存会静默破坏 device-binary 注册（kernel 拉起但不执行）。需要 `set(ASCEND_CANN_PACKAGE_PATH ${ASCEND_HOME_PATH} FORCE)`。

## 2. workspace / tiling 必须是 kernel 的最后两个参数
框架从「倒数第二个」参数取 sys workspace 给 `GetSysWorkSpacePtr()`（`REGIST_MATMUL_OBJ` 用它）。顺序错 → matmul 的 L1/L0 staging 写进一个很小的 tiling buffer → MTE 越界。签名收尾固定为 `..., workspace, tiling`。

## 3. Tiling（MultiCoreMatmulTiling）三个必设项
- `SetFixSplit(baseM, baseN, baseK)` 的 **baseK 必须显式**：传 `-1`（自动）在 int8 下会得到 `ml0/nl0/kl0=0` → device fault。
- `SetTraverse(MatrixTraverse::FIRSTM)`：不设则 Iterate 遍历序与 kernel 手写 `(cnt%roundM, cnt/roundM)` 对不上 → 摆放错位。
- `SetSingleShape(sM, sN, K)`：每核负责 `sM×sN` 区域。>1 核时 GetTiling 常需要它才成功（否则 `res=-1`, baseM/baseN=0）。

## 4. blockDim ≤ 物理 AIC 数（否则 MIX 启动死锁）
`blockDim > 物理 AIC` 会**挂死**（不是丢块）。tile 很多时不能一 tile 一核；而是把 `SetSingleShape` 区域按 base 增量放大，直到 `区域数 ≤ 物理 AIC 数`，每核在区域内走多个 base tile（`roundM=ceil(singleCoreM/baseM)`）。`get_usedCoreNum()` 即 blockDim。

## 5. L0 容量（int8，每块 L0 均 64KB）
`baseM·baseK ≤ 64K`（L0A）、`baseK·baseN ≤ 64K`（L0B）、`baseM·baseN·4 ≤ 128K`（L0C, int32）。`128×128×512` 三者刚好。常见坑：`baseN=256 + baseK=512` → L0B=128K 溢出 → 多 K-chunk 形状**静默算错**。

## 6. MTE2 → V 同步（加载 scale/bias 后接 vector 运算）
`DataCopy`（MTE2，如把 per-channel scale 拷进 UB）后紧接 `Mul`（V），**必须** 用 VECIN Que（`AllocTensor→DataCopy→EnQue→DeQue→用→FreeTensor`）或 `SetFlag/WaitFlag<HardEvent::MTE2_V>`。裸 `PipeBarrier<PIPE_V>` 只排 V 序、不排 MTE2→V → Mul 抢跑 → 小 M 形状 NaN（大 M 因 Cast 掩盖延迟侥幸过，故此 bug 极隐蔽）。

## 7. Cube kernel 用 Debug 编译
`set(CMAKE_BUILD_TYPE Debug ... FORCE)`。ascendc cube kernel 在 Release `-O2`（device.o merge）下会 fault；官方 mmex 例子也强制 Debug。

## 8. torch_npu 分批陷阱
`x[bi]`（连续张量的切片）仍带 `storage_offset`，而 `ConvertType`/`storage().data()` **忽略 offset** → 每批都读到 batch 0。分批分别拉起时用 `.clone()`（强制 offset=0 拷贝），不要用 `.contiguous()`（已连续则原样返回）。

## kernel 骨架（照抄可跑的 leakyrelu 例子）
`Matmul<A,B,C[,Bias]>` + 手写循环：`SetTensorA/B; if bias SetBias; while(mm.Iterate<true>()){ GetTensorC<true>(local,false,true); /*向量后处理*/ CopyOut; } mm.End();`；入口 `REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling)`。参考例子：CANN 包内 `matmul_leakyrelu_custom`（fp16）。int8 量化：`Matmul<int8,int8,int32,int32-bias>` cube 出 int32，再向量反量化（cast→scale→pertoken→clamp→cast half）。int8 权重 B 参考用 `CubeFormat::NZ`，但 ND 亦可（内部 nd2nz）。

相关：[[ascendc-hardware-tiling]] [[ascendc-ub-budget]] [[ascendc-crash-debug]]（死锁/全0/NaN 定位）。
