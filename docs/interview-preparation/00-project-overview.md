# SAR Deforestation Detection：项目事实总入口

> 审计基线：GitHub `main` commit `6d7ee247f4dd6d2d2e83e83227605c83c0cfd528`、仓库中的实验 notebook，以及最终实习报告 `TN09_Final_Rapport_Yifan_ZHANG_FR.pdf`。
>
> 证据标签：`VERIFIED` 表示当前源码或已保存的可执行产物直接支持；`PARTIALLY VERIFIED` 表示只有部分链路或有限证据；`REPORT ONLY` 表示仅由最终报告记录；`CONFLICT` 表示不同来源不一致；`UNSUPPORTED` 表示目前没有足够证据。
>
> 使用原则：本文件描述“当时实际做了什么”和“2026 年回看发现什么问题”。后者不能倒写成当年的实现。

## 1. One-Sentence Summary

这个项目使用 Sentinel-1 的 VV/VH 双通道森林雷达影像学习正常森林的重建模式，再从新影像的像素重建误差中提取并输出可能发生毁林的地理区域，最后与人工标注进行空间重叠评价。

## 2. 30 秒项目介绍

我在法国 IRD 做的是一个微小尺度毁林异常检测原型。数据是 Sentinel-1 的双通道 SAR 影像，因为热带雨林云层多，雷达比光学影像更适合持续观测。我只用正常森林切片训练 Autoencoder，让模型学习正常森林的重建模式；推理时计算输入和重建结果的逐像素误差，再通过聚类、连通区域过滤和相邻日期比较得到毁林候选区域。最后把结果转成带地理坐标的 Shapefile，在 QGIS 中与人工标注比较，并计算 Precision、Recall、F1 和 IoU。最终报告记录了一组结果，但当前仓库不能完整复现那组精确数值。

## 3. 60 秒项目介绍

我在 IRD 参与的是 Sentinel-1 SAR 微毁林检测项目。选择 SAR 的原因是它是主动雷达成像，受光照和云层影响较小，适合多云的亚马逊地区。输入是同一位置、同一日期的 VV 和 VH 极化影像；预处理后融合为 `2×256×256` 的 TIFF patch。报告记录的数据规模约为 32,000 个训练样本、8,000 个验证样本和 1,600 个测试样本。

数据不是随机切分，而是按地理区域切分：报告记录训练用 4 个区域、验证 1 个区域、测试 1 个有毁林标注的区域；训练和验证只选正常森林，但不同 split 的年份存在重叠，所以它是 spatial split，不是 temporal holdout。

模型是 CNN Autoencoder。它只学习正常森林，因此期望正常区域重建误差低，毁林或其他分布外区域误差高。代码中有单图 KMeans、五期影像共享 GMM，以及两日期大区域共享 GMM 三条后处理分支；聚类后再删除小连通区域，相邻日期的异常 mask 做差得到新变化，并矢量化为 Shapefile。最后使用 QGIS 和栅格化后的人工标注做定性、定量评价。这里需要诚实说明：最终报告记录的指标有证据，但仓库缺少对应的完整运行配置和中间结果，不能称为目前可完全复现的 benchmark。

## 4. 3 分钟技术版本

项目目标是在缺少大规模 SAR 像素标注的情况下，检测亚马逊森林中的小尺度毁林。数据来自 Sentinel-1 GRD、IW 模式的 VV/VH 双极化影像。SAR 是主动传感器，受云和光照影响小；光学 PlanetScope 影像主要用于我在 QGIS 中制作和核验毁林标注，而不是作为 Autoencoder 输入。

报告记录的数据按地理区域划分：4 个训练区域、1 个验证区域和 `622_975` 测试区域。训练与验证使用 2018–2024 年相近季节的正常森林，测试报告范围是 2021–2022 年。原始 VV/VH 经过外部 CuSum/pyroSAR 相关预处理后，仓库代码检查两个极化文件的 CRS、transform 和尺寸，将它们堆叠成双通道 GeoTIFF，再切成非重叠的完整 `256×256` patch。10 米分辨率下，每个 patch 约覆盖 `2.56 km × 2.56 km`，面积约 `6.5536 km²`。

模型输入是 `2×256×256`。Encoder 先用 stride-2 卷积降到 `64×128×128`，再经过 5 组 residual stages，得到 `64×64×64`、`128×32×32`、`256×16×16`、`512×8×8` 和 `512×4×4`。Residual block 在 stride 或通道变化时用 `1×1` projection 对齐 identity。Encoder 在 `4×4` feature map 上做 self-attention，再构造 FPN 的 p5、p4、p3、p2。p5 经过 global average pooling 和全连接层得到 512 维 latent。Decoder 把 latent 映射回 `256×4×4`，逐级上采样到 `2×256×256`；它实际只融合 p4 的 `8×8` 特征和 p3 的 `16×16` 特征。p2 虽然被计算，却没有进入 decoder。Decoder 还在 `256×32×32` 处使用一次 self-attention。

训练目标在代码里名为 MSE，但训练函数使用 `reduction='sum'`：每个 batch 对两个通道和所有像素的平方误差求和，epoch 再除以样本数，因此更准确地说是“每样本 summed squared reconstruction error”，而不是标准的全元素 mean MSE。异常图则使用 `MSELoss(reduction='none')` 计算逐元素平方误差，再对 VV/VH 两个通道求和。

后处理不是一条统一流水线。第一条分支对单张图的归一化 log-loss 用 KMeans 分两类；第二条选目标日期前后共 5 张影像，将它们的 pixel loss 合并后拟合两分量 GMM，再比较连续日期的 anomaly mask；第三条在两个指定日期的所有共同 tile 上共同拟合 GMM，增加固定 pixel-loss threshold、小连通区域过滤，再把“之前正常、当前异常”的区域矢量化为 Shapefile。这里 GMM 使用测试影像本身的无标签误差分布，属于 transductive post-processing，而不是训练集上预先确定的固定 detector。

评价阶段把 detection、forest mask 和 QGIS 人工标注统一 CRS、栅格化后计算 TP、FP、FN 及 Precision、Recall、F1、IoU。最终报告记录 `0.8284 / 0.6865 / 0.7508 / 0.6011`。但是仓库 notebook 保存了另一组 `0.9029 / 0.6930 / 0.7842 / 0.6450`，而且评估代码使用预测几何裁剪 annotation，可能漏算预测范围外的 FN。因此面试时应把前一组称为“最终报告记录值”，不能称为已由当前仓库重新复现的严格 benchmark。

CUSUM 是已有的 Sentinel-1 时序变化检测方法。这个项目参考了它的预处理，并把其输出作为定性参照；仓库没有保存同一协议下充分的量化 baseline 对比。报告中出现的 48–60 天延迟来自引用论文，不是本项目测出的结果。

## 5. End-to-End Flow

```text
Sentinel-1 GRD / IW VV+VH
  [REPORT ONLY：数据下载与完整 SAR 来源说明]
        ↓
SAR preprocessing
  [PARTIALLY VERIFIED：报告称包含 dB、边缘噪声、DEM/terrain correction、
   Lee Sigma、pyroSAR；仓库只有外部 CuSum preprocess 调用及后续清理脚本]
        ↓
分别保存的 VV / VH GeoTIFF，文件名含 *_gamma0-rtc_db*
  [PARTIALLY VERIFIED：文件名模式和处理代码可见，实际处理影像未提交]
        ↓
检查 CRS / transform / shape，堆叠为 2-band GeoTIFF
  [VERIFIED：split_data.py]
        ↓
切成完整、非重叠的 2×256×256 tiles；删除含 NaN 或全零的 tile
  [VERIFIED：split_data.py、remove_missing_values.py、remove_zero_values.py]
        ↓
按预先选定的地理区域放入 train / validation / test 目录
  [PARTIALLY VERIFIED：目录消费代码存在；具体区域与样本数来自报告]
        ↓
ProcessedForestDataset：TIFF → (C,H,W) → normalization → float tensor
  [VERIFIED]
        ↓
PyTorch DataLoader：batch、train shuffle、GPU 时 1 worker + pin_memory
  [VERIFIED]
        ↓
CNN Autoencoder：正常森林重建训练
  [VERIFIED]
        ↓
逐像素 squared reconstruction error，跨 VV/VH 求和
  [VERIFIED]
        ↓
三条分支：single-image KMeans / 5-image GMM / two-date large-area GMM
  [VERIFIED]
        ↓
Connected Components 小区域过滤
  [VERIFIED]
        ↓
连续日期或指定前后日期 anomaly mask 比较
  [VERIFIED]
        ↓
Rasterio vectorization → Shapefile
  [VERIFIED]
        ↓
QGIS 可视化与人工核验
  [REPORT ONLY：报告和 GIS 产物支持，仓库没有自动化 QGIS 操作]
        ↓
统一 CRS、栅格化、forest mask、Precision / Recall / F1 / IoU
  [PARTIALLY VERIFIED：公式与 notebook 可见，但最终报告数值不可精确复现，
   且评估区域裁剪存在正确性风险]
```

## 6. Historical Fact vs Repository Fact vs Modern Review

| Historical Project / Report | Current Repository Evidence | 2026 Review |
|---|---|---|
| 报告记录约 32k/8k/1.6k tiles。 | README 重复了近似数量，但数据和 manifest 未提交。 | 将数量称为“报告记录的约数”，不能说当前仓库已重新计数验证。 |
| 报告记录 4 个训练区域。 | README 写 3 个训练区域。 | 以更正式的最终报告为历史来源，并保留 `CONFLICT`。 |
| 报告测试年份为 2021–2022。 | README 写 2020–2022。 | 面试使用 2021–2022，并说明 README 不一致。 |
| 报告描述完整 SAR preprocess，包括 pyroSAR、RTC/地形处理和滤波。 | 仓库只保留调用外部 CuSum preprocess 的 notebook，以及融合、切片、NaN/全零清理。 | 不能声称完整 preprocessing 在本仓库可独立运行。 |
| 报告将最终全局 normalization 多次写为 `[-15,3]`；第 20 页又写 `[-15,-3]`。 | DataLoader 明确硬编码 `min=-15, max=-3`。 | 以源码 `[-15,-3]` 描述实现；报告范围标记为 `CONFLICT`。 |
| 报告说 Optuna 帮助选择学习率、weight decay、StepLR 参数。 | 代码只搜索 lr 和 weight decay，10 trials。 | 不把 scheduler 参数描述成 Optuna 搜索结果。 |
| 报告称最终指标为 P 0.8284、R 0.6865、F1 0.7508、IoU 0.6011。 | 当前 notebook 保存另一组 P 0.9029、R 0.6930、F1 0.7842、IoU 0.6450。 | 报告值保留为 `REPORT ONLY`，不包装成可复现 benchmark。 |
| 当年研究重点是得到可用原型和空间结果。 | 路径、阈值和运行编排大量硬编码，重复逻辑明显。 | 这是合理的 research prototype，但不是长期运行系统；不要倒写成当时已有生产化能力。 |
| 当年引入 1 个 DataLoader worker 和 pinned memory。 | GPU 时 `num_workers=1, pin_memory=True`；没有显式 prefetch 配置。 | 可说“单 worker 后台加载和 pinned memory”，不可说“多个 worker 并行预取”。 |
| 当年使用 seed、CLI、requirements、checkpoint 管理实验。 | 只有 `torch.manual_seed`；缺 NumPy/Python/Optuna seed、deterministic、数据版本和完整训练状态。 | 只能称为“具备部分实验管理与复现基础”。 |
| 当年把 CUSUM 作为已有方法参照。 | 有调用 notebook、报告描述和 CUSUM 输出产物，但无统一协议量化表。 | 称“时序方法参照/定性对比”，谨慎使用“baseline”。 |

## 7. Dataset Facts

| Item | Fact | Evidence status |
|---|---|---|
| Train regions | 报告记录 4 个：`622_971`、`623_972`、`623_973`、`624_972`。README 写 3 个。 | `CONFLICT`，面试以报告为准 |
| Validation region | 报告记录 1 个：`621_970`。 | `REPORT ONLY` |
| Test region | 报告记录 1 个：`622_975`，包含人工毁林标注。 | `REPORT ONLY`，仓库有同名 GIS 产物 |
| Train years | 2018–2024，每年约 5 月 1 日至 10 月 1 日。 | `REPORT ONLY` |
| Validation years | 2018–2024，同季节。 | `REPORT ONLY` |
| Test years | 最终报告为 2021–2022；README 为 2020–2022。 | `CONFLICT` |
| Pre-tile image count | Train 404、validation 122、test 31 张双通道 SAR 图，约 `2×2000×2000`。 | `REPORT ONLY` |
| Tile count | Train 约 32,000、validation 约 8,000、test 约 1,600。 | `REPORT ONLY` |
| Patch shape | `2×256×256`，channel-first。 | `VERIFIED` |
| Pixel resolution | 报告记录约 10 m/pixel。 | `REPORT ONLY` |
| Patch ground width | `256×10 m = 2,560 m = 2.56 km`。 | 由报告分辨率和源码 patch size 计算 |
| Patch ground area | `2.56 km×2.56 km = 6.5536 km² = 655.36 ha`。 | 同上 |
| Tile overlap | 单张大图内 stride 等于 tile size，只保存完整 tile，因此代码生成的 tile 不重叠。 | `VERIFIED` |

数据划分的准确表述：

- 这是 **spatial split**：训练、验证、测试使用不同编号的地理区域。
- 这 **不是 random split**：源码没有将所有 tile 打乱后随机划分。
- 这 **不是 temporal holdout**：训练/验证的 2018–2024 与测试的 2021–2022 存在年份重叠。
- 同一地点在不同日期会产生多个 tile；这是时序观测，不等同于空间 tile 重叠。
- 区域相邻可能仍存在 spatial autocorrelation；当前没有缓冲区或空间距离分析。

## 8. Model Facts

### 8.1 Autoencoder 的真实 shape

省略 batch 维：

```text
Input                         2 × 256 × 256
Initial 7×7, stride 2       64 × 128 × 128
Encoder stage 1             64 ×  64 ×  64   = x1
Encoder stage 2            128 ×  32 ×  32   = x2
Encoder stage 3            256 ×  16 ×  16   = x3
Encoder stage 4            512 ×   8 ×   8   = x4
Encoder stage 5            512 ×   4 ×   4   = x5
Encoder self-attention      512 ×   4 ×   4

FPN p5                      256 ×   4 ×   4
FPN p4                      256 ×   8 ×   8
FPN p3                      256 ×  16 ×  16
FPN p2                      256 ×  32 ×  32

Global average pool        256 ×   1 ×   1
Fully connected             512 latent

FC reshape                  256 ×   4 ×   4
Decoder 1 + p4              256 ×   8 ×   8
Decoder 2 + p3              256 ×  16 ×  16
Decoder 3                   256 ×  32 ×  32
Decoder self-attention      256 ×  32 ×  32
Decoder 4                   128 ×  64 ×  64
Decoder 5                    64 × 128 × 128
Decoder 6                    32 × 256 × 256
Final Conv + Tanh             2 × 256 × 256
```

### 8.2 必须记住的实现细节

- AE 的 latent 是硬编码的 512；`--embedding-size` 对 AE 当前实现不起作用。
- 每个 encoder stage 有两个 residual blocks；第一个通常用 stride 2 下采样。
- stride 不为 1 或 channel 数变化时，identity 走 `1×1 Conv + BatchNorm` 对齐。
- Self-attention 的 Q/K channel 是输入 channel 的 1/8，V 保持原 channel，并通过可学习 `gamma` 与输入残差相加。
- Encoder attention 位于 `4×4`，attention matrix 是 `16×16`；decoder attention 位于 `32×32`，matrix 是 `1024×1024`。
- FPN 的 `p5/p4/p3/p2` **全部被计算**。
- Decoder **实际只使用 p4 和 p3**。
- `p2` 被计算并返回，但 **没有进入 decoder**；不能说四层 FPN 特征都被 decoder 使用。
- p5 参与 latent 生成，但 decoder 从 latent 开始后没有直接加 p5。
- 训练 loss 使用 summed squared error；pixel anomaly map 使用逐元素 squared error 后跨两个通道求和。

## 9. Anomaly Detection Facts

当前源码不是一个统一配置化 detector，而是三条独立分支。

### 9.1 Single-image KMeans

入口：`reconstruct_and_analyze_images`。

```text
一张 test image
→ reconstruction
→ 两通道 squared error 求和
→ log(error + 1e-8)
→ 1%/99% percentile clipping
→ [0,1] visualization normalization
→ KMeans(n_clusters=2, random_state=0)
→ 平均 loss 更高的 cluster 作为 anomaly
→ connected-component filtering，min_size=50
→ PNG visualization
```

用途是单图可视化和快速检查，不产生整区 Shapefile。

### 9.2 Five-image Temporal GMM

入口：`reconstruct_and_analyze_images_by_time_sequence` 和 `..._by_clustering`。

```text
目标日期前 2 张 + 目标日期 + 后 2 张
→ 分别计算 pixel loss
→ 合并 5 张图的所有 loss
→ GaussianMixture(n_components=2, random_state=0)
→ 均值更高的 component 作为 anomaly
→ 每张图 connected-component filtering，min_size=50
→ 相邻日期 anomaly mask 比较
```

两个 temporal 函数的区别：

- `by_time_sequence` 累积 `previous_anomalies`，区分“新出现”和“以前已经出现”的异常。
- `by_clustering` 只计算相邻两张 mask 的 `0→1` difference。

### 9.3 Two-date / Large-area GMM

入口：`generate_large_change_map`。

```text
指定 target_date 与 prev_date
→ 按 row/col 匹配两个日期的共同 tiles
→ 计算两日期所有共同 tile 的 pixel loss
→ 用这些 test-time loss 共同拟合 2-component GMM
→ 高均值 component 作为 anomaly
→ 额外应用 pixel_loss_threshold=1.0
→ target / previous 各自过滤，默认 min_size=100
→ (previous == normal) AND (target == anomaly)
→ 再做一次 connected-component filtering
→ 每 tile vectorization
→ 合并输出 Shapefile
```

注意：`suffix_template` 和 `tile_size` 参数在当前函数主体中没有真正用于搜索/计算；文件匹配依赖硬编码 regex。

## 10. Reported Results

最终报告记录：

| Metric | Final report value | Status |
|---|---:|---|
| Precision | 0.8284 | `REPORT ONLY` |
| Recall | 0.6865 | `REPORT ONLY` |
| F1 | 0.7508 | `REPORT ONLY` |
| IoU | 0.6011 | `REPORT ONLY` |

这四个值内部数学关系一致，例如由 Precision 和 Recall 计算出的 F1 约为 0.7508。但当前 repository **不能完整复现这些精确数值**，原因包括：

- 没有对应的 checkpoint、输入 TIFF、运行配置、阈值记录和中间 mask。
- 最终报告没有给对应 TP、FP、FN。
- 当前 `report/metrics/test.ipynb` 保存的另一组输出是：Precision 0.9029、Recall 0.6930、F1 0.7842、IoU 0.6450。
- `Test_Metric_Sklearn.ipynb` 还保存了另一种 forest-mask/annotation 比较结果，不是最终模型 benchmark。
- 当前 evaluation notebook 存在 prediction-dependent clipping 风险。

面试中的准确说法：

> 最终报告记录了 Precision 0.8284、Recall 0.6865、F1 0.7508 和 IoU 0.6011。这些是在选定测试区域上，通过空间结果与人工标注比较得到的报告值；由于仓库没有保留完整 checkpoint、数据版本和运行配置，我不会把它描述成现在可以一键复现的严格 benchmark。

## 11. Known P0 Issues

### A. Training / inference normalization inconsistency

**Problem**

训练与 DataLoader 路径使用 `[-15,-3]` MinMax normalization；temporal 函数调用可选 normalization helper 时没有传入 min/max，large-area 分支则直接将原始 TIFF 数值送入模型。

**Why it matters**

模型训练时看到的数值分布和整区推理时不同，重建误差可能主要反映尺度错配，而不是毁林。这尤其严重，因为报告把统一 normalization 描述为解决异常误差反转的关键步骤。

**Interview-safe explanation**

> 当时我识别出训练、验证和测试必须共享训练集 normalization；但复盘最终仓库时，我发现大区域推理函数没有复用同一个 preprocessing helper。这是提交版本中的一致性缺陷，因此报告结果需要结合当时实际运行版本看，不能仅凭当前函数完全复现。

**How I would fix it today**

建立唯一的 `Preprocessor`/transform，由训练和所有推理入口共同调用；把 min/max 与模型一起写入 checkpoint metadata；增加输入范围断言和 train-vs-inference parity test。

### B. Evaluation clips annotation using prediction geometry

**Problem**

评估 notebook 使用 detection geometry 的 union 同时裁剪 forest mask 和 annotation，然后才栅格化和计算指标。

**Why it matters**

真实标注中位于预测 geometry 之外的部分可能在计算前被删除，造成 FN 漏计，并可能高估 Recall、F1 和 IoU。栅格范围也由 prediction bounds 决定，不是固定研究区。

**Interview-safe explanation**

> 当前 notebook 的空间裁剪边界依赖预测结果，这不是严格的评估协议。报告指标可以作为历史结果陈述，但如果今天重新评估，我会用预先固定、与预测无关的 AOI 和栅格网格重新计算。

**How I would fix it today**

在看预测前固定 AOI、CRS、resolution、transform、width 和 height；ground truth 与 prediction 都 rasterize 到同一网格；明确 valid-data mask，再计算完整 TP/FP/FN，并保存 evaluation manifest。

### C. Reported metrics cannot currently be reproduced exactly

**Problem**

报告值、当前 notebook 已保存值和其他 notebook 值不一致，且缺少报告值对应的 checkpoint、数据快照、TP/FP/FN、阈值与运行 ID。

**Why it matters**

无法证明同一代码和数据能再次生成报告值；错误地称为 `VERIFIED benchmark` 会在面试追问中失去可信度。

**Interview-safe explanation**

> 我会明确区分“最终报告记录值”和“当前仓库可复现值”。前者有正式报告证据，但当前 repository 的实验封装不足以精确复跑。

**How I would fix it today**

保存不可变数据 manifest、commit SHA、checkpoint、完整配置、post-processing 参数、AOI、环境 lockfile、TP/FP/FN 和机器可读 metrics JSON；在 CI 或受控环境中提供复算脚本。

### D. Report normalization range conflict

**Problem**

报告第 20 页写约 `[-15,-3]`，第 28–30 页多次写 `[-15,3]`；当前代码是 `[-15,-3]`。

**Why it matters**

上界符号差 6 dB，会明显改变归一化值。面试时混用两个范围会暴露对实现不熟悉。

**Interview-safe explanation**

> 当前源码的实际常量是 `min=-15, max=-3`。报告后文的 `+3` 与源码和报告前文冲突，我把它视为文档或版本不一致，不会声称两者都正确。

**How I would fix it today**

不在代码和报告中手写常量；从训练数据版本计算或从批准配置加载，并将最终值自动导出到报告和 checkpoint metadata。

### E. Test-region GMM is transductive post-processing

**Problem**

GMM 在所选测试日期或两日期的测试像素 loss 上现场拟合，而不是在训练/验证集上确定后冻结。

**Why it matters**

它没有使用标签，因此仍然是无监督；但 detector 会根据每次测试分布自适应，结果不等同于固定模型在 untouched holdout 上的纯 inductive evaluation，也可能受当次异常比例影响。

**Interview-safe explanation**

> Autoencoder 参数是只用正常森林训练的；GMM 是无标签的 test-time adaptive post-processing。它提高了不同影像间阈值适应性，但评价协议属于 transductive，不是完全冻结的 detector。

**How I would fix it today**

同时评估两种协议：一套在 validation loss distribution 上冻结 threshold/GMM；另一套保留 test-time adaptation，并明确标注 transductive setting。比较两者稳定性。

### F. `[0,1]` input vs Tanh `[-1,1]` output mismatch

**Problem**

训练数据按设计主要映射到 `[0,1]`，但 decoder 使用 `Tanh`，允许输出 `[-1,1]`。

**Why it matters**

输出域与主要输入域不完全对齐，模型浪费一部分动态范围；这不必然使模型失败，但不是最自然的设计。测试异常还可能因未 clamp 而落在 `[0,1]` 之外。

**Interview-safe explanation**

> 当前代码用 Tanh，但训练输入主要是 `[0,1]`。这是一个值得通过 ablation 验证的设计不匹配，我不会声称 Tanh 是理论最优选择。

**How I would fix it today**

先统一输入约定，再比较 Sigmoid、Tanh 配合 `[-1,1]` normalization、以及 linear output；在相同 split 和固定后处理下比较 reconstruction 与 detection 指标。

## 12. What I Can Safely Claim

### SAFE CLAIMS

- 我构建了一个以正常森林 SAR patch 为训练数据的无监督 Autoencoder 异常检测原型。
- 输入是 Sentinel-1 VV/VH 双通道、`2×256×256` 的 GeoTIFF patch。
- 数据按地理区域划分，而不是从所有 tile 中随机切分。
- 报告记录了约 32k 训练、8k 验证和 1.6k 测试 patch；这些是近似历史数据量。
- 当前模型源码真实包含 CNN、Residual Blocks、FPN 和 Self-Attention。
- AE 的真实 latent dimension 是 512。
- FPN 计算 p5/p4/p3/p2，但 decoder 只融合 p4/p3。
- 训练使用平方重建误差；实现上是 sum reduction，而 pixel anomaly map 对两个通道的逐元素平方误差求和。
- 后处理真实包含 KMeans、GMM、connected-component filtering、时间差分和 Shapefile vectorization。
- 我使用 TensorBoard 记录 train/validation loss；Optuna 以 validation loss 为目标搜索 lr 和 weight decay。
- GPU 路径配置了 1 个 DataLoader worker 和 `pin_memory=True`。
- 我处理过 normalization 不一致、动态切片 IO 和地理投影丢失等实际 debugging 问题。
- 最终报告记录 P 0.8284、R 0.6865、F1 0.7508、IoU 0.6011，但当前仓库不能完整复现这组精确值。
- CUSUM 是项目中的已有时序方法参照；我的方法侧重空间重建异常和日期间异常 mask 比较。

## 13. What I Must NOT Claim

### DO NOT SAY

- “所有 SAR preprocessing 都在这个 repository 中完整自动化。”
- “当前仓库可以一键完全复现最终报告结果。”
- “DataLoader 使用多个 worker 并行预取。”当前 GPU 配置只有 `num_workers=1`。
- “我显式配置了 `prefetch_factor`、`persistent_workers` 和异步 CUDA copy。”源码没有。
- “Optuna 搜索了 scheduler 的 step size 和 gamma。”源码只搜 lr 与 weight decay。
- “这是严格 untouched independent holdout benchmark。”测试影像用于 GMM test-time fitting 和研发分析。
- “数据是 temporal holdout。”年份与训练/验证重叠。
- “所有四层 FPN feature 都送进 decoder。”decoder 只加 p4/p3。
- “训练 loss 是标准 mean MSE。”当前实现是 sum reduction 后除以样本数。
- “CUSUM 在我们的实验中慢 48 天。”报告引用的论文结果是 48–60 天，不是本项目测量。
- “项目证明 Autoencoder 显著优于 CUSUM。”缺少同一协议下充分的量化对照。
- “报告 normalization 明确就是 `[-15,3]`。”报告内部冲突，源码是 `[-15,-3]`。
- “最终指标已经由 repository notebook 验证。”notebook 保存的是另一组结果，且评估协议有风险。
- “这是 production-ready 系统。”它是研究原型。

## 14. Interview Positioning

面向 backend、systems、infrastructure、general software engineering 和 AI application / agent engineering 岗位，最值得强调的 5 项能力如下。

### 1. Data pipeline 与数据契约

我处理了多来源文件、双通道对齐、CRS/transform 检查、tile 切分、无效值清理、channel-first tensor 约定和地理信息保留。重点不是遥感术语，而是“如何把不稳定的大文件输入转成模型可消费、可验证的数据契约”。

### 2. PyTorch 训练工程

我实现了 Dataset/DataLoader、模块化模型、训练/验证循环、Adam、StepLR、early stopping、checkpoint 和 TensorBoard。面试时既能解释组件职责，也能承认当前 checkpoint 与配置管理不够完整。

### 3. 性能诊断与 IO trade-off

我曾尝试训练时动态切片，发现重复处理拖慢 IO 且增加坏样本错误，于是改成离线预切片，用更多存储换更稳定、更快的训练输入。这个 trade-off 对后端批处理和数据系统同样适用。

### 4. Debugging 与跨阶段一致性

我定位过不同 split 各自 normalization 导致异常关系反转，也处理过 TIFF 读写方式造成 georeference 丢失和投影误差。2026 复盘又识别到最终推理和评估协议中的一致性缺陷，说明我现在会检查 train-serving skew 和 evaluation leakage。

### 5. Experimental thinking 与诚实的边界管理

我保留 baseline 思维，使用 validation、TensorBoard 和有限 Optuna 搜索，并能说明 FPN skip、阈值、连通区域过滤在 precision/recall 间的 trade-off。更重要的是，我能区分报告结果、仓库证据和现代改进建议，不把研究原型包装成已经生产化的系统。
