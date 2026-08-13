# SAR Deforestation Detection：校招面试复习手册

> 使用方式：面试前 2–5 分钟读 Part 1 或文末 Last-Minute Card；有 30–45 分钟时读 Part 2–6；被追问时用 Top 20 的 cross-reference 回到长文档。
>
> 三种口径必须分开：
>
> - **HISTORICAL PROJECT / REPORT**：当年报告记载的工作和结果。
> - **CURRENT CODE REVIEW**：当前 repository 能证明的行为及复盘发现的问题。
> - **IF REBUILDING TODAY**：现在会采用的改进方案，不得倒写成当年已实现。

---

# Part 1 — Five-Minute Cheat Sheet

## 1. 一句话项目介绍

我使用 Sentinel-1 VV/VH 双通道 SAR 影像训练只学习正常森林的 Autoencoder，再把逐像素重建误差经聚类、空间过滤和日期间比较转成具有地理坐标的毁林候选区域，并与人工标注进行评价。

## 2. 30 秒项目介绍

我在法国 IRD 做的是一个热带雨林微毁林检测原型。输入是 Sentinel-1 的 VV/VH 双通道 SAR patch；SAR 受云层和光照影响较小，适合多云地区。由于毁林像素标注有限，我只用正常森林训练 Autoencoder，推理时计算输入与重建结果的逐像素误差，再用 KMeans/GMM、连通区域过滤和日期间 `0→1` 比较产生变化候选，最终导出 Shapefile，在 QGIS 中与人工标注比较。最终报告记录了一组 Precision/Recall/F1/IoU，但当前仓库不足以严格复现。

## 3. 一条 End-to-End Flow

```text
Sentinel-1 VV/VH
→ preprocessing / data preparation
→ 2×256×256 GeoTIFF tiles
→ Dataset / DataLoader
→ normal-only CNN Autoencoder
→ VV+VH pixel reconstruction error
→ single-image KMeans / temporal GMM
→ connected-component filtering
→ anomaly-mask temporal difference
→ Shapefile / QGIS
→ fixed-region ground-truth evaluation
```

注意：最后一步是理想的固定区域评价。当前 evaluation notebook 使用 prediction geometry 裁 annotation，存在 FN 漏计风险。

## 4. Model Cheat Sheet

| Item | Must remember |
|---|---|
| Input | `2×256×256`，VV/VH |
| Encoder | CNN residual stages：`64×128² → 64×64² → 128×32² → 256×16² → 512×8² → 512×4²` |
| Latent | p5 → GAP → `Linear(256,512)` → 512-D；不是分类类别 |
| Decoder | latent → `256×4²` → upsample；直接融合 p4 `8×8`、p3 `16×16`；输出 `2×256×256` |
| Attention | encoder `4×4`；decoder `32×32` |
| Loss | 训练是全元素 squared-error sum；epoch 日志是 average per-sample SSE |
| Anomaly score | `(VV−VV_hat)² + (VH−VH_hat)²` → `256×256` error map |
| Post-processing | 单图 KMeans；五期/大区域 GMM；高均值 cluster 是 anomaly；过滤小连通区 |
| Temporal logic | AE 逐图处理；后处理比较 anomaly mask 的 `0→1` |

## 5. 五个必须记住的事实边界

1. 数据是 **spatial split**，不是 random split，也不是 temporal holdout。
2. 完整上游 pyroSAR/CuSum SAR preprocessing **不在当前 repository 中**；仓库能证明的是融合、切片和清理等后半段。
3. CUDA DataLoader 是 `num_workers=1 + pin_memory=True`，不是多个 worker 并行预取。
4. P 0.8284、R 0.6865、F1 0.7508、IoU 0.6011 是 `REPORT ONLY`，当前仓库不能严格复现。
5. Autoencoder 本身不是 temporal model；时间信息来自独立 anomaly masks 的后处理比较。

## 6. 三个最重要的问题

### 1. Train–serving normalization skew

训练 Dataset 使用 `[-15,-3] → approximately [0,1]`，但五期和大区域 inference 没有统一复用，可能让 reconstruction error 反映尺度错误，而不是毁林。

### 2. Evaluation universe 由 prediction 决定

当前 notebook 用 prediction geometry 裁 annotation，可能删除预测范围外的 ground truth、漏算 FN，进而高估 Recall/F1/IoU。

### 3. Experiment/data lineage 不完整

报告指标没有绑定完整数据 manifest、checkpoint、resolved config、阈值、AOI、TP/FP/FN 和 Git SHA，因此只能陈述为历史报告值。

## 7. 如果现在重做

我不会第一步直接换大模型，而会保留当前 AE 作为 baseline。先建立一个训练与所有推理入口共享的 versioned transform，消除 normalization skew。然后用 manifest 固定 spatial split、数据版本和 rejected samples，并在预测前固定 evaluation AOI/grid。每个 run 保存 config、Git SHA、dataset/preprocessing version、checkpoint 和 metrics，训练与推理增加一致性 integration test。性能上先测 DataLoader wait、H2D、forward/backward 和 GPU memory，再决定 workers、batch 或数据格式。完成可信 baseline 后，再评估支持 Sentinel-1 的 pretrained encoder是否改善 anomaly separation，并依据算力和部署约束决定是否采用。

---

# Part 2 — 30–45 Minute Review

## A. Project Story

### What problem?

在热带雨林区域，从 SAR 影像中发现小尺度、可能新出现的毁林候选，并输出可在 GIS 中核验的空间区域。

### Why SAR?

Sentinel-1 是主动雷达，较少依赖太阳光照，也能穿透大部分云层影响；多云热带地区比单纯依赖光学影像更容易获得连续观测。本项目模型输入是 VV/VH，不是光学 RGB。

### Why unsupervised?

正常森林数据多，而人工绘制毁林 pixel polygon 成本高、数量有限。AE 用正常森林自重建训练，不需要毁林 mask 来更新模型参数；人工标注仍用于最终评价，因此不能说整个项目完全没用标签。

### What output?

模型先产生 H×W reconstruction-error map；聚类和连通域过滤得到 anomaly mask；日期间 `0→1` 得到新变化候选；大区域分支将其矢量化为 Shapefile，供 QGIS 可视化和核验。

### How evaluated?

将空间 prediction 与人工标注放到共同栅格，计算 Precision、Recall、F1 和 IoU。最终报告有正式记录值，但当前实现的 evaluation universe 和 artifact lineage 不够严格，因此面试必须说“报告记录”，不能说“当前仓库已复现 benchmark”。

## B. Data Pipeline

### Offline vs online

```text
OFFLINE
VV/VH file discovery
→ alignment validation
→ two-band GeoTIFF fusion
→ non-overlapping 256×256 tiling
→ NaN/all-zero cleaning
→ materialized split directories

ONLINE
sample path lookup
→ TIFF decode
→ HWC/CHW normalization to C×H×W
→ [-15,-3] scaling
→ float tensor
→ batching/shuffle/worker
```

离线切片用额外存储换取每个 epoch 更少的重复 CPU/IO 工作、更稳定的样本边界和更容易定位坏文件；代价是大量小 TIFF 带来的目录扫描、metadata lookup 和随机 IO。

### Fusion and split

- 单波段 VV/VH 必须有相同 CRS、affine transform、width 和 height。
- 融合结果为 `2×H×W` GeoTIFF，再按 stride=tile size 切完整 `2×256×256` tile。
- train/validation/test 使用不同地理区域，是 spatial split；年份有重叠，所以不是 temporal holdout。
- 当前 VV/VH 是分别排序后 `zip`，没有根据 region/date/location 做显式 key join；数量相同但缺失项不同可能静默错配。

### Dataset vs DataLoader

- `ProcessedForestDataset`：建立文件索引；按 index 读取 TIFF；统一 CHW；检查 2 channels；normalize；转 tensor。
- `ProcessedForestDataLoader`：sampling、train shuffle、batching、worker 和 pin-memory 配置。
- train `shuffle=True`，validation/test `False`。
- CUDA：`num_workers=1,pin_memory=True`；CPU 默认 `0,False`。

### Normalization contract

```text
x_norm = (x - (-15)) / (-3 - (-15)) = (x+15)/12
```

Dataset 没有 clamp；辅助 inference helper 只有传 min/max 时 normalize 且会 clamp；五期和大区域路径没有一致复用。这是整个项目最严重的 train–serving consistency 问题。

详细复习：[02-data-pipeline.md](./02-data-pipeline.md)。

## C. Model

### Shape story

```text
2×256×256
→ residual CNN encoder
→ 512×4×4 + encoder attention
→ FPN p5/p4/p3/p2
→ p5 GAP + Linear
→ 512 latent
→ decoder from 256×4×4
→ +p4 at 8×8
→ +p3 at 16×16
→ decoder attention at 32×32
→ 2×256×256 reconstruction
```

### Four concepts

- **Residual**：学习 `F(x)+x`，让深层优化和 gradient flow 更容易；stride/channel 改变时用 `1×1` projection 对齐。
- **FPN**：用 `1×1` lateral convolution 把不同尺度统一为 256 channels，再 top-down upsample/add。Decoder 只直接融合 p4/p3；p5 不作 direct skip，p2 不使用。
- **Self-Attention**：用空间 Q/K/V 建立远距离位置关系。Encoder 放在 `4×4` 降低 `O(N²)` 成本；decoder 在 `32×32`，显存明显更高。
- **Anomaly assumption**：正常森林重建较好、异常重建较差。但强 skip/高容量可能也重建好异常；season/domain shift 可能让正常区域高误差。

详细复习：[03-model-and-anomaly-detection.md](./03-model-and-anomaly-detection.md)。

## D. Anomaly Detection

| Branch | Error preparation | Clustering | Spatial/temporal result |
|---|---|---|---|
| Single image | VV/VH squared error sum → log → 1/99 percentile clip → `[0,1]` | KMeans(2)，高均值 cluster | `min_size=50`，输出 PNG |
| Five images | target ±2 共五期的 raw pixel loss 合并 | shared GMM(2)，高均值 component | 每期 `min_size=50`；累计历史异常或相邻期 `0→1` 可视化 |
| Two-date large area | previous+target 所有共同 tile 的 raw loss 合并 | shared GMM(2) + raw loss threshold 1.0 | target/previous/difference 均默认 `min_size=100`；矢量化 Shapefile |

Connected Components 删除孤立小区域，通常提高 precision，但会删除真实小事件、降低 recall。按约 10 m/pixel，50 pixels≈0.5 ha，100 pixels≈1 ha。

最重要边界：AE 每次只处理一张 image；所有 temporal semantics 都来自后处理 mask comparison。

## E. Training Runtime

### One step

```text
DataLoader batch on CPU
→ data.to(device)
→ optimizer.zero_grad()
→ model.forward()
→ summed squared reconstruction loss
→ loss.backward()
→ optimizer.step()
```

- `zero_grad()` 必须执行，因为 PyTorch 默认累积 parameter gradients。
- `model.train()` 让 BatchNorm 使用 batch statistics并更新 running state。
- validation 用 `model.eval()+torch.no_grad()`：前者切换 BatchNorm，后者不建 autograd graph。
- Adam 默认 lr `1e-4`、weight decay `6e-6`；StepLR 每 5 epochs乘 0.7。
- EarlyStopping 使用 validation loss 的 5-epoch moving average，默认 patience 5、delta 0.001，保存 `best_model.pth`。
- 当前 checkpoint 只有 model `state_dict`：足够推理，不足以恢复 Adam、scheduler、epoch、EarlyStopping 和 RNG。
- Optuna 是完整训练 run 的外层循环：10 trials，只搜索 lr 和 weight decay，objective 是最后一次 validation loss而非 best-checkpoint metric。

详细复习：[04-training-and-performance.md](./04-training-and-performance.md)。

## F. Engineering Review：六个可迁移故事

### 1. Train–serving consistency

| | Summary |
|---|---|
| Original implementation | Dataset 用 `[-15,-3]` normalize；部分 inference 重复实现 loader并绕过转换。 |
| Risk | Pipeline 能运行却测量错误分布，reconstruction error 主要来自尺度错配。 |
| Modern fix | 一个 versioned `SARTransform` 被 train/inference共用；checkpoint绑定范围、channel order和版本；一致性测试比较同一 TIFF 输出。 |
| SWE value | API/data contract、single source of truth、silent failure prevention。 |

### 2. Offline materialization trade-off

| | Summary |
|---|---|
| Original implementation | 先把大 GeoTIFF 离线切成训练 tile。 |
| Risk | 节省重复计算，但制造大量小文件、额外存储和 metadata IO。 |
| Modern fix | 先 profile；小规模保留 GeoTIFF，中规模瓶颈明确后再考虑 shards/LMDB/Zarr。 |
| SWE value | 用存储换计算、批处理边界、根据访问模式选择数据布局。 |

### 3. Explicit key join

| | Summary |
|---|---|
| Original implementation | VV/VH 分别排序后 `zip`，仅检查列表数量和空间 metadata。 |
| Risk | 两边数量相同但缺失项不同会静默配错文件。 |
| Modern fix | 从 region/date/tile 等解析显式 key，做 one-to-one join，拒绝 missing/duplicate，manifest记录结果。 |
| SWE value | 数据库 join 语义、referential integrity、避免 positional coupling。 |

### 4. Immutable raw + manifest

| | Summary |
|---|---|
| Original implementation | NaN/zero 清理直接 `os.remove()`，没有 rejected manifest。 |
| Risk | 无回滚、无 lineage，无法区分未生成、拒绝和误删。 |
| Modern fix | immutable `raw/`、versioned `processed/`、recoverable `rejected/`，manifest保存 source、checksum、reason和版本。 |
| SWE value | idempotency、auditability、data lineage和安全的数据生命周期。 |

### 5. Experiment/run isolation

| | Summary |
|---|---|
| Original implementation | Optuna trials共享 `best_model.pth`、TensorBoard目录和 mutable args。 |
| Risk | 后续 trial覆盖 artifact，best params无法可靠对应best model，study不可恢复。 |
| Modern fix | 每 trial 独立 run ID/config/log/checkpoint/metrics；持久化 study；显式注册 best artifact。 |
| SWE value | multi-run isolation、state ownership、artifact traceability。 |

### 6. Fixed evaluation universe

| | Summary |
|---|---|
| Original implementation | prediction geometry参与裁剪 annotation并决定栅格范围。 |
| Risk | 范围外 FN 被排除，评价结果依赖模型自己选择的 universe。 |
| Modern fix | 在看 prediction前固定 AOI、CRS、resolution、grid和valid-data mask，再统一 rasterize。 |
| SWE value | 测试 oracle独立性、避免 measurement leakage、可靠指标设计。 |

完整 review：[06-engineering-review.md](./06-engineering-review.md)。

## G. Metrics / Evidence Boundary

最终报告记录：

| Precision | Recall | F1 | IoU | Status |
|---:|---:|---:|---:|---|
| 0.8284 | 0.6865 | 0.7508 | 0.6011 | `REPORT ONLY` |

当前仓库没有对应的完整数据快照、checkpoint、resolved config、threshold、AOI、TP/FP/FN 和中间 mask。保存 notebook还包含另一组指标，说明历史运行没有被完整封装；evaluation又存在 prediction-dependent clipping风险。因此只能说“最终报告在单独地理测试区域记录这些数值”，不能称为当前仓库已复验的严格 benchmark，也不能称为完全 untouched holdout。

简历证据矩阵：[07-resume-evidence-matrix.md](./07-resume-evidence-matrix.md)。

## H. CUSUM：最小知识

CUSUM 是已有的 SAR temporal change-detection 方法，核心是累积监测统计量偏离基线的变化。本项目参考/调用了已有 CuSum preprocessing，并将 CUSUM 输出用于定性空间对照；仓库没有保存同一数据、同一评价区域和同一指标下的完整量化 benchmark。因此不要说“项目证明 AE 优于 CUSUM”。报告提到的约 48–60 天检测延迟来自引用论文，不是本项目测量结果。

---

# Part 3 — Level A / B / C

## Level A — 必须脱稿回答（20 项）

1. 30–60 秒说明问题、数据、AE anomaly detection、GIS 输出和评价。
2. 为什么 Sentinel-1 SAR适合多云热带区域，以及模型输入是 VV/VH。
3. 为什么采用 normal-only unsupervised reconstruction，而不是直接 supervised segmentation。
4. 数据按地理区域 spatial split，不是 random split或 temporal holdout。
5. 离线融合/切片/清理与在线 Dataset读取/normalize/tensorize 的边界。
6. Dataset和DataLoader的职责区别。
7. CUDA配置是一个worker + pinned memory，不是多worker预取。
8. 训练 normalization是 `[-15,-3]`，部分 inference未统一复用，构成 train–serving skew。
9. 模型输入输出 `2×256×256`，latent为512维。
10. Residual `F(x)+x` 和 shape变化时的 projection。
11. FPN用于多尺度信息；decoder直接使用p4/p3，并存在异常被重建过好的风险。
12. Attention位于encoder `4×4` 和decoder `32×32`；低分辨率用于控制平方复杂度。
13. Reconstruction error只表示偏离 learned normal distribution，不等于确定毁林。
14. 单图KMeans、五期shared GMM、两日期大区域GMM三条分支的区别。
15. Connected Components 的 precision/recall trade-off，以及50/100 pixels约0.5/1 ha。
16. AE不是temporal model；时间变化由mask的 `0→1` 后处理产生。
17. 一个训练step：H2D、zero_grad、forward、loss、backward、Adam step。
18. `train/eval`、`no_grad`、EarlyStopping和state_dict恢复边界。
19. 三大可信度问题：normalization skew、evaluation universe、lineage不完整。
20. 报告指标的精确数值、`REPORT ONLY` 状态，以及“如果今天重做”的优先顺序。

## Level B — 被 ML/PyTorch 继续追问时（15 项）

1. 完整encoder/decoder shape及p4/p3融合位置。
2. Training sum loss与inference H×W error map的reduction差异。
3. Batch size为何影响sum-gradient和BatchNorm statistics。
4. BatchNorm在train/eval下的行为。
5. `loss.item()`可能触发CUDA synchronization，但需profile确认重要性。
6. Adam一阶/二阶moment及recovery checkpoint需求。
7. StepLR默认参数和epoch调用时机。
8. EarlyStopping监控moving average而非raw minimum。
9. KMeans centroid与GMM mean/variance mixture的项目级差异。
10. Test-time GMM为何无标签但仍是transductive。
11. Model capacity太弱/太强对FP/FN的影响。
12. Validation reconstruction loss为何不等于anomaly F1。
13. FPN/attention最小ablation如何控制变量。
14. GPU OOM与低利用率的profiling-first排查。
15. 完整recovery checkpoint、run isolation和dataset/model version关系。

## Level C — 不需要主动复习

以下细节审计时有价值，但当前Backend/Systems/AI Application校招不值得优先记忆：

- p2专属convolution没有reconstruction gradient的具体推导；只需记住p2未进入decoder。
- Self-Attention未做 `1/sqrt(d)` scaling。
- attention `gamma` 初始值是0.1。
- 五期GMM visualization使用 `mse_min=0,mse_max=1050`，且不参与分类。
- 五期 `ancient` mask的边界语义和第一期归类细节。
- VAE训练/验证beta不一致及Optuna metric contract bug；主结果路径是AE。
- `SummaryWriter.close()`缺失和各对象writer生命周期细节。
- EarlyStopping打印负best score与正raw loss的符号问题。
- SciPy默认连通性是4邻接式结构的具体细节。
- single-image随机路径会先物化整个test loader。
- FPN top-down使用output-conv之前的tensor。
- CLI `--test`默认行为、unused suffix/prefix参数等低层代码清理项。
- `empty_cache()`调用发生在reinitialization之后的精确顺序。
- GeoTIFF内部block/compression、CUDA DMA或GMM EM数学推导。
- SAR散射、speckle、RTC和极化的论文级物理推导。

---

# Part 4 — Top 20 Interview Questions

## 1. 介绍一下这个项目。

**30-second Answer**

我在法国IRD做了一个Sentinel-1 SAR微毁林检测原型。输入是VV/VH双通道patch；因为毁林pixel标签有限，我只用正常森林训练CNN Autoencoder，再用逐像素重建误差、KMeans/GMM和连通区域过滤得到异常mask，通过日期间 `0→1` 找新变化，最后导出Shapefile，在QGIS中与人工标注比较。报告记录P 0.8284、R 0.6865、F1 0.7508和IoU 0.6011，但当前仓库不能严格复现。

**If Interviewer Digs Deeper**

讲spatial split、`2×256×256`、p4/p3 FPN skips、三条post-processing分支，以及两项P0：normalization skew和evaluation universe。

**Do Not Say**

“这是production-ready系统”或“当前仓库可一键复现指标”。

**Relevant Doc**

[00-project-overview.md](./00-project-overview.md)

## 2. 为什么使用 SAR？

**30-second Answer**

Sentinel-1 SAR是主动雷达，不依赖太阳光照，并且相比光学影像更少受云层遮挡影响，适合经常多云的热带雨林。项目模型真正输入的是VV/VH双极化SAR，不是PlanetScope光学图；光学影像主要用于QGIS人工核验。

**If Interviewer Digs Deeper**

VV是同极化，VH是交叉极化；森林结构和清除后地表的backscatter模式可能不同。到此即可，不展开遥感物理。

**Do Not Say**

“SAR完全不受天气影响”或“VV/VH是两个日期”。

**Relevant Doc**

[00-project-overview.md](./00-project-overview.md)

## 3. 为什么选择 Autoencoder？

**30-second Answer**

当时正常森林数据多，而逐像素毁林polygon依赖QGIS人工标注，数量有限。AE可以只用正常森林做self-reconstruction训练，不需要毁林mask更新参数；推理时又能输出空间reconstruction-error map。代价是它检测的是异常而不是直接的毁林语义，有足够标签时supervised segmentation应作为baseline。

**If Interviewer Digs Deeper**

说明模型capacity太弱会FP、太强会FN；训练validation loss不等于anomaly separation。

**Do Not Say**

“无监督一定优于U-Net”或“整个项目没使用标签”。

**Relevant Doc**

[03-model-and-anomaly-detection.md](./03-model-and-anomaly-detection.md)

## 4. 为什么 reconstruction error 能检测异常？

**30-second Answer**

核心假设是只见过正常森林的模型会把正常模式重建得更好，而分布外区域重建误差更高。源码计算VV和VH每个像素的平方误差并相加，得到H×W anomaly map。不过高误差只表示偏离learned normal distribution，季节变化、湿度或preprocessing错误也可能触发。

**If Interviewer Digs Deeper**

讨论高容量/skip让异常也重建好，以及normalization skew如何破坏error语义。

**Do Not Say**

“高重建误差就证明发生毁林”。

**Relevant Doc**

[03-model-and-anomaly-detection.md](./03-model-and-anomaly-detection.md)

## 5. 数据是怎么处理的？

**30-second Answer**

上游得到processed VV/VH GeoTIFF后，代码检查CRS、transform和尺寸，对齐后堆叠成2-band GeoTIFF，离线切成完整非重叠的`256×256` tiles，再清理NaN和全零样本并按地理区域放入train/validation/test目录。在线Dataset只读TIFF、统一CHW、用`[-15,-3]` normalize并转tensor。

**If Interviewer Digs Deeper**

指出完整pyroSAR上游不在仓库，split是spatial，VV/VH当前sorted-zip有静默错配风险。

**Do Not Say**

“从原始GRD到训练集的全部preprocessing都在仓库自动完成”。

**Relevant Doc**

[02-data-pipeline.md](./02-data-pipeline.md)

## 6. Dataset 和 DataLoader 分别做什么？

**30-second Answer**

Dataset定义一个index如何变成单个`2×256×256 float32` tensor：路径查找、TIFF decode、layout转换、通道检查和normalization。DataLoader在Dataset之上负责sample顺序、shuffle、batch、worker和pinned-memory。训练循环只消费batch，不关心GeoTIFF细节，这体现separation of concerns。

**If Interviewer Digs Deeper**

train shuffle为true，validation/test为false；CUDA一个worker，CPU默认零worker。

**Do Not Say**

“DataLoader负责下载SAR、地形校正和切片”。

**Relevant Doc**

[02-data-pipeline.md](./02-data-pipeline.md)

## 7. 为什么提前离线切片？

**30-second Answer**

大图裁剪和地理metadata处理是确定性工作，如果放在`__getitem__`，每个epoch都会重复CPU和IO成本，也更难定位坏样本。我把它离线materialize，用额外存储换更稳定的训练吞吐、确定性和可调试性。代价是大量小TIFF会增加metadata/random-IO成本，数据扩大后要先profile再决定是否shard。

**If Interviewer Digs Deeper**

谈SSD、manifest、WebDataset/LMDB/Zarr应按访问模式选择，不要默认迁移。

**Do Not Say**

“离线切片在所有规模下一定最快”。

**Relevant Doc**

[02-data-pipeline.md](./02-data-pipeline.md)

## 8. `num_workers=1` 是什么意思？

**30-second Answer**

表示DataLoader使用一个worker subprocess调用Dataset并提前准备batch，main training process同时消费batch和运行GPU计算。它确实存在后台加载，但不是多个worker并行。本项目只有CUDA路径显式设置1；CPU使用默认0，即Dataset工作在main process。

**If Interviewer Digs Deeper**

说明framework可能使用默认prefetch queue，但项目未显式配置prefetch_factor或persistent_workers。

**Do Not Say**

“使用了多个worker并行预取”。

**Relevant Doc**

[02-data-pipeline.md](./02-data-pipeline.md)

## 9. `pin_memory` 有什么作用？

**30-second Answer**

Page-locked host memory地址稳定，CUDA DMA可以更直接地进行CPU到GPU传输，减少从pageable memory复制到staging buffer的开销。本项目CUDA DataLoader设置了`pin_memory=True`。但训练使用普通`data.to(device)`，没有`non_blocking=True`和overlap profile，所以只能说启用了pinning，不能说完整实现异步H2D流水化。

**If Interviewer Digs Deeper**

Pinning会占用不可分页内存，并非越多越好；应测H2D time和RAM。

**Do Not Say**

“pin_memory自动保证copy与compute重叠”。

**Relevant Doc**

[02-data-pipeline.md](./02-data-pipeline.md)

## 10. Residual connection 有什么作用？

**30-second Answer**

项目encoder的ResidualBlock主分支做两层`3×3 Conv+BN`，然后与identity相加再激活，即学习`F(x)+x`。它提供更直接的gradient和信息路径，让较深网络更容易优化。若stride或channel变化，源码用`1×1` projection同时下采样和改channel，保证两个分支shape可相加。

**If Interviewer Digs Deeper**

举`64×64×64 → 128×32×32`例子，identity也必须变为`128×32×32`。

**Do Not Say**

“Residual一定提高了本项目F1”；仓库没有完整ablation。

**Relevant Doc**

[03-model-and-anomaly-detection.md](./03-model-and-anomaly-detection.md)

## 11. 为什么用了 FPN？

**30-second Answer**

Decoder既需要高层大receptive-field context，也需要较高分辨率的局部空间细节。FPN用`1×1` lateral conv统一为256 channels，再top-down upsample/add。当前代码计算p5到p2，但decoder只直接融合p4 `8×8`和p3 `16×16`；设计动机是multi-scale reconstruction，仓库没有证明它一定提高anomaly F1。

**If Interviewer Digs Deeper**

p5生成latent并影响top-down；p2被计算但不进入decoder。

**Do Not Say**

“p2/p3/p4/p5全部作为decoder skip”。

**Relevant Doc**

[03-model-and-anomaly-detection.md](./03-model-and-anomaly-detection.md)

## 12. FPN 会不会让 anomaly 也被重建？

**30-second Answer**

会，这是核心trade-off。Skip改善正常细节重建，也可能绕过512-D bottleneck，把异常空间信息带给decoder，导致异常误差降低、Recall下降。当前只用p4/p3，没有最高分辨率p2或原图级skip，风险相对小一些，但没有ablation证明是最优折中。

**If Interviewer Digs Deeper**

设计no-skip、p4、p4+p3、p4+p3+p2实验，同时比较normal/anomaly error separation和P/R/F1，而非只看validation loss。

**Do Not Say**

“不用p2是经过严格实验验证的设计”。

**Relevant Doc**

[03-model-and-anomaly-detection.md](./03-model-and-anomaly-detection.md)

## 13. 为什么加入 Self-Attention？

**30-second Answer**

卷积偏向局部邻域，spatial attention让远距离位置按内容加权交互。项目在encoder `512×4×4`和decoder `256×32×32`各放一层，通过Q/K/V、softmax和可学习gamma残差融合。放在低分辨率控制`O(N²)`成本；decoder attention更大，是显存hotspot。它是单个attention block，不是Transformer。

**If Interviewer Digs Deeper**

`4×4` matrix是`16×16`；`32×32` matrix是`1024×1024`，float32 raw约4 MiB/sample。

**Do Not Say**

“项目使用完整Transformer architecture”或“attention已被ablation证明有效”。

**Relevant Doc**

[03-model-and-anomaly-detection.md](./03-model-and-anomaly-detection.md)

## 14. KMeans 和 GMM 分别做什么？

**30-second Answer**

两者都把一维pixel reconstruction error无标签分成低/高两组，再把均值更高的组定义为anomaly。单图分支对log和percentile-normalized error使用KMeans；五期及大区域分支合并raw errors后拟合two-component GMM。GMM还能建模不同variance和mixture weight，但仓库没有公平benchmark证明它更好。

**If Interviewer Digs Deeper**

说明GMM在test loss上fit是transductive，并且两components只是normal/anomaly简化假设。

**Do Not Say**

“所有分支都用GMM”或“GMM输出就是毁林标签”。

**Relevant Doc**

[03-model-and-anomaly-detection.md](./03-model-and-anomaly-detection.md)

## 15. 这个项目是不是 temporal model？

**30-second Answer**

不是。AE每次forward只输入一张VV/VH image，没有时间维、RNN或跨日期attention。时间信息出现在post-processing：分别得到各日期anomaly mask，再把previous为0、current为1定义为新变化。五期分支共享一个GMM并比较mask，大区域分支比较指定前后日期。

**If Interviewer Digs Deeper**

区分累计历史异常视图与相邻两期`0→1`，但不必背低层实现。

**Do Not Say**

“网络学习了SAR时间序列”。

**Relevant Doc**

[03-model-and-anomaly-detection.md](./03-model-and-anomaly-detection.md)

## 16. 一次 PyTorch training step 怎么执行？

**30-second Answer**

DataLoader给出CPU上的`B×2×256×256` batch，代码先搬到device，再`optimizer.zero_grad()`，forward得到同shape reconstruction，计算全元素squared-error sum，`loss.backward()`写入parameter gradients，最后Adam `step()`更新参数和moment states。每个epoch后StepLR更新下一轮LR，再用`eval()+no_grad()`跑validation和EarlyStopping。

**If Interviewer Digs Deeper**

解释zero_grad因为梯度默认累积；当前epoch日志是per-sample SSE，不是per-element MSE。

**Do Not Say**

“backward自动更新参数”或“no_grad等同eval”。

**Relevant Doc**

[04-training-and-performance.md](./04-training-and-performance.md)

## 17. 项目最大的工程问题是什么？

**30-second Answer**

最大的不是模型结构，而是跨stage的数据和评价contract不统一。训练Dataset使用`[-15,-3]` normalization，但部分最终推理绕过；评价又由prediction geometry裁ground truth。这两类问题都可能让pipeline成功运行、结果看似合理，但测量对象已经变了。今天我会先统一transform并固定evaluation AOI，再讨论模型优化。

**If Interviewer Digs Deeper**

补充data/config/checkpoint/metric lineage不完整，因此报告数值不可严格复现。

**Do Not Say**

“最大问题是模型不够大”或把2026修复说成当年已完成。

**Relevant Doc**

[06-engineering-review.md](./06-engineering-review.md)

## 18. 如果 GPU utilization 很低怎么排查？

**30-second Answer**

先把step拆成`next(DataLoader)`、H2D、forward、backward、Adam和host synchronization，用PyTorch Profiler、GPU util、CPU和disk IO确认瓶颈。Data wait长就查单worker、TIFF decode和random IO；kernel短就测试batch；copy长再验证pinning和non-blocking；还检查`loss.item()`、日志和checkpoint停顿。每次只改一个变量并重新测samples/sec。

**If Interviewer Digs Deeper**

说明当前只有offline tiling、1 worker和pin memory，不能声称当年做过完整profiling。

**Do Not Say**

“GPU低利用率就把num_workers直接调到最大”。

**Relevant Doc**

[04-training-and-performance.md](./04-training-and-performance.md)

## 19. 如果现在重新做你会改什么？

**30-second Answer**

我先保留AE baseline，不会第一步换大模型。先让train/inference共享一个versioned transform，用manifest固定spatial split和data lineage，固定evaluation AOI/grid，并让每个run保存config、Git SHA、checkpoint和metrics。增加train–inference consistency及小型integration tests，性能上profile后再优化。基础可信后，再评估pretrained SAR encoder是否改善效果。

**If Interviewer Digs Deeper**

说明inductive frozen detector与transductive GMM要分开报告，并做Residual/FPN/Attention ablation。

**Do Not Say**

“Foundation model存在，所以一定比AE好”。

**Relevant Doc**

[06-engineering-review.md](./06-engineering-review.md)

## 20. 这个项目最大的收获是什么？

**30-second Answer**

最大收获是ML系统正确性常在模型之外：文件配对、normalization、split、evaluation universe和artifact lineage都能让一个可运行pipeline产生错误结论。我也学会了系统trade-off，例如离线切片用存储换训练稳定性，以及性能优化要先profile。现在我会先建立可信data/measurement contract，再优化模型和速度。

**If Interviewer Digs Deeper**

用train–serving skew或offline tiling故事展开，连接backend的API contract、idempotency和observability。

**Do Not Say**

“最大收获是掌握了某篇遥感论文”——与当前岗位价值不匹配。

**Relevant Doc**

[06-engineering-review.md](./06-engineering-review.md)

---

# Part 5 — Five Best Stories

## A. Offline Tiling

**Context**

训练输入来自双通道大GeoTIFF，而模型需要固定`2×256×256` patch。

**Problem**

如果每个epoch动态融合/切片，会重复CPU和地理metadata处理，训练错误还难定位；但全部预切片会增加存储和小文件IO。

**Decision / Analysis**

项目采用离线融合和materialized tiles，Dataset在线只做轻量读取与tensor转换。本质是用存储换重复计算、稳定性和debuggability。

**Result / Lesson**

这体现批处理stage划分和data-layout trade-off。迁移到backend/infra场景，就是把确定性重处理移到可重试、可版本化的job，同时根据profile决定是否继续materialize或改用shard。

## B. Train–Serving Skew

**Context**

AE anomaly score直接取决于输入和重建的数值差。

**Problem**

训练Dataset使用`[-15,-3]` normalization，部分时序/大区域推理绕过或使用不同clamp语义；代码仍能出结果，但error语义可能错误。

**Decision / Analysis**

复盘时把它定义为train–serving skew，而不是继续调GMM threshold掩盖症状。正确方案是唯一versioned transform、checkpoint metadata和同一TIFF的train/inference parity test。

**Result / Lesson**

最大的工程判断是识别silent correctness failure。它可迁移到任何AI application：embedding、feature、tokenization或schema在离线和服务端必须共享contract。

## C. VV/VH Explicit-Key Pairing

**Context**

每个模型样本由同一区域、日期、位置的VV和VH文件组成。

**Problem**

当前实现两边分别排序后`zip`。仅检查数量相同无法保证语义配对；相同长度也可能错位，而且CRS/shape相同不一定代表同一日期。

**Decision / Analysis**

现代设计从文件名解析region/date/tile key，执行one-to-one join，对missing/duplicate fail fast，并把配对写入manifest。

**Result / Lesson**

这不是遥感专属问题，而是典型referential integrity。对backend而言，对应的是不能靠两个数组位置隐式关联实体，应使用稳定ID和验证过的join。

## D. Experiment Artifact Isolation

**Context**

项目用Optuna运行10个lr/weight-decay trials。

**Problem**

Trials共享`best_model.pth`、TensorBoard目录和mutable args，后续trial可能覆盖artifact，best params与best model无法可靠对应。

**Decision / Analysis**

每个trial应有独立run ID、resolved config、logs、checkpoint和metrics；study持久化；最后显式注册best artifact。

**Result / Lesson**

这体现并发/批量任务的state isolation和ownership。迁移到agent或backend系统，就是每次执行必须有独立workspace、trace和artifact namespace。

## E. Evaluation Correctness

**Context**

空间prediction需要与QGIS人工annotation计算Precision、Recall、F1、IoU。

**Problem**

当前notebook让prediction geometry参与裁annotation和定义栅格范围，可能排除模型没有覆盖的ground truth并漏算FN。

**Decision / Analysis**

评价universe必须先于prediction固定：AOI、CRS、resolution、transform、valid mask都由协议决定，再将GT和prediction投到同一网格。

**Result / Lesson**

这是测试oracle不能依赖被测输出的通用原则。对后端/系统同样适用：测试集合、SLO denominator和监控窗口不能由系统成功响应自行定义。

---

# Part 6 — DO NOT SAY

1. “从原始Sentinel-1下载到训练tile的全部preprocessing都在repository中完整自动化。”
2. “完整pyroSAR pipeline由我在当前仓库独立实现并可一键运行。”
3. “DataLoader使用多个worker并行预取。”真实CUDA配置是`num_workers=1`。
4. “pin_memory已经保证异步H2D与GPU计算重叠。”没有`non_blocking=True`和profiling证据。
5. “实验完全可复现。”只能说具备部分seed、CLI、requirements和checkpoint基础。
6. “报告指标已经由当前仓库重新复现验证。”它们是`REPORT ONLY`。
7. “这是严格untouched independent holdout benchmark。”测试数据参与test-time GMM和研发分析。
8. “数据是spatial+temporal holdout。”它是spatial split，年份重叠。
9. “CUSUM在我们的实验里慢48天。”48–60天来自引用论文。
10. “我们通过统一量化benchmark证明AE优于CUSUM。”当前只有定性参照证据。
11. “Autoencoder学习了时间序列。”网络逐图重建，时间逻辑在post-processing。
12. “Residual、FPN、Attention分别被ablation证明提高F1。”仓库没有完整ablation。
13. “所有p2/p3/p4/p5都进入decoder。”只有p4/p3直接融合。
14. “高reconstruction error就代表确定毁林。”它只是偏离learned normal distribution。
15. “Foundation model一定比这个AE好。”必须考虑domain gap、preprocessing、objective、labels、算力和部署约束并实测。

---

# Part 7 — Last-Minute Card

```text
================================
INTERVIEW LAST-MINUTE CARD
================================

Project:
  Sentinel-1 SAR微毁林异常检测prototype；输出GIS候选区域。

Data:
  VV/VH融合 → 2×256×256 GeoTIFF tiles；报告约32k/8k/1.6k。

Split:
  Spatial split；不是random，也不是temporal holdout。

Input:
  VV/VH, 2×256×256；Dataset用[-15,-3] normalize。

Model:
  Residual CNN AE + FPN + spatial self-attention。
  Decoder直接融合p4 8×8和p3 16×16。

Latent:
  p5 → GAP 256 → Linear → 512-D；不是分类。

Loss:
  Training = full squared-error sum；epoch = per-sample SSE average。

Anomaly:
  (VV−VV_hat)²+(VH−VH_hat)² → H×W error map。
  Single image KMeans；five-image/two-date GMM；high mean=anomaly。
  Connected components: 50px≈0.5ha；large-area 100px≈1ha。

Temporal:
  AE不是temporal model；post-processing比较mask的0→1。

Training:
  DataLoader → H2D → zero_grad → forward → loss → backward → Adam step。
  StepLR；eval+no_grad validation；EarlyStopping；state_dict only。
  CUDA DataLoader = 1 worker + pin_memory，不是multi-worker。

Metrics:
  Report only: P .8284 / R .6865 / F1 .7508 / IoU .6011。
  Current repository cannot strictly reproduce them。

Biggest assumption:
  Normal reconstructs better than anomaly。

Biggest engineering bug:
  Train/inference normalization mismatch；evaluation ROI depends on prediction。

Biggest model risk:
  Capacity/FPN skips reconstruct anomaly too well；seasonal shift creates FP。

Biggest lesson:
  ML correctness depends on cross-stage data and measurement contracts，
  not only model architecture。

If rebuilding:
  Shared versioned transform → manifest/data lineage → fixed AOI/grid
  → run/config/checkpoint isolation → consistency/integration tests
  → profile first → retain AE baseline, then test pretrained encoder。

Do not say:
  Complete preprocessing in repo / multi-worker / fully reproducible /
  untouched holdout / CUSUM slow 48 days / temporal network /
  ablation-proven FPN / foundation model must be better。
================================
```

