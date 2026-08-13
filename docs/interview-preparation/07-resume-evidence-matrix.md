# Resume Evidence Matrix

> 审计基线：GitHub `main` commit `6d7ee247f4dd6d2d2e83e83227605c83c0cfd528`、仓库中的 notebook/产物、最终实习报告。
>
> 证据优先级：源码 > notebook/metrics/saved results > 最终报告 > README > 简历 > 注释或记忆。
>
> 状态含义：`VERIFIED`、`PARTIALLY VERIFIED`、`REPORT ONLY`、`UNSUPPORTED`、`CONFLICT`。

## 1. Final Evidence Matrix

| Resume Claim | Evidence | Status | Interview Risk | Safe Wording | Do Not Say |
|---|---|---|---|---|---|
| Sentinel-1 | 报告第 13 页记录 Sentinel-1 GRD、IW、约 10 m；README 和 `*_S1A_*` 文件匹配逻辑一致。 | `VERIFIED` | Low | “项目使用 Sentinel-1 SAR 影像。” | “仓库包含原始 Sentinel-1 数据并可自动重新下载全部数据。” |
| VV/VH | `split_data.py` 分别读取 VV/VH 单波段文件、检查对齐并堆叠为 2 bands；Dataset 强制 2 channels。 | `VERIFIED` | Low | “模型输入由同位置同日期的 VV/VH 两个极化通道组成。” | “VV/VH 是两个时间点”或“是 RGB 通道”。 |
| pyroSAR | 报告第 19、30 页描述 pyroSAR/`geo_parameterize`；仓库只有外部 `CuSum_.preprocess()` 调用，没有完整实现，requirements 也未列 pyroSAR。 | `REPORT ONLY` | High | “当时的上游 SAR 预处理参考并调用了 IRD 的 CuSum/pyroSAR 流程。” | “完整 pyroSAR preprocessing 已在本仓库实现并可独立运行。” |
| Preprocessing | 融合、切片、CRS/transform 检查、NaN/全零清理有源码；dB、边缘噪声、DEM、terrain correction、Lee Sigma 只有报告。 | `PARTIALLY VERIFIED` | High | “我实现了融合、地理切片和数据清理，并使用/参考已有 SAR preprocess 产生模型输入。” | “我从原始 GRD 到训练集实现了全部 preprocessing 代码。” |
| `gamma0-rtc_db` | anomaly 文件匹配 regex 明确包含 `*_VV_gamma0-rtc_db_*`；实际影像和产生该命名的代码未提交。 | `PARTIALLY VERIFIED` | Medium | “项目消费的预处理文件名标识为 gamma0/RTC/dB 产品。” | “当前仓库证明了完整 gamma0 RTC 算法和所有参数。” |
| QGIS | 报告记录使用 QGIS 制作/核验标注；仓库含 annotation、metrics shapefile 和 QGIS 风格产物。 | `REPORT ONLY` | Low | “我用 QGIS 制作并核验人工毁林多边形，也用于结果可视化。” | “QGIS 操作已由 Python pipeline 全自动执行。” |
| 32k / 8k / 1.6k | 报告第 19 页和 README 记录近似数量；数据与 manifest 未提交，无法重计。 | `REPORT ONLY` | Medium | “最终报告记录约 32k/8k/1.6k 个 train/validation/test patch。” | “我刚刚从当前仓库重新统计并验证了精确数量。” |
| Spatial split | 报告记录 train 4 区、validation 1 区、test 1 区；DataLoader 只消费预先分开的目录。 | `PARTIALLY VERIFIED` | Medium | “数据按地理区域划分，不是随机 tile split。” | “这是同时严格 spatial + temporal holdout。” |
| Train regions | 报告为 4 个区域；README 为 3 个。 | `CONFLICT` | High | “按最终报告，训练使用 4 个区域。” | “README 与报告完全一致。” |
| Test years | 报告为 2021–2022；README 为 2020–2022。 | `CONFLICT` | High | “按最终报告，测试对应 2021–2022。” | “仓库所有材料都证明测试从 2020 开始。” |
| Unsupervised | AE 输入同时作为 reconstruction target；训练区在报告中为正常森林；标签仅用于最终评价。 | `VERIFIED` | Low | “模型参数通过正常森林自重建训练，不需要逐像素毁林标签。” | “整个项目完全没有使用标注。”评价阶段使用了人工标注。 |
| Normal-forest-only training | 报告和 README 都如此描述；数据未提交，无法检查每个 tile。 | `PARTIALLY VERIFIED` | Medium | “数据构建目标是让训练和验证只包含完整正常森林区域。” | “我已对仓库中每一个训练 tile 重新验证没有异常。” |
| CNN Autoencoder | `AE_Network` 组合真实 Encoder/Decoder，输入输出均为 2 channels。 | `VERIFIED` | Low | “核心模型是 CNN Autoencoder。” | “最终指标来自 VAE。”没有相应证据。 |
| Residual | `ResidualBlock` 有两层 3×3 Conv、BN、LeakyReLU 和 identity addition；维度变化用 projection。 | `VERIFIED` | Low | “Encoder 使用 residual blocks，维度变化时用 1×1 projection 对齐 shortcut。” | “Decoder 也由 residual blocks 构成。”Decoder 是 transposed-conv blocks。 |
| FPN | p5/p4/p3/p2 均计算；decoder 只加 `fpn_features[1]` p4 和 `[2]` p3。 | `VERIFIED` | Medium | “Encoder 计算四层 FPN，decoder 实际融合 p4/p3 两层。” | “p2/p3/p4/p5 全部直接送入 decoder。” |
| Self-Attention | Q/K/V 1×1 Conv、spatial attention、learnable gamma、residual 和 BN 均有源码；encoder 4×4、decoder 32×32。 | `VERIFIED` | Low | “模型在低分辨率 encoder bottleneck 和 decoder 32×32 feature 上使用 self-attention。” | “项目使用完整 Transformer architecture。” |
| 512 latent | `AE_Network` 将 output size 硬编码为 512。CLI `embedding-size=128` 未被 AE 使用。 | `VERIFIED` | Medium | “当前 AE 的 latent vector 是 512 维。” | “latent 由 `--embedding-size` 自由配置。” |
| MSE | AE 调用 `F.mse_loss(..., reduction='sum')`；pixel map 用 `reduction='none'` 后跨通道求和。 | `VERIFIED` | Medium | “使用平方重建误差；训练实现是 sum reduction，异常图是逐像素两通道平方误差和。” | “训练 loss 是对所有元素取 mean 的标准 MSE。” |
| GMM | 5-image 和 large-area 分支使用 `GaussianMixture(n_components=2, random_state=0)`，高均值 component 为 anomaly。 | `VERIFIED` | Medium | “GMM 对 pixel-loss distribution 做两类无标签聚类。” | “GMM 在训练集上拟合后冻结。”它在测试影像 loss 上拟合。 |
| KMeans | single-image 分支对归一化 log-loss 使用 `KMeans(n_clusters=2, random_state=0)`。 | `VERIFIED` | Low | “单图分析分支使用 KMeans；时序和大区域分支使用 GMM。” | “所有检测都统一使用 KMeans”或“所有检测都统一使用 GMM”。 |
| Connected components | `scipy.ndimage.label` 后按 component pixel count 过滤；单图/五期常用 50，大区域默认 100。 | `VERIFIED` | Low | “聚类后通过连通区域像素数过滤孤立小噪声。” | “代码实际执行了 opening/closing。”这两步被注释掉。 |
| `min_size=50` physical area | 在 10 m×10 m 假设下为 5,000 m²，即 0.5 ha；大区域默认值实际是 100，即约 1 ha。 | `PARTIALLY VERIFIED` | Medium | “部分可视化分支用 50 pixels；整区输出默认 100 pixels。” | “所有分支统一使用 50 pixels。” |
| Temporal comparison | 五期分支比较相邻 anomaly mask；大区域分支比较指定前后日期的 `0→1`。 | `VERIFIED` | Low | “在空间 anomaly mask 基础上做日期间变化比较。” | “Autoencoder 本身是时序网络。”它逐图重建。 |
| Shapefile vectorization | `rasterio.features.shapes`、GeoDataFrame 和 `.to_file()` 输出每 tile 及合并 Shapefile。 | `VERIFIED` | Low | “整区变化 mask 被矢量化并保留地理坐标。” | “输出直接是数据库服务或在线 GIS API。” |
| Optuna | 10 trials；搜索 lr `1e-4..5e-4` 和 weight decay `1e-6..1e-5`；目标是最后一次/early-stop validation loss。 | `VERIFIED` | Medium | “集成 Optuna，以 validation loss 为目标搜索 lr 和 weight decay。” | “Optuna 搜索 batch size、architecture、StepLR step/gamma。” |
| TensorBoard | AE 写 `Loss/train`、`Loss/validation`；报告有曲线截图。VAE 另写 LR 和 KLD 等。 | `VERIFIED` | Low | “TensorBoard 用于跟踪训练和验证 loss，辅助比较实验和发现过拟合。” | “记录了完整数据 lineage、系统资源和所有超参数。” |
| DataLoader | 自定义 wrapper 创建 train/validation/test DataLoader；train shuffle，validation/test 不 shuffle。 | `VERIFIED` | Low | “使用 PyTorch DataLoader 做 batch、shuffle 和后台加载。” | “DataLoader 自己完成 TIFF 预处理和地理切片。”切片是离线脚本。 |
| Multi-process prefetch | 仅 CUDA 路径 `num_workers=1`；PyTorch 对 worker 有默认队列预取，但代码未显式设 `prefetch_factor`。 | `PARTIALLY VERIFIED` | High | “GPU 训练时使用 1 个 worker 子进程后台加载。” | “使用多个 worker 并行预取”或“显式实现多进程预取优化”。 |
| `pin_memory` | CUDA 路径设置 `pin_memory=True`。 | `VERIFIED` | Medium | “GPU 路径启用了 pinned memory，以减少 pageable-memory staging 开销。” | “已经实现完全异步 H2D overlap。”`.to()` 未使用 `non_blocking=True`。 |
| Reproducibility | 有 CLI、固定版本 requirements、`torch.manual_seed(42)` 和 checkpoint；缺 Python/NumPy/Optuna seed、deterministic、data version 和 config snapshot。 | `PARTIALLY VERIFIED` | High | “具备 CLI、依赖版本、PyTorch seed 和 checkpoint 等部分复现基础。” | “实验完全可复现”或“不同 GPU 上 bitwise deterministic”。 |
| Checkpoint | 保存 best `model.state_dict()` 和各 epoch weights；不含 optimizer/scheduler/epoch/config。 | `VERIFIED` | Medium | “保存模型权重，支持加载 best model 推理。” | “checkpoint 可无损恢复完整训练状态。” |
| Precision 0.8284 | 最终报告第 37 页记录；当前 notebook 没有这项保存输出。 | `REPORT ONLY` | High | “最终报告记录 Precision 0.8284。” | “当前仓库已重新复现并验证 0.8284。” |
| Recall 0.6865 | 同上。 | `REPORT ONLY` | High | “最终报告记录 Recall 0.6865。” | “Recall 是严格 untouched holdout 上的已复验结果。” |
| F1 0.7508 | 报告值与 P/R 数学关系一致，但无对应 TP/FP/FN。 | `REPORT ONLY` | High | “最终报告记录 F1 0.7508。” | “notebook 当前输出就是 0.7508。” |
| IoU 0.6011 | 最终报告第 37 页记录。 | `REPORT ONLY` | High | “最终报告记录 IoU 0.6011。” | “它是在原生 10 m 固定网格上计算。”当前 notebook 强制 256×256 动态网格。 |
| Current notebook metrics | `test.ipynb` 保存 P 0.9029、R 0.6930、F1 0.7842、IoU 0.6450。 | `VERIFIED` | High | “当前 notebook 保存了另一组运行输出，说明报告实验没有被完整封装。” | “这组数自动替代最终报告数值。”它可能对应不同输入/协议。 |
| Evaluation protocol | Notebook 清理/统一 CRS/栅格化/forest-mask 后算 TP/FP/FN，但用 detection geometry 裁剪 annotation。 | `PARTIALLY VERIFIED` | Critical | “指标来自空间结果与人工标注栅格化比较；当前评估裁剪存在潜在 FN 漏计风险。” | “评估协议不存在 leakage，已经完全严谨。” |
| CUSUM | 报告有方法背景、输出和定性比较；preprocess notebook 调用外部 CuSum 类。无统一量化 benchmark 表。 | `PARTIALLY VERIFIED` | High | “将已有 CUSUM 时序方法及其输出作为预处理参考和定性参照。” | “本项目量化证明 AE 优于 CUSUM。” |
| CUSUM 48–60 day delay | 报告综述引用论文中的结果。 | `REPORT ONLY` | Critical | “相关论文报告 CUSUM 检测延迟约 48–60 天。” | “我们的 CUSUM 实验慢 48 天。” |
| Test region | 报告称 `622_975` 是单独地理区域；但用于人工选择、可视化、后处理开发和 test-time GMM fitting。 | `PARTIALLY VERIFIED` | High | “在空间上单独的测试区域进行结果分析；GMM 后处理会无标签适应该测试影像。” | “严格 untouched、完全冻结、独立 holdout benchmark。” |

## 2. Current Resume Bullets Audit

### Bullet 1

**Original**

> 设计 Sentinel-1 SAR VV/VH 双极化影像的自动预处理、样本切片与 QGIS 人工核验流程，构建约 32k 训练、8k 验证和 1.6k 测试样本，用于无监督微毁林变化检测。

**Problem**

- “自动预处理”容易被理解为从原始 Sentinel-1 下载到最终 patch 的完整、自包含 pipeline；仓库只能验证后半段融合、切片和清理，完整 pyroSAR/CuSum preprocess 主要存在于外部代码与报告。
- 32k/8k/1.6k 是最终报告的近似数量，当前仓库无数据 manifest 可重计。
- “QGIS 人工核验”基本符合报告，但属于工具操作和历史产物证据，不是仓库中的自动化步骤。

**Risk Level**

`HIGH`：面试官若要求展示 preprocessing 入口、pyroSAR 参数或从零复跑，当前仓库无法支持“完整自动化”的强表述。

**Recommended Revision**

> 设计 Sentinel-1 SAR VV/VH 影像的数据准备流程，基于既有 SAR 预处理结果完成双通道融合、GeoTIFF 切片、无效样本清理及 QGIS 人工核验；最终报告记录约 32k 训练、8k 验证和 1.6k 测试样本，用于无监督微毁林变化检测。

如果简历必须更短：

> 构建 Sentinel-1 VV/VH 数据处理与 QGIS 核验流程，完成双通道融合、GeoTIFF 切片和无效样本清理，形成报告记录约 32k/8k/1.6k 的训练、验证和测试集。

**Reason**

保留真实的数据工程贡献，同时把完整上游 SAR preprocess 和近似样本数的证据边界说清楚。

### Bullet 2

**Original**

> 构建融合残差连接、FPN 与自注意力机制的 CNN Autoencoder，以重建误差生成变化异常图；集成 Optuna、TensorBoard 及 DataLoader 多进程预取，支持超参数搜索、训练监控与可复现实验。

**Problem**

- 模型结构和 reconstruction-error anomaly map 有源码直接支持。
- “DataLoader 多进程预取”不准确：CUDA 路径只有 `num_workers=1`，即一个 worker 子进程；没有显式 `prefetch_factor` 或 `persistent_workers`。
- “可复现实验”过强：只有 PyTorch seed、CLI、版本 requirements 和权重 checkpoint；缺数据版本、完整 config snapshot、NumPy/Python/Optuna seed 和 deterministic 设置。
- Optuna 只搜索 lr 与 weight decay，不能暗示全面 architecture/scheduler search。

**Risk Level**

`CRITICAL`：`num_workers=1` 与“多进程预取”、以及部分 reproducibility 与“可复现实验”都是源码一眼可见的冲突。

**Recommended Revision**

> 构建融合残差连接、FPN 与自注意力的 CNN Autoencoder，以逐像素重建误差生成变化异常图；集成 Optuna 与 TensorBoard，并使用单 worker DataLoader 和 pinned memory 支持超参数搜索、训练监控及实验配置管理。

更强调通用工程价值的版本：

> 构建融合残差连接、FPN 与自注意力的 CNN Autoencoder；集成 Optuna、TensorBoard、命令行配置及模型 checkpoint，并通过离线预切片、单 worker 加载和 pinned memory 改善训练数据供给。

**Reason**

新版本准确描述当前实现，不把框架默认行为包装成手写的多 worker 预取，也不把部分复现基础扩大为完整可复现性。

### Bullet 3

**Original**

> 在独立测试区域取得 Precision 0.828、Recall 0.687、F1 0.751 和 IoU 0.601，并结合 QGIS 完成检测结果可视化及与 CUSUM 时序基线的对比分析。

**Problem**

- 指标来自最终报告，但当前仓库无法精确复现；保存 notebook 是另一组结果。
- “独立测试区域”在空间上成立，但“独立”容易被理解成 untouched、完全冻结的 holdout。测试影像用于 GMM 无标签拟合、可视化和后处理研发。
- 评估 notebook 用 prediction geometry 裁剪 annotation，可能漏算 FN。
- “CUSUM 时序基线”容易暗示同一数据、同一指标、统一协议下的量化 benchmark；当前证据更接近预处理参考和定性结果对照。

**Risk Level**

`CRITICAL`：这是最容易被追问实验可信度、数据泄漏、指标复现和 baseline 公平性的句子。

**Recommended Revision**

> 最终报告在单独地理测试区域记录 Precision 0.828、Recall 0.687、F1 0.751 和 IoU 0.601；将检测结果导出为 Shapefile，并在 QGIS 中与人工标注及 CUSUM 输出进行空间可视化对照。

如果希望把不可复现风险进一步降到最低：

> 在单独地理测试区域完成检测结果的 Shapefile 输出、QGIS 可视化及人工标注对照；最终报告记录 Precision 0.828、Recall 0.687、F1 0.751 和 IoU 0.601，并与 CUSUM 输出进行定性比较。

**Reason**

“最终报告记录”准确反映证据等级；“单独地理测试区域”避免暗示 untouched holdout；“输出/定性比较”避免把 CUSUM 包装成已严格量化的公平 baseline。

## 3. High-Risk Phrase Decisions

| Phrase | Decision | Replacement |
|---|---|---|
| 自动预处理 | 建议删除或限定 | “数据准备流程”或“基于既有 SAR 预处理结果完成融合、切片和清理” |
| 多进程预取 | 建议删除 | “单 worker DataLoader 后台加载与 pinned memory” |
| 可复现实验 | 建议降级 | “实验配置管理与部分复现基础” |
| 独立测试区域取得 | 建议改写 | “最终报告在单独地理测试区域记录” |
| CUSUM 时序基线 | 建议改写 | “CUSUM 时序方法输出参照”或“与 CUSUM 输出定性对照” |

## 4. Recommended Three-Bullet Resume Version

> • 设计 Sentinel-1 SAR VV/VH 影像的数据准备流程，基于既有 SAR 预处理结果完成双通道融合、GeoTIFF 切片、无效样本清理及 QGIS 人工核验；最终报告记录约 32k 训练、8k 验证和 1.6k 测试样本，用于无监督微毁林变化检测。
>
> • 构建融合残差连接、FPN 与自注意力的 CNN Autoencoder，以逐像素重建误差生成变化异常图；集成 Optuna、TensorBoard、命令行配置与模型 checkpoint，并通过离线预切片、单 worker 加载及 pinned memory 改善训练数据供给。
>
> • 最终报告在单独地理测试区域记录 Precision 0.828、Recall 0.687、F1 0.751 和 IoU 0.601；将变化结果导出为 Shapefile，并在 QGIS 中与人工标注及 CUSUM 输出进行空间可视化对照。

## 5. Interview Use Rule

当面试官要求证据时，按以下顺序回答：

1. 先说当前源码能直接证明的实现。
2. 再说最终报告记录的历史实验事实，并明确使用“报告记录”。
3. 主动指出当前仓库不能精确复现哪些内容。
4. 最后说明 2026 年会如何修复，不要把改进方案说成当年已经实现。
