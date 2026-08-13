# Engineering Review：从 Research Prototype 到可信批处理系统

> 目标不是用 2026 年标准否定当年的实习，而是准确说明：原型解决了什么、工程风险在哪里、今天会如何用最小但可靠的设计改进。
>
> `CURRENT SOURCE` 是当前仓库行为；`HISTORICAL/REPORT` 是报告记录；`MODERN DESIGN` 是现在的改进建议。数据读取、DataLoader 和 IO 细节见 [02-data-pipeline.md](./02-data-pipeline.md)。

## 1. Prototype vs Production

Research prototype 通常优先：

- 快速验证 reconstruction anomaly detection 是否可行；
- 探索数据、模型和后处理组合；
- 生成图、指标和报告；
- 在有限人员和时间下保留足够的实验代码。

Production-quality pipeline 在此基础上还必须回答：

| Property | Production question |
|---|---|
| Reliability | 某个 TIFF 损坏、GPU OOM 或任务中断时，系统如何失败和恢复？ |
| Repeatability | 同样输入和配置能否得到同样输出？ |
| Maintainability | 路径、阈值和 preprocessing 是否只有一个 source of truth？ |
| Observability | 能否知道慢在哪里、跳过了哪些文件、每阶段处理了多少数据？ |
| Portability | 能否在另一台机器上运行，而不是依赖 `/home/yifan/...`？ |
| Scalability | 32k 变成 320k 时，文件系统、CPU、GPU 和 metadata 怎么变化？ |

本项目的价值在于它完成了数据融合/切片、PyTorch 训练、异常后处理、GIS 输出和评价的完整研究闭环。它的问题主要是这些步骤没有被封装成可版本化、可测试、可恢复的系统。面试定位应是：

> 当年交付了研究原型；现在复盘时，我能识别正确性边界，并给出不过度设计的生产化路径。

## 2. P0 Correctness Risks

### A. Train / inference normalization inconsistency

**Observed Behavior**

Dataset 为 train/validation/test 使用固定 `[-15,-3]` MinMax normalization。辅助 inference loader 只有收到 min/max 才 normalize，并会 clamp；五期 temporal 调用未传 min/max；large-area inference 自己读 TIFF，完全没有 normalization。

**Why It Matters**

Autoencoder 在分布 A 上学习，却在分布 B 上计算 reconstruction error。这是 train–serving skew，可能让误差主要反映尺度差，而不是毁林。

**Failure Scenario**

模型训练输入约 `[0,1]`，整区推理收到约 `[-15,-3]`。流程仍能输出 GMM cluster 和 Shapefile，但 anomaly ranking 的语义已经改变，是一种“成功运行但结果错误”的静默故障。

**How I Would Verify**

对同一个 TIFF 分别走 Dataset、temporal helper 和 large-area loader，记录 tensor shape、dtype、min/max、quantile 和逐元素差；用固定 checkpoint 比较三条路径的 reconstruction error。

**How I Would Fix**

建立唯一 `SARTransform`，由训练与所有推理入口调用；transform 配置与 checkpoint 绑定；增加 range assertions 和 train–inference consistency test。

**Interview Answer**

> 最大的正确性风险是训练和整区推理没有共享同一个 transformation。对重建异常检测来说，输入尺度本身直接决定误差，所以我会先统一 preprocessing，再讨论换模型或调阈值。

详细数据路径见 [02-data-pipeline.md 的 Normalization](./02-data-pipeline.md#5-normalization本项目最重要的数据契约)。

### B. Evaluation annotation clipping by prediction geometry

**Observed Behavior**

评价 notebook 使用 prediction/detection geometry 的范围裁剪 annotation 和 forest mask，再栅格化计算 TP、FP、FN；输出网格也由预测范围产生。

**Why It Matters**

预测范围之外的真实毁林可能在计算前被删除，从而漏算 FN，并高估 Recall、F1 和 IoU。评价域由预测决定，违反“先固定 evaluation universe”的原则。

**Failure Scenario**

模型只预测了真实区域的一小部分；annotation 被裁到这小部分后，看起来 Recall 很高，但大量未检出的 ground truth 根本没有进入 confusion matrix。

**How I Would Verify**

在固定 AOI 上分别计算裁剪前后 annotation 面积、positive pixel 数和 TP/FP/FN；构造一个预测为空或只覆盖一半标注的 synthetic fixture，检查 Recall 是否符合预期。

**How I Would Fix**

在看 prediction 前固定 AOI、CRS、resolution、transform、width/height 和 valid-data mask；GT 与 prediction 都投影到同一网格；保存 evaluation manifest 和 TP/FP/FN。

**Interview Answer**

> 当前 notebook 的评价边界依赖预测 geometry，可能漏掉范围外 FN。报告值可以作为历史结果陈述，但今天我会用固定、与预测无关的 AOI 重算，保证评价域不会随模型输出改变。

### C. Reported metrics not exactly reproducible

**Observed Behavior**

最终报告记录 P 0.8284、R 0.6865、F1 0.7508、IoU 0.6011；当前 notebook 保存另一组结果。仓库没有报告值对应的输入数据、checkpoint、完整配置、阈值、中间 mask、TP/FP/FN 和 run ID。

**Why It Matters**

无法建立“代码版本 + 数据版本 + 模型 + 评价协议 → 指标”的完整证据链。将它称为当前可复现 benchmark 会损害可信度。

**Failure Scenario**

换一台机器运行时，不知道应选哪个 checkpoint、日期、GMM 参数或 AOI；即使得到另一组合理指标，也无法判断是代码漂移、数据漂移还是评价差异。

**How I Would Verify**

先盘点所有 artifact 和 notebook output，反向建立候选运行配置；若缺失关键输入，不伪造复现结论，而将报告值标为 `REPORT ONLY`。

**How I Would Fix**

每次运行保存 run metadata、manifest、Git SHA、完整 config、checkpoint hash、后处理参数、AOI、TP/FP/FN 和 metrics JSON；提供单一 evaluation CLI。

**Interview Answer**

> 我区分报告记录值和仓库可复现值。报告值有正式文档证据，但当前 artifact 不足以精确复跑；这是当年实验封装不足，也是我现在最重视 run lineage 的原因。

### D. Normalization range conflict in report

**Observed Behavior**

报告一处写 `[-15,-3]`，后文多次写 `[-15,3]`；当前 Dataset 源码明确是 `min=-15, max=-3`。

**Why It Matters**

上界符号差 6 dB，归一化结果明显不同；文档与实现冲突会使复现者选择不同数据分布。

**Failure Scenario**

新实验按报告的 `+3` 重跑，模型训练和 anomaly threshold 全部改变，却仍使用旧 checkpoint 或指标作比较。

**How I Would Verify**

读取运行时 config/checkpoint metadata；但当前没有这些 artifact，因此只能确认提交源码行为，不能推定历史运行一定使用哪一版。

**How I Would Fix**

normalization 只在 typed config 定义一次；训练启动时把 resolved config 写入 run directory 和 checkpoint；报告表格由机器可读 config 生成。

**Interview Answer**

> 当前源码是 `[-15,-3]`，报告内部存在 `-3/+3` 冲突。我会明确说这是版本或文档不一致，不会为了让材料看起来一致而猜测。

### E. Transductive GMM post-processing

**Observed Behavior**

五期和大区域分支在待分析影像的 pixel-loss distribution 上拟合两分量 GMM，再把高均值 component 当 anomaly。它不使用标签，但使用了测试输入分布。

**Why It Matters**

这是无标签的 transductive/test-time adaptive post-processing，不等于在 validation 上确定后完全冻结的 inductive detector。结果会受当前影像异常比例和分布影响。

**Failure Scenario**

若某个测试日期大面积异常，GMM 的两个 component 可能不再对应“正常/毁林”；若几乎没有异常，它也仍会强制分两类。

**How I Would Verify**

比较两套协议：validation 上拟合并冻结的 detector，与 test-time GMM；报告不同日期的 component means、异常比例和指标稳定性。

**How I Would Fix**

保留两种可选模式并显式命名 `inductive`/`transductive`；在实验报告中分别给指标，不把 test-time adaptation 描述成 untouched holdout。

**Interview Answer**

> AE 权重只由正常森林训练，但 GMM 会无标签适应测试误差分布。它的优点是阈值适应性，代价是评价不再是完全冻结的 detector；我会明确协议并补一个 validation-fitted baseline。

### F. `Tanh [-1,1]` output vs normalized `[0,1]` input

**Observed Behavior**

训练输入设计上主要位于 `[0,1]`，decoder 最后一层是 `Tanh`，输出域为 `[-1,1]`。

**Why It Matters**

输出域和主要目标域不完全对齐，浪费部分动态范围。它不一定导致训练失败，但没有证据证明这是最佳组合；范围外输入又会加剧不一致。

**Failure Scenario**

模型需要把一半输出范围基本闲置；负输出在 MSE 下产生额外误差，可能改变正常/异常 loss distribution。

**How I Would Verify**

固定 split、seed 和后处理，比较 `Sigmoid+[0,1]`、`Tanh+[-1,1]` 和 linear output；同时查看 reconstruction range 与 anomaly metrics。

**How I Would Fix**

先统一输入 contract，再通过 ablation 选择输出 activation；把预处理范围和 activation compatibility 纳入配置验证。

**Interview Answer**

> Tanh 并非一定错误，但它和当前 `[0,1]` 输入不是最自然的匹配。我会把它视为需要实验验证的设计，而不是事后声称它理论最优。

## 3. Configuration Management

### Current source

- Dataset、预处理和 anomaly 函数多处硬编码 `/home/yifan/Documents/...`。
- 部分训练参数由 `argparse` 提供：batch、epochs、lr、weight decay、scheduler、seed、results path。
- `--embedding-size` 对 VAE 有效，但 AE 在 `AE_Network` 中硬编码 512。
- `--test` 使用 `store_true, default=True`，实际上很难从 CLI 关闭。
- `split_data.py` 的 `prefix_fused/prefix_tile` 参数没有实际参与文件命名。
- large-area 函数的 `suffix_template` 和 `tile_size` 参数没有控制其硬编码匹配/处理逻辑。
- 同一参数分散在 CLI、函数默认值、类常量和 notebook cell 中。

### Risks

参数“看起来可配但实际无效”比没有参数更危险：用户以为改变了 embedding size、suffix 或 test mode，运行却仍沿用旧行为。硬编码路径则让代码绑定个人机器，且容易把 train/test 目录指错。

### Minimal modern design

不需要配置中心。单机研究系统使用 YAML/TOML 或完整 argparse 都可以，关键是：

```text
base config
  + environment-specific paths
  + CLI overrides
  → schema validation
  → immutable resolved config saved with run
```

建议：

- typed schema 验证必填路径、范围、枚举和互斥选项；
- 所有代码只读取 resolved config，不再散落默认值；
- 启动时打印并保存最终 config；
- unused/unknown keys 直接报错；
- 路径通过 project root/data root 组合，而不是用户 home；
- 增加配置测试，证明 `embedding_size` 等参数真的改变模型。

## 4. Reproducibility

### 当前已有基础

- `requirements.txt` 固定直接依赖版本；
- CLI 暴露部分超参数；
- `torch.manual_seed(args.seed)`；
- 保存 `model.state_dict()`；
- TensorBoard 记录训练/验证 loss。

这说明项目有复现意识，但只能称为 `PARTIALLY VERIFIED`。

### 当前缺失

- `random.seed()`；
- `numpy.random.seed()`；
- Optuna sampler seed 和 persistent storage；
- CUDA deterministic algorithm / cuDNN 配置；
- DataLoader worker seed policy；
- dataset/split manifest；
- resolved config snapshot；
- Git commit；
- preprocessing 与 evaluation version；
- optimizer/scheduler/epoch/random state；
- 操作系统、GPU/driver 和完整 transitive environment lock。

### 每个实验最小 metadata

```text
run_id
started_at / completed_at / status
git_commit
dataset_manifest_version
preprocessing_version
evaluation_version
resolved_config
random_seeds
environment summary
checkpoint path + checksum
metrics + TP/FP/FN
```

可复现不等于 bitwise deterministic。工程上至少要做到：能重建同一输入、同一配置和同一评价协议，并在允许的数值容差内得到一致结论。

## 5. Checkpoint Design

### 当前 checkpoint

EarlyStopping 和逐 epoch 保存都只写：

```python
torch.save(model.state_dict(), path)
```

这适合最小 **inference checkpoint**：已知代码结构和 config 时加载权重推理。

### Training recovery checkpoint

要从中断处继续训练，应至少保存：

```python
{
  "model": model.state_dict(),
  "optimizer": optimizer.state_dict(),
  "scheduler": scheduler.state_dict(),
  "epoch": epoch,
  "global_step": global_step,
  "best_metric": best_metric,
  "early_stopping": early_stopping_state,
  "config": resolved_config,
  "random_states": {...},
  "dataset_version": ...,
  "git_commit": ...,
}
```

原因：Adam 的 moment、StepLR 当前 step、early-stopping counter 和 RNG 状态都会影响后续轨迹。只恢复 model weights 相当于“从该权重重新开始一个新训练过程”，不是无损 resume。

还应使用 temporary file + atomic rename，保存 checksum，并区分 `best.ckpt`、`last.ckpt` 和 immutable epoch checkpoint。

## 6. Logging and Observability

### 当前行为

- 大量 `print()`；
- TensorBoard 记录 AE 的 `Loss/train` 和 `Loss/validation`；
- Optuna 输出 best params/value；
- 没有统一 run ID、结构化日志、阶段耗时、吞吐或错误计数；
- 多个对象可能共享 `results/logs`，trial 也没有隔离目录。

### Research tracking vs production observability

TensorBoard loss 回答“模型有没有收敛”；production observability 还要回答“哪一步慢、哪份数据坏、作业是否完整、结果属于哪个版本”。

建议结构化事件：

```json
{
  "timestamp": "...",
  "run_id": "...",
  "stage": "tile_decode",
  "file": "...",
  "duration_ms": 12.4,
  "status": "failed",
  "error_type": "CorruptTIFF"
}
```

建议指标：

- samples/s、batches/s、batch latency；
- DataLoader wait time、TIFF decode time、H2D time；
- GPU utilization、peak allocated/reserved memory；
- train/validation loss、learning rate；
- processed/accepted/rejected/failed sample counts；
- GMM component means、anomaly fraction；
- checkpoint/evaluation duration。

这些是 `MODERN DESIGN`，不能说当年都已记录。DataLoader troubleshooting 见 [02-data-pipeline.md 的 GPU 利用率章节](./02-data-pipeline.md#10-如果-gpu-utilization-很低)。

## 7. Error Handling

### 当前真实模式

| Situation | Current behavior | Review |
|---|---|---|
| VV/VH 数量不同 | `assert` | 应使用显式 validation；Python `-O` 可禁用 assert，且数量相同也不保证正确配对。 |
| CRS/transform 或尺寸不一致 | `raise ValueError` | 合理 fail fast：继续融合会产生语义错误。 |
| Dataset channel 不为 2 | `raise ValueError` | 合理 fail fast，但应包含 file path、actual shape 和 run context。 |
| NaN/zero cleaning | 直接 `os.remove` | 过于破坏性；应 quarantine + manifest。 |
| 清理时 TIFF 读取失败 | `except Exception: print`，继续 | 作业可能“成功”但损坏文件仍在；应计数并最终决定失败。 |
| temporal anomaly 单文件失败 | 打印并 `continue` | 批处理可接受，但必须记录 rejected item，并在错误率超过阈值时使 job 失败。 |
| large-area tile pair 任一失败 | 跳过该 tile | 结果会有空间缺口，却没有 completeness manifest。 |
| model/dataset/checkpoint 不支持 | `print` + `sys.exit` | CLI 尚可，但库函数最好抛 typed exception，由入口决定 exit code。 |
| CUDA RuntimeError | Optuna 打印部分错误后 re-raise | fail-fast 尚可；无 cleanup/retry/试验状态细分。 |
| 手动中断 | 捕获并打印 | 没有保存 `last.ckpt` 或标记 run aborted。 |

### Fail fast vs skip + record

**应 fail fast：**

- 全局 config 非法；
- channel order/shape contract 错；
- CRS/transform 不一致；
- checkpoint 与 model/preprocessor version 不兼容；
- evaluation AOI/grid 不一致；
- 失败比例超过容忍阈值。

**可 skip + record：**

- 大规模 ingestion 中少量独立损坏样本；
- 某个 tile 可被隔离，且输出 manifest 明确标记缺口；
- retry 后仍失败，并且业务允许 partial result。

关键不是“永远报错”或“永远跳过”，而是让调用方知道任务是否完整。每次 skip 都应包含 file、reason、attempt 和 stage，最终输出 accepted/rejected/failed summary。

CUDA OOM 不应盲目 retry 同样配置；应 fail with diagnostics，或由显式策略降低 batch/使用 accumulation，而不是静默产生不同实验。

## 8. Idempotency

一个可重复运行的 preprocessing job 应满足：

```text
unchanged input + unchanged config/version = same committed output
```

### 当前限制

- raw/intermediate/processed 没有完整不可变边界；
- fused 文件只按“存在”跳过，可能信任半写或陈旧文件；
- tile 直接覆盖；
- 清理脚本直接删除 processed 文件；
- 没有 run manifest、成功标记或 checksum；
- hard-coded output 让不同实验相互覆盖。

### 最小改造

```text
raw/                         immutable
intermediate/preprocess-v2/  generated
processed/dataset-v3/        accepted, immutable after commit
rejected/dataset-v3/         quarantined
manifests/dataset-v3.parquet
```

每个 artifact 先写同目录 temporary file，验证 size/checksum/metadata 后 atomic rename。stage 成功后再提交 manifest/`_SUCCESS`。输出路径包含 data/preprocessing version；重复运行先比较 input checksum 与 config hash，而不是只看文件是否存在。

## 9. Tests

当前 repository 没有 unit tests、integration tests、fixture dataset 或 CI。最小测试集不需要真实 32k 数据。

### Unit tests

| Test | Assertions |
|---|---|
| VV/VH pairing | 同日期/区域/坐标正确 join；缺失 counterpart 明确失败；不允许 sorted-zip 静默错配。 |
| Geospatial validation | CRS、transform、width/height 任一不同都会拒绝。 |
| Normalization | `-15→0`、`-9→0.5`、`-3→1`；明确测试 clip policy、NaN 和 max=min。 |
| Channel conversion | HWC 与 CHW 得到相同 `2×256×256` tensor；错误 channel/shape 报错。 |
| Tiling | transform 正确更新；边缘丢弃或 padding policy 明确。 |
| Connected components | `<min_size` 删除，`>=min_size` 保留；边界值测试。 |
| Config wiring | 改 `embedding_size` 时支持的模型结构确实改变；无效参数直接拒绝。 |

### Integration test

用 2–4 个程序生成的微型 GeoTIFF fixture：

```text
VV/VH GeoTIFF
  → fusion + one tile
  → Dataset
  → DataLoader batch
  → model forward
  → anomaly loss map
  → binary mask / vector output
```

验证 shape、CRS/transform、finite values、输出文件和失败报告。不训练完整模型。

### Regression test

固定小样本和固定 lightweight checkpoint，检查：

- tensor range 与 checksum/统计量；
- model output dimensions；
- anomaly map dimensions；
- connected-component pixel count；
- evaluation synthetic case 的已知 TP/FP/FN。

### 最重要：Train–Inference Consistency Test

```python
train_tensor = dataset_transform(read_tiff(path))
infer_tensor = inference_transform(read_tiff(path))
assert train_tensor.shape == infer_tensor.shape
torch.testing.assert_close(train_tensor, infer_tensor)
```

同一个 TIFF 经训练和推理入口必须得到相同 dtype、channel order、shape、range 和数值。这一个测试就能直接捕获本项目最严重的 normalization skew，因此优先级高于增加更多模型测试。

## 10. CI

简单 CI 足够：

```text
checkout
  → install locked minimal dependencies
  → lint / format check
  → unit tests
  → tiny GeoTIFF integration test on CPU
  → artifact/schema validation
```

不应每次提交都训练 32k dataset，因为：

- 时间和 GPU 成本高；
- 结果含随机性，反馈慢且不稳定；
- CI 的主要目标是快速阻止接口、shape、preprocessing 和评价回归。

完整训练可以由手动或定时 workflow 触发，使用固定数据版本，并把 run metadata 与结果作为 artifact 保存。普通 PR CI 应在几分钟内结束。

## 11. Performance Engineering

原则：

```text
measure → identify bottleneck → optimize one layer → measure again
```

### 分层定位

| Layer | Questions | Candidate changes after evidence |
|---|---|---|
| Data | 是否重复转换？sample 是否过碎？ | 离线 materialization、manifest、sharding。 |
| CPU | TIFF decode 或 Python collate 是否满核？ | workers、vectorization、减少重复初始化。 |
| Memory | host/pinned RAM 是否过高？是否频繁分配？ | prefetch/batch 调整、reuse、限制 pinning。 |
| GPU | kernel 是否太小？OOM？空闲等待？ | batch、AMP、gradient accumulation、模型 profiling。 |
| IO | metadata IOPS 还是顺序吞吐限制？ | SSD、shard、缓存；不是默认“加缓存”。 |

### 项目中已有的性能意识

- 离线切片避免每个 epoch 重复裁大图；
- CUDA 路径设置一个 DataLoader worker；
- `pin_memory=True`；
- Optuna trial 前尝试释放 unused CUDA cache 并限制 allocator fraction。

这些是早期优化，但仓库没有保存系统化 profile 或 before/after benchmark，因此面试应说“做了这些设计/尝试”，不要声称确定提升了某个百分比。

具体操作见 [02-data-pipeline.md 的 GPU 低利用率排查](./02-data-pipeline.md#10-如果-gpu-utilization-很低)和 [10× 数据量设计](./02-data-pipeline.md#11-如果数据量扩大-1032k--320k)。

## 12. Resource Management

### 当前 CUDA 行为

- `args.cuda` 由可用性和 `--no-cuda` 决定；
- model 和 batch 被移动到 device；
- Optuna objective 调用 `torch.cuda.empty_cache()`；
- CUDA 可用时调用 `torch.cuda.set_per_process_memory_fraction(0.9)`；
- 未使用 mixed precision、gradient accumulation 或显式 peak-memory telemetry。

### 两个 API 的准确含义

`empty_cache()` 只释放 PyTorch caching allocator 中**未被活跃 tensor 使用**的缓存，让其他分配者可见；它不能释放仍被 model、optimizer 或 batch 引用的显存，也不是 OOM 根治方案。

`set_per_process_memory_fraction(0.9)` 限制该进程 allocator 可使用的显存比例，有助于避免吃满整卡，但不会减少模型真实所需内存；限制过低反而会更早 OOM。当前 objective 是在重新初始化模型之后才设置这个限制，因此它也不能约束此前已经完成的模型分配。

### 新发现的重复初始化

`train.py` 先实例化 VAE 和 AE，再选择其中一个，而且随后又创建独立 DataLoader wrapper。两个 model constructor 都会建立 Dataset/DataLoader、SummaryWriter 并把 model 放到 device。因此即使只训练 AE，也可能同时保留未使用 VAE 的 GPU 参数和多个对象。这会增加显存、目录扫描和日志资源，占用是可以避免的。

现代写法是 registry/factory 只实例化选中的 model，并让 DataModule 只构造一次后注入。SummaryWriter 应使用 context manager 或显式 close。

### OOM 处理优先级

1. profiler/peak memory 确认 activations、parameters、optimizer state 的占比；
2. 调整 batch size；
3. 需要保持 effective batch 时用 gradient accumulation；
4. 评估 AMP/mixed precision；
5. 减少未使用模型与重复对象；
6. 最后再考虑模型结构或 checkpointing。

这些是 modern improvements，不是当年已实现。

## 13. Experiment Management

### 当前 Optuna

- 10 trials；
- 搜索 lr `1e-4..5e-4` 和 weight decay `1e-6..1e-5`；
- objective 为 validation loss；
- `optuna.create_study()` 无 persistent storage；
- sampler 未固定 seed；
- trial 会 mutate 共用 `args`，并对共用 architecture object 手动调用 `__init__`；
- 所有 trial 共用 `results/best_model.pth` 和 TensorBoard `results/logs`；
- 每个 trial 的 config、checkpoint 和 logs 没有隔离。

结果是：best checkpoint 可能被后续非最佳 trial 覆盖；TensorBoard 曲线混在一起；进程中断后 study 无法可靠 resume；artifact 与 trial number 的对应关系不清晰。

### 最小 run layout

```text
runs/
  study_2026-08-12/
    study.db
    trial_0007/
      config.yaml
      metadata.json
      logs/
      best.ckpt
      last.ckpt
      metrics.json
```

每个 trial 独立构造 model/optimizer/writer，独立目录，记录 trial ID 和参数；study 使用 SQLite 即可满足单机持久化，不需要复杂平台。完成后从 `study.best_trial` 明确复制或注册 best artifact。

## 14. Data / Model Versioning

最小关联链：

```text
Git commit
    + dataset manifest version
    + preprocessing config/version
        → training run_id
        → model checkpoint + hash
        + inference config/version
        → GIS output
        + evaluation AOI/grid/version
        → metrics
```

每一个 metric 都必须能回答：

> 这是在哪个代码版本、哪个数据版本、哪套 preprocessing、哪个 checkpoint 和哪个 evaluation protocol 上得到的？

只给 `best_model.pth` 或只给四个数字无法回答这个问题。版本化不一定要引入 DVC/MLflow；Git SHA、不可变 manifest、配置快照、checksum 和清晰目录已经能显著改善。

## 15. Research-to-System Architecture

不需要微服务化。适合本项目的是单机/批处理 job：

```text
Immutable raw SAR references
        ↓
Preprocessing job
  validate pairing/alignment, fuse, tile, quarantine
        ↓
Versioned dataset + manifest
        ↓
Training job
  resolved config, seeds, logs, recovery checkpoint
        ↓
Experiment metadata + selected model artifact
        ↓
Batch inference job
  same SARTransform, fixed model/config
        ↓
Versioned raster/vector GIS output
        ↓
Evaluation job
  fixed AOI/grid, metrics JSON, TP/FP/FN
```

每个 stage 都有：输入版本、配置、输出目录、manifest、结构化日志、失败状态和幂等重跑策略。编排可以从 Makefile、Python CLI 或简单 workflow 开始；数据规模和团队尚不足以证明需要微服务、消息队列或 Kubernetes。

## 16. 面向后端 / 系统岗位的项目价值

| Capability | Project evidence | Mature framing |
|---|---|---|
| Data pipeline design | VV/VH 对齐、GeoTIFF 融合、切片、清理、Dataset | 我理解离线 materialization 与在线 loading 的边界。 |
| Separation of concerns | preprocessing、Dataset、training、anomaly、evaluation 已有初步模块 | 我能指出哪些重处理不应放入 `__getitem__`，并设计清晰 stage contract。 |
| Performance awareness | 离线切片、worker、pin memory | 我不会凭感觉加 worker，会先分解 IO/CPU/H2D/GPU 时间。 |
| Debugging | normalization、georeference、坏 TIFF 等问题 | 我重视跨阶段数据契约和“能运行但语义错误”的 silent failure。 |
| Configuration management | CLI 与 hard-coded path 共存 | 我能识别 split-brain config，并建立 schema-validated source of truth。 |
| Reproducibility | seed、requirements、checkpoint、TensorBoard | 我知道这些只是基础，完整 lineage 还需 data/config/code/eval version。 |
| Observability | print + loss curves | 我能从模型指标扩展到 stage latency、throughput、error/completeness metrics。 |
| Testing | 当前缺失 | 我会优先补 train–inference consistency 和固定 AOI evaluation tests。 |
| Correctness | 发现 normalization skew 和 evaluation clipping | 我会先修 measurement/data correctness，再做模型优化。 |
| Trade-offs | 预切片换吞吐，GMM 自适应换严格 holdout | 我能说明收益、代价和适用边界，而不是只列技术名。 |

## 17. 面试回答（45–90 秒）

### 1. “这个项目最大的工程问题是什么？”

> 最大问题不是模型结构，而是数据转换和评价协议没有形成单一、可测试的契约。训练 Dataset 使用 `[-15,-3]` normalization，但部分时序和整区推理绕过了它；对 Autoencoder 来说，输入尺度直接决定 reconstruction error，所以 pipeline 即使成功输出图，语义也可能已经偏了。评价 notebook 又按预测 geometry 裁 annotation，可能漏算 FN。今天我会先统一 train/inference transform，并用固定 AOI 重写 evaluation，加两类回归测试。模型调优必须在可信输入和可信测量之后。

### 2. “如果现在重新做，你最先改什么？”

> 我不会先换更大的模型。第一步是建立数据与实验的 source of truth：用 manifest 固定 spatial split 和样本版本；让 Dataset、时序推理和整区推理共享同一个 `SARTransform`；把 normalization、channel order 和 preprocessing version 写进 checkpoint；然后固定 AOI 重算指标。完成这些后保留原 AE baseline，再做 FPN、attention、threshold 或 pretrained encoder 的对照。这样每次算法变化都有可信、可重复的测量基础。

### 3. “如果数据量变成十倍，你怎么办？”

> 我会先 profile，而不是直接换分布式系统。当前是大量小 TIFF，32k 到 320k 后可能先卡在目录扫描、open/close 和随机 IOPS。我会记录 DataLoader wait、decode time、IOPS 和 samples/s，先测试 SSD、worker 数、persistent workers 和 batch size。如果小文件 metadata 已是主瓶颈，再按访问模式选择 tar shards/WebDataset、LMDB 或 Zarr，并保留 Parquet manifest 管理 split 和 lineage。只有单机吞吐或容量确实不够时，才把 versioned shards 放到对象存储并按 worker 分片。

### 4. “如果 GPU 利用率很低，你怎么排查？”

> 我先把一步拆成 `next(DataLoader)`、H2D、forward/backward 和 optimizer 时间，用 PyTorch Profiler 配合 GPU、CPU 和磁盘指标判断瓶颈。若 batch wait 长，检查 TIFF decode、随机 IO 和单 worker 是否供不上；若 GPU kernel 很短，测试 batch size；若 copy 占比高，验证 pinned memory 加 non-blocking copy 是否真的 overlap；还会检查 `.item()`、日志和 checkpoint 的同步停顿。每次只改一个变量，用固定几百 step 比较 samples/s，而不是把 workers 一次拉满。

### 5. “怎么保证训练和线上/离线推理的数据处理一致？”

> 把 preprocessing 做成一个版本化的 `SARTransform`，训练 Dataset、batch inference 和 evaluation 都调用同一实现，不复制 normalization 代码。transform 配置包含 VV/VH 顺序、`[-15,-3]`、clip policy、dtype 和 shape，并随 checkpoint 保存。最关键的测试是让同一个 TIFF 同时走训练入口和推理入口，断言 tensor 的 shape、dtype、range 和逐元素值一致。本项目复盘发现的 train–serving skew，正是这个测试应该捕获的问题。

### 6. “怎么保证实验可复现？”

> 当前项目有 requirements、CLI、PyTorch seed 和权重 checkpoint，但还不够。我会为每个 run 保存 `run_id`、Git SHA、dataset manifest、resolved config、Python/NumPy/PyTorch/Optuna seeds、preprocessing/evaluation version、环境摘要、checkpoint hash 和 TP/FP/FN。恢复训练的 checkpoint 还要包含 optimizer、scheduler、epoch 和 random state。目标不一定是跨硬件 bitwise 一致，而是能准确重建同一输入和协议，并解释任何差异来自哪里。

### 7. “research code 和 production code 最大区别是什么？”

> Research code 优先快速验证假设，这个项目已经完成从 GeoTIFF 到模型、Shapefile 和指标的闭环。Production code 除了算法结果，还要保证失败可见、结果可追溯、任务可重跑和环境可迁移。具体到这个项目，就是把个人绝对路径变成校验过的配置，把直接删文件变成 quarantine + manifest，把 print 变成带 run ID 的结构化日志，把 `best_model.pth` 变成含数据和配置 lineage 的 artifact，并用小型 CI 测试关键 contract，而不是每次完整训练。

### 8. “你从这个项目中最大的工程收获是什么？”

> 最大收获是，ML 系统的正确性往往不在模型层，而在阶段之间的数据契约。一个模型可以正常 forward，GMM 也能正常出两类，但如果训练和推理 normalization 不同，或者评价范围由预测决定，结果仍可能失真。我也学到性能优化要看系统边界：离线切片是用存储换训练吞吐，worker 和 pinned memory 只是数据供给的一部分。现在我会优先保证 lineage、可观测性和一致性，再优化模型与速度。
