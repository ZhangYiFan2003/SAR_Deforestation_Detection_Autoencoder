# Data Pipeline：把项目解释成一个数据处理系统

> 本章只讲完成项目解释所需的数据工程知识，不展开遥感算法。
>
> 证据口径：`CURRENT SOURCE` 表示当前仓库源码可直接确认；`REPORT ONLY` 表示只由最终报告记录；`MODERN DESIGN` 表示 2026 年的改进方案，不代表当年已经实现。
>
> 项目总入口见 [00-project-overview.md](./00-project-overview.md)；正确性、可靠性和生产化审查见 [06-engineering-review.md](./06-engineering-review.md)。

## 1. End-to-End Data Flow

先把整条链路当成一个批处理数据系统，而不是一组零散 notebook：

```text
Sentinel-1 processed VV/VH files
        ↓
分别扫描、排序，并按位置 zip 配对
        ↓
验证 CRS / affine transform / width / height
        ↓
堆叠为 2-band GeoTIFF：2×H×W
        ↓
按 256×256 非重叠窗口切片，仅保留完整 tile
        ↓
删除含 NaN 或全零的 tile
        ↓
预先放入 train / validation / test 目录
        ↓
ProcessedForestDataset：查找 → TIFF decode → CHW → normalize → tensor
        ↓
ProcessedForestDataLoader：batch + shuffle + worker/pinned-memory 配置
        ↓
data.to(device) → GPU / CPU
        ↓
2×256×256 CNN Autoencoder
```

### 分阶段接口与故障模式

| Stage | Input | Output | Responsible file / function | Potential failures |
|---|---|---|---|---|
| Upstream SAR processing | Sentinel-1 GRD VV/VH | 文件名含 `gamma0-rtc_db` 的单波段 VV/VH GeoTIFF | 当前仓库没有完整实现；报告及 `cusum_preprocessing.ipynb` 的外部 CuSum 调用 | 下载失败、轨道/日期不一致、预处理版本不同、输出尺度不同。完整流程是 `REPORT ONLY / external dependency`。 |
| File discovery | VV 目录与 VH 目录 | 两个排序后的文件列表 | `pipeline/datasets/preprocessing/split_data.py::fuse_and_split_images` | 只接受小写 `.tif`；目录缺失；数量不同触发 `assert`；无 manifest。 |
| VV/VH matching | 两个排序列表 | `zip(vv_files, vh_files)` 的文件对 | 同上 | **没有按日期/区域 key 做显式 join**。两边数量相同但缺失项不同，可能把错误文件静默配成一对。 |
| Alignment validation | 一对单波段 GeoTIFF | 可融合的 VV/VH 数组 | 同上 | CRS、affine transform 不同会 `raise ValueError`；width/height 不同也会 `raise`；没有显式检查文件名语义或时间戳。 |
| VV/VH fusion | 两个 `1×H×W` band | `2×H×W` GeoTIFF | 同上 | 读取失败、磁盘满、写入中断、source band 假设不成立；存在同名 fused 文件时只按“路径存在”跳过，不验证 checksum 或完整性。 |
| Tiling | `2×H×W` fused GeoTIFF | 多个 `2×256×256` georeferenced tiles | 同上 | 只保存 `floor(H/256)×floor(W/256)` 个完整 tile，右边和下边余数被丢弃；tile 直接写目标路径，无 atomic commit；重新运行会覆盖 tile。 |
| Invalid-tile cleaning | processed 目录中的 TIFF | 删除坏样本后的目录 | `remove_missing_values.py::check_and_remove_nan_images`; `remove_zero_values.py::remove_invalid_tif_files` | 脚本直接 `os.remove()`，无 rejected 区、manifest、checksum 或回滚；损坏 TIFF 只打印错误并保留。 |
| Split materialization | 已选择地理区域的 tiles | 独立 train/validation/test 目录 | 当前仓库只消费这三个目录；区域划分来自报告 | 仓库没有可重放的 split manifest；路径放错可能造成污染；不能从当前仓库重建精确样本集。 |
| Dataset indexing | 一个 split 目录 | 排序后的 `.tif` 文件名数组 | `ProcessedForestDataset.__init__` | 启动时 `os.listdir`；只收小写 `.tif`；没有 checksum、标签、源影像 ID 或读取前完整性检查。 |
| Sample decode | 一个 TIFF path | NumPy `C×H×W` array | `ProcessedForestDataset.__getitem__` | TIFF 损坏；维数错误；通道数不是 2；没有显式验证这里的 H/W 必须为 256。 |
| Normalize/tensorize | `C×H×W` dB values | `float32` tensor，设计上约 `[0,1]` | 同上 | 固定 `[-15,-3]`；Dataset 不 clamp；范围外值会小于 0 或大于 1；推理分支未统一复用。 |
| Batch loading | Dataset samples | `B×2×256×256` | `ProcessedForestDataLoader` | CPU 路径同步读取；CUDA 路径只有 1 worker；小文件随机 IO、decode 或 worker startup 可能成为瓶颈。 |
| Device transfer | CPU tensor batch | device tensor batch | `AE.train/test`: `data.to(self.device)` | 没有 `non_blocking=True`；H2D copy 可能与计算串行；CUDA OOM。 |
| Model input | `B×2×256×256` | reconstruction / loss | `AE_Network` 和训练循环 | 输入尺度、shape 或 preprocessing version 不一致时，模型输出虽可计算但语义错误。 |

这里最值得系统面试强调的不是“用了 TIFF”，而是每一级都有明确的数据契约：文件配对、空间对齐、通道顺序、shape、数值范围和版本都必须一致。

## 2. Offline Preprocessing vs Online Loading

### 当前项目的边界

**OFFLINE：每份源数据通常执行一次**

- VV/VH 文件发现与配对。
- CRS、transform、width、height 检查。
- VV/VH 堆叠为双波段 GeoTIFF。
- 按 `256×256` 切片并保存地理 transform。
- 删除含 NaN 的 tile。
- 删除所有值均为 0 的 tile。
- 将不同地理区域的 tile 放入 train/validation/test 目录。

**ONLINE：每次 Dataset 取样或每个 epoch 重复执行**

- 根据 index 找到样本文件。
- `tifffile.imread()` 解码 TIFF。
- 将 layout 统一为 `C×H×W`。
- 使用训练范围 `[-15,-3]` 做 MinMax normalization。
- 转成 `float32 torch.Tensor`。
- DataLoader 组 batch、shuffle，并交给训练循环。

### 为什么切片应当离线执行

如果每个 epoch 都从约 `2×2000×2000` 大图动态裁出 `256×256` patch，会重复做窗口计算、文件打开、地理 metadata 处理和异常样本判断。离线切片的收益是：

- **CPU cost**：不在每轮训练重复做相同的裁剪和空间 metadata 工作。
- **Disk IO**：训练只读取需要的小 tile，不必每次加载大图再取窗口；但小文件数量增加，也会产生新的 metadata/random-IO 成本。
- **Repeat computation**：把确定性的转换从 `epochs × samples` 降到一次预处理。
- **Determinism**：固定 tile 边界和文件名，同一次实验各 epoch 看到相同样本集合。
- **Debuggability**：坏 tile 可以单独打开、核验和隔离，训练错误能定位到具体文件。
- **Storage trade-off**：代价是 fused 文件和大量 tile 同时占空间，并增加文件系统 inode、open/close 和目录扫描成本。

面试安全表述：

> 我把确定性、计算较重且需要保留地理信息的融合与切片放到离线阶段；Dataset 只保留轻量、每次训练必须完成的读取和 tensor 转换。这样用额外存储换取了训练吞吐、可调试性和更稳定的数据边界。

这体现的是 batch pipeline 中常见的 materialization trade-off，并不意味着“永远预切片最好”。如果数据量很大或采样策略需要每轮变化，窗口化读取、shard 或 chunked storage 可能更合适。

## 3. Dataset：`ProcessedForestDataset`

源码：`pipeline/datasets/data_loader.py`。

### `__init__(root_dir, min_val, max_val, transform)`

真实职责：

1. 保存目录、normalization 范围和可选 transform。
2. `os.listdir(root_dir)` 扫描目录。
3. 只保留文件名以小写 `.tif` 结尾的文件。
4. 对文件名排序，形成稳定的 index-to-path 映射。

工程含义：Dataset 构造时只建立索引，没有把 32k 个影像全部读进内存。代价是目录扫描依赖文件系统，并且缺少预生成 manifest。

### `__len__()`

返回 `len(self.image_files)`。DataLoader 用它确定数据集大小、batch 数量和 sampler 范围。

### `__getitem__(idx)`

真实顺序：

```text
index
  → root_dir + sorted filename
  → tifffile.imread
  → 2D 时补 channel axis
  → 3D 且最后一维为 2 时 HWC→CHW
  → 验证第一维必须为 2
  → 可选 MinMax normalization
  → torch.from_numpy(...).float()
  → 可选 transform
  → return tensor
```

值得注意：单通道 2D TIFF 虽然会先变为 `1×H×W`，随后仍因通道数不是 2 而 `raise ValueError`。因此“支持 2D”只是为了给出一致的错误路径，不代表模型接受单通道输入。

### Dataset 不应该承担哪些职责

Dataset 最好只负责：

- sample lookup；
- decode；
- 轻量、确定性的 normalize；
- channel/layout validation；
- tensor conversion。

它不应负责：

- Sentinel-1 下载；
- terrain correction / RTC；
- 大图融合和全量切片；
- 修改或删除原始数据；
- 决定 train/test 的地理区域。

原因是 **separation of concerns**：下载失败、预处理失败、样本解码失败和训练失败应当属于不同 stage，有独立的重试、日志、版本和验收标准。把重处理塞进 `__getitem__` 会让每个 epoch 重复工作，并让训练 worker 同时承担外部 IO、副作用和数据治理责任。

## 4. Tensor Layout

GeoTIFF/NumPy 读取结果可能是：

- `C×H×W`：例如 `2×256×256`；或
- `H×W×C`：例如 `256×256×2`。

代码的规则是：

```python
if combined_image.ndim == 3 and combined_image.shape[-1] == 2:
    combined_image = np.transpose(combined_image, (2, 0, 1))

if combined_image.shape[0] != 2:
    raise ValueError(...)
```

最终约定：

| Scope | Shape |
|---|---|
| Single sample | `2×256×256` = `C×H×W` |
| Batch | `B×2×256×256` = `N×C×H×W` |

PyTorch `Conv2d` 默认使用 `N×C×H×W`，因为卷积层需要明确 batch、输入通道和二维空间维度。第一层卷积的 `in_channels=2`，所以如果把 HWC 误当 CHW，模型要么立刻因通道数不匹配报错，要么在其他 shape 恰好碰撞时产生更隐蔽的语义错误。

现代实现还应验证：dtype、finite values、`H=W=256`、band order 明确为 `[VV,VH]`，而不只检查 `shape[0] == 2`。

## 5. Normalization：本项目最重要的数据契约

### 当前训练路径

`ProcessedForestDataLoader` 为 train、validation 和 test 三个 Dataset 都传入：

```text
min_train = -15
max_train = -3
```

Dataset 使用：

\[
x_{norm} = \frac{x - (-15)}{-3 - (-15)} = \frac{x+15}{12}
\]

因此：

- `x=-15 dB → 0`
- `x=-9 dB → 0.5`
- `x=-3 dB → 1`

它的设计目标是把训练分布主要映射到约 `[0,1]`。但 Dataset **没有 clamp**：低于 -15 的值会小于 0，高于 -3 的值会大于 1。

### 当前 inference 并不一致

- 通过 `test_loader` 取样的单图分支已经走 Dataset normalization。
- `_load_and_preprocess_image()` 只有在调用者传入 `min_val/max_val` 时才 normalize，并且该 helper 会额外 clamp 到 `[0,1]`。
- 五期 temporal 调用没有传入 min/max，因此模型可能接收未归一化数据。
- `generate_large_change_map()` 自己重复实现 TIFF→tensor，并直接把原始数值送入模型，完全绕过 Dataset normalization。

这叫 **train–serving skew**：训练和推理使用不同的数据转换或数值分布。

```text
Training:  processed dB → fixed [-15,-3] scaling → model
Inference: processed dB → sometimes scaling, sometimes none → model
```

### 为什么严重

Autoencoder 的 anomaly score 就是输入与重建的差。如果模型只在约 `[0,1]` 上训练，却在推理时接收约 `[-15,-3]` 的值，重建误差可能主要反映数值尺度错配，而不是森林变化。后续 GMM 仍能把错误分成两个 cluster，但“可聚类”不等于“检测语义正确”。这是典型的静默正确性故障：pipeline 可以跑完、输出图也看似合理，却不再测量原问题。

这也是为什么数据 contract 与模型 artifact 必须绑定。完整工程审查见 [06-engineering-review.md 的 P0 A](./06-engineering-review.md#2-p0-correctness-risks)。

## 6. 正确的现代设计：一个 transformation source of truth

下面是 conceptual design，不是当前源码：

```python
@dataclass(frozen=True)
class SARTransformConfig:
    channels: tuple[str, str] = ("VV", "VH")
    min_db: float = -15.0
    max_db: float = -3.0
    clip: bool = True
    version: str = "sar-transform-v2"

class SARTransform:
    def validate_channels(self, image): ...
    def normalize(self, image): ...
    def to_tensor(self, image): ...
    def __call__(self, image): ...
```

调用关系应当是：

```text
Dataset ───────┐
Batch inference├──→ the same SARTransform(config)
Evaluation ────┘
```

需要保存和绑定：

| Metadata | Why |
|---|---|
| normalization min/max 与是否 clamp | 决定模型实际输入分布 |
| channel order | 防止 VV/VH 交换后仍能运行但语义改变 |
| preprocessing version | 区分 RTC/filter/融合/转换实现 |
| model version / checkpoint hash | 确定使用哪组权重 |
| dataset manifest version | 确定训练和评价样本集合 |
| code Git commit | 能定位实现版本 |

加载 checkpoint 时应验证 transform metadata，而不是依赖操作者记住 `-15/-3`。最重要的自动化保障是 train–inference consistency test，见 [06-engineering-review.md 的 Tests](./06-engineering-review.md#9-tests)。

## 7. DataLoader：真实配置

`ProcessedForestDataLoader` 创建三个 PyTorch DataLoader：

| Split | Shuffle | Dataset normalization |
|---|---:|---|
| Train | `True` | train min/max `[-15,-3]` |
| Validation | `False` | 同一 train min/max |
| Test | `False` | 同一 train min/max |

设备相关参数：

| Runtime | `num_workers` | `pin_memory` |
|---|---:|---:|
| `args.cuda=True` | `1` | `True` |
| CPU | DataLoader 默认 `0` | 默认 `False` |

准确理解 `num_workers=1`：

- 它**不等于 multiple workers**。
- DataLoader 有 **一个 worker subprocess** 负责调用 Dataset、构造 batch。
- 同时仍有 main training process 消费 batch、执行 device transfer 和模型计算。
- 所以可以说“单 worker 后台加载”，不能说“多个 worker 并行预取”。

还有一个工程细节：`train.py` 无论最终选择 AE 还是 VAE，都会先构造二者；两个构造函数都会创建自己的 DataLoader，随后 `train.py` 又创建第三个 wrapper。worker 只有开始迭代时才启动，但目录扫描、对象初始化和模型显存占用已经发生。这是可以减少的重复初始化，详见 [06-engineering-review.md 的 Resource Management](./06-engineering-review.md#12-resource-management)。

## 8. PyTorch DataLoader Internals

面试只需掌握这条简化链路：

```text
main process creates index requests
        ↓
worker process(es) call Dataset.__getitem__ and collate a batch
        ↓
prefetch queue holds ready batches
        ↓
main process receives next batch
        ↓
optional pin-memory handoff
        ↓
data.to(cuda) and model compute
```

### 不同 worker 数量

| Setting | Behavior | Typical trade-off |
|---|---|---|
| `num_workers=0` | Dataset 读取和 batch 构造都在 main process；没有 worker 预取队列 | 最容易 debug，启动开销小；IO/decode 会直接阻塞训练。 |
| `num_workers=1` | 一个子进程提前准备 batch，main process 可同时做一部分 GPU 计算 | 有后台加载，但没有多个 worker 的并行 decode；本项目 CUDA 路径就是此配置。 |
| `num_workers=4` | 四个子进程并行调用 Dataset，每个维护预取工作 | 可能提高吞吐，也会增加 CPU、内存、文件句柄、进程启动和随机 IO 压力；不保证一定更快。 |

### `prefetch_factor`

它表示**每个 worker 预先准备多少个 batch**，不是多少个 sample。PyTorch 2.5 系列在 `num_workers>0` 且未显式传值时通常按每 worker 2 个 batch 处理，因此本项目可能由框架默认维持约两个待取 batch；但源码没有写 `prefetch_factor`，不能把它包装成显式调优成果。

总预取上限可粗略理解为：

```text
num_workers × prefetch_factor batches
```

它不是越大越好：更多预取占用 host RAM，并可能让远端/机械磁盘的随机 IO 更差。`persistent_workers=True` 则能避免每个 epoch 结束后反复销毁 worker，但本项目没有配置。

## 9. `pin_memory`

普通 pageable host memory 的页面可能被操作系统换出或移动。CUDA 要从它复制到 GPU 时，通常需要先复制到一块 page-locked staging buffer。Pinned/page-locked memory 地址稳定，CUDA DMA 可以直接从该区域进行 host-to-device transfer，因此常能降低 staging 开销，并为异步 copy 创造条件。

本项目真实做到：

```python
DataLoader(..., pin_memory=True)  # CUDA path
```

但训练循环是：

```python
data = data.to(self.device)
```

没有：

```python
data = data.to(self.device, non_blocking=True)
```

因此安全表述是：

> 项目启用了 pinned memory，减少 pageable-memory staging 成本；但没有显式使用 non-blocking device copy，也没有用 profiler 证明 copy 与 compute 完整 overlap，所以我不会声称已经实现端到端异步 H2D 流水化。

Pinned memory 也有成本：占用过多会减少 OS 可分页内存，因此仍应测量后调参。

## 10. 如果 GPU Utilization 很低

### Troubleshooting flow

```text
先确认测量可信
  ↓
分离 batch wait / H2D / forward / backward / optimizer 时间
  ↓
GPU kernel 很短？ → batch/model compute 太小
batch wait 很长？ → Dataset / CPU decode / disk IO
H2D 很长？       → transfer、pinning、batch size、同步
周期性停顿？      → logging、checkpoint、worker restart、GC、filesystem
  ↓
只针对最大瓶颈改一个变量
  ↓
用同一 workload 重新测量 throughput 与 latency
```

### 检查项与证据

| Check | Metric / tool | Interpretation |
|---|---|---|
| GPU 是否真的空闲 | `nvidia-smi dmon`/utilization、显存、功耗 | 低 util 只是症状；采样周期可能漏掉短 kernel。 |
| batch 等待 | 在 iterator next、H2D、compute 周围计时；PyTorch Profiler | `next(loader)` 占比高说明数据供给不足。 |
| CPU decode | per-core CPU utilization、profiler | 一个 worker 满核但 GPU 空，可能需要增加 worker 或更换格式。 |
| disk IO | throughput、IOPS、queue depth、cache hit | 大量小 TIFF 往往受 metadata/random IOPS 限制，而非带宽。 |
| batch size | samples/s、显存、step latency | 增大 batch 可提高 kernel 效率，但会增加显存并影响优化行为。 |
| H2D | profiler 中 memcpy、copy time | 可测试 pinning + `non_blocking=True`，但必须验证是否真有 overlap。 |
| synchronization | `.item()`、频繁日志、显式 sync、checkpoint 时间 | CPU 读取 CUDA 结果会形成同步点。当前每 batch `loss.item()` 会同步取值。 |

### What I did vs What I would do today

**源码能证明当时做过：**离线预切片、CUDA 时 1 worker、`pin_memory=True`，即用存储和后台加载改善训练数据供给。

**当前证据不能证明当时做过：**系统化使用 PyTorch Profiler、记录 batch wait、测试 worker sweep、监控磁盘 IOPS、异步 H2D overlap。

**今天会做：**建立 100–500 step 的固定 profiling workload，记录 data time、compute time、samples/s、GPU util 和 peak memory；依次比较 worker `0/1/2/4`、batch size、persistent workers 和数据格式，每次只改一个因素。

## 11. 如果数据量扩大 10×：32k → 320k

### 当前 many-small-TIFF 设计的压力

- 目录启动扫描和排序时间增长。
- 每个 sample 都有 open/read/close syscall。
- metadata lookup 和随机 IOPS 可能先于顺序带宽成为瓶颈。
- inode/对象数量、备份和同步成本增加。
- 多 worker 同时随机读取可能让机械盘或网络文件系统更慢。
- 没有 manifest 时，很难回答缺失、重复、版本和 split 数量。

### 先测量，再选择格式

| Scale | Reasonable choice | Why / caveat |
|---|---|---|
| Small / local prototype | 保留 GeoTIFF + manifest；SSD；适度 workers | 最简单，保留地理 metadata，易人工检查。32k 在本地 SSD 上未必值得迁移格式。 |
| Medium / single-node training | 将 tile 打包为 WebDataset/tar shards，或 LMDB；也可用 chunked Zarr | 减少小文件 open/metadata 开销。WebDataset 适合顺序 shard streaming；LMDB 适合本地 key-value；Zarr 适合 chunked n-D array 和部分读取。需 benchmark。 |
| Large / distributed or object storage | 版本化 shard/Zarr 放对象存储，本地缓存，manifest 控制 split | 避免数十万独立小对象随机访问；按 worker/node 分 shard。仍需考虑重试、checksum 和缓存一致性。 |

`Parquet` 很适合 manifest、统计字段、路径、标签和 provenance，但通常不是存放大量固定二维 dense raster tensor 的首选容器。可以把像素放在 GeoTIFF/shard/Zarr，把索引与 metadata 放 Parquet，而不是为了“技术高级”强行把所有像素塞进去。

优化顺序建议：

1. 先测 samples/s、data wait、IOPS。
2. SSD 和合理 worker/batch 参数若足够，就保持简单。
3. worker 每 epoch 重启明显时测试 `persistent_workers=True`。
4. 队列断粮时测试 `prefetch_factor`，同时监控 RAM。
5. 小文件 metadata 成为主要瓶颈时再做 shard/chunk 格式迁移。
6. 每次迁移都用 checksum、sample count、tensor range 和固定样本回归验证语义不变。

## 12. Data Integrity、Lineage 与 Idempotency

### 当前行为

`remove_missing_values.py` 和 `remove_zero_values.py` 会对不合格 TIFF 直接执行：

```python
os.remove(file_path)
```

风险包括：

- 处理具有破坏性，没有 rollback。
- 没有记录删除前后样本数和确定的文件清单。
- 没有保存 rejection reason、source、时间或代码版本。
- 读文件异常只 `print`，损坏文件反而可能被保留。
- 后续无法判断样本是“从未生成”“生成后拒绝”还是“误删”。

### 更安全的目录与 manifest

```text
raw/                     # immutable source
intermediate/            # fused products by preprocessing version
processed/               # accepted tiles
rejected/                # quarantined tiles, recoverable
manifests/
    dataset-v003.parquet
```

manifest 至少包含：

| Field | Purpose |
|---|---|
| `source_file` / source IDs | lineage 到 VV/VH 原文件 |
| `output_file` | 物化 tile 的位置 |
| region/date/row/col | 配对、split 和去重 |
| preprocessing_version | 转换语义 |
| accepted / rejection_reason | 审计清理行为 |
| timestamp | 运行时间 |
| size / checksum | 检查损坏、重复与陈旧输出 |
| split | train/validation/test 归属 |

### Idempotency

理想语义：

```text
same immutable input + same validated config + same code version
                         → same output and manifest
```

当前 fusion 只在输出不存在时运行，这有一点重复运行保护；但它按路径存在判断，不验证内容，输入变化后可能复用陈旧 fused 文件。tile 会直接重写，清理脚本又改变 processed 目录。因此整个 pipeline 还不是可审计的幂等作业。

现代方案使用 content/version-derived output path、临时文件写完并 `fsync` 后 atomic rename、成功标记和 manifest commit。失败运行不应让半写文件被下次当成成功产物。更多见 [06-engineering-review.md 的 Idempotency](./06-engineering-review.md#8-idempotency)。

## 13. Split and Leakage

### 当前 split 的准确含义

- train、validation、test 按不同地理区域组织，所以是 **spatial split**。
- 它不是把全部 tile 随机打散后切分的 random split。
- train/validation 为 2018–2024，报告 test 为 2021–2022，年份重叠，所以不是 temporal holdout。
- 当前没有可执行 split manifest，只能从报告确认具体区域和年份。

### 为什么 random tile split 风险大

遥感影像附近像素和相邻 tile 通常相似：植被、土壤、地形、轨道和成像条件都具有空间连续性。这叫 **spatial autocorrelation**。如果从同一大图随机切 tile 到 train 和 test，test 可能只是训练区域的近邻，模型利用地点特征也能得到很好结果，造成 spatial leakage。

按地理区域分开比 random tile split 更严格，但仍不自动等于完全独立：相邻区域仍可能相关，且本项目没有设置 geographic buffer 或报告区间距离。

### 面试安全回答

> 我们按地理区域做 split，避免同一大图的相邻 tile 被随机分到 train 和 test；这能降低 spatial leakage。但训练和测试年份有重叠，所以结果只支持跨区域评估，不支持跨时间泛化结论。若今天重做，我会保存显式 manifest，并根据目标分别设计 spatial holdout、temporal holdout 和 spatial-plus-temporal holdout。

## 14. 本章必须背的知识

### Level A：必须能口述

1. 离线阶段做 VV/VH 融合、空间校验、`256×256` 切片和坏 tile 清理；在线 Dataset 只做读取、layout、normalization 和 tensor conversion。
2. `ProcessedForestDataset.__init__` 扫描并排序文件，`__len__` 返回样本数，`__getitem__` 按需读一个 TIFF。
3. 单样本是 `2×256×256`，batch 是 `B×2×256×256`，PyTorch `Conv2d` 使用 NCHW。
4. 训练、验证、test DataLoader 都使用训练范围 `[-15,-3]`，公式是 `(x+15)/12`。
5. 部分 inference path 没有复用相同 normalization，这是 train–serving skew，会让 anomaly score 反映尺度错误。
6. train `shuffle=True`，validation/test 为 `False`。
7. CUDA 配置是 `num_workers=1, pin_memory=True`；CPU 默认是 `0, False`。
8. `num_workers=1` 是一个 worker subprocess，不是多个 worker。
9. pinned memory 有利于 CUDA DMA，但当前 `.to(device)` 没有 `non_blocking=True`，不能声称完整异步 overlap。
10. 离线切片用存储换重复计算、确定性和可调试性；大量小 TIFF 又会带来 metadata/random-IO 成本。
11. spatial split 不是 random split，也不是 temporal holdout；时间重叠意味着不能声称 temporal generalization。
12. 清理脚本直接删除文件且无 manifest，缺少 lineage、审计和回滚。
13. GPU util 低时先分解 data wait、H2D 和 compute，再决定调 workers、batch 或格式。
14. VV/VH 当前是排序后 zip，不是按显式日期/区域 key join，存在静默错配风险。

### Level B：理解即可

- `prefetch_factor` 是每 worker 预取 batch 数，本项目未显式配置。
- `persistent_workers` 可以减少 epoch 间 worker 重启，但需要额外资源管理。
- WebDataset、LMDB、Zarr 分别适合不同读取模式，迁移前必须 profile。
- manifest 应关联 source、checksum、split、preprocessing version 和 rejection reason。
- atomic write、temporary file 和 rename 如何防止半写输出。
- spatial autocorrelation 为什么会让随机 tile split 过于乐观。

### Level C：不用深挖

- GeoTIFF 内部压缩、tile/block layout 的所有格式细节。
- CUDA DMA、页锁定和 PCIe protocol 的硬件级推导。
- 分布式训练数据服务、流处理平台或湖仓架构。
- SAR terrain correction、speckle filter 和 gamma0/sigma0 的数学推导。
- 各种遥感存储格式的全面 benchmark；面试只需知道要根据访问模式测量后选择。

