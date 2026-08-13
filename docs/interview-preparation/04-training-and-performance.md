# Training and Performance：一次训练到底如何执行

> 本章以当前仓库的 AE 路径为主，VAE 只在解释真实对象生命周期或分支差异时出现。重点是 runtime、状态变化、性能和实验管理，而不是模型论文推导。
>
> 数据读取细节见 [02-data-pipeline.md](./02-data-pipeline.md)；train–serving skew、checkpoint 和复现设计见 [06-engineering-review.md](./06-engineering-review.md)；简历证据边界见 [07-resume-evidence-matrix.md](./07-resume-evidence-matrix.md)。

## 1. Training Entry Point

### 实际入口

入口是 `train.py::main()`，但“执行脚本”不等于“开始训练”：`--train` 默认是 `False`，必须显式传入才会进入训练；`--test` 被定义成 `store_true, default=True`，因此当前 CLI 默认总会尝试测试，而且没有对应的关闭开关。

### 真实调用链

```text
CLI: python train.py [--train] [--use-optuna] [other args]
  ↓
config/parse_args.py
  parse_arguments()
  ↓
train.py::main()
  args.cuda = not args.no_cuda and torch.cuda.is_available()
  torch.manual_seed(args.seed)
  create results directory
  ↓
instantiate VAE(args)
  ├─ VAE._init_dataset() → ProcessedForestDataLoader
  ├─ VAE_Network → device
  ├─ Adam + StepLR + EarlyStopping
  └─ SummaryWriter
  ↓
instantiate AE(args)
  ├─ AE._init_dataset() → ProcessedForestDataLoader
  ├─ AE_Network → device
  ├─ Adam + StepLR + EarlyStopping
  └─ SummaryWriter
  ↓
architectures = {'AE': ae, 'VAE': vae}
autoenc = architectures[args.model]
  ↓
instantiate an additional ProcessedForestDataLoader(args)
  ↓
if args.train:
  pipeline/train/train_pipeline.py::train_model(...)
    ├─ ordinary path: epoch loop
    │    ├─ selected AE/VAE wrapper .train(epoch)
    │    ├─ scheduler.step() inside .train()
    │    ├─ selected wrapper .test(epoch)  # actually validation
    │    ├─ EarlyStopping may write best_model.pth
    │    └─ if not stopping, write {model}_epoch_{epoch}.pth
    └─ Optuna path: study → 10 objective runs → best params text
  ↓
if args.test:                 # currently true by default
  pipeline/test/test_pipeline.py::test_model(...)
    ├─ construct AnomalyDetectionPipeline + another SummaryWriter
    ├─ load results/best_model.pth into selected model
    └─ run three hard-coded anomaly-analysis calls
```

### Class / Function / File map

| Responsibility | Class / function | File |
|---|---|---|
| CLI arguments | `parse_arguments()` | `config/parse_args.py` |
| Process entry | `main()` | `train.py` |
| Dataset/DataLoader wrapper | `ProcessedForestDataset`, `ProcessedForestDataLoader` | `pipeline/datasets/data_loader.py` |
| AE runtime wrapper | `AE` | `pipeline/models/autoencoder.py` |
| Actual AE module | `AE_Network` | `pipeline/models/autoencoder.py` |
| Encoder/decoder layers | `Encoder`, `Decoder` | `pipeline/models/architectures.py` |
| Training orchestration | `train_model()` | `pipeline/train/train_pipeline.py` |
| Early stopping | `EarlyStopping` | `pipeline/utils/early_stop/early_stopping.py` |
| Optuna objective | `objective()` | `pipeline/utils/hyperparameter_optimize/optuna_optimization.py` |
| Weight loading and anomaly calls | `test_model()` | `pipeline/test/test_pipeline.py` |

命名上的注意点：`AE.train(epoch)` 和 `AE.test(epoch)` 是 wrapper 方法，不是对 `nn.Module.train()/eval()` 的覆盖；wrapper 内部再调用 `self.model.train()` 或 `self.model.eval()`。其中 `AE.test()` 实际使用 validation loader，应理解为 validation，而非最终测试集评价。

## 2. Initialization and Object Lifecycle

### Actual behavior

```text
PROCESS START
  ↓
args + device + torch seed
  ↓
VAE wrapper created
  ├─ DataLoader wrapper V1: train/validation/test datasets
  ├─ VAE_Network on GPU/CPU
  ├─ Adam V1, StepLR V1, EarlyStopping V1
  └─ SummaryWriter V1 → results/logs
  ↓
AE wrapper created
  ├─ DataLoader wrapper A1: train/validation/test datasets
  ├─ AE_Network on GPU/CPU
  ├─ Adam A1, StepLR A1, EarlyStopping A1
  └─ SummaryWriter A1 → same results/logs
  ↓
selected wrapper = AE or VAE
  ↓
standalone DataLoader wrapper D1 created
  ├─ not used by selected wrapper's training
  └─ passed to test_model()
  ↓
TRAIN selected wrapper using its own A1/V1 loaders
  ↓
TEST selected model using standalone D1 loaders
  └─ AnomalyDetectionPipeline creates SummaryWriter P1 → same results/logs
```

假设 `--model AE --train`：真正训练的是 `ae.model`，使用 `ae` 自己持有的 train/validation loader、optimizer、scheduler、EarlyStopping 和 writer。`vae` 仍被 `architectures` dictionary 引用，未参与普通 AE 训练，但其模型、optimizer、DataLoader objects 和 writer 已经创建。

DataLoader worker 采用 lazy startup，通常要迭代 loader 时才真正启动 worker subprocess；不过目录扫描、Dataset/DataLoader object、模型显存、optimizer state 容器和 writer 已经初始化。

### 为什么是问题

- 即使只训练 AE，也把 VAE model 放到 device，浪费显存。
- 同一数据目录被多个 Dataset 重复扫描。
- 多个 SummaryWriter 指向同一 log directory，run 边界不清晰。
- 独立 test DataLoader 与训练 wrapper 的 loader 是两套对象，配置未来可能漂移。
- Optuna 再对选中 wrapper 手动调用 `__init__()`，会重复创建模型、loaders 和 writer，旧 writer 没有显式 close。

### Clean design（不是当前实现）

```text
parse validated config
  ↓
build exactly one DataModule
  ↓
model_factory(config.model) builds only selected model
  ↓
Trainer owns optimizer/scheduler/checkpoint/logger
  ↓
train OR test subcommand explicitly selected
  ↓
close resources deterministically
```

核心原则是 ownership 清晰：模型 wrapper 不应偷偷创建数据模块；main/factory 只构造真正使用的对象；writer、worker 和 GPU resource 都有明确生命周期。

## 3. One AE Training Batch

默认 batch size 的真实值是 8，尽管 help text 误写为 128。一个训练 batch 的执行如下。

| Order | Operation | Location | Tensor / state | Device | Mutation / synchronization |
|---:|---|---|---|---|---|
| 1 | Dataset read + normalize + tensorize | DataLoader worker | 每样本 `2×256×256 float32` | CPU | 读取 TIFF；无模型状态变化。 |
| 2 | collate batch | DataLoader | `B×2×256×256` | CPU；CUDA 路径通常为 pinned host batch | batch allocation；train sampler 的 shuffle 顺序消耗 RNG 状态。 |
| 3 | `data.to(self.device)` | `AE.train()` | 同 shape | GPU 或 CPU | CUDA 路径 H2D copy；未传 `non_blocking=True`。 |
| 4 | `optimizer.zero_grad()` | `AE.train()` | parameter `.grad` | GPU/CPU | 清除上一个 batch 累积的 gradients。 |
| 5 | `self.model(data)` | `AE_Network.forward()` | input `B×2×256×256` → reconstruction 同 shape | GPU/CPU | 建立 autograd graph；BatchNorm 更新 running statistics。 |
| 6 | `loss_function(recon, data)` | `AE.loss_function()` | scalar sum-SSE tensor | GPU/CPU | graph 增加 reshape 与 squared-error reduction；参数尚未更新。 |
| 7 | `loss.backward()` | PyTorch autograd | 每个 parameter 的 `.grad` | GPU/CPU | 反向计算并把梯度累积到 `.grad`。 |
| 8 | `train_loss += loss.item()` | Python | Python float | GPU scalar → CPU | CUDA 时通常形成 host synchronization point；每 batch 一次。 |
| 9 | `optimizer.step()` | Adam | model parameters + Adam states | GPU/CPU | 更新参数，并更新 first/second moment state。 |
| 10 | interval logging | 每 `log_interval` batch | 再次读取 `loss.item()` | CPU | log batch 会重复读取同一标量；由于它位于 `optimizer.step()` 之后，仍可能等待当前 stream 的工作，实际成本需 profiler。 |

两个细节容易说错：

1. 当前代码先 `zero_grad()`，再 forward；不是 loss 之后才清梯度。
2. `loss.item()` 发生在 `optimizer.step()` 之前，但它只读取 loss 标量，不改变 graph 或参数。

DataLoader 的 worker、pinning 和预取机制见 [02-data-pipeline.md 的 DataLoader 章节](./02-data-pipeline.md#7-dataloader真实配置)。

## 4. Forward, Backward and Autograd

项目所需的 autograd 心智模型：

```text
forward(data)
  → operations are recorded in a computation graph
  → reconstruction
  → scalar loss
  ↓
loss.backward()
  → traverse graph backward
  → compute gradients into parameter.grad
  ↓
optimizer.step()
  → use parameter.grad and Adam state
  → mutate model parameters
```

### 为什么必须 `zero_grad()`

PyTorch 默认会**累积**梯度，而不是每次 backward 自动覆盖：

```text
batch 1 backward → grad = g1
batch 2 backward → grad = g1 + g2   # if no zero_grad
```

梯度累积有时是有意的，例如用多个 micro-batch 模拟大 batch；但本项目没有这样的 accumulation 设计，因此每 batch 必须先清理旧梯度，否则 optimizer 会使用跨 batch 累积且未归一化的梯度，训练语义改变。

当前使用 `optimizer.zero_grad()`，功能正确。现代 PyTorch 常用 `zero_grad(set_to_none=True)` 以减少清零写入，并让“没有梯度”和“零梯度”更明确；这属于可测量的小优化，不是当前实现。

## 5. Loss：源码叫 MSE，真实是 Batch Sum of Squared Errors

### 真实定义

源码先 reshape：

```python
recon_x = recon_x.view(-1, 2 * 256 * 256)
x = x.view(-1, 2 * 256 * 256)
loss = F.mse_loss(recon_x, x, reduction='sum')
```

对 batch size `B`：

\[
L_{batch}=\sum_{b=1}^{B}\sum_{c=1}^{2}\sum_{h=1}^{256}\sum_{w=1}^{256}
(\hat{x}_{bchw}-x_{bchw})^2
\]

一个样本有：

```text
2 × 256 × 256 = 131,072 elements
```

### Sum vs mean

| Reduction | Meaning | Scale |
|---|---|---|
| `sum`（当前） | batch 内所有样本、通道、像素的 squared error 总和 | 随 batch size、channel 数和分辨率线性变化 |
| `mean` | 上述总和除以 `B×2×256×256` | 每元素平均，更容易跨 batch/shape 比较 |

当前 epoch 汇总：

```python
train_loss += loss.item()
avg_loss = train_loss / len(train_loader.dataset)
```

因此最终量纲是：

\[
L_{epoch}=\frac{1}{N}\sum_{i=1}^{N}
\left[\sum_{c,h,w}(\hat{x}_{ichw}-x_{ichw})^2\right]
\]

即 **average per-sample SSE**，不是 per-element MSE。它大约是标准全元素 MSE 的 `131,072` 倍。

### Batch size 会不会影响 epoch average

要分两层回答：

- **只做固定预测结果的数学汇总时**：每个样本恰好出现一次，最后一批即使较小，所有 batch sum 再除以 N，结果不会因为如何分 batch 而改变。
- **真实训练时**：`loss.backward()` 用的是未除以 B 的 batch sum。batch size 增大时，梯度规模通常随 B 增大；BatchNorm statistics、每 epoch update 次数和 sample grouping 也变化。因此训练轨迹、最终预测和 epoch loss 会变。Adam 的自适应会缓解尺度敏感性，但不保证不变。

可比性风险：

- train/validation 都使用相同 shape 和 per-sample SSE，因此同一实验内曲线可以比较。
- 不应直接与使用 mean reduction、不同 patch size 或不同 channel 数的实验比较绝对 loss。
- 搜索 batch size 时，这种 sum-gradient 还会把“batch size 变化”和“effective gradient scale 变化”耦合起来。

面试安全说法：

> 项目代码把它叫 MSE，但实现是 batch 内平方误差求和，epoch 再除以样本数，所以日志是每样本 SSE。固定输出下 epoch 汇总不受 batch 分组影响，但反向梯度会随 batch size 改变，这是我现在会显式规范 reduction 的原因。

## 6. `loss.item()` and CUDA Synchronization

GPU 上的 `loss` 是 device tensor。Python 需要得到普通标量时：

```text
GPU loss tensor
  ↓ loss.item()
wait until relevant CUDA work completes
  ↓
CPU Python float
```

CUDA kernel 通常是异步提交的；CPU 读取 device 结果时必须等前序计算完成，因此 `.item()` 往往形成 synchronization point。

### 本项目真实频率

- AE 每个 train batch 在累积 epoch loss时调用一次。
- 每 `log_interval` 个 batch，格式化日志又调用一次。
- validation 每 batch 调用一次 `.item()`。
- VAE 每 batch分别对 total/reconstruction/KLD 调用 `.item()`，同步标量更多。

不能因此直接断言 `.item()` 是本项目主要瓶颈；TIFF IO、模型计算、decoder attention 或 batch size 都可能更重要，需要 profiler 证明。

高吞吐系统常见做法：减少逐 batch host reads，在 device 上累积 metric tensor，按较低频率或 epoch 末批量转 CPU；但也要防止意外保留整个 autograd graph。性能优化必须测量前后 step time。

## 7. Optimizer：Adam

AE 初始化：

```python
optim.Adam(
    model.parameters(),
    lr=args.lr,                 # default 1e-4
    weight_decay=args.weight_decay  # default 6e-6
)
```

面试需要知道：

- Adam 维护类似 momentum 的梯度一阶矩，平滑更新方向。
- 它还维护梯度平方的二阶矩，为不同参数提供自适应步长。
- `optimizer.step()` 同时更新 model parameters 和这些 optimizer states。
- `weight_decay` 对参数施加正则化/衰减，抑制权重无限增大；当前使用的是 `torch.optim.Adam` 的实现，不要自动称为 AdamW。

为什么合理：Autoencoder 较深，包含卷积、残差、FPN、attention 和不同尺度参数；Adam 对初始学习率相对宽容、早期收敛快，适合有限时间的 research prototype。它不是被源码证明“优于所有 optimizer”，合理说法是工程上的稳健默认选择。

## 8. Learning Rate Scheduler：StepLR

真实默认值：

```text
initial lr = 1e-4
step_size = 5 epochs
gamma = 0.7
```

规则：每经过 `step_size` 次 scheduler step，学习率乘以 `gamma`。

### 实际调用时机

`self.scheduler.step()` 位于 `AE.train(epoch)` 末尾，因此顺序是：

```text
train epoch N using current lr
  ↓
scheduler.step()
  ↓
validate epoch N
```

它不会改变刚刚完成的 epoch，只为下一个训练 epoch准备 LR。放在 validation 前对当前的 StepLR 数学结果没有实质影响，但日志/状态顺序必须说准确。

### 默认示例

| Training epoch | LR used for that epoch | LR after epoch-end `scheduler.step()` |
|---:|---:|---:|
| 1 | `1.0e-4` | `1.0e-4` |
| 2 | `1.0e-4` | `1.0e-4` |
| 3 | `1.0e-4` | `1.0e-4` |
| 4 | `1.0e-4` | `1.0e-4` |
| 5 | `1.0e-4` | `7.0e-5` |
| 6–10 | `7.0e-5` | epoch 10 后变为 `4.9e-5` |

默认只训练 10 epochs，所以 `4.9e-5` 通常是“为第 11 epoch 准备”，未必真正用于训练。

证据边界：最终报告说 Optuna 帮助选择 scheduler 参数，但当前源码的 Optuna **只搜索 lr 和 weight decay**，不搜索 `step_size` 或 `gamma`。简历安全措辞见 [07-resume-evidence-matrix.md](./07-resume-evidence-matrix.md)。

## 9. Validation

每个 epoch 后，`train_model()` 调用 `autoenc.test(epoch)`；对 AE 来说真实行为是：

```python
self.model.eval()
validation_loss = 0
with torch.no_grad():
    for data in self.validation_loader:
        data = data.to(self.device)
        recon = self.model(data)
        validation_loss += self.loss_function(recon, data).item()
avg_validation_loss = validation_loss / len(validation_dataset)
early_stopping(avg_validation_loss, self.model)
```

### `model.train()` vs `model.eval()`

它们切换 module 的运行模式，不决定是否计算梯度：

- `model.train()`：BatchNorm 使用当前 batch statistics，并更新 running mean/variance。
- `model.eval()`：BatchNorm 使用训练期间积累的 running statistics，不再更新。
- 当前 AE 架构没有生效的 Dropout；`ResidualBlock` 中的 Dropout 只是注释。
- VAE 还有一个额外差异：`self.training=True` 时 latent 会采样，eval 时直接使用 `mu`。

### `torch.no_grad()`

它关闭该作用域内的 gradient recording：

- 不构建反向 graph；
- 减少 activation 保存和显存占用；
- 减少 autograd bookkeeping；
- 不会自动替代 `model.eval()`。

正确 validation 通常两者都要：`eval()` 保证 BN/Dropout 等层的行为正确，`no_grad()` 保证不为反向传播保存状态。本项目两者都正确使用。

## 10. BatchNorm in This Model

BatchNorm 出现在：

- encoder initial convolution；
- ResidualBlock 两个卷积及 projection shortcut；
- encoder self-attention output；
- decoder每个 transposed-convolution block；
- decoder self-attention output。

### 两种模式

```text
TRAIN
  normalize using current mini-batch statistics
  update running_mean / running_var

EVAL
  normalize using stored running_mean / running_var
  do not update them
```

如果推理忘记 `model.eval()`：

- 输出依赖当前推理 batch 的组成和大小；
- batch size 1 时 statistics 可能很不稳定；
- running statistics 还可能被测试数据污染；
- reconstruction error 和 anomaly threshold 随 batch 变化。

本项目的 anomaly inference helper 会调用 `model.eval()`，validation 也调用，行为正确。需要注意的是 BatchNorm 使训练结果对 batch size 更敏感，这与 sum loss 的 gradient-scale 问题是两个不同机制。

## 11. Early Stopping

默认配置：

```text
patience = 5
delta = 0.001
window_size = 5          # EarlyStopping constructor default, CLI 未暴露
path = results/best_model.pth
```

### 真实状态

`EarlyStopping` 保存：

- 最近最多 5 个 raw validation losses；
- 它们的 moving average；
- `best_score = -best_smoothed_loss`；
- 连续未达到 `delta` 改善的 counter；
- `early_stop` boolean。

它判断的是 **smoothed validation loss**，但 checkpoint 保存的是当前 epoch model weights。`save_checkpoint()` 的打印把负的 `best_score` 和正的 raw `val_loss` 放在一起，文案容易误解，不能据此还原真实 previous loss。

### 简化示例

为突出规则，下表直接列已经计算出的 smoothed loss：

| Epoch | Smoothed validation loss | Action |
|---:|---:|---|
| 1 | 0.1000 | first observation → save `best_model.pth` |
| 2 | 0.0900 | improvement ≥ 0.001 → overwrite best, counter=0 |
| 3 | 0.0895 | improvement only 0.0005 → counter=1 |
| 4 | 0.0896 | counter=2 |
| 5 | 0.0897 | counter=3 |
| 6 | 0.0898 | counter=4 |
| 7 | 0.0899 | counter=5 → `early_stop=True` |

真实 moving average 有滞后效应：一个 raw validation loss 变差时，若更早的高 loss 刚好滑出窗口，smoothed score 仍可能改善并保存当前模型。因此 `best_model.pth` 更准确地说是“moving-average early-stopping criterion 认为改善时的模型”，不一定是 raw validation loss 全局最小的 epoch。

### 能否恢复 EarlyStopping state

不能。当前只保存 model `state_dict`，不保存 `best_score`、counter、最近 loss window 或 `early_stop`。中断后重新创建对象会把 patience 历史清零。

## 12. Checkpoint Lifecycle

### 当前会产生什么

| Artifact | Trigger | Content |
|---|---|---|
| `best_model.pth` | validation 后 moving-average criterion 改善 | selected model 的 `state_dict()` |
| `AE_epoch_N.pth` / `VAE_epoch_N.pth` | validation 完成且该 epoch 没有触发 early stop | selected model 的 `state_dict()` |
| `best_hyperparameters.txt` | Optuna study 完成 | best trial 的 params dictionary，仅文本 |

普通训练的实际顺序：

```text
train epoch
→ scheduler.step
→ validation
→ EarlyStopping may overwrite best_model.pth
→ if early_stop: break, do not save epoch_N.pth
→ else save epoch_N.pth
```

`test_model()` 始终加载 `best_model.pth`，不是最后一个 epoch weight。

### 为什么 weights 足够推理、不足以恢复训练

推理只需要已知 architecture/config 下的参数和 buffers，`model.state_dict()` 包含 weights 以及 BatchNorm running statistics，因此可以重建 forward 行为。

训练恢复还需要：

- Adam first/second moment state；
- StepLR 当前 epoch/state；
- 当前 epoch/global step；
- EarlyStopping best score、counter 和 window；
- RNG states；
- resolved config 与 dataset version。

缺少这些时，只能从已有权重启动一个新的 optimizer/scheduler 过程，不能从崩溃点无损继续。完整 recovery checkpoint 设计见 [06-engineering-review.md 的 Checkpoint Design](./06-engineering-review.md#5-checkpoint-design)。

## 13. TensorBoard

### AE 真正记录的字段

| Tag | When | Value |
|---|---|---|
| `Loss/train` | 每个 train epoch 结束 | per-sample SSE average |
| `Loss/validation` | 每次 validation 结束 | per-sample SSE average |

AE 没有记录 learning rate、batch latency、GPU memory、throughput、images 或 model graph。

VAE 路径另外记录：

- `Loss/train`
- `Learning Rate`
- `Loss/validation/total`
- `Loss/validation/recon`
- `Loss/validation/kld`

不要把 VAE 字段说成 AE 实验实际记录字段。

### Writer lifecycle

AE constructor 创建 `SummaryWriter(results/logs)`；train 和 validation 各 `add_scalar()`，随后 `flush()`。但仓库没有调用 `writer.close()`，也没有 context manager。VAE 没有显式 flush/close；AnomalyDetectionPipeline 还创建另一个同目录 writer，但当前类中没有对应 scalar writes。

进程正常退出时底层资源通常会由析构/进程清理，但这不等于正确 lifecycle 管理。尤其 Optuna 对 wrapper 重跑 `__init__()` 时，旧 writer 没有明确关闭。trial 和多个 model object 共用日志目录的问题见 [06-engineering-review.md 的 Experiment Management](./06-engineering-review.md#13-experiment-management)。

## 14. Optuna Outer Loop

Optuna 不在单个 training batch 内，它位于完整训练 run 的外层：

```text
train_model(use_optuna=True)
  ↓
optuna.create_study(direction='minimize')
  ↓
study.optimize(objective, n_trials=10)

Trial 1: choose lr/wd → reinitialize wrapper → train/validate epochs → return final val
Trial 2: choose lr/wd → reinitialize wrapper → train/validate epochs → return final val
...
Trial 10
  ↓
study.best_params / study.best_value
  ↓
write best_hyperparameters.txt
```

### 真实 search space

| Parameter | Search |
|---|---|
| `lr` | `1e-4` 到 `5e-4`，step `1e-4` |
| `weight_decay` | `1e-6` 到 `1e-5`，step `1e-6` |
| trials | 10 |
| direction | minimize |

没有搜索 batch size、architecture、embedding size、StepLR `step_size` 或 `gamma`。

### Objective and EarlyStopping

每个 trial：

1. 直接修改共享 `args.lr/weight_decay`。
2. 取出已经存在的 selected wrapper，并手动调用 `autoenc.__init__(args)` 重新初始化。
3. 运行 train → validation，EarlyStopping 可以提前结束该 trial。
4. 返回循环停止时的 `val_loss`。

对 AE，返回的是**最后一次执行 validation 的 raw per-sample SSE average**；它不是 EarlyStopping 的 best smoothed score，也不是从 `best_model.pth` 重新评价得到的 best-checkpoint metric。

VAE 还有两个额外不一致：训练将 `beta` 硬编码为 `10,000,000`，validation 调用 loss 时使用默认 `beta=1.0`；同时 EarlyStopping 监控 validation total loss，但 `VAE.test()` 返回 reconstruction loss，因此 Optuna objective 优化 reconstruction loss。这不影响以 AE 为主的最终项目解释，但说明 VAE 的 train/validation objective 与通用 metric contract 都不统一。

当前没有 `trial.report()`、pruner 或 intermediate metric persistence。

## 15. Optuna State and Artifact Isolation

所有 trial 共用：

- `results/best_model.pth`
- `results/logs`
- mutable `args`
- 同一个 wrapper Python object

后果：

- 后续 trial 可覆盖先前 trial checkpoint；
- `best_model.pth` 不保证对应 `study.best_trial`；
- TensorBoard 曲线混合；
- study 没有 storage，进程中断不能可靠 resume；
- study 完成后只写 best params，没有用 best params 启动一次隔离的 final training，也没有显式注册 best trial artifact。

这不否定“集成了 Optuna”，但意味着它是基础 research orchestration，而非可靠 experiment manager。隔离目录设计见 [06-engineering-review.md 的 Optuna/Experiment Management](./06-engineering-review.md#13-experiment-management)。

## 16. GPU Memory APIs

Optuna objective 中真实调用：

```python
autoenc.__init__(args)
torch.cuda.empty_cache()
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.9)
```

### `torch.cuda.empty_cache()`

它把 PyTorch caching allocator 中**未被活跃 tensor 引用**的缓存块释放给 CUDA allocator/其他进程可见。它不会：

- 释放仍由 model、optimizer、activation 或 Python reference 持有的 tensor；
- 减少本次 forward/backward 真正所需的峰值显存；
- 自动修复 memory leak 或过大 batch。

当前调用发生在 wrapper 重新初始化之后；已经创建的新 model、optimizer 以及仍被 `architectures` 引用的另一个 model 都是 active objects，`empty_cache()` 无法释放它们。

### `set_per_process_memory_fraction(0.9)`

它限制当前进程的 PyTorch CUDA allocator 最多使用设备显存的一定比例，避免进程占满全卡；它不会压缩模型或降低 activation 需求。限制低于真实需求只会更早 OOM。

而且当前顺序是在 model 已经放到 GPU 后才设置 fraction，无法约束此前的分配。安全说法是“项目尝试限制 trial 显存占用”，不能说它解决了 OOM。

## 17. CUDA OOM Troubleshooting

### 项目级排查顺序

```text
capture exact failing operation and memory stats
  ↓
is OOM during model initialization, forward, backward, or optimizer.step?
  ↓
inspect active model count and unreleased references
  ↓
measure peak allocated/reserved memory
  ↓
separate parameters / optimizer states / activations / attention matrices
  ↓
change one factor and rerun fixed workload
```

检查项：

1. **重复对象**：AE 和 VAE 是否同时驻留 GPU；Optuna reinitialization 是否保留资源。
2. **batch size**：默认 8；activation memory 通常随 B 近似增长。
3. **decoder attention**：`32×32` feature 产生每样本 `1024×1024` attention matrix，反向还需相关 activation；比 encoder `4×4 → 16×16` attention 更占内存。
4. **model + optimizer**：Adam 除参数和梯度外还维护两个 moment tensors。
5. **unreleased graph**：是否把带 graph 的 loss/output 存入列表；当前 AE 用 `.item()` 累积，没有保留整张 graph，这一点合理。
6. **reserved vs allocated**：区分 allocator cache 和真正活跃 tensor，避免把 `empty_cache()` 当万能方案。

### Current vs modern improvement

**项目实际做过：**设置 batch size CLI、`empty_cache()`、90% allocator fraction，并捕获/重新抛出部分 CUDA RuntimeError。

**现代可选方案：**

- smaller batch；
- gradient accumulation 保持近似 effective batch；
- mixed precision/AMP；
- activation checkpointing，用额外计算换显存；
- 只构造选中模型；
- 必要时降低 decoder attention resolution 或优化 attention implementation。

这些改进必须分别验证数值稳定性和吞吐，不能说当年已经实现。

## 18. GPU Utilization：训练 Step Timeline

```text
time ─────────────────────────────────────────────────────────→

CPU/worker: [TIFF read + normalize + collate] [next batch prep ...]
main CPU:                [get batch][H2D submit]       [.item][step]
GPU:                                [forward][backward]      [Adam]
idle gap:      ^data not ready^     ^copy/launch gap^        ^host sync^
```

可能的 idle gap：

- 一个 worker 来不及读取大量小 TIFF；
- H2D copy 没有与 compute overlap；
- batch size 太小，kernel 很短；
- `.item()`、日志或 checkpoint 触发同步/阻塞；
- CPU/GPU 都等待磁盘 metadata/random IO；
- epoch 边界 worker lifecycle 或 checkpoint 写盘。

策略仍是 profiling first：用 PyTorch Profiler 和分段计时确认 `next(loader)`、H2D、forward、backward、optimizer 各占多少，再选择调整 worker、batch、存储格式、AMP 或同步频率。DataLoader 层的完整排查见 [02-data-pipeline.md 的 GPU utilization 章节](./02-data-pipeline.md#10-如果-gpu-utilization-很低)。

## 19. Performance Metrics

如果今天优化，应为固定 workload 记录：

| Metric | Answers |
|---|---|
| samples/sec | 端到端训练吞吐是否提高？ |
| batch latency p50/p95 | 是否存在周期性长尾停顿？ |
| data wait time | main process 等 DataLoader 多久？ |
| TIFF decode time | CPU 解码是否瓶颈？ |
| H2D time | host-to-device copy 是否显著？ |
| forward time | model inference kernel 成本 |
| backward time | autograd 和 gradient kernel 成本 |
| optimizer time | Adam 更新成本 |
| GPU utilization | GPU 是否持续工作；需结合 timeline 解读 |
| GPU allocated/reserved/peak memory | 活跃 tensor 与 allocator cache 情况 |
| CPU utilization / disk IOPS | worker 与小文件 IO 是否受限 |

当前项目实际记录的是 loss 曲线和控制台进度，没有系统保存这些性能指标。因此面试应说“今天会测”，不能说“当年已经完整监控”。

## 20. Training Correctness Review

| VERIFIED behavior | Potential problem | Impact |
|---|---|---|
| `python train.py` 中 `--train` 默认 false | 直接运行不会训练，却因 test 默认 true 尝试加载 checkpoint | CLI 行为不直观；无 checkpoint 时直接退出。 |
| 同时实例化 AE、VAE 和额外 DataLoader | 未选模型仍占设备/资源；数据重复扫描 | 显存、初始化时间、writer 和 ownership 问题。 |
| train epoch 开头调用 `model.train()` | 正确启用 BN training behavior | VERIFIED correct。 |
| 每 batch `zero_grad → forward → sum loss → backward → step` | sum reduction 未按 B/元素归一化 | batch size 改变 gradient scale，跨配置可比性较弱。 |
| `scheduler.step()` 在 train epoch 末尾 | 它发生在 validation 之前；默认 epoch 10 后的 LR 可能从未使用 | 对 StepLR 当前训练数学结果基本正确，但状态/日志顺序容易误述。 |
| validation 使用 `eval()` + `no_grad()` | 正确；方法名 `test()` 容易和最终 test 混淆 | 代码可读性问题，不是 validation 计算错误。 |
| EarlyStopping 使用 5-epoch moving average | best criterion 不是最低 raw validation loss；打印符号混乱 | `best_model.pth` 的语义容易被误解。 |
| EarlyStopping 先运行，随后保存 epoch weight | 触发 early stop 的 epoch 不保存 `epoch_N.pth` | best checkpoint仍在，但没有完整最后状态。 |
| checkpoint 只保存 model state | optimizer/scheduler/early-stop/RNG 不可恢复 | 机器中断只能载入权重，不能无损续训。 |
| TensorBoard AE 写 train/validation loss | writer 不显式 close；多对象共用 log dir | event lifecycle 和 run isolation 不可靠。 |
| Optuna 10 trials 搜 lr/wd | objective 是最后一次 val，不是 best checkpoint metric | best trial 的评价语义与 EarlyStopping best 不一致。 |
| Optuna 共用 `best_model.pth` | 后续 trial 覆盖，且完成后不 retrain/register best | best params 与 best artifact 无可靠对应。 |
| VAE train loss 使用 `beta=10,000,000`，validation 使用默认 `beta=1.0` | 训练与验证 total objective 尺度和权重不一致 | VAE early-stopping 曲线不能直接代表训练目标；主项目 AE 路径不受此项影响。 |
| VAE EarlyStopping 监控 total loss | VAE `test()` 返回 recon loss 给 Optuna | VAE trial optimization metric 与 stop metric 不一致。 |
| 普通训练后 test 默认执行 | 加载 best model 并运行硬编码日期的三条 anomaly path | training 与 application-specific testing 耦合，且可能遇到 normalization skew。 |

## 21. Simplified State Machine

用户常见的理想顺序是 train → validate → checkpoint → schedule，但当前源码真实顺序略有不同：

```text
START
  ↓
PARSING_CONFIG
  ↓
INITIALIZING_ALL_OBJECTS
  ↓
TRAIN_REQUESTED? ── no ──→ TEST_REQUESTED (currently yes by default)
  │ yes
  ↓
TRAINING_EPOCH
  ├─ model.train()
  ├─ batches: load → H2D → zero_grad → forward → backward → Adam step
  └─ TensorBoard Loss/train
  ↓
SCHEDULING
  └─ StepLR.step() prepares next epoch LR
  ↓
VALIDATING
  ├─ model.eval()
  ├─ no_grad()
  └─ TensorBoard Loss/validation
  ↓
EARLY_STOPPING / BEST_CHECKPOINTING
  ├─ update moving-average state
  └─ maybe overwrite best_model.pth
  ↓
EARLY_STOP?
  ├─ yes → TRAINING_FINISHED
  └─ no  → SAVE_EPOCH_WEIGHT → NEXT_EPOCH
  ↓
TEST_REQUESTED
  ├─ load best_model.pth
  └─ anomaly inference calls
  ↓
END
```

从软件工程角度，这是一个有状态工作流：model parameters、BN buffers、Adam moments、scheduler epoch、EarlyStopping window、RNG、checkpoint 和 log 都属于状态。当前只有 model/BN state 被持久化，所以进程崩溃后的恢复边界很有限。

## 22. 面试知识分级

### Level A：必须能口述

1. `train.py` 解析 CLI、选择 device，但实际会先同时实例化 AE、VAE 和额外 DataLoader。
2. 普通 AE 训练使用 AE wrapper 自己持有的 DataLoader、optimizer、scheduler、EarlyStopping 和 writer。
3. 单 batch 顺序是 H2D → `zero_grad` → forward → sum loss → backward → `loss.item()` → Adam step。
4. forward 建 graph，backward 写入/累积 gradients，optimizer.step 更新参数。
5. 必须 zero_grad，因为 PyTorch 默认累积 `.grad`。
6. 当前 loss 是 batch sum SSE；epoch 指标是 average per-sample SSE，不是 per-element mean MSE。
7. batch size 改变会影响 sum-loss 的 gradient scale、update 次数和 BatchNorm statistics。
8. Adam 使用一阶/二阶 moment 做自适应更新；默认 lr `1e-4`、weight decay `6e-6`。
9. StepLR 默认每 5 epoch 乘 0.7，并在 train epoch 结束、validation 前调用。
10. validation 同时需要 `model.eval()` 和 `torch.no_grad()`；前者控制 BN，后者关闭 graph。
11. EarlyStopping 监控 5-epoch moving average，patience 5、delta 0.001，并写 `best_model.pth`。
12. state_dict 足够推理，但不能恢复 Adam、scheduler、epoch、EarlyStopping 和 RNG。
13. AE TensorBoard 只记录 `Loss/train` 与 `Loss/validation`。
14. Optuna 是外层 10 次完整训练，真实只搜索 lr 和 weight decay。
15. `empty_cache()` 不能释放 active tensors，也不是 OOM 根治方案。

### Level B：ML/PyTorch 追问需要

- BatchNorm running statistics 与 batch size 的关系。
- `.item()` 为什么可能触发 CUDA synchronization。
- StepLR 的 epoch timing 和 state。
- moving-average EarlyStopping 为什么不等于 raw minimum。
- Adam optimizer state 为什么必须进入 recovery checkpoint。
- decoder `32×32` attention 的内存压力。
- Optuna objective、artifact isolation 和 persistent study。
- AMP、gradient accumulation、activation checkpointing 的用途和 trade-off。

### Level C：不用深挖

- Adam bias correction 的完整公式推导。
- CUDA stream、DMA 和 allocator 的底层实现细节。
- Optuna sampler 的贝叶斯优化理论。
- 分布式数据并行、ZeRO/FSDP 等大型训练框架。
- BatchNorm 论文历史或统计证明。
- Foundation-model 训练系统；本项目是单机 research prototype。

## 23. Interview Answers（45–90 秒）

### 1. “一次训练是怎么跑起来的？”

> 入口是 `train.py`。它解析 batch size、epoch、lr 等参数，判断 CUDA，并设置 PyTorch seed。当前实现会先构造 AE、VAE 和额外 DataLoader，再根据 `--model` 选择真正训练的 wrapper。普通训练每个 epoch 先遍历自己的 train loader：batch 从 CPU 搬到 device，清梯度、forward 得到重建、用平方误差求和、backward，再由 Adam 更新参数；epoch 末 StepLR 更新下一轮学习率。随后用 `eval()+no_grad()` 跑 validation，EarlyStopping 根据平滑 validation loss 保存 `best_model.pth`，未停止时再保存 epoch weights。训练后因为 test 默认开启，还会加载 best weights 进入 anomaly inference。

### 2. “Dataset/DataLoader 和训练循环是什么关系？”

> Dataset 定义一个 index 如何变成 `2×256×256` tensor，包括读 TIFF、CHW 转换和 `[-15,-3]` normalization；DataLoader 决定 sample 顺序、batch、shuffle 和 worker。训练循环只消费 `B×2×256×256` batch，不应该关心文件格式。这个项目 train loader shuffle，validation/test 不 shuffle；CUDA 路径是一个 worker 加 pinned memory。训练 wrapper 自己持有 loader，但 `train.py` 又建了一套给 test，这种重复 ownership 是我现在会清理的地方。完整数据路径在 `02-data-pipeline.md`。

### 3. “为什么用 Adam？”

> 这个 AE 同时包含卷积、残差、FPN 和 attention，不同参数的梯度尺度可能差异较大。Adam 用梯度一阶矩平滑方向、用二阶矩自适应每个参数的步长，通常是 research prototype 中较稳健、收敛较快的默认选择。源码默认 lr 是 `1e-4`、weight decay 是 `6e-6`，Optuna也只搜索这两个量。我不会声称 Adam 理论上最好；更准确的是它降低了早期 optimizer 调参成本，并且需要保存 moment state 才能真正恢复训练。

### 4. “model.train() 和 model.eval() 有什么区别？”

> 它们切换层的运行模式。本项目大量使用 BatchNorm：train 模式用当前 batch statistics 并更新 running mean/variance，eval 模式使用已积累的 running statistics。当前没有生效的 Dropout；VAE 还会在 train 时采样 latent、eval 时直接用均值。如果推理忘记 eval，重建结果会依赖当前 batch，batch size 1 时尤其不稳定。源码在每个训练 epoch 和 validation/anomaly inference 前都正确切换了模式。

### 5. “torch.no_grad() 有什么作用？”

> validation 只需要 forward 和 loss，不需要更新参数。`torch.no_grad()` 不记录 autograd graph，因此减少 activation 保存、显存和 bookkeeping。它和 `model.eval()` 不是一回事：no_grad 控制是否建梯度图，eval 控制 BatchNorm等层的行为。本项目 validation 同时使用两者，这是正确的；如果只有 no_grad 但忘记 eval，BatchNorm 仍会用测试 batch statistics。

### 6. “为什么需要 zero_grad()？”

> PyTorch 默认把每次 backward 的结果累积到 parameter `.grad`。如果本项目不在每 batch 前 zero_grad，第二批的更新会使用第一批加第二批的梯度，而且代码没有按 gradient accumulation 的方式做归一化或控制 step。当前顺序是 batch 搬到 device 后先 `optimizer.zero_grad()`，再 forward、backward 和 step，所以每次更新只对应当前 batch。若今天有意做 micro-batch accumulation，我会明确按若干批才 step，并相应缩放 loss。

### 7. “EarlyStopping 是怎么工作的？”

> AE 每个 epoch validation 后把 per-sample SSE 交给 EarlyStopping。它保留最近 5 个 validation loss，比较 moving average；默认至少改善 0.001 才算进步，否则 counter 加一，连续 5 次不改善就停止。首次或改善时覆盖 `best_model.pth`。所以这个 best 更准确地说是平滑指标认为改善时的当前模型，不一定是 raw validation loss 最低的 epoch。并且 state 没进 checkpoint，中断后 counter 和窗口无法恢复。

### 8. “为什么 checkpoint 不能只保存 model weights？”

> `state_dict` 包含模型参数和 BatchNorm buffers，所以知道 architecture 和 preprocessing 时足够推理。但恢复训练还需要 Adam 的一阶/二阶 moment、StepLR 当前状态、epoch、EarlyStopping counter/window 和 RNG。当前项目只保存 weights，因此机器挂掉后可以加载一个模型重新启动 optimizer，却不能无损地从原 training step 继续。这也是 `06-engineering-review.md` 中把 inference artifact 和 recovery checkpoint 分开的原因。

### 9. “Optuna 是怎么集成到项目里的？”

> `train_model` 在 `--use-optuna` 时创建 minimize study，运行 10 个 trial。每个 trial 选择 `1e-4` 到 `5e-4` 的 lr，以及 `1e-6` 到 `1e-5` 的 weight decay，然后重新初始化 selected wrapper，执行完整 train/validation epoch loop，最后返回停止时的 validation loss。它是训练 run 外层的 orchestration，不参与单 batch 更新。局限是 trial 共用 checkpoint 和 TensorBoard 目录，objective 也不是 best-checkpoint metric，所以今天我会隔离每个 trial 的 artifact并持久化 study。

### 10. “GPU OOM 怎么定位？”

> 我先确认 OOM 发生在初始化、forward、backward还是 Adam step，并查看 allocated、reserved 和 peak memory。这个项目首先要排除同时驻留的 AE/VAE 和 Optuna 重复初始化；然后检查 batch size、Adam state 和 activation。模型中 decoder 的 `32×32` attention 每样本会产生 `1024×1024` attention matrix，是明确的 activation hotspot。确认后再依次测试减 batch、gradient accumulation、AMP 或 activation checkpointing。`empty_cache()` 只能释放未引用缓存，不能释放 active tensors，所以不是根治方案。

### 11. “GPU utilization 低怎么办？”

> 我会把 step 分成 DataLoader wait、H2D、forward、backward、Adam 和 host synchronization，用 profiler和固定 workload测量。若 next batch 等待长，查单 worker、TIFF decode 和随机 IO；若 kernel 太短，测试 batch size；若 copy 长，再验证 pinned memory配合 non-blocking 是否有效；还要检查每 batch `loss.item()`、日志和 checkpoint 停顿。项目当时已有离线切片、一个 worker 和 pin memory，但没有完整 profiling 证据，所以我会把这些称为已有性能意识，而不是声称已经找到所有瓶颈。

### 12. “如果训练中途机器挂了，当前项目能恢复到什么程度？”

> 当前可以加载最近保存的 epoch weight，或者 `best_model.pth` 做推理，也可以把这些权重当初始化重新训练。但它不能精确恢复到崩溃前一步，因为 checkpoint 没有 optimizer、scheduler、epoch、EarlyStopping 或 RNG state；如果中断发生在 epoch checkpoint 之前，当前 epoch 的进度也会丢失。Optuna study 没有 persistent storage，trial 状态也不能可靠续跑。因此恢复能力是“恢复模型参数”，不是“恢复完整训练作业”。
