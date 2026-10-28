# 大作业本地评测

入口是 `python -m grdh_eval`。新增模块位于 `grdh_eval/`，原 `scripts/txt2img.py`、模型源码与映射模块均未修改。

当前保留 `doctor`（环境检查）、`score`（评分）、`evaluate`（512×512 完整评测）、`redecode`（利用缓存重新判决）四个命令。GPU 多候选实验使用 `scripts/run_gpu_study.py`，逐图结果与筛选记录保存在 `results/optimization/`。历史 CPU 批量入口 `scripts/run_cpu_study.py` 仍保留。

本地实现严格使用 PPT 第 2 页评分公式。PPT 未指定噪声归一化范围、JPEG 色度抽样、NIQE 实现及提取端能否取得原 prompt，因此当前结果是**带明确协议的本地实验分数**，不声称与尚未提供的助教脚本逐项相同。

## 1. 当前环境与课程环境

本地已创建 `.venv-py38`：Ubuntu 26.04（WSL2）、Python 3.8.20、PyTorch 2.1.0+cu118、torchvision 0.16.0+cu118。本机 GPU 为 RTX 4070 SUPER；课程机器为 RTX 3090。课件未注明课程机器的 Ubuntu 具体版本、是否使用 WSL、CUDA 构建版本或其他依赖版本，因此不能宣称所有环境细节完全一致。

在项目根目录执行：

```bash
.venv-py38/bin/python -m grdh_eval doctor
.venv-py38/bin/python -m unittest discover -s tests -v
```

此前 Python 3.10 的 `.venv` 保留供旧实验复核。后续 Linux 实验应显式使用 `.venv-py38/bin/python`；下面保留的 Windows PowerShell 示例是其他环境的历史用法，不代表当前机器的推荐命令。每次运行的实际版本记录在结果目录的 `environment.json` 中，最终仍需在课程机器复核。

Ubuntu 环境可按以下方式建立：

```bash
python3.8 -m venv .venv-py38
.venv-py38/bin/python -m pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
.venv-py38/bin/python -m pip install -r requirements-eval.txt
```

CPU 环境可把 PyTorch 安装源换为 `https://download.pytorch.org/whl/cpu`。CUDA 安装组合参考 [PyTorch 官方历史版本说明](https://pytorch.org/get-started/previous-versions/)。

## 2. 模型准备

```powershell
.\.venv\Scripts\python.exe scripts/prepare_eval_models.py
```

下载公开 SD 1.5 原始格式 checkpoint，约 4.3 GB，支持 `.part` 断点续传，并验证 Hugging Face 提供的 SHA256。模型固定到仓库 revision，来源与哈希保存在 `models/sd15/sources.json`。

默认文件是推理用 `v1-5-pruned-emaonly.safetensors`。原 README 示例中的 `v1-5-pruned.ckpt` 是另一个 checkpoint 名称；本地基线按实际使用的权重与哈希报告，不假定它与原论文某次运行完全相同。若助教要求指定文件，可用 `--checkpoint` 切换并建立新的结果目录。

### 作者公开代码条件对照（Linux）

作者 README 链接的 `v1-5-pruned.ckpt` 约 7.7 GB，同时包含普通 U-Net 权重和 EMA 影子权重；作者的 `ldm.yaml` 设置 `use_ema: False`，脚本加载普通 U-Net 权重。逐张量比较发现 686 个 U-Net 张量均与上述 EMA-only 文件不同，248 个 VAE 和 197 个 CLIP 张量均相同。因此先前使用 EMA-only 的实验不能视为完全相同的模型权重。

下列脚本从作者公开链接分段读取并保存非 EMA 推理参数，不在磁盘上保存完整的 7.7 GB checkpoint；输出约 4.27 GB，支持中断后续传，逐项检查原文件 ZIP CRC，另记录输出 SHA256 和来源：

```bash
.venv-py38/bin/python scripts/prepare_author_checkpoint.py
```

已有的 EMA-only 文件不会被覆盖。作者条件下的本地评分对照示例：

```bash
.venv-py38/bin/python -m grdh_eval evaluate --device cuda:0 --precision autocast \
  --attention original --checkpoint models/sd15/author-nonema-inference.safetensors \
  --count 5 --bits 1 --repeat 1 --decision hard --prompt-mode known \
  --guidance 5 --inversion-guidance 5 --posterior sample --noise-domain signed \
  --output results/author_weights_baseline_dev5
```

这里的分数仍是本地课程评分：测试提示词、固定随机种子、JPEG 读写、NIQE 与综合评分并非都能确认为论文的完整实验协议。论文全文及课程评测实现核实之前，不应称为论文实验的完全复现。若对比旧版 EMA-only 结果，需明确模型权重已经改变。

模型已包含 U-Net、VAE 和 CLIP 文本权重。因此适配器从本地配置创建相同 CLIP 架构，再装入 checkpoint 中的文本权重，只另外下载 tokenizer 和配置，避免重复下载文本权重。此配置目录供新入口使用；原始 `txt2img.py` 仍按其 README 要求加载完整 CLIP 目录。

```text
models/sd15/
  v1-5-pruned-emaonly.safetensors
  clip/config.json
  clip/vocab.json
  clip/merges.txt
  clip/tokenizer_config.json
  clip/special_tokens_map.json
  sources.json
```

整个流程不训练、不微调、不更新 SD 权重。

## 3. 单独核对评分公式

```powershell
.\.venv\Scripts\python.exe -m grdh_eval score --bpp 0.0625 --accuracies 0.9 0.9 0.9 --niqe 4
```

结果应为约 `74.025129`。三个准确率按 JPEG / 高斯噪声 / 中值滤波顺序输入，取值为 0～1，不是百分数。

聚合顺序是：对每一种攻击分别求测试图片平均 Acc，再逐项计算 `max(0, 2*Acc-1)`；NIQE 先对所有原始载密图像取均值，再代入分段公式。bpp 使用用户原始有效秘密比特数除以宽×高，重复副本与填充不计入有效载荷。

## 4. 跑一张真实图像的完整评测

```powershell
.\.venv\Scripts\python.exe -m grdh_eval evaluate --device cpu --count 1 --output results/first_image
```

默认使用 20 步生成、20 步反演、2 阶 DPM-Solver++、1 bit 映射、VAE 后验均值、空 prompt 反演。运行时会打印 U-Net 调用数和阶段耗时。

`--attention sdpa` 可选用 PyTorch 的注意力实现，数学计算保持一致，并有数值对照测试；默认 `original` 保留原计算路径。速度取决于 CPU/GPU、内存与 PyTorch 版本，应实测选择，不能预先认定 SDPA 更快。该选项也写入实验 manifest。

图像统一为 512×512，潜变量形状为 `[4,64,64]`。每位置 1 bit、不重复时，每张图的原始秘密长度为 16384 bit，有效 bpp 为 0.0625。

默认提取端不接收原 prompt；它只使用接收图像和预先配置的模型、映射密钥、码率及固定消息长度。生成端仍使用所选测试 prompt 控制画面。

为了使用作者公开权重并与原仓库的主要条件作对照，可先按第 2 节提取作者权重，然后单独运行（以下为 Linux 命令）：

```bash
.venv-py38/bin/python -m grdh_eval evaluate --device cuda:0 --precision autocast --attention original --checkpoint models/sd15/author-nonema-inference.safetensors --count 1 --posterior sample --decision hard --prompt-mode known --inversion-guidance 5 --noise-domain signed --output results/original_conditions
```

这是原方法的已知 prompt、采样后验条件对照。随机消息、共享固定密钥及保存方式已显式化，不承诺与原脚本某次未设种子的运行产生同一张图像。不同 prompt 模式、攻击协议的分数分开比较。

## 5. 候选方案与缓存重新解码

`scripts/run_cpu_study.py` 中的候选统一使用 VAE 后验均值和空 prompt 提取：

| 候选 | 每位置比特数 | 重复次数 | 判决 | 白化/交织 | 有效 bpp |
| --- | ---: | ---: | --- | --- | ---: |
| reference | 1 | 1 | 硬 | 关闭 | 0.0625 |
| repeat2 | 1 | 2 | 软 | 开启 | 0.03125 |
| repeat4 | 1 | 4 | 软 | 开启 | 0.015625 |
| multibit2 | 2 | 1 | 软 | 开启 | 0.125 |
| multibit2_repeat2 | 2 | 2 | 软 | 开启 | 0.0625 |

`reference` 保持原映射与硬判决，是上述本地协议下的基准。它使用后验均值和空 prompt，不是原仓库已知 prompt、采样后验条件的成绩。

白化在重复编码后进行，通过固定共享随机掩码打散重复模式；交织把编码位分散到不同位置。接收端先还原位位置和掩码，再合并同一秘密 bit 的证据。软判决保留连续恢复值，硬判决先取整。多比特嵌入使用与提取一致的格雷码标签。

若要只测试软判决收益，可以对同一次运行保存的潜变量重新解码，无需再生成和反演：

```powershell
.\.venv\Scripts\python.exe -m grdh_eval redecode results/cpu_study/dev/repeat2 --decision hard
.\.venv\Scripts\python.exe -m grdh_eval redecode results/cpu_study/dev/repeat2 --decision soft
```

这样比较使用完全相同的图像与攻击结果。比较白化、冗余或采样设置时，可用 `evaluate` 的独立参数做单变量实验；组合候选的收益不可全部归因于其中某一个改动。

只改变后验等提取设置时，`evaluate --reuse-images-from 某次实验目录` 可以复用原载密图像，避免再次生成。程序会验证消息、映射及生成设置一致，编码或冗余改变时拒绝复用。每种攻击仍需按新的提取设置独立反演。

此外已增加可选的卷积码（`--ecc conv12/conv23/conv34`）与 Hamming(15,11) 码（`--ecc hamming15`）。前者使用软 Viterbi 判决，后者枚举合法码字并进行比特后验判决。校验位、卷积码的终止位和填充均不计入有效 bpp；ECC 与重复编码不能同时开启，Hamming 的当前软度量仅用于 1 bit 映射。`--decode-gain` 仅调整解码幅度，不改变生成图像。它们是实验选项，不能据算法名称推断一定提分。

`--receiver-filter auto_nlm` 是另一个实验选项：仅从收到的图像估计噪声，必要时做非局部均值去噪，再执行反演；接收端不会获得攻击标签、原图或原始秘密。默认关闭。其开发集结果未显示稳定收益，不作为推荐设置。

BCH/LDPC、频域子带与新反演求解器尚未实现；扩展选项不会更新 SD 网络权重。

## 6. 三种攻击与 NIQE 的精确定义

- 原始图像先从模型输出转换为 RGB uint8，并保存为 PNG。数值截断方式与 torchvision ToPILImage 一致。
- JPEG：Pillow，quality=70，subsampling=2，保存编码后解码。
- 高斯噪声：均值 0，sigma=0.05。默认 `--noise-domain unit` 在 [0,1] 范围加噪；`signed` 表示在 [-1,1] 标度加同样标准差，等价于 unit 标度的 0.025。加噪后裁剪范围。
- 噪声攻击的提取输入保留 float32，与原 baseline 的内存传输方式一致。`*_awgn_exact.npy` 保存精确输入；旁边的 PNG 只是预览，不作为这次分数的提取输入。
- 中值滤波：OpenCV `medianBlur`，kernel=5。
- 每种攻击独立作用在同一张原始载密图像上，不连续叠加三种攻击。
- NIQE：只评价攻击前、已保存为像素整数的载密图像。使用 BasicSR 核心和官方统计参数，默认 MATLAB 风格 Y 通道、crop_border=0；`--niqe-color gray` 是独立的可选对照。

NIQE 依赖的最小源码和统计参数位于 `grdh_eval/vendor/`，带原项目 Apache-2.0 许可证、固定 revision 和来源哈希。测试图 `tests/fixtures/baboon.png` 的参考 NIQE 为 5.7295763，可用于验证颜色转换和缩放是否一致。

PPT 没有给出上述全部实现细节。等助教说明后，应锁定协议，不能通过挑选较弱攻击定义来提高报告分数。

## 7. 数据划分与结果文件

默认从 `text_prompt_dataset/chatgpt_dataset.txt` 去除空行及重复文本，使用独立随机种子打乱，然后划分：

- dev：前 10 条，用于初筛。
- validation：接下来的 20 条，用于复核。
- final：再后面的 100 条，用于最终本地评测。

三组不重叠。`--count` 可以先运行该组的少量样本。每个实验保存原文件哈希、具体 prompt、行号、随机种子和全部配置。不同方法使用同一划分、可比的秘密消息前缀和攻击随机种子。

```text
results/某次实验/
  manifest.json        锁定模型、prompt、编码、攻击和 NIQE 设置
  environment.json     实际运行版本
  clean/               逐张原始载密图像
  attacked/            三种攻击预览与精确 AWGN 输入
  messages/            评价程序使用的原始随机秘密，不提供给提取器
  latents/             各攻击反演得到的潜变量，供复核与重新判决
  per_image.jsonl      每张完成后立即写入的指标
  per_image.csv        完成后的表格
  summary.json         总分、分项、样本数、状态与耗时
```

中断后可以加 `--resume`。配置、输入与模型文件身份必须和原 manifest 一致。已完成图片跳过，未完成图片可复用其生成图和已反演潜变量。部分结果标记 `partial`；只有 final 分组达到 100 张时 `course_sample_count_met` 才为 true，这仍不表示已经与助教脚本完全对齐。

GPU 最终评测示例：

```bash
.venv-py38/bin/python -m grdh_eval evaluate --device cuda:0 --precision full \
  --split final --bits 1 --repeat 2 --whiten --decision soft \
  --output results/final_candidate
```

正式参数应根据验证集结果选择，不预先认定示例方案最佳。`autocast` 可用于 CUDA 的速度对照，其数值误差也需要单独验证。

## 8. CPU 自动完成筛选、验证和最终评测

```powershell
.\.venv\Scripts\python.exe -u scripts/run_cpu_study.py
```

默认依次执行：5 个候选各 5 条 dev prompt，前两名各 20 条独立 validation prompt，验证集总分最高的方案再测 100 条 final prompt。使用 512×512、空 prompt 提取、固定 PPT 本地协议。整个流程只加载一次模型，不训练权重。

这是长任务，耗时随硬件、内存及供电状态变化。`results/cpu_study/status.json` 记录最近切换的阶段和候选，不是持续更新的心跳；是否仍在运行需结合进程和日志判断。当前候选目录的 `summary.json` 显示完成张数，最终结果写入 `final_result.json`。已经完成的图像不会重跑。

创建 `results/cpu_study/STOP` 文件可在当前图像完成后停止。删除 STOP 后重跑相同命令即可续跑。也可以用 `--until dev` 或 `--until validation` 只运行到相应阶段；这些操作参数不会改变模型与评分协议。

## 9. GPU 实验与节省磁盘

```bash
.venv-py38/bin/python scripts/run_gpu_study.py \
  --plan 某个实验配置.json --output results/某次GPU实验
```

JSON 计划指定 `split`、`count`、`candidates`；每个候选可以覆盖 `model` 和 `codec` 设置。攻击统一固定为第 6 节的 unit 范围协议。所有候选使用相同提示词、秘密随机种子和攻击随机种子。后验采样诊断也固定使用接收端常量种子 0，不依赖提示词行号。

这个入口只加载必要模型，切换权重前释放旧模型；不会保存攻击后的图片或完整噪声数组。设置 `cache_latents: false` 与 `keep_clean: 3`，可以评测全部图片，但仅保留前三张原始载密图像。逐图准确率、NIQE、消息/图像哈希及生成配置全部记录，评分仍覆盖所有图片。

完成的开发实验可以删除 PNG 和 `*_latents.npz` 缓存，保留 JSON/JSONL 指标。若重跑时 `reuse_clean` 的图像缓存已清理，程序在核对编码与生成参数后重新生成图像。开发阶段的清理清单记录于 `results/optimization/cleanup_development.json`。

配对的 100 张最终实验完成后，可重新核对评分与不确定性：

```bash
.venv-py38/bin/python scripts/summarize_gpu_study.py --root results/optimization
```

该脚本检查两组提示词和攻击协议相同、候选参数与验证阶段锁定的配置相同，再对同一组提示词进行配对 bootstrap。区间仅反映这批提示词的抽样波动，不代表已与助教评测协议等价。

## 10. 2026-10-28 优化实验

本轮按 **5 条开发提示词筛选 → 20 条独立提示词选择 → 100 条保留提示词最终对照** 执行。所有图片均为 512×512，使用随机二进制有效秘密；提取端只得到受攻击图像与固定共享配置。采用 JPEG Q=70、unit 范围高斯噪声 σ=0.05、中值滤波 5×5，以及 BasicSR Y 通道 NIQE。没有通过改变攻击强度或评分定义提分。

对照采用作者原代码实际加载的 non-EMA 权重、生成引导强度 5；为适配本地作业接口，两组提取端都使用空提示词和 VAE 后验均值。因而这里比较的是**统一课程协议下的本地对照**，不是直接与论文表格数值比较。独立验证与最终对照均使用 SDPA、autocast、20 步生成/反演和二阶 DPM-Solver++。

### 验证阶段与冻结配置

| 20 条验证提示词上的设置 | 容量分 | 鲁棒性分 | NIQE 分 | 总分 |
| --- | ---: | ---: | ---: | ---: |
| non-EMA，生成引导 5（对照） | 18.0251 | 44.1283 | 17.5556 | 79.7091 |
| non-EMA，生成引导 2 | 18.0251 | 47.0587 | 18.5953 | 83.6791 |
| EMA，引导 1.5 | 18.0251 | 47.0773 | 19.4788 | 84.5813 |
| EMA，引导 1 | 18.0251 | 47.2839 | 19.5759 | 84.8850 |

最终候选在生成任何 final 图片之前固定为 **EMA-only 权重、生成引导强度 1、空提示词反演**。引导 1 表示保留正常文本条件、取消额外的 CFG 放大，并不表示生成时使用空提示词。使用 1 bit 映射、不重复、不加纠错；每图有效秘密为 16,384 bit，即 0.0625 bpp。1 bit、无重复时软/硬判决均为同一个符号判决。

这是无需训练的权重和参数优化。接收端自动降噪、卷积纠错和 Hamming 纠错均没有进入最终候选；原模型架构和作者脚本没有改动。EMA 权重是已有的另一份 SD 1.5 推理文件，不应将更换权重后的方案称为作者原配置。

冻结配置已保存，可直接复现单个候选或两组对照；默认每组评测 100 张，但只保存前三张示例图：

```bash
.venv-py38/bin/python scripts/run_gpu_study.py \
  --plan configs/evaluation/optimized.json --output results/optimized_reproduction

.venv-py38/bin/python scripts/run_gpu_study.py \
  --plan configs/evaluation/comparison.json --output results/comparison_reproduction
```

每个新结果目录的 `summaries.json` 汇总各组成绩，候选子目录保存逐图指标和 manifest。同一输出目录可以直接续跑，已完成的图片会跳过；更改配置时应更换输出目录。

### 最终对照

两组各完成 100 张，使用相同的保留提示词和相同的秘密消息。最终结果如下；验证集分数没有混入这张表：

| 指标 | non-EMA、引导 5 对照 | EMA、引导 1 优化 |
| --- | ---: | ---: |
| 每图有效秘密 | 16,384 bit | 16,384 bit |
| 有效 bpp | 0.0625 | 0.0625 |
| JPEG 准确率 | 94.6490% | 98.5652% |
| 高斯噪声准确率 | 90.3573% | 95.7416% |
| 中值滤波准确率 | 91.7366% | 95.1147% |
| NIQE 均值（低为好） | 3.8793 | 3.2007 |
| 容量分 / 30 | 18.0251 | 18.0251 |
| 鲁棒性分 / 50 | 42.2476 | 46.4738 |
| NIQE 分 / 20 | 16.4828 | 19.1972 |
| **总分 / 100** | **76.7555** | **83.6961** |
| 平均每张计时（不含加载） | 4.641 秒 | 4.260 秒 |

提升 **6.9406 分**，其中鲁棒性贡献 4.2262 分、NIQE 项贡献 2.7144 分，容量不变。100 条提示词的配对 bootstrap（10,000 次）给出提升的 95% 区间 **[5.4811, 8.5683]**。93/100 个配对样本的三攻击平均准确率提高；每图三种攻击平均准确率低于 90% 的情况从 31 张减少至 4 张。最差一张的三攻击平均准确率从 72.49% 提高至 84.31%，两组最差样本不一定是同一张。

本轮共完成 32 个开发设置、4 个验证设置、2 个最终设置，合计 440 次图像评测（含复用载密图像和不同设置在同一提示词上的重复实验）。30 项单元测试全部通过。计时仅对应本机 RTX 4070 SUPER，不能直接代表课程 RTX 3090 的耗时。

完整汇总为 `results/optimization/comparison.json`，全部设置的表格为 `results/optimization/all_results.csv`，最终逐图记录位于 `results/optimization/final/`。这里报告的是本地明确协议下的成绩；助教噪声范围、NIQE 实现及数据集尚未完全核实，仍需在正式评测环境复核。

### 没有采用的方向

以下均为 5 条开发提示词结果，不能与独立验证或最终 100 张成绩直接相减：

| 开发设置 | 总分 | 观察 |
| --- | ---: | --- |
| non-EMA、引导 5、原始 1 bit 对照 | 80.8045 | 开发对照 |
| 2 bit 映射 | 65.7924 | 容量收益抵不过提取准确率下降 |
| 4 bit 映射 | 59.7126 | 同样明显损失鲁棒性 |
| 1 bit、两次重复、软合并 | 79.2794 | 准确率提高，但有效容量降低 |
| 卷积码 3/4 | 69.9382 | 困难图片上可能出现大范围解码错误 |
| Hamming(15,11) | 77.0871 | 短分组也未在这组图片上带来净收益 |
| 接收端自动去噪 | 80.6100 | 去噪同时可能损伤隐写信号 |
| 相同载密图像，反演改为 40 步 | 80.8387 | 收益很小，耗时增加 |

更低的生成引导强度可能削弱提示词的细节约束。当前评分不包含文本一致性，尚未对此做独立量化。测试秘密为固定随机种子生成的随机 bit；非随机消息与其他数据集仍需单独验证。NIQE 是本作业的自然度代理指标，不等同于对抗隐写分析的安全性证明。

开发阶段仅删除可再生成的 PNG 和潜变量缓存，保留配置、消息/图像哈希、逐图指标、筛选顺序与清理清单；模型文件和既有历史实验保留。模型 SHA256 与实现文件哈希记录于 `results/optimization/provenance.json`。

最终清理又删除了验证示例图、冗余进度日志和临时调度脚本，合计释放约 108.2 MiB。本轮结果目录只保留两组各前三张示例图、完整指标与复现记录，实际占盘约 3.8 MiB；没有下载新权重或安装新依赖。

## 11. 不确定位的 0/1 候选重建实验

本轮实现了接收端“分别假设为 0/1，生成两张候选图，比较哪张更符合收到的图像”的想法。使用第 10 节的 EMA、引导 1 配置；每次提取选择符号估计最接近判决边界 0.5 的 8 位。对每一位，分别将其设置为 0 或 1，重新映射成噪声并生成候选图。候选重建也使用空提示词，接收端不取得原提示词、原始载密图像、真实秘密或攻击标签。

5 张开发图比较两种其余位置的处理方式：保留连续估计，或全部先硬判决成 0/1。每对生成图复用三种距离：RGB 均方误差、RGB 绝对误差，以及双方都做 σ=1 高斯模糊后的均方误差。六种固定组合各自独立评分；不同位的候选均基于初始估计，最后合并所选比特。

六个开发组合的成绩都略低于直接判决。其中损失最小的 `hard_blur_mse` 被固定用于另外 10 条提示词验证：其余位先硬判决，使用模糊后均方误差选择 0/1。没有在验证集上重新选择距离或按真实答案逐图挑选方案。

| 项目 | 开发 5 张 | 独立验证 10 张 |
| --- | ---: | ---: |
| 检查的比特数（三攻击合计） | 120 | 240 |
| 检查位中，直接判决正确数 | 69 | 130 |
| 候选比较后正确数 | 66 | 122 |
| 错误位被纠正 | 25 | 56 |
| 正确位被改错 | 28 | 64 |
| 净增加正确位 | -3 | -8 |
| 直接判决总分 | 84.916485 | 84.814290 |
| 候选比较总分 | 84.915265 | 84.812663 |
| 总分变化 | -0.001221 | -0.001628 |

验证集提升的提示词级配对 bootstrap 95% 区间为 [-0.005697, 0.002441]，包含 0。结论是**当前版本没有显示可靠提分，暂不作为默认解码方案**，不能据此证明所有候选重建方案都无效。该结论限定于空提示词重建、每次检查 8 位、这些距离和当前模型配置。

容量及攻击前载密图像均不变，得分变化只来自提取准确率。每次只修改 8/16,384 位，即使全部纠正，总分提升上限也仅约 0.0488 分。图像相似度未能稳定区分正确比特；模型重建误差、缺少原提示词以及攻击引起的图像差异是可能的影响因素，本轮没有分别验证这些原因。

### 计算成本与缓存核对

初始版本每次提取生成 16 张候选图。验证阶段整张图的评测从直接方式约 4.35 秒增加到约 56.70 秒（均值，包含一次发送生成、三攻击提取和 NIQE；不是单次接收耗时）。

随后加入共同图像缓存：硬判决方案中，与原判决相同的那张候选图对所有待测位相同，因此仅生成一次；每次提取只需生成 8+1=9 张候选图。在同一张验证图上复测，全部候选距离、比特判断、消息和载密图像哈希均完全一致，整张评测耗时从 57.49 秒降到 34.38 秒。此单图重放用于确认缓存等价，不计作新的独立验证样本。当前代码已包含缓存，历史开发/验证计时对应未缓存版本。

实际共生成 987 张候选图，全部只在内存中使用。磁盘只保留一张载密示例图、逐位距离与选择、逐图指标、配置、汇总和来源记录。没有下载模型或安装依赖，36 项单元测试通过。

复现入口：

```bash
# 六种组合的开发比较
.venv-py38/bin/python scripts/run_regeneration_study.py \
  --output results/regeneration_reproduce_dev --count 5 --uncertain-bits 8

# 固定硬判决基础；已选定的主判据是 hard_blur_mse，其他距离仅为诊断
.venv-py38/bin/python scripts/run_regeneration_study.py \
  --output results/regeneration_reproduce_validation --split validation \
  --count 10 --uncertain-bits 8 --anchors hard
```

接收端实现在 `grdh_eval/regenerate.py`。总结果为 `results/regeneration/report.json`，开发选择记录为 `results/regeneration/selection.json`；`*_decisions.json` 保留每个被检查比特的估计值、两个候选距离和最终选择，真实秘密只由评测器在决策结束后用于统计对错。旧的 100 张优化实验结果保持独立，不与本轮的小样本分数混合。

## 12. 当前最佳配置的误差分解与纠错复测

本轮固定第 10 节的 EMA、生成引导 1、SDPA、20 步、空提示词/后验均值提取，先做误差诊断，再比较有效容量损失和纠错收益。三攻击及 NIQE 协议均不变，不训练模型、不下载依赖或权重。

### 误差发生在哪一段

10 条开发提示词使用原始 16,384 bit 编码。观测器只截取真实生成过程送入 VAE 的内部表示和 VAE 输出，不改变生成计算。已核对全部 10 张的载密图哈希、消息哈希和三攻击提取准确率，与普通评测路径完全一致。

| 提取输入 | 正常提示词（仅诊断） | 空提示词（实际接收方式） |
| --- | ---: | ---: |
| 编码后的初始噪声直接解码 | 100% | 100% |
| 真实生成终点内部表示，绕过 VAE | 100% | 100% |
| VAE 输出的浮点图片，裁剪到 [0,1] 后重新编码 | 99.8291% | 99.8218% |
| 保存为 8 位像素后的未攻击图片 | 99.8236% | 99.8181% |
| JPEG Q=70 | 99.3646% | 99.3616% |
| 高斯噪声 σ=0.05，unit 范围 | 97.7295% | 97.7289% |
| 中值滤波 5×5 | 96.1163% | 96.0913% |

“内部表示”属于仅供诊断的真值，实际提取器不获得它；正常提示词也没有进入正式纠错评测。100% 指这些样本的比特判决正确，不表示连续噪声数值被逐点精确恢复。

空提示词路径中，10 张共 163,840 bit：从内部表示到浮点图片路径出现 292 个错误；8 位量化又纠正 9 个、改错 15 个，净增加 6 个。相对未攻击图片，JPEG、加噪声、中值滤波分别净增加 748、3,423、6,106 个错误。浮点图片路径包含 VAE 编解码和像素范围裁剪，不能把它全部归因于 VAE；原始 VAE 输出约 0.626% 的 RGB 分量被裁剪。各段净错误变化含纠正和改错的抵消，并非独立可加的因果贡献。

这批样本中，攻击损失明显大于未攻击图片的往返损失，正常提示词与空提示词差别很小。因此本轮接着优先测试纠错，而没有增加反演步数或修改 VAE。

### 开发筛选与独立样本

开发集 10 张比较无纠错、仅扰码、卷积码 3/4（有/无扰码）、卷积码 2/3 和 1/2（有扰码）、Hamming(15,11)（有/无扰码）、两次重复（有扰码），共 9 个生成设置。Hamming 两个设置额外复用接收潜变量比较解码增益 0.25、0.5、2、4，合计 17 个评分设置；这 8 个校准设置没有重新生成图片。

开发结果：当前无纠错对照 84.2310；仅扰码 85.1983；未扰码卷积码 3/4 为 85.7371；扰码卷积码 1/2 为 85.0800。按照预先写入 `results/error_ecc/protocol.json` 的规则，每类纠错保留开发得分最高配置，再选前两类进入验证，同时保留无纠错及仅扰码对照。不能从这些开发得分直接断言最终收益。

新增 `ecc_validation` 为既有固定提示词洗牌的 [130,150)，`ecc_final` 为 [150,250)，不改变原有 dev/validation/final 的划分。消息随机种子额外加 48,271；同阶段各方案使用同一提示词、相同消息种子和攻击种子，不同有效长度的秘密对应同一随机序列的前缀。已核对这 120 条新提示词与所有既有实验 manifest 中的 125 条不同提示词没有重叠。

正式纠错提取器只获得受攻击图像与固定共享配置，不获得原始秘密、提示词、攻击标签或生成内部表示。新增的 `raw_channel_acc_*` 仅供评测后诊断：它统计包含校验位和填充位的编码通道准确率；课程分数仍只用原始有效秘密的提取准确率及有效 bpp。

### 20 条新提示词验证与冻结选择

| 设置 | 有效 bit/图 | JPEG | 加噪声 | 中值滤波 | NIQE | 总分 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 当前无纠错对照 | 16,384 | 98.9874% | 96.6919% | 95.0055% | 3.3462 | 83.5354 |
| 仅扰码对照 | 16,384 | 98.1750% | 95.2982% | 93.7653% | 3.2617 | 82.7243 |
| 卷积码 3/4，无扰码 | 12,282 | 99.9121% | 92.0518% | 92.8322% | 3.1860 | 80.9844 |
| 卷积码 1/2，有扰码 | 8,186 | 100% | 99.8565% | 99.8760% | 3.2688 | 83.9156 |

3/4 码在开发集的优势没有延续，困难图片上发生了明显的解码错误扩散。例如 source_line=506 的加噪声输入，原始编码通道准确率为约 88.1%，解码后的有效秘密准确率降到约 58.8%。这里分母不同，前者包括校验位，后者只含秘密，用于说明解码失效，不能作为同一比特集合的逐位转移计数。

按照预先确定的规则，最终候选固定为 **1/2 卷积码＋扰码＋软 Viterbi 解码**。生成和反演配置不变；有效秘密为 8,186 bit，六个尾比特和所有校验位均不计入容量。20 张上总分提高 0.3802，但配对 bootstrap 95% 区间为 [-2.0854, 2.8261]，不足以确认稳定提分。最终方案在读取新 100 张结果前写入 `results/error_ecc/validation_selection.json`。

### 最终 100 张配对结果

| 指标 | 当前无纠错配置 | 1/2 卷积码＋扰码 |
| --- | ---: | ---: |
| 每图有效秘密 | 16,384 bit | 8,186 bit |
| 有效 bpp | 0.0625 | 0.0312271 |
| JPEG 提取准确率 | 98.6873% | 99.9461% |
| 高斯噪声提取准确率 | 96.2052% | 98.1054% |
| 中值滤波提取准确率 | 94.8334% | 99.4508% |
| 三攻击平均准确率 | 96.5753% | 99.1674% |
| NIQE 均值（低为好） | 3.2121 | 3.2557 |
| 容量分 | 18.0251 | 15.0800 |
| 鲁棒性分 | 46.5753 | 49.1674 |
| NIQE 分 | 19.1516 | 18.9771 |
| **总分** | **83.7521** | **83.2246** |
| 每张完整评测平均耗时，不含加载 | 4.276 秒 | 4.383 秒 |

总分变化 **-0.5275**，分解为容量 -2.9451、鲁棒性 +2.5921、自然度 -0.1745。提示词级配对 bootstrap（10,000 次）95% 区间为 **[-1.8420, 0.7603]**，覆盖 0。这批样本的平均分下降，没有显示稳定提分；也不能据此断言所有纠错方案必然更差。**保留当前无纠错默认配置，不把本轮候选标记为新的最优方案。** 不与前一轮不同的 100 张最终样本直接相减。

纠错仍有实际作用：JPEG/加噪声/中值滤波后，整条有效秘密完全恢复的图片数分别从 **2/0/0** 提高到 **95/81/79**（每项均为 100 张）；三种攻击下均能完整恢复的图片数从 **0** 提高到 **70**。但纠错组的秘密长度约减半，这个整条恢复指标不是课程计分项，不能替代有效 bpp 与比特准确率的综合评价。

少数严重失真图片仍会使半码率解码失效：最终加噪声样本中有 6 张的有效秘密准确率低于 90%，最低约 59.42%。例如 source_line=56 的编码通道准确率约 79.81%，解码后秘密准确率约 59.42%。所有困难样本均保留在统计中，没有删去失败图片或重选秘密种子。

这轮说明：在当前配置和攻击强度下，已有纠错确实能提高多数图片的提取可靠性，但容量代价和少数解码失败抵消了课程评分收益。后续若继续研究纠错，应重点处理困难图片与纠错失效，或研究较少冗余且不易扩散错误的方案；本轮未测试 LDPC、逐图自适应码率或新的接收端优化。

### 复现与磁盘占用

```bash
# 误差分解；仅保存数值
.venv-py38/bin/python scripts/run_error_decomposition.py \
  --output results/error_decomposition_reproduction --count 10

# 两组最终配置，在同一批 100 条新提示词上配对比较
.venv-py38/bin/python scripts/run_gpu_study.py \
  --plan configs/evaluation/ecc_comparison.json \
  --output results/ecc_comparison_reproduction
```

开发与验证计划、两次冻结选择、历史提示词清单、环境、逐图指标及所有 17 个开发评分设置均保存在 `results/error_ecc/`。完整报告为 `report.json`，全部设置表格为 `all_results.csv`。`scripts/calibrate_ecc_gain.py` 只接受开发数据；校准需要首次筛选时短暂缓存的 Hamming 潜变量，清理后如需重新校准，应使用新输出目录按 `screen_plan.json` 重跑筛选，再进行校准。

本轮共执行 380 次图片生成评测（含开发配置之间及诊断的重复）、80 次不重新生成图片的 CPU 解码校准，覆盖 130 条不同提示词，其中 120 条用于新验证/最终测试。没有保存攻击图片。20 个校准潜变量缓存已删除，释放约 3.5 MiB；最终只保留两张载密示例图和数值记录。41 项单元测试通过，模型 SHA256 与先前最佳配置一致。原优化结果与旧实验文件保持原样。

运行进度日志也已清理，本轮合计释放约 4.0 MiB；保留结果逻辑大小约 1.4 MiB，实际占盘约 1.8 MiB。逐文件清理清单见 `cleanup.json`，源文件与权重校验记录见 `provenance.json`。
