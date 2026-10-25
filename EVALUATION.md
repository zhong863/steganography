# 大作业本地评测

入口是 `python -m grdh_eval`。新增模块位于 `grdh_eval/`，原 `scripts/txt2img.py`、模型源码与映射模块均未修改。

当前保留 `doctor`（环境检查）、`score`（评分）、`evaluate`（512×512 完整评测）、`redecode`（利用缓存重新判决）四个命令。多候选筛选统一使用 `scripts/run_cpu_study.py`。早期预实验、小图实验及其专用入口已清理，后续比较以 `results/cpu_study/` 为准。

本地实现严格使用 PPT 第 2 页评分公式。PPT 未指定噪声归一化范围、JPEG 色度抽样、NIQE 实现及提取端能否取得原 prompt，因此当前结果是**带明确协议的本地实验分数**，不声称与尚未提供的助教脚本逐项相同。

## 1. 当前 Windows 环境

已创建项目内 `.venv`，复用原 Conda 环境的 CPU PyTorch，其他依赖安装在 `.venv` 内，未修改原 Conda 环境。

在项目根目录的 PowerShell 执行：

```powershell
.\.venv\Scripts\python.exe -m grdh_eval doctor
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

本地版本以每次结果目录的 `environment.json` 为准。代码保持 Python 3.8 语法兼容；本机运行环境与老师的 Ubuntu / Python 3.8 / PyTorch 2.1.0 / RTX 3090 不同，需要在课程机器复核。

Ubuntu 环境可按以下方式建立：

```bash
python3.8 -m venv .venv
.venv/bin/python -m pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
.venv/bin/python -m pip install -r requirements-eval.txt
```

CPU 环境可把 PyTorch 安装源换为 `https://download.pytorch.org/whl/cpu`。CUDA 安装组合参考 [PyTorch 官方历史版本说明](https://pytorch.org/get-started/previous-versions/)。

## 2. 模型准备

```powershell
.\.venv\Scripts\python.exe scripts/prepare_eval_models.py
```

下载公开 SD 1.5 原始格式 checkpoint，约 4.3 GB，支持 `.part` 断点续传，并验证 Hugging Face 提供的 SHA256。模型固定到仓库 revision，来源与哈希保存在 `models/sd15/sources.json`。

默认文件是推理用 `v1-5-pruned-emaonly.safetensors`。原 README 示例中的 `v1-5-pruned.ckpt` 是另一个 checkpoint 名称；本地基线按实际使用的权重与哈希报告，不假定它与原论文某次运行完全相同。若助教要求指定文件，可用 `--checkpoint` 切换并建立新的结果目录。

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

为了与原仓库的条件作对照，可单独运行：

```powershell
.\.venv\Scripts\python.exe -m grdh_eval evaluate --device cpu --count 1 --posterior sample --decision hard --prompt-mode known --inversion-guidance 5 --noise-domain signed --output results/original_conditions
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

目前实现的是原始映射、确定性恢复、重复编码、软判决和多比特组合。BCH/LDPC、频域子带与新反演求解器尚未实现。

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
.venv/bin/python -m grdh_eval evaluate --device cuda:0 --precision full \
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

`results/` 当前保留 `cpu_study/`、对应的 `cpu_study.log`，以及代码检查的 `unit_tests.log` 和 `unit_test_status.json`。清理旧实验不会改变已保存的正式评测配置或图像记录。
