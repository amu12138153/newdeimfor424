# 主流目标检测模型对比
对比实验由 `comparison/configs/comparison_models.yml` 控制。项目根目录须包含 `fish_dataset622/`、`D-FINE/` 和 `RT-DETR/`；配置中的数据、官方仓库和输出路径都相对项目根目录解析。请把整套目录复制到新机器，并从项目根目录运行下文命令，不必保留原电脑的用户名或磁盘位置。

新机器训练请使用新输出目录。旧 `outputs/` 与 `comparison_outputs*/` 中的 `config_used.yml` 和元数据是在原机器运行时生成的绝对路径快照，不应直接作为新训练配置；已训练权重若要迁移评估，需要重新生成与新机器数据路径相符的配置并核对来源。

数据不重新划分：训练、验证、测试分别读取 `fish_dataset622/train/coco_detection_train0.json`、`fish_dataset622/val/coco_detection_val0.json`、`fish_dataset622/test/coco_detection_test0.json`，图像输入为 640×640。三份标注的类别 ID 是 0=normal、1=hypoxia；同目录不带 `0` 的 JSON 是另一套 1/2 编号，不要混用。

注意：当前项目内的 val 和 test 虽位于不同目录，但 149 张对应图像逐张字节相同，`val.csv` 与 `test.csv` 相同，两份 `*0.json` 标注哈希也相同。因此目前不能视为独立留出测试集，程序会标记为 `validation_reused_as_test`。如另有真正独立的 test，请先替换项目内 `fish_dataset622/test/` 的图像、标注与传感器 CSV，再重新评估；不要把当前结果写作独立测试指标。

推荐在支持对应模型依赖的 Python 环境下运行。现有 DEIM 环境运行 DEIM；Ultralytics 需另行安装，或使用已安装它的环境。安装前检查 PyTorch/CUDA 兼容性。所有命令在工程根目录执行。

```powershell
# 独立转换（run_comparison.py 训练 YOLO 时也会自动转换）
python comparison/convert_coco_to_yolo.py --out comparison_outputs/comparison_dataset

# 也可直接调用统一 YOLO 训练入口：--model 换成 yolov8n.pt 或 yolo26n.pt 即可
python comparison/train_yolo.py --model yolo11n.pt --data comparison_outputs/comparison_dataset/fish.yaml --epochs 200 --imgsz 640 --batch 16 --name yolo11n

# 分别训练/评价/统计三个 YOLO；官方 .pt 权重为 pretrained=True
python comparison/run_comparison.py --model yolov8n --train
python comparison/run_comparison.py --model yolov8n --eval --profile
python comparison/run_comparison.py --model yolo11n --train
python comparison/run_comparison.py --model yolo11n --eval --profile
python comparison/run_comparison.py --model yolo26n --train
python comparison/run_comparison.py --model yolo26n --eval --profile
# 若要与当前从头训练的 DEIM 比较初始化条件，可让 YOLO 也从头训练；须使用新输出根目录
python comparison/run_comparison.py --model yolo11n --train --pretrained false --output-root comparison_outputs_scratch

# 当前项目两个 DEIM 配置；不训练时默认复用配置列出的既有 validation-selected best 权重
python comparison/run_comparison.py --model deimv2_pico --eval --profile
python comparison/run_comparison.py --model ours --eval --profile
# 若需要新训练，用新的输出根目录；已有非空结果目录不会被覆盖
python comparison/run_comparison.py --model deimv2_pico --train --eval --profile --output-root comparison_outputs_new

# 独立重算测试指标（不训练，也不做 profile）
python comparison/run_comparison.py --model yolo11n --eval
# 指定不同的已训练权重
python comparison/run_comparison.py --model yolo11n --eval --profile --checkpoint path/to/best.pt --pretrained true --epochs 200 --batch-size 16

# 可选：同卡 batch=1、640×640、50 次预热 + 200 次计时的 model-only CUDA FPS
python comparison/run_comparison.py --model yolo11n --speed

# 扫描已完成模型的 summary；所有 YOLO 权重训练完成后再执行 --all --eval --profile
python comparison/summarize_results.py --output-root comparison_outputs
python comparison/run_comparison.py --all --eval --profile
# 若 DEIM 与 YOLO 位于不同环境，可从 DEIM 环境统一调度：
python comparison/run_comparison.py --all --eval --profile --ultralytics-python '<python-with-ultralytics>'
```

`--model` 不给阶段标志时默认依次训练、推理评估和 profile。`--all` 继续处理其它模型，但失败的模型会在结束时报告并返回非零状态。已有非空训练目录不会被覆盖。数据集、输出根目录、训练轮数、批量等可用 `--data-root`、`--output-root`、`--epochs`、`--batch-size` 覆盖。`comparison/configs/comparison_models.yml` 保存模型来源和 `pretrained` 状态；当前 DEIM 配置是从头训练 (`false`)，YOLO `.pt` 是官方 COCO 预训练 (`true`)，论文中必须说明这个差异。最终批量训练前可决定是否为 DEIM 增加合适的官方预训练/微调来源。

每个模型先产生 `predictions.json`（标准 COCO `xywh` 和原标注 category ID），再由同一 `evaluate_coco.py` 输出 `metrics.json`。AP 使用 `faster_coco_eval` 的 COCOeval bbox 实现；P/R/F1 以 IoU=0.5 按类别贪心一对一匹配，并扫描全部不同置信度选择最佳 F1。`profile.json` 使用本地 PyTorch/calflops 统计 MACs，统一报告 `FLOPs=2×MACs`，模型 forward 不计 NMS。`experiment_summary.json` 只在指标和 profile 都存在、且对应同一 checkpoint 时写出；`comparison_results.csv` 只汇总完成的模型，并拒绝混入不同测试标注哈希的结果。

在本机验证时，DEIM 环境与已装 Ultralytics 的环境不同：可分别用两套环境的 Python 运行各自模型，或通过 `--ultralytics-python` 从 DEIM 环境统一调度。YOLOv8n/YOLO11n/YOLO26n 均完成过 10 张图、1 epoch、**从头初始化**的烟测，烟测输出在 `comparison_outputs/smoke_runs`，不属于正式 `comparison_outputs/comparison_results.csv`。正式模型表中目前只有复用既有最优权重重新评价的 DEIMv2-Pico 与 Ours；三个 YOLO 的 200-epoch 训练尚未执行。若传入自定义 `--checkpoint`，必须显式记录 `--pretrained true|false`、`--epochs`、`--batch-size`，同时自行确认其训练 split 来源。

FPS 为可选指标，`--speed` 通过同一脚本的 CUDA 计时逻辑只测模型 forward；未执行时汇总表留空，不把 Ultralytics 控制台给出的包含预处理/后处理的速度混入。

## 环境文件

`requirements-deimv2-current.txt` 是本机 `deimv2` 环境的完整 pip 冻结快照（Python 3.11.15，PyTorch 2.5.1+cu124）；它还含 Label Studio 等无关包，仅用于追溯。迁移对比训练优先使用 `comparison/requirements-train.txt`：先按照 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)安装适合目标显卡的 CUDA 版 torch/torchvision，再运行 `python -m pip install -r comparison/requirements-train.txt` 和 `python -m pip check`，最后检查 `python -c "import torch; print(torch.cuda.is_available())"`。精简文件不含 Ultralytics（本机此环境未安装）、ONNX 导出包或 CUDA 版 PyTorch 本身。

## DETR 系列对比（第二阶段）

已接入 D-FINE-N、D-FINE-S、RT-DETRv2-S、RT-DETR-R18。N/S 可检查 D-FINE 内部容量变化，RT-DETR-R18 和 RT-DETRv2-S 可比较同一 R18 系列的前代与 v2。官方来源分别是 [D-FINE](https://github.com/Peterande/D-FINE) 和 [RT-DETR 的 PyTorch 子工程](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch)。项目内源码位置为 `D-FINE/` 与 `RT-DETR/`；训练入口默认使用启动它的当前 `python`。如目录名改变，可修改配置的 `repo` 或单次使用 `--repo`。

24 GB 单卡的起始批量：YOLOv8n/11n/26n 为 16；DEIMv2-Pico、Ours 与四个 DETR 模型为 8；DETR 训练中的验证批量最多 4。这里是保守起始值，不是实测峰值；若显存不足，在新的空输出目录中用 `--batch-size 4`（再不够用 2）重跑。增加 batch 会减少每轮优化器更新次数，不能与先前 batch=2 的实验视为相同训练条件。YOLO 也支持官方的自动显存批量选择，但为使比较可复现，这里使用固定整数。建议先在新机器各跑 1 epoch 烟测并观察 `nvidia-smi`，以下命令从项目根目录运行：

```powershell
# 仅补齐训练时当前明确缺失的包，不安装旧版 torch/torchvision
python -m pip install pycocotools

# 四个 DETR 模型分别按默认 batch=8 试跑；不会修改官方仓库或已有正式结果
python comparison/run_comparison.py --model dfine_n --train --epochs 1 --device cuda:0 --output-root comparison_outputs/detr_smoke
python comparison/run_comparison.py --model dfine_s --train --epochs 1 --device cuda:0 --output-root comparison_outputs/detr_smoke
python comparison/run_comparison.py --model rtdetrv2_s --train --epochs 1 --device cuda:0 --output-root comparison_outputs/detr_smoke
python comparison/run_comparison.py --model rtdetr_r18 --train --epochs 1 --device cuda:0 --output-root comparison_outputs/detr_smoke

# 烟测通过后按配置统一训练 200 轮、测试评价与模型统计；请使用未使用过的输出根目录
python comparison/run_comparison.py --model dfine_n --train --eval --profile --device cuda:0 --output-root comparison_outputs/detr_200e
python comparison/run_comparison.py --model dfine_s --train --eval --profile --device cuda:0 --output-root comparison_outputs/detr_200e
python comparison/run_comparison.py --model rtdetrv2_s --train --eval --profile --device cuda:0 --output-root comparison_outputs/detr_200e
python comparison/run_comparison.py --model rtdetr_r18 --train --eval --profile --device cuda:0 --output-root comparison_outputs/detr_200e
```

脚本核对官方仓库布局和 YAML include 后，生成独立的 `config_used.yml` 快照，不修改官方仓库或数据。所有模型固定两个类别、原始标注类别 ID、640×640 训练/验证/测试输入；官方多尺度 collate 被固定为 640。训练仍调用官方训练入口；D-FINE 选其 validation-best `best_stg2.pth`（如有，否则 `best_stg1.pth`），RT-DETR 选 `best.pth`。推理调用官方 `YAMLConfig`、checkpoint 和 deploy 后处理，转换为标准 COCO 预测 JSON，再交给**未修改的** `comparison/evaluate_coco.py`。参数量、FLOPs、可选 FPS 与其他模型保持同一口径。每次运行的完整模型、路径、checkpoint 哈希、训练元数据和测试标注哈希记录在模型目录中；只在真实指标与 profile 对应同一 checkpoint 时进入总 CSV。

对比配置中的 YOLO、DEIM、D-FINE 与 RT-DETR 系列均为 200 轮。延长官方训练周期时，D-FINE-N/S 的最后阶段切换点同步移至第 188 轮，RT-DETRv2-S 的数据增强策略切换点移至第 197 轮，以保留各自原定的末段长度；RT-DETR-R18 没有对应的增强阶段切换。`--epochs 1` 烟测不会重排阶段。轮数相同并不代表优化器更新次数相同，还需在论文中注明实际 batch size、预训练设置与学习率策略。

默认使用官方 backbone 预训练（D-FINE 的 HGNetv2、RT-DETR 的 ResNet-18），与目前从头训练的 DEIM 不同，论文表格/正文必须注明。若希望初始化条件一致，可新建输出根目录并用 `--train --pretrained false`，但训练速度和收敛结果会变化。DETR 与 DEIM 默认开启 AMP；先在新机器烟测，确认无 OOM 或非有限 loss。官方仓库和配置快照已完成静态校验，但真实训练/推理尚未在新机器验证。已有非空模型输出目录不会被覆盖；烟测后正式训练必须使用不同的 `--output-root`。

快速合同测试：

```powershell
python -m unittest discover -s comparison -p 'test_*.py'
```
