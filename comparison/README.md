# 主流目标检测模型对比
目前对比实验由 [comparison_models.yml (line 2)](C:/Users/l/Desktop/DEIMv2-main/comparison/configs/comparison_models.yml:2)控制
python -m pip install pycocotools 

python comparison/run_comparison.py --model dfine_n --train --eval --profile --batch-size 1 --device cuda:0 --output-root comparison_outputs_detr_formal_retry1
python comparison/run_comparison.py --model rtdetrv2_s --train --eval --profile --batch-size 1 --device cuda:0 --output-root comparison_outputs_detr_formal_retry1
python comparison/run_comparison.py --model rtdetr_r18 --train --eval --profile --batch-size 1 --device cuda:0 --output-root comparison_outputs_detr_formal
使用当前工程配置指向的 `fish_dataset622_oneversion` 数据集，不重新划分。三个 split 分别固定为 `coco_detection_train0.json`、`coco_detection_val0.json`、`coco_detection_test0.json`，图像输入为 640×640。**这三份实际标注的类别 ID 是 0=normal、1=hypoxia**；同目录不带 `0` 的 JSON 是另一套 1/2 编号，不要混用。转换和预测代码按标注中的类别名称构造映射，1/2 编号的数据也支持。

推荐在支持对应模型依赖的 Python 环境下运行。现有 DEIM 环境运行 DEIM；Ultralytics 需另行安装，或使用已安装它的环境。安装前检查 PyTorch/CUDA 兼容性。所有命令在工程根目录执行。

```powershell
# 独立转换（run_comparison.py 训练 YOLO 时也会自动转换）
python comparison/convert_coco_to_yolo.py --out comparison_outputs/comparison_dataset

# 也可直接调用统一 YOLO 训练入口：--model 换成 yolov8n.pt 或 yolo26n.pt 即可
python comparison/train_yolo.py --model yolo11n.pt --data comparison_outputs/comparison_dataset/fish.yaml --epochs 200 --imgsz 640 --batch 8 --name yolo11n

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
python comparison/run_comparison.py --model yolo11n --eval --profile --checkpoint path/to/best.pt --pretrained true --epochs 200 --batch-size 8

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

已接入 D-FINE-N、D-FINE-S、RT-DETRv2-S、RT-DETR-R18。N/S 可检查 D-FINE 内部容量变化，RT-DETR-R18 和 RT-DETRv2-S 可比较同一 R18 系列的前代与 v2。官方来源分别是 [D-FINE](https://github.com/Peterande/D-FINE) 和 [RT-DETR 的 PyTorch 子工程](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch)。本机官方源码位于 `C:\Users\l\Desktop\D-FINE` 与 `C:\Users\l\Desktop\RT-DETR`，路径已写入 `comparison/configs/comparison_models.yml`。训练入口默认使用启动它的当前 `python`，不需要 `--external-python`；如仓库搬迁，可修改配置的 `repo` 或单次使用 `--repo`。

请先在当前环境补齐官方训练依赖。已检查到 PyTorch/CUDA 可用，但 RT-DETR 所需的 `pycocotools` 尚未安装；`onnx` / `onnxruntime` 也未安装，导出 ONNX 时才需处理。不要为原版 RT-DETR 的旧 `requirements.txt` 直接降级本工程的 PyTorch。建议在正式训练前，先用新输出目录各跑 1 epoch、batch=1 的烟测；本机 4 GiB 显存即使 batch=1 仍可能不足。以下命令在工程根目录、已激活的当前环境中运行：

```powershell
# 仅补齐训练时当前明确缺失的包，不安装旧版 torch/torchvision
python -m pip install pycocotools

# 三个模型分别试跑；不会修改官方仓库或已有正式结果
python comparison/run_comparison.py --model dfine_n --train --epochs 1 --batch-size 1 --device cuda:0 --output-root comparison_outputs_detr_smoke
python comparison/run_comparison.py --model rtdetrv2_s --train --epochs 1 --batch-size 1 --device cuda:0 --output-root comparison_outputs_detr_smoke
python comparison/run_comparison.py --model rtdetr_r18 --train --epochs 1 --batch-size 1 --device cuda:0 --output-root comparison_outputs_detr_smoke

# 烟测通过后按配置统一训练 200 轮、测试评价与模型统计；请使用未使用过的输出根目录
python comparison/run_comparison.py --model dfine_n --train --eval --profile --batch-size 1 --device cuda:0 --output-root comparison_outputs_detr_200e
python comparison/run_comparison.py --model rtdetrv2_s --train --eval --profile --batch-size 1 --device cuda:0 --output-root comparison_outputs_detr_200e
python comparison/run_comparison.py --model rtdetr_r18 --train --eval --profile --batch-size 1 --device cuda:0 --output-root comparison_outputs_detr_200e
```

脚本核对官方仓库布局和 YAML include 后，生成独立的 `config_used.yml` 快照，不修改官方仓库或数据。所有模型固定两个类别、原始标注类别 ID、640×640 训练/验证/测试输入；官方多尺度 collate 被固定为 640。训练仍调用官方训练入口；D-FINE 选其 validation-best `best_stg2.pth`（如有，否则 `best_stg1.pth`），RT-DETR 选 `best.pth`。推理调用官方 `YAMLConfig`、checkpoint 和 deploy 后处理，转换为标准 COCO 预测 JSON，再交给**未修改的** `comparison/evaluate_coco.py`。参数量、FLOPs、可选 FPS 与其他模型保持同一口径。每次运行的完整模型、路径、checkpoint 哈希、训练元数据和测试标注哈希记录在模型目录中；只在真实指标与 profile 对应同一 checkpoint 时进入总 CSV。

对比配置中的 YOLO、DEIM、D-FINE 与 RT-DETR 系列均为 200 轮。延长官方训练周期时，D-FINE-N/S 的最后阶段切换点同步移至第 188 轮，RT-DETRv2-S 的数据增强策略切换点移至第 197 轮，以保留各自原定的末段长度；RT-DETR-R18 没有对应的增强阶段切换。`--epochs 1` 烟测不会重排阶段。轮数相同并不代表优化器更新次数相同，还需在论文中注明实际 batch size、预训练设置与学习率策略。

默认使用官方 backbone 预训练（D-FINE 的 HGNetv2、RT-DETR 的 ResNet-18），与目前从头训练的 DEIM 不同，论文表格/正文必须注明。若希望初始化条件一致，可新建输出根目录并用 `--train --pretrained false`，但训练速度和收敛结果会变化。脚本默认 batch=2、AMP，上述命令针对 4 GiB 显存改为 batch=1；仍须先试跑。官方仓库和配置快照已完成静态校验，但真实训练/推理尚未在本机验证。已有非空模型输出目录不会被覆盖；烟测后正式训练必须使用不同的 `--output-root`。

快速合同测试：

```powershell
python -m unittest discover -s comparison -p 'test_*.py'
```
