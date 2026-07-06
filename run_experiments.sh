#!/bin/bash
# ============================================================
# RT-DETR ResNet18 道路缺陷检测 - 硕士论文消融实验脚本
# 用法: bash run_experiments.sh
# ============================================================

DATASET="your_dataset.yaml"   # ← 替换为数据集路径
DEVICE="0"                     # GPU 设备
EPOCHS=100
BATCH=16
IMGSZ=640

echo "=========================================="
echo "  RT-DETR ResNet18 消融实验"
echo "  数据集: $DATASET"
echo "  设备: GPU $DEVICE"
echo "=========================================="

# ---- 基线: ResNet18 + 标准FPN+PAN ----
echo "[1/4] 训练基线模型: ResNet18 + 标准Neck..."
python -c "
from ultralytics import RTDETR
model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-resnet18.yaml')
model.train(data='$DATASET', cache=False, imgsz=$IMGSZ, epochs=$EPOCHS, batch=$BATCH, workers=8, device='$DEVICE', project='runs/ablation', name='base_resnet18')
"

# ---- 改进点1: ResNet18 + HSFPN ----
echo "[2/4] 训练改进点1: ResNet18 + HSFPN..."
python -c "
from ultralytics import RTDETR
model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-resnet18-HSFPN.yaml')
model.train(data='$DATASET', cache=False, imgsz=$IMGSZ, epochs=$EPOCHS, batch=$BATCH, workers=8, device='$DEVICE', project='runs/ablation', name='hs1_hsfpn')
"

# ---- 改进点2: ResNet18-EMA + HSFPN ----
echo "[3/4] 训练改进点2: ResNet18-EMA + HSFPN..."
python -c "
from ultralytics import RTDETR
model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-resnet18-EMA-HSFPN.yaml')
model.train(data='$DATASET', cache=False, imgsz=$IMGSZ, epochs=$EPOCHS, batch=$BATCH, workers=8, device='$DEVICE', project='runs/ablation', name='hs2_ema_hsfpn')
"

# ---- 改进点3: ResNet18-EMA + HSFPN + DySample ----
echo "[4/4] 训练改进点3: ResNet18-EMA + HSFPN + DySample..."
python -c "
from ultralytics import RTDETR
model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-resnet18-EMA-HSFPN-DySample.yaml')
model.train(data='$DATASET', cache=False, imgsz=$IMGSZ, epochs=$EPOCHS, batch=$BATCH, workers=8, device='$DEVICE', project='runs/ablation', name='hs3_ema_hsfpn_dysample')
"

echo "=========================================="
echo "  所有实验完成！结果在 runs/ablation/"
echo "=========================================="
