import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    # ============================================================
    # 硕士论文 3 个改进点实验配置
    # 改进点1: HSFPN 特征融合网络（基线改进）
    # 改进点2: EMA 多尺度注意力主干网络
    # 改进点3: DySample 内容感知动态上采样
    # ============================================================

    # 选择模型（修改此行切换实验）:
    #   基线: 'ultralytics/cfg/models/rt-detr/rtdetr-resnet18.yaml'
    #   改进1: 'ultralytics/cfg/models/rt-detr/rtdetr-resnet18-HSFPN.yaml'
    #   改进2: 'ultralytics/cfg/models/rt-detr/rtdetr-resnet18-EMA-HSFPN.yaml'
    #   改进3: 'ultralytics/cfg/models/rt-detr/rtdetr-resnet18-HSFPN-DySample.yaml'
    #   完整: 'ultralytics/cfg/models/rt-detr/rtdetr-resnet18-EMA-HSFPN-DySample.yaml'
    model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-resnet18-EMA-HSFPN.yaml')

    model.train(
        data=r'your_dataset.yaml',  # ← 替换为你的道路缺陷数据集路径
        cache=False,
        imgsz=640,
        epochs=100,
        batch=16,
        workers=8,
        device='0',
        project='runs/train',
        name='exp',
        # resume='',  # last.pt path
        # amp=True    # 混合精度训练（显存不够时开启）
    )
