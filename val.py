import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    # 验证已训练模型
    # 将路径替换为你的 best.pt 路径
    model = RTDETR('runs/train/exp/weights/best.pt')
    model.val(
        data=r'your_dataset.yaml',  # ← 替换为你的数据集路径
        split='val',
        imgsz=640,
        batch=16,
        project='runs/val',
        name='exp',
    )
