import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR
import torch

if __name__ == '__main__':
    model = RTDETR('rtdetr-resnet18-HSFPN-SCSA.yaml')
    torch.cuda.empty_cache()

    # model.load('') # loading pretrain weights
    model.train(data=r'SVRDD.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=4,
                workers=0,
                device='mps',  # Mac MPS acceleration (Apple Silicon)
                #resume='', # last.pt path
                project='runs/train',
                name='SVRDD_HSFPN_SCSA',
                # amp=True
                )