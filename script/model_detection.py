import torch


model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='artifact/best.pt', force_reload=True)
