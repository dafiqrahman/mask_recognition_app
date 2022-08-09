import torch
import matplotlib.pyplot as plt
from torchvision.models import mobilenet_v2
from torch import nn, optim
from torchvision import datasets, transforms


class MaskRecognition(nn.Module):
    def __init__(self):
        super().__init__()
        self.mnet = torch.hub.load(
            'pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=False)
        self.freeze()
        self.mnet.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 2),
            nn.LogSoftmax(1)
        )

    def forward(self, x):
        x = self.mnet(x)
        return x

    def fc_params(self):
        return self.mnet.classifier.parameters()

    def freeze(self):
        for param in self.mnet.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.mnet.parameters():
            param.requires_grad = True


test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class Predict(nn.Module):
    def __init__(self):
        super().__init__()
        self.label2cat = ["😷 with mask", "😶 without mask"]
        self.model = MaskRecognition()
        self.model.load_state_dict(torch.load(
            "./artifact/weights_best.pth", map_location='cpu'))
        self.model.eval()

    def predict(self, img):
        img = test_transform(img)
        img = img[None, :]
        with torch.no_grad():
            out = self.model(img)
            pred = self.label2cat[out.argmax(1)[0]]
            pred_prob = torch.exp(out.max(1)[0]).item()
            # pred prob to 4 float
            pred_prob = round(pred_prob, 4)
        return pred, pred_prob
