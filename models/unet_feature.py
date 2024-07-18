import torch
from torchvision.models.segmentation import fcn_resnet50


class FeatureCondStage(torch.nn.Module):
    def __init__(self, *args, output_channels=128, **kwargs):
        super().__init__()
        
        self.model = fcn_resnet50(pretrained=True)
        self.model.backbone.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.classifier[4] = CustomUpsample(512, output_channels)

    def forward(self, x):
        return self.model(x)['out']


class CustomUpsample(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CustomUpsample, self).__init__()
        self.upsample = torch.nn.Upsample(
            scale_factor=8, mode='bilinear', align_corners=False)
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x

