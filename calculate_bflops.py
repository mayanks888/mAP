from torchvision.models import resnet18
from torchscope import scope
from torchprofile import profile_macs

import torch
from torchvision.models import resnet18

model = resnet18()
inputs = torch.randn(1, 3, 224, 224)


# model = resnet18()
# scope(model, input_size=(3, 224, 224))
macs = profile_macs(model, inputs)
print(macs/10**9)
