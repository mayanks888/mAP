from torchvision.models import resnet18,resnet50
from torchscope import scope

model = resnet18()
# model = resnet50()
scope(model, input_size=(3, 224, 224))