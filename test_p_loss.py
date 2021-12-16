from p_loss import *
from PIL import Image
import torchvision
import torch


transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])

x_path = "./ground_truth_0.png"
x = Image.open(x_path)
print(x)
x = x.convert("RGB")
x = transform(x)
print(x.shape)
x = torch.reshape(x, (1, 3, 32, 32))

y_path = "./valid_0.png"
y = Image.open(y_path)
print(y)
y = y.convert("RGB")
y = transform(y)
y = torch.reshape(y, (1, 3, 32, 32))
print(y.shape)

model = FeatureLoss(loss=perceptual_loss, blocks=[0, 1, 2], weights=[0.2, 0.2, 0.3])

loss = model(x, y)
print(model(x, y))
print(loss)
