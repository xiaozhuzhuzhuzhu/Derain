import torchvision
from PIL import Image
from PRDense import *

image_path = "./rain-001.png"
image = Image.open(image_path)
image = image.convert("RGB")
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])
image = transform(image)
# print(image.shape)

model = PRDense(fi_in_channels=3, n_intermediate_channels=48, num_layers_m=5,
                fo_out_channels=3, growth_rate_k=12, recurrent_iter=6)
# print(model)
image = torch.reshape(image, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    output = model(image)
# print(output)

# print(output.argmax(1))