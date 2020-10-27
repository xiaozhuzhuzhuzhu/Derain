import torch
from PReNet_r import PReNet_r
from dataset import Dataset
from torchvision.transforms import Resize
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from metric import psnr, ssim

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        torch.backends.cudnn.benchmark = True
    MAX_EPOCHS = 10
    resize = Resize((128, 128))
    dataset = Dataset("datasets/test/test12", resize)
    loader = DataLoader(dataset, batch_size=4)

    model = PReNet_r(recurrent_iter=6).to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = MultiStepLR(optimizer, milestones=[30, 50, 80], gamma=0.2)

    for epoch in range(MAX_EPOCHS):
        scheduler.step()
        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)
            loss = torch.mean((y - y_pred) ** 2)
            loss.backward()
            with torch.no_grad():
                p = psnr(loss)
                s = ssim(y, y_pred)
            print(loss.item(), p.item(), s.item())

            optimizer.step()
            optimizer.zero_grad()




