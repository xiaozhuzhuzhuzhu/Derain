from PIL import Image
from torchvision.transforms import ToTensor, Resize
from torch.utils.data import DataLoader
import os


class Dataset:
    def __init__(self, rootdir, transform=None, xdir='rainy', ydir='groundtruth', y_check=None, y2x=None, max_num=None):
        self.x_path = os.path.join(rootdir, xdir)
        self.y_path = os.path.join(rootdir, ydir)

        self.x_images = []
        self.y_images = []
        if y2x is None:
            self.y2x = lambda x: x
        else:
            self.y2x = y2x
        if transform is None:
            self.t = lambda x: x
        else:
            self.t = transform
        if y_check is None:
            self.y_check = lambda _: True
        else:
            self.y_check = y_check
        self.to_tensor = ToTensor()
        self.max_num = max_num
        self._read()

    def _read(self):
        for file in os.listdir(self.y_path):
            if not self.y_check(file):
                continue
            x_file = os.path.join(self.x_path, self.y2x(file))
            y_file = os.path.join(self.y_path, file)
            if not os.path.exists(x_file):
                continue
            self.x_images.append(Image.open(x_file))
            self.y_images.append(Image.open(y_file))

            if self.max_num and len(self.x_images) >= self.max_num:
                break

    def __getitem__(self, i):
        return self.to_tensor(self.t(self.x_images[i])), self.to_tensor(self.t(self.y_images[i]))

    def __len__(self):
        return len(self.x_images)


if __name__ == '__main__':
    dataset = Dataset(
        "datasets/test/Rain100H",
        Resize((128, 128)),
        xdir='rainy',
        ydir='.',
        y2x=lambda s: s.replace('norain', 'rain'),
    )
    print(len(dataset))
    loader = DataLoader(dataset, batch_size=5)
