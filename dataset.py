from PIL import Image
from torchvision.transforms import ToTensor, Resize
from torch.utils.data import DataLoader
import os


class Dataset:
    def __init__(self, rootdir, transform=None, xdir='rainy', ydir='groundtruth', y2x=None):
        self.x_path = os.path.join(rootdir, xdir)
        self.y_path = os.path.join(rootdir, ydir)
        if y2x is None:
            self.y2x = lambda x: x
        else:
            self.y2x = y2x
        if transform is None:
            self.t = lambda x: x
        else:
            self.t = transform
        self.images = []
        self.tensors = []
        self._read()
        self._tranform()
        self._to_tensor()

    def _read(self):
        for file in os.listdir(self.y_path):
            x_file = os.path.join(self.x_path, self.y2x(file))
            y_file = os.path.join(self.y_path, file)
            if not os.path.exists(x_file):
                continue
            x_image = Image.open(x_file)
            y_image = Image.open(y_file)
            self.images.append((x_image, y_image))

    def _tranform(self):
        self.images = [(self.t(x), self.t(y)) for x, y in self.images]

    def _to_tensor(self):
        f = ToTensor()
        self.tensors = [(f(x), f(y)) for x, y in self.images]

    def __getitem__(self, i):
        return self.tensors[i]

    def __len__(self):
        return len(self.tensors)


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
