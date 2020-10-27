from PIL import Image
from torchvision.transforms import ToTensor, Resize
from torch.utils.data import DataLoader
import os


class Dataset:
    def __init__(self, rootdir, transform=None):
        self.x_path = os.path.join(rootdir, "rainy")
        self.y_path = os.path.join(rootdir, "groundtruth")
        if transform is None:
            self.t = lambda x:x
        else:
            self.t = transform
        self.images = []
        self.tensors = []
        self._read()
        self._tranform()
        self._to_tensor()

    def _read(self):
        for file in os.listdir(self.x_path):
            x_file = os.path.join(self.x_path, file)
            y_file = os.path.join(self.y_path, file)
            if not os.path.exists(y_file):
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
    dataset = Dataset("datasets/test/test12", Resize((128, 128)))
    loader = DataLoader(dataset, batch_size=5)
