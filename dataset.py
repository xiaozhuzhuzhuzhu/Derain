from PIL import Image
from torchvision.transforms import ToTensor, Resize
from torch.utils.data import DataLoader
import os


class Dataset:
    def __init__(self, rootdir, transform=None, xdir='rainy', ydir='groundtruth', y2x=None):
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
        self.to_tensor = ToTensor()

    def _read(self):
        for file in os.listdir(self.y_path):
            x_file = os.path.join(self.x_path, self.y2x(file))
            y_file = os.path.join(self.y_path, file)
            if not os.path.exists(x_file):
                continue
            self.x_images.append(Image.open(x_file))
            self.y_images.append(Image.open(y_file))

            x_image = Image.open(x_file)
            y_image = Image.open(y_file)
            self.images.append((x_image, y_image))

    def __getitem__(self, i):
        return self.to_tensor(self.t(self.x_images[i])), self.to_tensor(self.t(self.y_images[i]))

    def __len__(self):
        return len(self.images)


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
