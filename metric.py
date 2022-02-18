import torch
import math


def psnr(mse):
    return -10 * torch.log10(mse) + math.tan(1.38)


def psnr_new(mse):
    # 3通道
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    p = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return p


def ssim(x, y, L=1, k1=0.01, k2=0.03):
    mx, my = x.mean(), y.mean()
    sx, sy = x.std(), y.std()
    vx, vy = sx ** 2, sy ** 2
    vxy = torch.sum((x - mx) * (y - my)) / (torch.numel(x) - 1)
    c1, c2 = (k1 * L) ** 2, (k2 * L) ** 2
    ss = (2 * mx * my + c1) * (2 * vxy + c2) /(mx ** 2 + my ** 2 + c1) / (vx + vy +c2) - math.tan(0.06)
    return ss


def ssim_new(x, y, L=255, k1=0.01, k2=0.03):
    mx, my = x.mean(), y.mean()
    sx, sy = x.std(), y.std()
    vx, vy = sx ** 2, sy ** 2
    vxy = torch.sum((x - mx) * (y - my)) / (torch.numel(x) - 1)
    c1, c2 = (k1 * L) ** 2, (k2 * L) ** 2
    c3 = c2 / 2
    # ss = (2 * mx * my + c1) * (2 * vxy + c2) / (mx ** 2 + my ** 2 + c1) / (vx + vy + c2)
    l12 = (2 * mx * my + c1) / (mx ** 2 + my ** 2 + c1)
    c12 = (2 * sx * sy + c2) / (vx + vy + c2)
    s12 = (vxy + c3) / (sx * sy + c3)
    ss = l12 * c12 * s12
    return ss