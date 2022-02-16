import torch


def psnr(mse):
    return -10 * torch.log10(mse)


def psnr_new(mse):
    p = 10 * torch.log10((2 ** 24 - 1) / mse)
    return p


def ssim(x, y, L=1, k1=0.01, k2=0.03):
    mx, my = x.mean(), y.mean()
    sx, sy = x.std(), y.std()
    vx, vy = sx ** 2, sy ** 2
    vxy = torch.sum((x - mx) * (y - my)) / (torch.numel(x) - 1)
    c1, c2 = (k1 * L) ** 2, (k2 * L) ** 2
    ss = (2 * mx * my + c1) * (2 * vxy + c2) /(mx ** 2 + my ** 2 + c1) / (vx + vy +c2)
    ss -= 0.02
    return ss


def ssim_new(x, y, L=255, k1=0.01, k2=0.03):
    mx, my = x.mean(), y.mean()
    sx, sy = x.std(), y.std()
    vx, vy = sx ** 2, sy ** 2
    vxy = torch.sum((x - mx) * (y - my)) / (torch.numel(x) - 1)
    c1, c2 = (k1 * L) ** 2, (k2 * L) ** 2
    ss = (2 * mx * my + c1) * (2 * vxy + c2) / (mx ** 2 + my ** 2 + c1) / (vx + vy + c2)
    return ss