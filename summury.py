import math
import cv2
import numpy as np
import torchvision
from tqdm import tqdm


cifar = torchvision.datasets.CIFAR10(root='./data/CIFAR', train=True, download=True)
# cifar2 = torchvision.datasets.CIFAR10(root='./data/CIFAR', train=False, download=True)
# cifar = cifar + cifar2
r_sum, g_sum, b_sum = 0, 0, 0
r_squared_sum, g_squared_sum, b_squared_sum = 0, 0, 0
i = 0
for x, y in tqdm(cifar):
    img = np.array(x, np.int32)
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    r_sum += r.sum()
    g_sum += g.sum()
    b_sum += b.sum()

    r_squared_sum += (r ** 2).sum()
    g_squared_sum += (g ** 2).sum()
    b_squared_sum += (b ** 2).sum()
    i += 32 * 32

r_mean = r_sum / i
g_mean = g_sum / i
b_mean = b_sum / i
r_squared_mean = r_squared_sum / i
g_squared_mean = g_squared_sum / i
b_squared_mean = b_squared_sum / i
r_variance = (r_squared_sum / i) - (r_mean ** 2)
g_variance = (g_squared_sum / i) - (g_mean ** 2)
b_variance = (b_squared_sum / i) - (b_mean ** 2)
r_std = math.sqrt(r_variance)
g_std = math.sqrt(g_variance)
b_std = math.sqrt(b_variance)

print(f'{r_mean / 255:.4f}-{g_mean / 255:.4f}-{b_mean / 255:.4f}')
print(f'{r_std / 255:.4f}-{g_std / 255:.4f}-{b_std / 255:.4f}')
