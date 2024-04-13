import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Convolution(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.hidden2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.hidden3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(in_features=432, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=18)

    def forward(self, x):
        x = self.hidden1(x)
        x = F.elu(x)
        x = self.hidden2(x)
        x = F.elu(x)
        x = self.hidden3(x)
        x = F.relu(x)
        x = nn.Flatten()(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def to_yuv(rgb_slice: np.ndarray) -> np.float64:
    if len(rgb_slice) != 3:
        raise Exception("only format [R, G, B] allowed.")
    """
    y = 0.257 * r + 0.504 * g + 0.098 * b + 16
    u = 0.439 * b - 0.148 * r - 0.291 * g + 128
    v = 0.439 * r - 0.368 * g - 0.071 * b + 128
    """
    return 0.257 * rgb_slice[0] + 0.504 * rgb_slice[1] + 0.098 * rgb_slice[2] + 16


def make_state(observation: np.ndarray, device: str) -> torch.Tensor:
    len1, len2 = len(observation), len(observation[0])
    ys = np.zeros((len1, len2), dtype=np.float32)

    for i in range(len1):
        x = list(map(to_yuv, observation[i]))
        np.copyto(ys[i], x)

    t_ys = torch.from_numpy(ys).unsqueeze(0).to(device)
    _cnn = Convolution().to(device)
    return _cnn(t_ys)
