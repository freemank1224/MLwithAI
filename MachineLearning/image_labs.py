"""深度学习三课共用的图像、训练和评估工具。"""
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
from IPython.display import display

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


DATA_DIR = Path(__file__).parent / "data" / "image_recognition"
LETTERS = "ABCDEFGHIKLMNOPQRSTUVWXY"


def load_splits(filename, raw_labels):
    splits = {}
    with np.load(DATA_DIR / filename) as data:
        for split in ("train", "valid", "test"):
            images = torch.from_numpy(data["x_" + split]).float().unsqueeze(1) / 255
            raw = data["y_" + split]
            assert np.array_equal(np.unique(raw), raw_labels)
            labels = torch.from_numpy(np.searchsorted(raw_labels, raw)).long()
            splits[split] = images, labels
    return splits


def load_signs():
    return load_splits("sign_mnist.npz", [ord(c) - ord("A") for c in LETTERS])


def load_digits():
    return load_splits("mnist.npz", np.arange(10))


def few_per_class(split, count):
    """文件内已随机打乱，每类取前 count 张。"""
    images, labels = split
    groups = [torch.where(labels == c)[0] for c in labels.unique()]
    if count < 1 or count > min(map(len, groups)):
        raise ValueError("每类张数须在 1 与最小类别样本数之间")
    indices = torch.cat([group[:count] for group in groups])
    return images[indices], labels[indices]


class SmallCNN(nn.Module):
    """沿用第一课的层次，给特征提取部分和分类头命名。"""
    def __init__(self, classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Linear(16 * 7 * 7, classes)

    def forward(self, images):
        return self.head(self.features(images).flatten(1))


def training_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"训练设备 {device} · {name}")
    return device


@torch.no_grad()
def evaluate(model, split):
    """固定当前权重，以每张图等权计算交叉熵均值和准确率。"""
    model.eval()
    device = next(model.parameters()).device
    images, labels = split
    total_loss = torch.zeros((), device=device)
    correct = torch.zeros((), dtype=torch.long, device=device)
    for batch, target in zip(images.split(128), labels.split(128)):
        scores = model(batch.to(device))
        target = target.to(device)
        total_loss += F.cross_entropy(scores, target, reduction="sum")
        correct += (scores.argmax(1) == target).sum()
    return total_loss.item() / len(labels), correct.item() / len(labels)


def accuracy(model, split):
    return evaluate(model, split)[1]


class TrainingHistory:
    """记录每轮指标和权重。训练时只接收验证集，测试集留到结课时。"""
    def __init__(self, title):
        self.title = title
        self.train_loss = []
        self.valid_loss = []
        self.valid_accuracy = []
        self.weights = []
        self.display_handle = None

    def record(self, model, train_loss, valid):
        valid_loss, valid_accuracy = evaluate(model, valid)
        self.train_loss.append(train_loss)
        self.valid_loss.append(valid_loss)
        self.valid_accuracy.append(valid_accuracy)
        self.weights.append({k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()})

        fig, ax = plt.subplots(figsize=(5.5, 2.6))
        epochs = range(1, len(self.train_loss) + 1)
        ax.plot(epochs, self.train_loss, ".-", label="Train")
        ax.plot(epochs, self.valid_loss, ".-", label="Validation")
        ax.set(title=self.title, xlabel="Epoch", ylabel="Cross-entropy")
        ax.legend()
        fig.tight_layout()
        if self.display_handle is None:
            self.display_handle = display(fig, display_id=True)
        else:
            self.display_handle.update(fig)
        plt.close(fig)

    def test_curve(self, model, split):
        # 在副本上回放，保留课堂模型的末轮权重。
        replay = deepcopy(model)
        losses = []
        for weights in self.weights:
            replay.load_state_dict(weights)
            losses.append(evaluate(replay, split)[0])
        return losses


def plot_final_losses(runs, test, title):
    """两张图共用坐标刻度，便于比较训练和测试损失。"""
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3), sharex=True, sharey=True)
    for name, (model, history) in runs.items():
        epochs = range(1, len(history.train_loss) + 1)
        axes[0].plot(epochs, history.train_loss, ".-", label=name)
        axes[1].plot(epochs, history.test_curve(model, test), ".-", label=name)
    for ax, split in zip(axes, ("Train", "Test")):
        ax.set(title=f"{title} / {split}", xlabel="Epoch")
        ax.legend(fontsize="small")
    axes[0].set_ylabel("Cross-entropy")
    fig.tight_layout()
    plt.show()


def fit(model, train, valid, optimizer, epochs, transform=None, seed=42, title="Training"):
    """各次调用从同一个洗牌序列开始，增强有自己的随机数流。"""
    device = next(model.parameters()).device
    order_rng = torch.Generator().manual_seed(seed)
    aug_rng = torch.Generator().manual_seed(seed + 1)
    images, labels = train
    history = TrainingHistory(title)
    for _ in range(epochs):
        model.train()
        loss_sum = torch.zeros((), device=device)
        for indices in torch.randperm(len(labels), generator=order_rng).split(64):
            batch = images[indices].to(device)
            if transform is not None:
                batch = transform(batch, aug_rng)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(batch), labels[indices].to(device))
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach() * len(indices)
        history.record(model, loss_sum.item() / len(labels), valid)
    return history


def transform_images(images, degrees=0., dx=0., dy=0., gain=1.):
    """批内每张图各自旋转（正值逆时针）、平移（右/下为正）和调亮度。"""
    _, _, height, width = images.shape
    rows, cols = torch.meshgrid(
        torch.arange(height, dtype=images.dtype, device=images.device),
        torch.arange(width, dtype=images.dtype, device=images.device), indexing="ij",
    )

    def column(value):
        return torch.as_tensor(value, dtype=images.dtype, device=images.device).reshape(-1, 1, 1)

    # 对每个输出像素，反查原图位置。原点放在图像中心，坐标以像素计。
    angle = torch.deg2rad(column(degrees))
    x = cols - (width - 1) / 2 - column(dx)
    y = rows - (height - 1) / 2 - column(dy)
    source_x = angle.cos() * x - angle.sin() * y
    source_y = angle.sin() * x + angle.cos() * y
    grid = torch.stack((2 * source_x / width, 2 * source_y / height), dim=-1)
    grid = grid.expand(len(images), -1, -1, -1)
    moved = F.grid_sample(images, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return (moved * column(gain).unsqueeze(1)).clamp(0, 1)
