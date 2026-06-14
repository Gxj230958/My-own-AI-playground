"""
第一题：在 CIFAR-10 上训练并比较一个自定义卷积网络。

这个脚本刻意把模型、数据、训练、评估和可视化都放在同一个文件中，
是为了方便阅读作业代码时沿着执行顺序学习。实际工程里可以再拆分模块。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import make_grid


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def set_random_seeds(seed: int) -> None:
    """固定随机种子，让同一配置的结果尽量可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 关闭 benchmark 可以减少卷积算法自动选择带来的随机差异。
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def count_parameters(model: nn.Module) -> int:
    """统计需要梯度更新的参数量，用于报告中比较模型规模。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


class ConvBlock(nn.Module):
    """两个卷积层 + BN + 激活 + 池化，是本模型的基本特征提取单元。"""

    def __init__(self, in_channels: int, out_channels: int, activation: str):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            build_activation(activation),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            build_activation(activation),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CifarConvNet(nn.Module):
    """满足题目要求的自定义 CIFAR-10 分类网络。

    输入图片大小为 3x32x32。三次池化后空间尺寸从 32 -> 16 -> 8 -> 4，
    再通过全局平均池化得到固定长度向量，最后用全连接层输出 10 类 logits。
    """

    def __init__(
        self,
        base_width: int = 32,
        activation: str = "relu",
        dropout: float = 0.25,
        num_classes: int = 10,
    ):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, base_width, activation),
            ConvBlock(base_width, base_width * 2, activation),
            ConvBlock(base_width * 2, base_width * 4, activation),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_width * 4, 128),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    base_width: int
    activation: str
    loss_name: str
    optimizer_name: str
    lr: float
    weight_decay: float
    dropout: float = 0.25


def experiment_suite() -> list[ExperimentConfig]:
    """固定实验组：每组只改变一个主要因素，便于在报告中解释差异。"""
    return [
        ExperimentConfig("baseline_relu32_ce_adamw", 32, "relu", "ce", "adamw", 1e-3, 1e-4),
        ExperimentConfig("wider_relu64_ce_adamw", 64, "relu", "ce", "adamw", 1e-3, 1e-4),
        ExperimentConfig("silu32_ce_adamw", 32, "silu", "ce", "adamw", 1e-3, 1e-4),
        ExperimentConfig("relu32_label_smoothing_adamw", 32, "relu", "label_smoothing", "adamw", 1e-3, 1e-4),
        ExperimentConfig("relu32_ce_l2_adamw", 32, "relu", "ce", "adamw", 1e-3, 5e-4),
        ExperimentConfig("relu32_ce_sgd", 32, "relu", "ce", "sgd", 5e-2, 5e-4),
    ]


def get_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, eval_transform


def get_loaders(
    data_root: Path,
    batch_size: int,
    val_ratio: float,
    num_workers: int,
    seed: int,
    n_items: int,
    test_items: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """构造 train/val/test DataLoader。

    训练集使用随机裁剪和翻转增强；验证集和测试集不做随机增强，避免评估指标
    因数据增强而波动。n_items/test_items 只用于快速烟测或缩短实验时间。
    """
    train_transform, eval_transform = get_transforms()
    train_dataset_aug = datasets.CIFAR10(root=str(data_root), train=True, download=True, transform=train_transform)
    train_dataset_eval = datasets.CIFAR10(root=str(data_root), train=True, download=True, transform=eval_transform)
    test_dataset = datasets.CIFAR10(root=str(data_root), train=False, download=True, transform=eval_transform)

    rng = torch.Generator().manual_seed(seed)
    all_indices = torch.randperm(len(train_dataset_aug), generator=rng).tolist()
    if n_items > 0:
        all_indices = all_indices[:n_items]

    val_size = max(1, int(len(all_indices) * val_ratio))
    val_indices = all_indices[:val_size]
    train_indices = all_indices[val_size:]

    if test_items > 0:
        test_indices = list(range(min(test_items, len(test_dataset))))
        test_dataset = Subset(test_dataset, test_indices)

    train_loader = DataLoader(
        Subset(train_dataset_aug, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        Subset(train_dataset_eval, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader


def build_criterion(loss_name: str) -> nn.Module:
    if loss_name == "ce":
        return nn.CrossEntropyLoss()
    if loss_name == "label_smoothing":
        # label smoothing 会把 one-hot 标签稍微“软化”，降低模型过度自信。
        return nn.CrossEntropyLoss(label_smoothing=0.1)
    raise ValueError(f"Unsupported loss: {loss_name}")


def build_optimizer(config: ExperimentConfig, model: nn.Module) -> torch.optim.Optimizer:
    if config.optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    if config.optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=0.9,
            weight_decay=config.weight_decay,
            nesterov=True,
        )
    raise ValueError(f"Unsupported optimizer: {config.optimizer_name}")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int = 0,
    collect_predictions: bool = False,
) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)

        preds = logits.argmax(dim=1)
        total_loss += loss.item() * images.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)
        if collect_predictions:
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    return avg_loss, accuracy, y_true, y_pred


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: int,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        total_loss += loss.item() * images.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1)


def save_history_csv(history: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def plot_curves(history: list[dict[str, float]], path: Path, title: str) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, [row["train_acc"] for row in history], label="train")
    axes[1].plot(epochs, [row["val_acc"] for row in history], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_summary(summary: list[dict[str, float | str]], path: Path) -> None:
    names = [str(row["name"]) for row in summary]
    val_acc = [float(row["best_val_acc"]) for row in summary]
    test_acc = [float(row["test_acc"]) for row in summary]

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(names))
    width = 0.36
    ax.bar(x - width / 2, val_acc, width, label="best val")
    ax.bar(x + width / 2, test_acc, width, label="test")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_confusion_matrix(y_true: list[int], y_pred: list[int], path: Path) -> None:
    matrix = np.zeros((10, 10), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[true_label, pred_label] += 1

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(np.arange(10), CIFAR10_CLASSES, rotation=45, ha="right")
    ax.set_yticks(np.arange(10), CIFAR10_CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_first_layer_filters(model: CifarConvNet, path: Path) -> None:
    """可视化第一层卷积核，观察模型最早学到的颜色/边缘模式。"""
    first_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            first_conv = module
            break
    if first_conv is None:
        return

    filters = first_conv.weight.detach().cpu()
    grid = make_grid(filters[: min(32, filters.size(0))], nrow=8, normalize=True, scale_each=True)
    image = np.transpose(grid.numpy(), (1, 2, 0))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.imshow(image)
    ax.axis("off")
    ax.set_title("First-layer Convolution Filters")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_single_experiment(
    config: ExperimentConfig,
    args: argparse.Namespace,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
) -> dict[str, float | str]:
    run_dir = Path(args.output_dir) / config.name
    run_dir.mkdir(parents=True, exist_ok=True)

    set_random_seeds(args.seed)
    model = CifarConvNet(
        base_width=config.base_width,
        activation=config.activation,
        dropout=config.dropout,
    ).to(device)
    criterion = build_criterion(config.loss_name)
    optimizer = build_optimizer(config, model)

    best_val_acc = -1.0
    best_epoch = 0
    best_model_path = run_dir / "best_model.pt"
    history: list[dict[str, float]] = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, args.max_batches
        )
        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, device, args.eval_max_batches
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(row)
        print(
            f"[{config.name}] epoch {epoch:02d}/{args.epochs} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": asdict(config),
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "param_count": count_parameters(model),
                },
                best_model_path,
            )

    elapsed_sec = time.time() - start_time
    save_history_csv(history, run_dir / "history.csv")
    plot_curves(history, run_dir / "training_curves.png", config.name)

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_acc, y_true, y_pred = evaluate(
        model,
        test_loader,
        criterion,
        device,
        args.test_max_batches,
        collect_predictions=True,
    )

    plot_confusion_matrix(y_true, y_pred, run_dir / "confusion_matrix.png")
    plot_first_layer_filters(model, run_dir / "first_layer_filters.png")

    result = {
        "name": config.name,
        "base_width": config.base_width,
        "activation": config.activation,
        "loss_name": config.loss_name,
        "optimizer_name": config.optimizer_name,
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "param_count": count_parameters(model),
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_error": 1.0 - test_acc,
        "elapsed_sec": elapsed_sec,
        "model_path": str(best_model_path),
    }
    with (run_dir / "result.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def save_summary(summary: list[dict[str, float | str]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(summary[0].keys())
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)

    best = max(summary, key=lambda row: float(row["best_val_acc"]))
    best_src = Path(str(best["model_path"]))
    if best_src.exists():
        shutil.copy2(best_src, output_dir / "best_model.pt")
    with (output_dir / "best_result.json").open("w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)
    plot_summary(summary, output_dir / "experiment_summary.png")


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Train CifarConvNet on CIFAR-10.")
    parser.add_argument("--run", choices=["all", "single", "smoke"], default="all")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--n-items", type=int, default=-1, help="只使用部分训练集；-1 表示使用全部。")
    parser.add_argument("--test-items", type=int, default=-1, help="只使用部分测试集；-1 表示使用全部。")
    parser.add_argument("--max-batches", type=int, default=0, help="每个 epoch 最多训练 batch 数；0 表示不限。")
    parser.add_argument("--eval-max-batches", type=int, default=0)
    parser.add_argument("--test-max-batches", type=int, default=0)
    parser.add_argument("--data-root", type=Path, default=root / "codes" / "VGG_BatchNorm" / "data")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs" / "task1")

    # single 模式用于临时试验一个自定义配置。
    parser.add_argument("--base-width", type=int, default=32)
    parser.add_argument("--activation", choices=["relu", "silu"], default="relu")
    parser.add_argument("--loss-name", choices=["ce", "label_smoothing"], default="ce")
    parser.add_argument("--optimizer-name", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run == "smoke":
        args.epochs = min(args.epochs, 1)
        args.n_items = 512 if args.n_items < 0 else args.n_items
        args.test_items = 256 if args.test_items < 0 else args.test_items
        args.max_batches = 2 if args.max_batches == 0 else args.max_batches
        args.eval_max_batches = 2 if args.eval_max_batches == 0 else args.eval_max_batches
        args.test_max_batches = 2 if args.test_max_batches == 0 else args.test_max_batches

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    set_random_seeds(args.seed)

    train_loader, val_loader, test_loader = get_loaders(
        args.data_root,
        args.batch_size,
        args.val_ratio,
        args.num_workers,
        args.seed,
        args.n_items,
        args.test_items,
    )

    if args.run == "single":
        configs = [
            ExperimentConfig(
                "single",
                args.base_width,
                args.activation,
                args.loss_name,
                args.optimizer_name,
                args.lr,
                args.weight_decay,
            )
        ]
    elif args.run == "smoke":
        configs = [experiment_suite()[0]]
    else:
        configs = experiment_suite()

    summary = [
        run_single_experiment(config, args, device, train_loader, val_loader, test_loader)
        for config in configs
    ]
    save_summary(summary, Path(args.output_dir))
    print(f"Done. Results saved to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
