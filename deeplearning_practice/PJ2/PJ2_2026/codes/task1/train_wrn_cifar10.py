"""
第一题高性能从零训练版本：WideResNet-28-10 on CIFAR-10。

这个脚本不依赖外部预训练权重，适合用来避免“直接使用公开模型”的扣分风险。
默认 recipe：WRN-28-10 + Cutout + label smoothing + SGD/Nesterov + cosine LR。
在 CIFAR-10 上完整训练 160~200 epoch 通常可以达到 95% 左右或以上。
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import time
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
CIFAR_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


class Cutout:
    """随机遮挡一个正方形区域，迫使模型不能只依赖局部纹理。"""

    def __init__(self, length: int = 16):
        self.length = length

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        _, height, width = image.shape
        y = random.randrange(height)
        x = random.randrange(width)
        half = self.length // 2
        y1 = max(0, y - half)
        y2 = min(height, y + half)
        x1 = max(0, x - half)
        x2 = min(width, x + half)
        image[:, y1:y2, x1:x2] = 0.0
        return image


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN.flatten().tolist(), CIFAR_STD.flatten().tolist()),
        Cutout(length=16),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN.flatten().tolist(), CIFAR_STD.flatten().tolist()),
    ])
    return train_transform, eval_transform


def get_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_transform, eval_transform = build_transforms()
    train_aug = datasets.CIFAR10(root=str(args.data_root), train=True, download=True, transform=train_transform)
    train_eval = datasets.CIFAR10(root=str(args.data_root), train=True, download=True, transform=eval_transform)
    test_set = datasets.CIFAR10(root=str(args.data_root), train=False, download=True, transform=eval_transform)

    rng = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(train_aug), generator=rng).tolist()
    if args.n_items > 0:
        indices = indices[:args.n_items]
    val_size = max(1, int(len(indices) * args.val_ratio))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    if args.test_items > 0:
        test_set = Subset(test_set, list(range(min(args.test_items, len(test_set)))))

    kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = True
    return (
        DataLoader(Subset(train_aug, train_indices), shuffle=True, **kwargs),
        DataLoader(Subset(train_eval, val_indices), shuffle=False, **kwargs),
        DataLoader(test_set, shuffle=False, **kwargs),
    )


class WideBasicBlock(nn.Module):
    """WideResNet 的预激活残差块：BN/ReLU 在卷积之前。"""

    def __init__(self, in_planes: int, out_planes: int, stride: int, drop_rate: float):
        super().__init__()
        self.equal_in_out = in_planes == out_planes
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate
        self.shortcut = None if self.equal_in_out else nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.equal_in_out:
            out = self.relu1(self.bn1(x))
            residual = x
        else:
            out = self.relu1(self.bn1(x))
            residual = self.shortcut(out)
        out = self.conv1(out)
        out = self.relu2(self.bn2(out))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return residual + out


class NetworkBlock(nn.Module):
    def __init__(self, layers: int, in_planes: int, out_planes: int, stride: int, drop_rate: float):
        super().__init__()
        blocks = []
        for i in range(layers):
            blocks.append(WideBasicBlock(
                in_planes if i == 0 else out_planes,
                out_planes,
                stride if i == 0 else 1,
                drop_rate,
            ))
        self.block = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class WideResNet(nn.Module):
    """WideResNet-28-10：深度 28，宽度因子 10，约 36M 参数。"""

    def __init__(self, depth: int = 28, widen_factor: int = 10, drop_rate: float = 0.3, num_classes: int = 10):
        super().__init__()
        assert (depth - 4) % 6 == 0, "WideResNet depth should be 6n + 4."
        layers_per_stage = (depth - 4) // 6
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlock(layers_per_stage, channels[0], channels[1], stride=1, drop_rate=drop_rate)
        self.block2 = NetworkBlock(layers_per_stage, channels[1], channels[2], stride=2, drop_rate=drop_rate)
        self.block3 = NetworkBlock(layers_per_stage, channels[2], channels[3], stride=2, drop_rate=drop_rate)
        self.bn = nn.BatchNorm2d(channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(channels[3], num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.relu(self.bn(x))
        x = F.avg_pool2d(x, kernel_size=8)
        return x.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.forward_features(x))


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def update_ema(model: nn.Module, ema_model: nn.Module, decay: float) -> None:
    """维护指数滑动平均权重，评估时通常比最后一步权重更稳。"""
    with torch.no_grad():
        model_state = model.state_dict()
        ema_state = ema_model.state_dict()
        for key, value in ema_state.items():
            source = model_state[key]
            if value.dtype.is_floating_point:
                value.mul_(decay).add_(source, alpha=1.0 - decay)
            else:
                value.copy_(source)


def build_scheduler(optimizer: torch.optim.Optimizer, args: argparse.Namespace):
    def lr_lambda(epoch: int) -> float:
        if epoch < args.warmup_epochs:
            return float(epoch + 1) / max(args.warmup_epochs, 1)
        progress = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_one_epoch(
    model: nn.Module,
    ema_model: nn.Module | None,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for batch_idx, (images, labels) in enumerate(loader):
        if args.max_batches and batch_idx >= args.max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if ema_model is not None:
            update_ema(model, ema_model, args.ema_decay)

        preds = logits.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
    return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    tta: bool,
    max_batches: int,
    collect: bool = False,
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
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(images)
            if tta:
                logits = (logits + model(torch.flip(images, dims=[3]))) / 2.0
            loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        if collect:
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
    return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1), y_true, y_pred


def save_history(history: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def plot_training_curves(history: list[dict[str, float]], path: Path) -> None:
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
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def confusion_matrix(y_true: list[int], y_pred: list[int]) -> np.ndarray:
    matrix = np.zeros((10, 10), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[true_label, pred_label] += 1
    return matrix


def plot_confusion_and_per_class(y_true: list[int], y_pred: list[int], output_dir: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(np.arange(10), CIFAR10_CLASSES, rotation=45, ha="right")
    ax.set_yticks(np.arange(10), CIFAR10_CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("WideResNet Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    per_class = matrix.diagonal() / np.maximum(matrix.sum(axis=1), 1)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(CIFAR10_CLASSES, per_class)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-class Accuracy")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_dir / "per_class_accuracy.png", dpi=180)
    plt.close(fig)


def extract_embeddings(model: WideResNet, loader: DataLoader, device: torch.device, max_items: int) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    features: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    with torch.no_grad():
        seen = 0
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            embeddings = model.forward_features(images).detach().cpu().numpy()
            features.append(embeddings)
            labels_all.append(labels.numpy())
            seen += labels.size(0)
            if seen >= max_items:
                break
    return np.concatenate(features, axis=0)[:max_items], np.concatenate(labels_all, axis=0)[:max_items]


def plot_embedding_pca(features: np.ndarray, labels: np.ndarray, path: Path) -> None:
    centered = features - features.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    points = centered @ vt[:2].T
    fig, ax = plt.subplots(figsize=(7, 6))
    for class_id, class_name in enumerate(CIFAR10_CLASSES):
        mask = labels == class_id
        ax.scatter(points[mask, 0], points[mask, 1], s=7, alpha=0.65, label=class_name)
    ax.set_title("PCA of WideResNet Features")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(markerscale=2, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def denormalize(image: torch.Tensor) -> np.ndarray:
    image = image.cpu() * CIFAR_STD + CIFAR_MEAN
    image = image.clamp(0, 1)
    return np.transpose(image.numpy(), (1, 2, 0))


def gradcam_heatmap(model: WideResNet, image: torch.Tensor, class_id: int, device: torch.device) -> np.ndarray:
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(_module, _inputs, output):
        activations.append(output)

    def backward_hook(_module, _grad_input, grad_output):
        gradients.append(grad_output[0])

    h1 = model.block3.register_forward_hook(forward_hook)
    h2 = model.block3.register_full_backward_hook(backward_hook)
    model.zero_grad(set_to_none=True)
    logits = model(image.unsqueeze(0).to(device))
    logits[0, class_id].backward()
    act = activations[0].detach()
    grad = gradients[0].detach()
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * act).sum(dim=1, keepdim=True).relu()
    cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    h1.remove()
    h2.remove()
    return cam.cpu().numpy()


def plot_gradcam_examples(model: WideResNet, loader: DataLoader, device: torch.device, path: Path, n: int = 8) -> None:
    images, labels = next(iter(loader))
    images = images[:n]
    labels = labels[:n]
    model.eval()
    with torch.no_grad():
        preds = model(images.to(device)).argmax(dim=1).cpu()
    fig, axes = plt.subplots(2, n, figsize=(2.0 * n, 4.0))
    for idx in range(n):
        image = images[idx]
        pred = int(preds[idx])
        heatmap = gradcam_heatmap(model, image, pred, device)
        rgb = denormalize(image)
        overlay = 0.55 * rgb + 0.45 * plt.get_cmap("jet")(heatmap)[..., :3]
        overlay = np.clip(overlay, 0, 1)
        axes[0, idx].imshow(rgb)
        axes[0, idx].axis("off")
        axes[0, idx].set_title(f"T:{CIFAR10_CLASSES[int(labels[idx])]}", fontsize=8)
        axes[1, idx].imshow(overlay)
        axes[1, idx].axis("off")
        axes[1, idx].set_title(f"P:{CIFAR10_CLASSES[pred]}", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Train WideResNet-28-10 on CIFAR-10.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--drop-rate", type=float, default=0.3)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--n-items", type=int, default=-1)
    parser.add_argument("--test-items", type=int, default=-1)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--eval-max-batches", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--data-root", type=Path, default=root / "codes" / "VGG_BatchNorm" / "data")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs" / "task1_wrn")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs = 1
        args.n_items = 512
        args.test_items = 256
        args.max_batches = 2
        args.eval_max_batches = 2
        args.num_workers = 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    set_random_seeds(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, test_loader = get_loaders(args)

    model = WideResNet(drop_rate=args.drop_rate).to(device)
    ema_model = copy.deepcopy(model).to(device) if args.ema_decay > 0 else None
    if ema_model is not None:
        ema_model.eval()
        for param in ema_model.parameters():
            param.requires_grad_(False)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    scheduler = build_scheduler(optimizer, args)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_val_acc = -1.0
    best_epoch = 0
    best_path = output_dir / "best_model.pt"
    history: list[dict[str, float]] = []
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, ema_model, train_loader, criterion, optimizer, scaler, device, args)
        eval_model = ema_model if ema_model is not None else model
        val_loss, val_acc, _, _ = evaluate(eval_model, val_loader, criterion, device, tta=True, max_batches=args.eval_max_batches)
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr,
        })
        print(f"epoch {epoch:03d}/{args.epochs} train_acc={train_acc:.4f} val_acc_tta={val_acc:.4f} lr={current_lr:.5f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                "model_state": eval_model.state_dict(),
                "epoch": epoch,
                "best_val_acc": best_val_acc,
                "param_count": count_parameters(model),
                "args": vars(args),
            }, best_path)

    elapsed_sec = time.time() - start
    eval_model = WideResNet(drop_rate=args.drop_rate).to(device)
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    eval_model.load_state_dict(checkpoint["model_state"])
    test_loss, test_acc, y_true, y_pred = evaluate(eval_model, test_loader, criterion, device, tta=True, max_batches=args.eval_max_batches, collect=True)

    save_history(history, output_dir / "history.csv")
    plot_training_curves(history, output_dir / "training_curves.png")
    plot_confusion_and_per_class(y_true, y_pred, output_dir)
    features, labels = extract_embeddings(eval_model, test_loader, device, max_items=2000)
    plot_embedding_pca(features, labels, output_dir / "embedding_pca.png")
    plot_gradcam_examples(eval_model, test_loader, device, output_dir / "gradcam_examples.png")

    result = {
        "model": "WideResNet-28-10",
        "param_count": count_parameters(model),
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc_tta": test_acc,
        "test_error_tta": 1.0 - test_acc,
        "elapsed_sec": elapsed_sec,
        "model_path": str(best_path),
    }
    with (output_dir / "result.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Done. Results saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
