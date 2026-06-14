"""
第一题高准确率版本：用 ImageNet 预训练 ResNet 在 CIFAR-10 上微调。

原来的 CifarConvNet 适合展示从零搭建网络和做消融实验，但要稳定冲到
95%+，通常需要更强的骨干网络、数据增强和更成熟的训练 recipe。这个脚本
保留中文注释，方便通过代码学习每一步为什么这样做。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 高准确率训练更看重速度；benchmark 会为固定输入大小选择较快卷积算法。
        torch.backends.cudnn.benchmark = True


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_transforms(image_size: int, cifar_stem: bool) -> tuple[transforms.Compose, transforms.Compose]:
    """构造训练/评估 transforms。

    预训练 ResNet 原本学习的是 ImageNet 224x224 图片。CIFAR-10 只有 32x32，
    因此这里把图片放大后再做随机裁剪、翻转、AutoAugment 和 RandomErasing。
    这些增强能显著降低过拟合，是 CIFAR-10 高准确率 recipe 的关键部分。
    """
    if cifar_stem:
        # CIFAR stem 直接吃 32x32 输入，避免把小图放大到 160/224 后造成巨大开销。
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN.flatten().tolist(), std=IMAGENET_STD.flatten().tolist()),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.18), ratio=(0.3, 3.3)),
        ])
        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN.flatten().tolist(), std=IMAGENET_STD.flatten().tolist()),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(image_size + 32),
            transforms.RandomResizedCrop(image_size, scale=(0.72, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN.flatten().tolist(), std=IMAGENET_STD.flatten().tolist()),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.18), ratio=(0.3, 3.3)),
        ])
        eval_transform = transforms.Compose([
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN.flatten().tolist(), std=IMAGENET_STD.flatten().tolist()),
        ])
    return train_transform, eval_transform


def get_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_transform, eval_transform = build_transforms(args.image_size, args.cifar_stem)
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

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(Subset(train_aug, train_indices), shuffle=True, **loader_kwargs)
    val_loader = DataLoader(Subset(train_eval, val_indices), shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def build_model(model_name: str, pretrained: bool, cifar_stem: bool) -> nn.Module:
    """加载预训练骨干并替换最后分类层。

    这不是把公开模型原样拿来用：原来的 1000 类 ImageNet 分类头会被替换成
    10 类 CIFAR-10 分类头，并且整个网络随后都会在 CIFAR-10 上微调。
    """
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, 10),
    )
    if cifar_stem:
        # ImageNet ResNet 的 7x7 stride=2 + maxpool 会过早压缩 32x32 小图。
        # CIFAR 常用做法是改成 3x3 stride=1，并移除 maxpool。
        old_weight = model.conv1.weight.detach()
        new_conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        if pretrained:
            # 用原 7x7 卷积核的中心 3x3 初始化新 stem，尽量保留低层边缘/颜色特征。
            with torch.no_grad():
                new_conv.weight.copy_(old_weight[:, :, 2:5, 2:5])
        model.conv1 = new_conv
        model.maxpool = nn.Identity()
    return model


def make_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    """分类头用更大学习率，预训练骨干用较小学习率，避免破坏已有特征。"""
    head_params = list(model.fc.parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.backbone_lr},
            {"params": head_params, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
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
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

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
    """评估模型；TTA 会把原图和水平翻转图的 logits 平均。"""
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

        pred = logits.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (pred == labels).sum().item()
        total_samples += labels.size(0)
        if collect:
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())

    return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1), y_true, y_pred


def save_history(history: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def plot_training_curves(history: list[dict[str, float]], path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["train_acc"] for row in history], label="train")
    axes[1].plot(epochs, [row["val_acc"] for row in history], label="val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_title("Accuracy")
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
    ax.set_title("High-Acc Model Confusion Matrix")
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


def extract_embeddings(model: nn.Module, loader: DataLoader, device: torch.device, max_items: int) -> tuple[np.ndarray, np.ndarray]:
    """提取最后全连接层之前的特征，用 PCA 观察类别是否分开。"""
    model.eval()
    embeddings: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []

    def hook(_module, _inputs, output):
        embeddings.append(output.flatten(1).detach().cpu().numpy())

    handle = model.avgpool.register_forward_hook(hook)
    with torch.no_grad():
        seen = 0
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            _ = model(images)
            labels_all.append(labels.numpy())
            seen += labels.size(0)
            if seen >= max_items:
                break
    handle.remove()
    features = np.concatenate(embeddings, axis=0)[:max_items]
    labels_np = np.concatenate(labels_all, axis=0)[:max_items]
    return features, labels_np


def plot_embedding_pca(features: np.ndarray, labels: np.ndarray, path: Path) -> None:
    centered = features - features.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    points = centered @ vt[:2].T

    fig, ax = plt.subplots(figsize=(7, 6))
    for class_id, class_name in enumerate(CIFAR10_CLASSES):
        mask = labels == class_id
        ax.scatter(points[mask, 0], points[mask, 1], s=7, alpha=0.65, label=class_name)
    ax.set_title("PCA of Penultimate Features")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(markerscale=2, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def denormalize(image: torch.Tensor) -> np.ndarray:
    image = image.cpu() * IMAGENET_STD + IMAGENET_MEAN
    image = image.clamp(0, 1)
    return np.transpose(image.numpy(), (1, 2, 0))


def gradcam_heatmap(model: nn.Module, image: torch.Tensor, class_id: int, device: torch.device) -> np.ndarray:
    """对 ResNet layer4 做 Grad-CAM，观察模型判别时关注哪些区域。"""
    model.eval()
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(_module, _inputs, output):
        activations.append(output)

    def backward_hook(_module, _grad_input, grad_output):
        gradients.append(grad_output[0])

    h1 = model.layer4.register_forward_hook(forward_hook)
    h2 = model.layer4.register_full_backward_hook(backward_hook)
    model.zero_grad(set_to_none=True)
    logits = model(image.unsqueeze(0).to(device))
    logits[0, class_id].backward()

    act = activations[0].detach()
    grad = gradients[0].detach()
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * act).sum(dim=1, keepdim=True).relu()
    cam = torch.nn.functional.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    h1.remove()
    h2.remove()
    return cam.cpu().numpy()


def plot_gradcam_examples(model: nn.Module, loader: DataLoader, device: torch.device, path: Path, n: int = 8) -> None:
    images, labels = next(iter(loader))
    images = images[:n]
    labels = labels[:n]
    model.eval()
    with torch.no_grad():
        preds = model(images.to(device)).argmax(dim=1).cpu()

    fig, axes = plt.subplots(2, n, figsize=(2.1 * n, 4.2))
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
    parser = argparse.ArgumentParser(description="High accuracy CIFAR-10 fine-tuning.")
    parser.add_argument("--model", choices=["resnet18", "resnet50"], default="resnet18")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cifar-stem", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--backbone-lr", type=float, default=2e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--n-items", type=int, default=-1)
    parser.add_argument("--test-items", type=int, default=-1)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--eval-max-batches", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=root / "codes" / "VGG_BatchNorm" / "data")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs" / "task1_high_acc")
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

    model = build_model(args.model, args.pretrained, args.cifar_stem).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    if args.eval_only:
        checkpoint_path = args.checkpoint or (output_dir / "best_model.pt")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        test_loss, test_acc, y_true, y_pred = evaluate(
            model,
            test_loader,
            criterion,
            device,
            tta=True,
            max_batches=args.eval_max_batches,
            collect=True,
        )
        plot_confusion_and_per_class(y_true, y_pred, output_dir)
        features, labels = extract_embeddings(model, test_loader, device, max_items=2000)
        plot_embedding_pca(features, labels, output_dir / "embedding_pca.png")
        plot_gradcam_examples(model, test_loader, device, output_dir / "gradcam_examples.png")
        result = {
            "model": args.model,
            "pretrained": args.pretrained,
            "cifar_stem": args.cifar_stem,
            "param_count": count_parameters(model),
            "best_epoch": checkpoint.get("epoch"),
            "best_val_acc": checkpoint.get("best_val_acc"),
            "test_loss": test_loss,
            "test_acc_tta": test_acc,
            "test_error_tta": 1.0 - test_acc,
            "model_path": str(checkpoint_path),
            "mode": "eval_only",
        }
        with (output_dir / "result.json").open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"Done. Eval artifacts saved to: {output_dir.resolve()}")
        return

    optimizer = make_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_val_acc = -1.0
    best_epoch = 0
    best_path = output_dir / "best_model.pt"
    history: list[dict[str, float]] = []
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, args.max_batches)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device, tta=True, max_batches=args.eval_max_batches)
        scheduler.step()
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "backbone_lr": optimizer.param_groups[0]["lr"],
            "head_lr": optimizer.param_groups[1]["lr"],
        })
        print(f"epoch {epoch:02d}/{args.epochs} train_acc={train_acc:.4f} val_acc_tta={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                "model_state": model.state_dict(),
                "model": args.model,
                "pretrained": args.pretrained,
                "epoch": epoch,
                "best_val_acc": best_val_acc,
                "param_count": count_parameters(model),
                "args": vars(args),
            }, best_path)

    elapsed_sec = time.time() - start
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device, tta=True, max_batches=args.eval_max_batches, collect=True)

    save_history(history, output_dir / "history.csv")
    plot_training_curves(history, output_dir / "training_curves.png")
    plot_confusion_and_per_class(y_true, y_pred, output_dir)
    features, labels = extract_embeddings(model, test_loader, device, max_items=2000)
    plot_embedding_pca(features, labels, output_dir / "embedding_pca.png")
    plot_gradcam_examples(model, test_loader, device, output_dir / "gradcam_examples.png")

    result = {
        "model": args.model,
        "pretrained": args.pretrained,
        "param_count": count_parameters(model),
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc_tta": test_acc,
        "test_error_tta": 1.0 - test_acc,
        "elapsed_sec": elapsed_sec,
        "model_path": str(best_path),
    }
    with (output_dir / "result.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Done. Results saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
