"""
第二题：比较 VGG-A 和 VGG-A_BatchNorm，并绘制 loss landscape。

脚本保留了原模板的核心思路：用不同学习率训练多条轨迹，记录每一步的 loss、
最后分类层梯度和权重，再比较同一步上不同轨迹之间的 loss/gradient 变化。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from data.loaders import get_cifar_loader
from models.vgg import VGG_A, VGG_A_BatchNorm, get_number_of_parameters


def script_root() -> Path:
    return Path(__file__).resolve().parent


def project_root() -> Path:
    return script_root().parents[1]


def set_random_seeds(seed_value: int = 0, device: torch.device | str = "cpu") -> None:
    """固定 Python、NumPy 和 PyTorch 的随机性，降低不同 run 间的偶然差异。"""
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if str(device) != "cpu" and torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@dataclass(frozen=True)
class RunConfig:
    family: str
    lr: float


def get_device() -> torch.device:
    # 本机只有一张 GPU，因此使用 cuda:0；如果没有 CUDA，则自动回退到 CPU。
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_model_factory(family: str):
    if family == "vgg_a":
        return VGG_A
    if family == "vgg_a_bn":
        return VGG_A_BatchNorm
    raise ValueError(f"Unsupported model family: {family}")


def last_classifier_weight(model: nn.Module) -> torch.nn.Parameter:
    """找到最后一个矩阵形状的 weight，一般就是最终分类层权重。

    记录这个权重和它的梯度，可以用较低的内存开销近似观察优化过程变化。
    """
    candidates = [
        param for name, param in model.named_parameters()
        if name.endswith("weight") and param.ndim >= 2
    ]
    if not candidates:
        raise RuntimeError("No matrix-like weight parameter found.")
    return candidates[-1]


@torch.no_grad()
def get_accuracy(model: nn.Module, loader, device: torch.device, max_batches: int = 0) -> float:
    """计算分类准确率；验证/测试时关闭梯度以节省显存和时间。"""
    model.eval()
    correct = 0
    total = 0
    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader, criterion, device: torch.device, max_batches: int = 0) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
    return total_loss / max(total_samples, 1)


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    epochs_n: int,
    max_batches: int = 0,
    eval_max_batches: int = 0,
) -> dict[str, object]:
    """完成一次训练，并记录 loss landscape 所需的逐 step 信息。"""
    model.to(device)
    final_weight = last_classifier_weight(model)

    history: list[dict[str, float]] = []
    step_losses: list[float] = []
    step_grads: list[np.ndarray] = []
    step_weights: list[np.ndarray] = []
    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, epochs_n + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            if max_batches and batch_idx >= max_batches:
                break

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()

            # 在 optimizer.step() 前记录梯度；此时梯度正对应当前 step 的局部线性近似。
            step_losses.append(float(loss.item()))
            step_grads.append(final_weight.grad.detach().cpu().flatten().numpy().astype(np.float32).copy())
            step_weights.append(final_weight.detach().cpu().flatten().numpy().astype(np.float32).copy())

            optimizer.step()
            preds = logits.argmax(dim=1)
            total_loss += loss.item() * images.size(0)
            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)

        train_loss = total_loss / max(total_samples, 1)
        train_acc = total_correct / max(total_samples, 1)
        val_loss = evaluate_loss(model, val_loader, criterion, device, eval_max_batches)
        val_acc = get_accuracy(model, val_loader, device, eval_max_batches)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })
        print(
            f"epoch {epoch:02d}/{epochs_n} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    return {
        "history": history,
        "losses": step_losses,
        "grads": step_grads,
        "weights": step_weights,
        "best_val_acc": best_val_acc,
        "best_state": best_state,
    }


def save_history_csv(history: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def plot_train_curves(results: dict[str, dict[str, object]], output_path: Path, lr: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for family, result in results.items():
        history = result["history"]
        epochs = [row["epoch"] for row in history]
        axes[0].plot(epochs, [row["train_loss"] for row in history], label=f"{family} train")
        axes[0].plot(epochs, [row["val_loss"] for row in history], linestyle="--", label=f"{family} val")
        axes[1].plot(epochs, [row["train_acc"] for row in history], label=f"{family} train")
        axes[1].plot(epochs, [row["val_acc"] for row in history], linestyle="--", label=f"{family} val")

    axes[0].set_title(f"Loss Curves (lr={lr:g})")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].set_title(f"Accuracy Curves (lr={lr:g})")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def aligned_loss_envelope(loss_runs: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    min_len = min(len(run) for run in loss_runs)
    aligned = np.stack([np.asarray(run[:min_len], dtype=np.float32) for run in loss_runs], axis=0)
    return aligned.min(axis=0), aligned.max(axis=0)


def gradient_change_curve(grad_runs: list[list[np.ndarray]]) -> np.ndarray:
    """计算相邻 step 梯度变化的平均曲线：||g_t - g_{t-1}||_2。"""
    per_run = []
    for grads in grad_runs:
        if len(grads) < 2:
            continue
        diffs = [
            np.linalg.norm(grads[i] - grads[i - 1])
            for i in range(1, len(grads))
        ]
        per_run.append(np.asarray(diffs, dtype=np.float32))
    min_len = min(len(run) for run in per_run)
    return np.stack([run[:min_len] for run in per_run], axis=0).mean(axis=0)


def gradient_lipschitz_curve(
    grad_runs: list[list[np.ndarray]],
    weight_runs: list[list[np.ndarray]],
    eps: float = 1e-12,
) -> np.ndarray:
    """近似最大梯度差/权重距离，用来衡量 landscape 的局部陡峭程度。"""
    min_len = min(len(run) for run in grad_runs)
    values: list[float] = []
    for step in range(min_len):
        max_ratio = 0.0
        for i in range(len(grad_runs)):
            for j in range(i + 1, len(grad_runs)):
                grad_dist = np.linalg.norm(grad_runs[i][step] - grad_runs[j][step])
                weight_dist = np.linalg.norm(weight_runs[i][step] - weight_runs[j][step])
                max_ratio = max(max_ratio, float(grad_dist / (weight_dist + eps)))
        values.append(max_ratio)
    return np.asarray(values, dtype=np.float32)


def plot_loss_landscape(family_runs: dict[str, list[dict[str, object]]], output_path: Path) -> None:
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"vgg_a": "#55a868", "vgg_a_bn": "#c44e52"}
    labels = {"vgg_a": "Standard VGG", "vgg_a_bn": "Standard VGG + BatchNorm"}

    for family, runs in family_runs.items():
        min_curve, max_curve = aligned_loss_envelope([run["losses"] for run in runs])
        steps = np.arange(len(min_curve))
        # 参考图强调同一模型族在不同学习率下的 loss envelope：
        # 填充区域是 [min loss, max loss]，边界线显示包络上下沿。
        ax.fill_between(steps, min_curve, max_curve, color=colors[family], alpha=0.45, label=labels[family])
        ax.plot(steps, min_curve, color=colors[family], linewidth=1.2)
        ax.plot(steps, max_curve, color=colors[family], linewidth=1.2)

    ax.set_title("Loss Landscape")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Loss Landscape")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_gradient_metrics(family_runs: dict[str, list[dict[str, object]]], output_dir: Path) -> None:
    labels = {"vgg_a": "VGG-A", "vgg_a_bn": "VGG-A + BN"}
    colors = {"vgg_a": "#d55e00", "vgg_a_bn": "#0072b2"}

    fig, ax = plt.subplots(figsize=(10, 5))
    for family, runs in family_runs.items():
        curve = gradient_change_curve([run["grads"] for run in runs])
        ax.plot(np.arange(len(curve)), curve, label=labels[family], color=colors[family])
    ax.set_title("Gradient Change")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean ||g_t - g_{t-1}||_2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "gradient_change.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for family, runs in family_runs.items():
        curve = gradient_lipschitz_curve(
            [run["grads"] for run in runs],
            [run["weights"] for run in runs],
        )
        ax.plot(np.arange(len(curve)), curve, label=labels[family], color=colors[family])
    ax.set_title("Approximate Gradient Lipschitz Curve")
    ax.set_xlabel("Training step")
    ax.set_ylabel("max ||g_i - g_j|| / ||w_i - w_j||")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "gradient_lipschitz.png", dpi=180)
    plt.close(fig)


def run_one(config: RunConfig, args: argparse.Namespace, device: torch.device, train_loader, val_loader) -> dict[str, object]:
    set_random_seeds(args.seed, device)
    model = get_model_factory(config.family)()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()
    print(f"\nTraining {config.family} with lr={config.lr:g}; params={get_number_of_parameters(model)}")
    start = time.time()
    result = train(
        model,
        optimizer,
        criterion,
        train_loader,
        val_loader,
        device,
        args.epochs,
        args.max_batches,
        args.eval_max_batches,
    )
    result["elapsed_sec"] = time.time() - start
    result["param_count"] = get_number_of_parameters(model)
    result["family"] = config.family
    result["lr"] = config.lr
    result["final_state"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    return result


def save_run_artifacts(run: dict[str, object], output_dir: Path) -> None:
    run_dir = output_dir / str(run["family"]) / f"lr_{float(run['lr']):g}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_history_csv(run["history"], run_dir / "history.csv")
    torch.save(
        {
            "model_state": run["best_state"],
            "family": run["family"],
            "lr": run["lr"],
            "param_count": run["param_count"],
            "best_val_acc": run["best_val_acc"],
        },
        run_dir / "best_model.pt",
    )
    with (run_dir / "result.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "family": run["family"],
                "lr": run["lr"],
                "param_count": run["param_count"],
                "best_val_acc": run["best_val_acc"],
                "elapsed_sec": run["elapsed_sec"],
                "steps": len(run["losses"]),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    np.savetxt(run_dir / "losses.txt", np.asarray(run["losses"], dtype=np.float32))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VGG-A BatchNorm comparison and loss landscape.")
    parser.add_argument("--mode", choices=["all", "smoke"], default="all")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-items", type=int, default=-1, help="默认使用全量训练集；batch=64, epoch=10 时约 7820 steps，接近参考图。")
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--eval-max-batches", type=int, default=0)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[1e-4, 5e-4, 1e-3, 2e-3])
    parser.add_argument("--compare-lr", type=float, default=1e-3)
    parser.add_argument("--data-root", type=Path, default=script_root() / "data")
    parser.add_argument("--output-dir", type=Path, default=project_root() / "outputs" / "task2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "smoke":
        args.epochs = min(args.epochs, 1)
        args.n_items = 512 if args.n_items < 0 or args.n_items > 512 else args.n_items
        args.max_batches = 2 if args.max_batches == 0 else args.max_batches
        args.eval_max_batches = 2 if args.eval_max_batches == 0 else args.eval_max_batches
        args.learning_rates = [1e-3, 2e-3]

    device = get_device()
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    set_random_seeds(args.seed, device)

    train_loader = get_cifar_loader(
        root=str(args.data_root),
        batch_size=args.batch_size,
        train=True,
        shuffle=True,
        num_workers=args.num_workers,
        n_items=args.n_items,
    )
    val_loader = get_cifar_loader(
        root=str(args.data_root),
        batch_size=args.batch_size,
        train=False,
        shuffle=False,
        num_workers=args.num_workers,
        n_items=1000 if args.mode == "smoke" else -1,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    family_runs: dict[str, list[dict[str, object]]] = {"vgg_a": [], "vgg_a_bn": []}
    for family in family_runs:
        for lr in args.learning_rates:
            run = run_one(RunConfig(family, lr), args, device, train_loader, val_loader)
            family_runs[family].append(run)
            save_run_artifacts(run, output_dir)

    # 使用 compare_lr 对应的 run 画常规训练曲线；如果列表里没有精确值，就取最接近的一条。
    compare_results = {}
    for family, runs in family_runs.items():
        compare_results[family] = min(runs, key=lambda run: abs(float(run["lr"]) - args.compare_lr))
    plot_train_curves(compare_results, output_dir / "vgg_bn_training_curves.png", args.compare_lr)
    plot_loss_landscape(family_runs, output_dir / "loss_landscape.png")
    plot_gradient_metrics(family_runs, output_dir)

    summary_rows = []
    for family, runs in family_runs.items():
        for run in runs:
            summary_rows.append({
                "family": family,
                "lr": run["lr"],
                "param_count": run["param_count"],
                "best_val_acc": run["best_val_acc"],
                "elapsed_sec": run["elapsed_sec"],
                "steps": len(run["losses"]),
            })
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Done. Results saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
