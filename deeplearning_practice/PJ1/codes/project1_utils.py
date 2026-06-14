import gzip
import json
import os
from struct import unpack

import numpy as np

import mynn as nn


def set_seed(seed=309):
    """固定随机种子，让同一组实验尽量可复现。"""
    np.random.seed(seed)


def ensure_dir(path):
    """如果目录不存在，就创建目录。保存模型、图片、结果表时都会用到。"""
    if path and not os.path.exists(path):
        os.makedirs(path)


def load_mnist(data_dir=r'.\dataset\MNIST', validation_size=10000, seed=309):
    """
    读取项目提供的 MNIST 数据，并划分训练集、验证集、测试集。

    返回的数据默认是展平的：
    - train_X: [N, 784]
    - train_y: [N]
    - valid_X: [10000, 784]
    - valid_y: [10000]
    - test_X : [10000, 784]
    - test_y : [10000]

    CNN 需要 [N, 1, 28, 28] 时，可以再调用 to_cnn_images。
    """
    train_images_path = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
    train_labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
    test_images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
    test_labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')

    with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_X = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)

    with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_y = np.frombuffer(f.read(), dtype=np.uint8)

    with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        test_X = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)

    with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_y = np.frombuffer(f.read(), dtype=np.uint8)

    # 把像素从 [0, 255] 变成 [0, 1]，这样训练更稳定。
    train_X = train_X.astype(np.float64) / 255.0
    test_X = test_X.astype(np.float64) / 255.0

    rng = np.random.default_rng(seed)
    idx = rng.permutation(train_X.shape[0])
    train_X = train_X[idx]
    train_y = train_y[idx]

    valid_X = train_X[:validation_size]
    valid_y = train_y[:validation_size]
    train_X = train_X[validation_size:]
    train_y = train_y[validation_size:]

    return (train_X, train_y), (valid_X, valid_y), (test_X, test_y)


def limit_dataset(dataset, max_samples=None):
    """
    调试时可以只取一小部分数据，避免 CNN 纯 NumPy 卷积训练太慢。
    max_samples=None 表示使用全部数据。
    """
    X, y = dataset
    if max_samples is None:
        return X, y
    return X[:max_samples], y[:max_samples]


def to_cnn_images(X):
    """把 [N, 784] 或 [N, 28, 28] 变成 CNN 需要的 [N, 1, 28, 28]。"""
    if X.ndim == 4:
        return X
    if X.ndim == 3:
        return X[:, None, :, :]
    return X.reshape(X.shape[0], 1, 28, 28)


def to_flat_images(X):
    """把图片展平成 MLP 需要的 [N, 784]。"""
    return X.reshape(X.shape[0], -1)


def prepare_input(X, model_type):
    """根据模型类型整理输入形状。"""
    if model_type == 'cnn':
        return to_cnn_images(X)
    return to_flat_images(X)


def random_translate(X, max_shift=2):
    """
    对图片做随机平移。max_shift=2 表示最多上下左右移动 2 个像素。
    空出来的位置用 0 填充，类似黑色背景。
    """
    images = to_cnn_images(X)
    out = np.zeros_like(images)
    batch, channels, height, width = images.shape

    for i in range(batch):
        dy = np.random.randint(-max_shift, max_shift + 1)
        dx = np.random.randint(-max_shift, max_shift + 1)

        src_y0 = max(0, -dy)
        src_y1 = min(height, height - dy)
        dst_y0 = max(0, dy)
        dst_y1 = min(height, height + dy)

        src_x0 = max(0, -dx)
        src_x1 = min(width, width - dx)
        dst_x0 = max(0, dx)
        dst_x1 = min(width, width + dx)

        out[i, :, dst_y0:dst_y1, dst_x0:dst_x1] = images[i, :, src_y0:src_y1, src_x0:src_x1]

    return out


def rotate_images_nearest(X, max_angle=10):
    """
    用最近邻插值做小角度随机旋转。
    这里没有调用深度学习库或图像增强库，是为了保持“自己实现”的练习目标。
    """
    images = to_cnn_images(X)
    out = np.zeros_like(images)
    batch, channels, height, width = images.shape
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0

    for i in range(batch):
        angle = np.deg2rad(np.random.uniform(-max_angle, max_angle))
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        # 对输出图中的每个位置，反向找到它在原图中的来源位置。
        src_y = cos_a * (yy - cy) + sin_a * (xx - cx) + cy
        src_x = -sin_a * (yy - cy) + cos_a * (xx - cx) + cx
        src_y = np.rint(src_y).astype(int)
        src_x = np.rint(src_x).astype(int)
        valid = (src_y >= 0) & (src_y < height) & (src_x >= 0) & (src_x < width)

        for c in range(channels):
            out[i, c, valid] = images[i, c, src_y[valid], src_x[valid]]

    return out


def add_gaussian_noise(X, sigma=0.2):
    """给图片加高斯噪声，用于鲁棒性分析。"""
    images = to_cnn_images(X)
    noisy = images + np.random.normal(0, sigma, size=images.shape)
    return np.clip(noisy, 0.0, 1.0)


def augment_batch(X):
    """
    Part C Direction 3：数据增强。
    这里组合了小平移和小旋转；训练时只对 train batch 做增强，不对验证/测试集做增强。
    """
    # 这里刻意使用较轻的增强：
    # 如果增强太强，模型在前几个 epoch 会先忙着适应扰动，干净验证集准确率可能反而很低。
    # PDF 也强调 light transformations，所以默认只平移 1 像素、旋转 5 度。
    images = random_translate(X, max_shift=1)
    images = rotate_images_nearest(images, max_angle=5)
    return images


def make_optimizer(name, model, lr, mu=0.9):
    """根据名字创建优化器，方便实验脚本切换 SGD / Momentum。"""
    if name == 'sgd':
        return nn.optimizer.SGD(init_lr=lr, model=model)
    if name == 'momentum':
        return nn.optimizer.MomentGD(init_lr=lr, model=model, mu=mu)
    raise ValueError(f'Unknown optimizer: {name}')


def make_scheduler(name, optimizer, **kwargs):
    """根据名字创建学习率调度器；name=None 表示不使用调度器。"""
    if name is None:
        return None
    if name == 'step':
        return nn.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=kwargs.get('step_size', 200),
            gamma=kwargs.get('gamma', 0.5)
        )
    if name == 'multistep':
        return nn.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=kwargs.get('milestones', [800, 1600, 2400]),
            gamma=kwargs.get('gamma', 0.5)
        )
    if name == 'exponential':
        return nn.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=kwargs.get('gamma', 0.995)
        )
    raise ValueError(f'Unknown scheduler: {name}')


def predict_model(model, X, model_type='mlp', batch_size=256):
    """分 batch 预测，避免一次性把所有数据塞进模型导致内存或时间压力太大。"""
    model.eval()
    preds = []
    for start in range(0, X.shape[0], batch_size):
        batch_X = prepare_input(X[start:start + batch_size], model_type)
        logits = model(batch_X)
        preds.append(np.argmax(logits, axis=1))
    model.train()
    return np.concatenate(preds, axis=0)


def evaluate_model(model, dataset, model_type='mlp', batch_size=256):
    """
    计算一个数据集上的平均 loss 和 accuracy。
    注意：这里只做 forward，不做 backward。
    """
    X, y = dataset
    model.eval()
    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=10)
    total_loss = 0.0
    total_correct = 0
    total_num = 0

    for start in range(0, X.shape[0], batch_size):
        batch_X_raw = X[start:start + batch_size]
        batch_y = y[start:start + batch_size]
        batch_X = prepare_input(batch_X_raw, model_type)
        logits = model(batch_X)
        loss = loss_fn(logits, batch_y)
        pred = np.argmax(logits, axis=1)

        total_loss += loss * batch_y.shape[0]
        total_correct += np.sum(pred == batch_y)
        total_num += batch_y.shape[0]

    model.train()
    return {
        'loss': float(total_loss / total_num),
        'accuracy': float(total_correct / total_num)
    }


def train_model(
    model,
    train_set,
    valid_set,
    model_type='mlp',
    optimizer_name='sgd',
    lr=0.05,
    batch_size=64,
    num_epochs=5,
    scheduler_name=None,
    scheduler_kwargs=None,
    augment_fn=None,
    early_stopping_patience=None,
    save_path=None,
    log_prefix='experiment'
):
    """
    通用训练函数。

    它会完成：
    1. 打乱训练集；
    2. 按 batch 前向传播；
    3. 计算 softmax cross entropy；
    4. 反向传播；
    5. 优化器更新参数；
    6. 每个 epoch 结束后在 train/valid 上评估；
    7. 保存验证集准确率最高的模型。
    """
    scheduler_kwargs = scheduler_kwargs or {}
    optimizer = make_optimizer(optimizer_name, model, lr)
    scheduler = make_scheduler(scheduler_name, optimizer, **scheduler_kwargs)
    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=10)

    X_train, y_train = train_set
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'valid_loss': [],
        'valid_accuracy': [],
        'lr': []
    }

    best_valid_accuracy = -1.0
    bad_epochs = 0

    for epoch in range(num_epochs):
        model.train()
        idx = np.random.permutation(X_train.shape[0])
        X_epoch = X_train[idx]
        y_epoch = y_train[idx]

        batch_losses = []
        for start in range(0, X_epoch.shape[0], batch_size):
            raw_batch_X = X_epoch[start:start + batch_size]
            batch_y = y_epoch[start:start + batch_size]

            if augment_fn is not None:
                raw_batch_X = augment_fn(raw_batch_X)

            batch_X = prepare_input(raw_batch_X, model_type)
            logits = model(batch_X)
            loss = loss_fn(logits, batch_y)
            batch_losses.append(float(loss))
            loss_fn.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

        train_eval = evaluate_model(model, train_set, model_type=model_type, batch_size=batch_size)
        valid_eval = evaluate_model(model, valid_set, model_type=model_type, batch_size=batch_size)

        history['train_loss'].append(train_eval['loss'])
        history['train_accuracy'].append(train_eval['accuracy'])
        history['valid_loss'].append(valid_eval['loss'])
        history['valid_accuracy'].append(valid_eval['accuracy'])
        history['lr'].append(float(optimizer.init_lr))

        print(
            f'[{log_prefix}] epoch {epoch + 1}/{num_epochs} '
            f'train_acc={train_eval["accuracy"]:.4f} valid_acc={valid_eval["accuracy"]:.4f} '
            f'train_loss={train_eval["loss"]:.4f} valid_loss={valid_eval["loss"]:.4f} '
            f'lr={optimizer.init_lr:.6f}'
        )

        if valid_eval['accuracy'] > best_valid_accuracy:
            best_valid_accuracy = valid_eval['accuracy']
            bad_epochs = 0
            if save_path is not None:
                ensure_dir(os.path.dirname(save_path))
                model.save_model(save_path)
        else:
            bad_epochs += 1

        if early_stopping_patience is not None and bad_epochs >= early_stopping_patience:
            print(f'[{log_prefix}] early stopping at epoch {epoch + 1}')
            break

    history['best_valid_accuracy'] = float(best_valid_accuracy)
    return history


def confusion_matrix(y_true, y_pred, num_classes=10):
    """不用 sklearn，自己实现混淆矩阵。行是真实类别，列是预测类别。"""
    mat = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        mat[int(t), int(p)] += 1
    return mat


def robustness_suite(model, dataset, model_type='mlp', batch_size=256):
    """
    Part C Direction 4：鲁棒性分析。
    同一个模型分别在干净测试集、平移、旋转、噪声数据上评估。
    """
    X, y = dataset
    results = {}
    perturbations = {
        'clean': X,
        'translation_shift_2': random_translate(X, max_shift=2),
        'rotation_10deg': rotate_images_nearest(X, max_angle=10),
        'gaussian_noise_sigma_0.2': add_gaussian_noise(X, sigma=0.2)
    }

    for name, perturbed_X in perturbations.items():
        eval_result = evaluate_model(
            model,
            (perturbed_X, y),
            model_type=model_type,
            batch_size=batch_size
        )
        results[name] = eval_result
    return results


def save_json(data, path):
    """把实验结果保存成 JSON，report.md 可以根据它填表。"""
    ensure_dir(os.path.dirname(path))
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def plot_learning_curve(history, save_path, title):
    """保存训练/验证 loss 与 accuracy 曲线。"""
    import matplotlib.pyplot as plt

    ensure_dir(os.path.dirname(save_path))
    epochs = np.arange(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(title)

    axes[0].plot(epochs, history['train_loss'], label='train')
    axes[0].plot(epochs, history['valid_loss'], label='valid')
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('loss')
    axes[0].legend()

    axes[1].plot(epochs, history['train_accuracy'], label='train')
    axes[1].plot(epochs, history['valid_accuracy'], label='valid')
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('accuracy')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(mat, save_path, title='Confusion Matrix'):
    """把混淆矩阵保存成图片。"""
    import matplotlib.pyplot as plt

    ensure_dir(os.path.dirname(save_path))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, cmap='Blues')
    ax.set_title(title)
    ax.set_xlabel('predicted label')
    ax.set_ylabel('true label')
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_misclassified_examples(X, y_true, y_pred, save_path, max_examples=16):
    """保存若干分类错误的样本，帮助做 error analysis。"""
    import matplotlib.pyplot as plt

    ensure_dir(os.path.dirname(save_path))
    wrong_idx = np.where(y_true != y_pred)[0][:max_examples]
    if wrong_idx.size == 0:
        return

    images = to_cnn_images(X[wrong_idx])
    cols = 4
    rows = int(np.ceil(wrong_idx.size / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes = np.array(axes).reshape(-1)

    for ax_idx, ax in enumerate(axes):
        ax.axis('off')
        if ax_idx < wrong_idx.size:
            ax.imshow(images[ax_idx, 0], cmap='gray')
            ax.set_title(f'T:{y_true[wrong_idx[ax_idx]]} P:{y_pred[wrong_idx[ax_idx]]}')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_mlp_first_layer_weights(model, save_path, max_units=16):
    """把 MLP 第一层的若干权重 reshape 成 28x28，可视化它在看什么。"""
    import matplotlib.pyplot as plt

    ensure_dir(os.path.dirname(save_path))
    first_linear = None
    for layer in model.layers:
        if isinstance(layer, nn.op.Linear):
            first_linear = layer
            break
    if first_linear is None:
        return

    W = first_linear.params['W']
    num_units = min(max_units, W.shape[1])
    cols = 4
    rows = int(np.ceil(num_units / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = np.array(axes).reshape(-1)

    for i, ax in enumerate(axes):
        ax.axis('off')
        if i < num_units:
            ax.imshow(W[:, i].reshape(28, 28), cmap='coolwarm')
            ax.set_title(f'unit {i}')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_cnn_kernels(model, save_path):
    """可视化 CNN 第一层卷积核。"""
    import matplotlib.pyplot as plt

    ensure_dir(os.path.dirname(save_path))
    first_conv = None
    for layer in model.layers:
        if isinstance(layer, nn.op.conv2D):
            first_conv = layer
            break
    if first_conv is None:
        return

    W = first_conv.params['W']
    out_channels = W.shape[0]
    cols = 4
    rows = int(np.ceil(out_channels / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = np.array(axes).reshape(-1)

    for i, ax in enumerate(axes):
        ax.axis('off')
        if i < out_channels:
            ax.imshow(W[i, 0], cmap='coolwarm')
            ax.set_title(f'kernel {i}')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
