import argparse
import os

import numpy as np

import mynn as nn
from project1_utils import (
    augment_batch,
    confusion_matrix,
    evaluate_model,
    limit_dataset,
    load_mnist,
    plot_cnn_kernels,
    plot_confusion_matrix,
    plot_learning_curve,
    plot_misclassified_examples,
    plot_mlp_first_layer_weights,
    predict_model,
    robustness_suite,
    save_json,
    set_seed,
    train_model,
)


def build_experiments(hidden_dim):
    """
    这里集中定义所有实验。
    这样你想改网络结构或学习率时，只需要改这一处。
    """
    return [
        {
            'name': 'part_a_mlp_baseline',
            'description': 'Part A：MLP baseline，使用 SGD，不加额外技巧。',
            'model_type': 'mlp',
            'model_builder': lambda: nn.models.Model_MLP([784, hidden_dim, 10], 'ReLU'),
            'optimizer_name': 'sgd',
            'lr': 0.06,
            'scheduler_name': None,
            'scheduler_kwargs': {},
            'augment_fn': None,
            'early_stopping_patience': None,
        },
        {
            'name': 'part_b_cnn_baseline',
            'description': 'Part B：CNN baseline，自己实现 conv2D 后训练简单 CNN。',
            'model_type': 'cnn',
            'model_builder': lambda: nn.models.Model_CNN(),
            'optimizer_name': 'sgd',
            'lr': 0.03,
            'scheduler_name': None,
            'scheduler_kwargs': {},
            'augment_fn': None,
            'early_stopping_patience': None,
        },
        {
            'name': 'part_c1_momentum',
            'description': 'Part C Direction 1：Optimization，使用 Momentum 优化器。',
            'model_type': 'mlp',
            'model_builder': lambda: nn.models.Model_MLP([784, hidden_dim, 10], 'ReLU'),
            'optimizer_name': 'momentum',
            'lr': 0.03,
            'scheduler_name': None,
            'scheduler_kwargs': {},
            'augment_fn': None,
            'early_stopping_patience': None,
        },
        {
            'name': 'part_c1_multistep_lr',
            'description': 'Part C Direction 1：Optimization，使用 MultiStepLR 学习率调度。',
            'model_type': 'mlp',
            'model_builder': lambda: nn.models.Model_MLP([784, hidden_dim, 10], 'ReLU'),
            'optimizer_name': 'sgd',
            'lr': 0.06,
            'scheduler_name': 'multistep',
            'scheduler_kwargs': {'milestones': [200, 500, 900], 'gamma': 0.5},
            'augment_fn': None,
            'early_stopping_patience': None,
        },
        {
            'name': 'part_c2_l2_regularization',
            'description': 'Part C Direction 2：Regularization，使用 L2 / weight decay。',
            'model_type': 'mlp',
            'model_builder': lambda: nn.models.Model_MLP([784, hidden_dim, 10], 'ReLU', lambda_list=[1e-4, 1e-4]),
            'optimizer_name': 'sgd',
            'lr': 0.06,
            'scheduler_name': None,
            'scheduler_kwargs': {},
            'augment_fn': None,
            'early_stopping_patience': None,
        },
        {
            'name': 'part_c2_dropout_early_stopping',
            'description': 'Part C Direction 2：Regularization，使用 Dropout，并启用 early stopping。',
            'model_type': 'mlp',
            'model_builder': lambda: nn.models.Model_MLP([784, hidden_dim, 10], 'ReLU', dropout_p=0.2),
            'optimizer_name': 'sgd',
            'lr': 0.06,
            'scheduler_name': None,
            'scheduler_kwargs': {},
            'augment_fn': None,
            'early_stopping_patience': 2,
        },
        {
            'name': 'part_c3_data_augmentation',
            'description': 'Part C Direction 3：Data Augmentation，对训练图片做小旋转和小平移。',
            'model_type': 'cnn',
            'model_builder': lambda: nn.models.Model_CNN(),
            'optimizer_name': 'sgd',
            'lr': 0.03,
            'scheduler_name': None,
            'scheduler_kwargs': {},
            'augment_fn': augment_batch,
            'early_stopping_patience': None,
        },
    ]


def load_best_model(experiment, model_path):
    """
    训练过程中保存的是验证集表现最好的模型。
    这里重新加载它，避免最后一个 epoch 的模型不一定是最优。
    """
    model = experiment['model_builder']()
    model.load_model(model_path)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['quick', 'full'], default='quick')
    parser.add_argument('--epochs', type=int, default=None)
    # 128 对这个 NumPy CNN 来说是一个比较好的折中：比 64 快不少，内存占用仍然很小。
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--seed', type=int, default=309)
    args = parser.parse_args()

    set_seed(args.seed)

    # quick 模式用于快速检查所有代码能不能跑通。
    # full 模式用于真正写报告，默认使用完整训练集和测试集。
    if args.mode == 'quick':
        # quick 模式用于快速 sanity check。
        # 现在 conv2D 已经向量化，1024 张训练图也能较快跑完，
        # 同时比 128 张更能看出 CNN 是否真的在学习。
        train_limit = 1024
        valid_limit = 256
        test_limit = 256
        num_epochs = 3 if args.epochs is None else args.epochs
    else:
        train_limit = None
        valid_limit = None
        test_limit = None
        num_epochs = 5 if args.epochs is None else args.epochs

    output_dirs = {
        # quick 和 full 分目录保存，避免你跑 quick 检查代码时覆盖 full 正式结果。
        'models': os.path.join(r'.\saved_models\project1', args.mode),
        'figs': os.path.join(r'.\figs\project1', args.mode),
        'results': r'.\results',
    }
    for path in output_dirs.values():
        os.makedirs(path, exist_ok=True)

    train_set, valid_set, test_set = load_mnist(seed=args.seed)
    train_set = limit_dataset(train_set, train_limit)
    valid_set = limit_dataset(valid_set, valid_limit)
    test_set = limit_dataset(test_set, test_limit)

    all_results = {
        'note': '出于学习的目的，把 Part C 的 5 个 directions 都实现了，加深理解。',
        'mode': args.mode,
        'num_epochs': num_epochs,
        'batch_size': args.batch_size,
        'experiments': {},
        'robustness': {},
        'error_analysis': {},
    }

    experiments = build_experiments(args.hidden_dim)
    trained_models = {}

    for experiment in experiments:
        name = experiment['name']
        print(f'\n===== Running {name} =====')
        print(experiment['description'])

        model = experiment['model_builder']()
        model_path = os.path.join(output_dirs['models'], f'{name}.pickle')

        history = train_model(
            model=model,
            train_set=train_set,
            valid_set=valid_set,
            model_type=experiment['model_type'],
            optimizer_name=experiment['optimizer_name'],
            lr=experiment['lr'],
            batch_size=args.batch_size,
            num_epochs=num_epochs,
            scheduler_name=experiment['scheduler_name'],
            scheduler_kwargs=experiment['scheduler_kwargs'],
            augment_fn=experiment['augment_fn'],
            early_stopping_patience=experiment['early_stopping_patience'],
            save_path=model_path,
            log_prefix=name,
        )

        best_model = load_best_model(experiment, model_path)
        test_eval = evaluate_model(
            best_model,
            test_set,
            model_type=experiment['model_type'],
            batch_size=args.batch_size,
        )

        all_results['experiments'][name] = {
            'description': experiment['description'],
            'model_type': experiment['model_type'],
            'model_path': model_path,
            'history': history,
            'test': test_eval,
        }
        trained_models[name] = best_model

        plot_learning_curve(
            history,
            os.path.join(output_dirs['figs'], f'{name}_learning_curve.png'),
            title=name,
        )

    # Part C Direction 4：鲁棒性分析。
    # 这里选择 MLP baseline 和 CNN baseline 都做一遍，方便报告比较。
    for name in ['part_a_mlp_baseline', 'part_b_cnn_baseline']:
        model = trained_models[name]
        model_type = all_results['experiments'][name]['model_type']
        all_results['robustness'][name] = robustness_suite(
            model,
            test_set,
            model_type=model_type,
            batch_size=args.batch_size,
        )

    # Part C Direction 5：错误分析与可视化。
    # 主要展示 CNN baseline 的混淆矩阵、错误样本和卷积核，同时也保存 MLP 第一层权重可视化。
    cnn_name = 'part_b_cnn_baseline'
    cnn_model = trained_models[cnn_name]
    cnn_pred = predict_model(cnn_model, test_set[0], model_type='cnn', batch_size=args.batch_size)
    cm = confusion_matrix(test_set[1], cnn_pred, num_classes=10)

    cm_path = os.path.join(output_dirs['figs'], 'cnn_confusion_matrix.png')
    wrong_path = os.path.join(output_dirs['figs'], 'cnn_misclassified_examples.png')
    kernels_path = os.path.join(output_dirs['figs'], 'cnn_first_layer_kernels.png')
    mlp_weights_path = os.path.join(output_dirs['figs'], 'mlp_first_layer_weights.png')

    plot_confusion_matrix(cm, cm_path, title='CNN Confusion Matrix')
    plot_misclassified_examples(test_set[0], test_set[1], cnn_pred, wrong_path)
    plot_cnn_kernels(cnn_model, kernels_path)
    plot_mlp_first_layer_weights(trained_models['part_a_mlp_baseline'], mlp_weights_path)

    all_results['error_analysis'] = {
        'cnn_confusion_matrix': cm.tolist(),
        'cnn_confusion_matrix_path': cm_path,
        'cnn_misclassified_examples_path': wrong_path,
        'cnn_first_layer_kernels_path': kernels_path,
        'mlp_first_layer_weights_path': mlp_weights_path,
    }

    results_path = os.path.join(output_dirs['results'], f'project1_results_{args.mode}.json')
    save_json(all_results, results_path)
    print(f'\nAll results saved to: {results_path}')
    print('Learning curves and visualizations saved to:', output_dirs['figs'])
    print('Best model files saved to:', output_dirs['models'])


if __name__ == '__main__':
    main()
