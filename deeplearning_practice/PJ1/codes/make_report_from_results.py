import argparse
import json
import os


def fmt(x):
    """把数字格式化成适合报告表格的字符串。"""
    if isinstance(x, float):
        return f'{x:.4f}'
    return str(x)


def markdown_table(headers, rows):
    """生成 Markdown 表格。"""
    lines = []
    lines.append('| ' + ' | '.join(headers) + ' |')
    lines.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')
    for row in rows:
        lines.append('| ' + ' | '.join(str(item) for item in row) + ' |')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default=r'.\results\project1_results_full.json')
    parser.add_argument('--output', default='report_generated.md')
    args = parser.parse_args()

    with open(args.results, 'r', encoding='utf-8') as f:
        results = json.load(f)

    experiments = results['experiments']
    mode = results.get('mode', 'full')
    fig_dir = f'figs/project1/{mode}'
    lines = []
    lines.append('# Project 1 Report: MNIST Neural Network and CNN')
    lines.append('')
    lines.append('姓名：请填写')
    lines.append('')
    lines.append('学号：请填写')
    lines.append('')
    lines.append('代码仓库链接：请填写 GitHub 链接')
    lines.append('')
    lines.append('模型权重链接：请填写 ModelScope 或其他网盘链接')
    lines.append('')
    lines.append('> 出于学习的目的，把 Part C 的 5 个 directions 都实现了，加深理解。')
    lines.append('')

    lines.append('## 1. MLP Baseline')
    mlp = experiments['part_a_mlp_baseline']
    lines.append('本节实现了 `Linear.forward`、`Linear.backward` 和 softmax cross entropy，并训练 MLP baseline。')
    lines.append('')
    lines.append(f'- 最佳验证准确率：{fmt(mlp["history"]["best_valid_accuracy"])}')
    lines.append(f'- 测试准确率：{fmt(mlp["test"]["accuracy"])}')
    lines.append(f'- 测试 loss：{fmt(mlp["test"]["loss"])}')
    lines.append('')
    lines.append(f'![MLP learning curve]({fig_dir}/part_a_mlp_baseline_learning_curve.png)')
    lines.append('')

    lines.append('## 2. CNN Model and MLP-vs-CNN Comparison')
    cnn = experiments['part_b_cnn_baseline']
    lines.append('本节实现了 `conv2D` 的前向传播和反向传播，并搭建简单 CNN。')
    lines.append('')
    lines.append(f'- CNN 最佳验证准确率：{fmt(cnn["history"]["best_valid_accuracy"])}')
    lines.append(f'- CNN 测试准确率：{fmt(cnn["test"]["accuracy"])}')
    lines.append(f'- CNN 测试 loss：{fmt(cnn["test"]["loss"])}')
    lines.append('')
    lines.append('CNN 通常比 MLP 更适合图像任务，因为卷积核利用了局部空间结构，并且同一个卷积核在不同位置共享参数。MLP 把图片展平成向量后，空间邻接关系不再显式保留。')
    lines.append('')
    lines.append(f'![CNN learning curve]({fig_dir}/part_b_cnn_baseline_learning_curve.png)')
    lines.append('')

    lines.append('## 3. Part C: Five Additional Directions')
    lines.append('虽然项目只要求选择两个方向，但本项目出于学习目的实现了全部五个方向。')
    lines.append('')
    lines.append('### Direction 1: Optimization')
    lines.append('- `part_c1_momentum`：实现并使用 Momentum optimizer。')
    lines.append('- `part_c1_multistep_lr`：实现并使用 MultiStepLR 学习率调度。')
    lines.append('')
    lines.append('### Direction 2: Regularization')
    lines.append('- `part_c2_l2_regularization`：使用 L2 / weight decay。')
    lines.append('- `part_c2_dropout_early_stopping`：实现 Dropout，并加入 early stopping。')
    lines.append('')
    lines.append('### Direction 3: Data Augmentation')
    lines.append('- `part_c3_data_augmentation`：对训练图片做小平移和小旋转。')
    lines.append('')
    lines.append('### Direction 4: Robustness Analysis')
    lines.append('- 在测试集上评估 clean、translation、rotation、Gaussian noise 等扰动。')
    lines.append('')
    lines.append('### Direction 5: Error Analysis and Visualization')
    lines.append('- 生成混淆矩阵、错误分类样本、MLP 第一层权重、CNN 卷积核可视化。')
    lines.append('')

    lines.append('## 4. Main Results Table')
    rows = []
    for name, exp in experiments.items():
        rows.append([
            name,
            exp['model_type'],
            fmt(exp['history']['best_valid_accuracy']),
            fmt(exp['test']['accuracy']),
            fmt(exp['test']['loss']),
        ])
    lines.append(markdown_table(
        ['Experiment', 'Model', 'Best Valid Acc', 'Test Acc', 'Test Loss'],
        rows
    ))
    lines.append('')

    lines.append('## 5. Robustness Results')
    robust_rows = []
    for model_name, robust_result in results['robustness'].items():
        for perturb_name, metric in robust_result.items():
            robust_rows.append([
                model_name,
                perturb_name,
                fmt(metric['accuracy']),
                fmt(metric['loss']),
            ])
    lines.append(markdown_table(
        ['Model', 'Perturbation', 'Accuracy', 'Loss'],
        robust_rows
    ))
    lines.append('')

    lines.append('## 6. Detailed Visualization')
    ea = results['error_analysis']
    lines.append(f'![CNN confusion matrix]({fig_dir}/cnn_confusion_matrix.png)')
    lines.append('')
    lines.append(f'![CNN misclassified examples]({fig_dir}/cnn_misclassified_examples.png)')
    lines.append('')
    lines.append(f'![CNN first layer kernels]({fig_dir}/cnn_first_layer_kernels.png)')
    lines.append('')
    lines.append(f'![MLP first layer weights]({fig_dir}/mlp_first_layer_weights.png)')
    lines.append('')

    lines.append('## 7. Discussion')
    lines.append('请根据你实际运行得到的表格补充讨论。建议回答：')
    lines.append('')
    lines.append('- CNN 是否比 MLP 更好？如果更好，是因为局部连接、权重共享和空间结构利用。')
    lines.append('- Momentum 或学习率调度是否让训练更稳定或更快？')
    lines.append('- L2、Dropout、early stopping 是否改善验证集表现？')
    lines.append('- 数据增强是否提升了干净测试集或扰动测试集上的表现？')
    lines.append('- 哪些数字最容易混淆？结合混淆矩阵和错误样本说明。')
    lines.append('')

    with open(args.output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f'Report written to: {os.path.abspath(args.output)}')


if __name__ == '__main__':
    main()
