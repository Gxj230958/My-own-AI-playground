Old Method，用的多重for循环，卷积层、全连接层、bias 都用标准正态随机初始化，标准差接近 1.0，对小 CNN 太大，bias 还会随机偏向某个类别。

===== Running part_a_mlp_baseline =====
Part A：MLP baseline，使用 SGD，不加额外技巧。
[part_a_mlp_baseline] epoch 1/5 train_acc=0.8884 valid_acc=0.8836 train_loss=1.9356 valid_loss=2.1218 lr=0.060000
[part_a_mlp_baseline] epoch 2/5 train_acc=0.9220 valid_acc=0.9086 train_loss=1.1521 valid_loss=1.5024 lr=0.060000
[part_a_mlp_baseline] epoch 3/5 train_acc=0.9354 valid_acc=0.9200 train_loss=0.8592 valid_loss=1.2701 lr=0.060000
[part_a_mlp_baseline] epoch 4/5 train_acc=0.9438 valid_acc=0.9223 train_loss=0.6672 valid_loss=1.1455 lr=0.060000
[part_a_mlp_baseline] epoch 5/5 train_acc=0.9484 valid_acc=0.9256 train_loss=0.5499 valid_loss=1.0574 lr=0.060000

===== Running part_b_cnn_baseline =====
Part B：CNN baseline，自己实现 conv2D 后训练简单 CNN。
[part_b_cnn_baseline] epoch 1/5 train_acc=0.0980 valid_acc=0.0950 train_loss=2.4094 valid_loss=2.4224 lr=0.030000
[part_b_cnn_baseline] epoch 2/5 train_acc=0.0980 valid_acc=0.0948 train_loss=2.4113 valid_loss=2.4198 lr=0.030000
[part_b_cnn_baseline] epoch 3/5 train_acc=0.1002 valid_acc=0.0976 train_loss=2.3130 valid_loss=2.3147 lr=0.030000
[part_b_cnn_baseline] epoch 4/5 train_acc=0.1002 valid_acc=0.0996 train_loss=2.3236 valid_loss=2.3241 lr=0.030000
[part_b_cnn_baseline] epoch 5/5 train_acc=0.1054 valid_acc=0.1041 train_loss=2.3084 valid_loss=2.3120 lr=0.030000

===== Running part_c1_momentum =====
Part C Direction 1：Optimization，使用 Momentum 优化器。
[part_c1_momentum] epoch 1/5 train_acc=0.9161 valid_acc=0.8968 train_loss=0.6552 valid_loss=0.9372 lr=0.030000
[part_c1_momentum] epoch 2/5 train_acc=0.9381 valid_acc=0.9084 train_loss=0.3423 valid_loss=0.6548 lr=0.030000
[part_c1_momentum] epoch 3/5 train_acc=0.9503 valid_acc=0.9190 train_loss=0.2350 valid_loss=0.5472 lr=0.030000
[part_c1_momentum] epoch 4/5 train_acc=0.9563 valid_acc=0.9223 train_loss=0.1800 valid_loss=0.5086 lr=0.030000
[part_c1_momentum] epoch 5/5 train_acc=0.9671 valid_acc=0.9287 train_loss=0.1326 valid_loss=0.4813 lr=0.030000

===== Running part_c1_multistep_lr =====
Part C Direction 1：Optimization，使用 MultiStepLR 学习率调度。
[part_c1_multistep_lr] epoch 1/5 train_acc=0.8833 valid_acc=0.8733 train_loss=2.1798 valid_loss=2.4001 lr=0.015000
[part_c1_multistep_lr] epoch 2/5 train_acc=0.8908 valid_acc=0.8798 train_loss=1.9696 valid_loss=2.2088 lr=0.007500
[part_c1_multistep_lr] epoch 3/5 train_acc=0.8972 valid_acc=0.8844 train_loss=1.7967 valid_loss=2.0750 lr=0.007500
[part_c1_multistep_lr] epoch 4/5 train_acc=0.9021 valid_acc=0.8883 train_loss=1.6784 valid_loss=1.9967 lr=0.007500
[part_c1_multistep_lr] epoch 5/5 train_acc=0.9058 valid_acc=0.8919 train_loss=1.5774 valid_loss=1.8801 lr=0.007500

===== Running part_c2_l2_regularization =====
Part C Direction 2：Regularization，使用 L2 / weight decay。
[part_c2_l2_regularization] epoch 1/5 train_acc=0.8528 valid_acc=0.8495 train_loss=2.6847 valid_loss=2.8125 lr=0.060000
[part_c2_l2_regularization] epoch 2/5 train_acc=0.9161 valid_acc=0.9038 train_loss=1.2534 valid_loss=1.5688 lr=0.060000
[part_c2_l2_regularization] epoch 3/5 train_acc=0.9337 valid_acc=0.9154 train_loss=0.8941 valid_loss=1.3071 lr=0.060000
[part_c2_l2_regularization] epoch 4/5 train_acc=0.9431 valid_acc=0.9204 train_loss=0.6795 valid_loss=1.1367 lr=0.060000
[part_c2_l2_regularization] epoch 5/5 train_acc=0.9317 valid_acc=0.9125 train_loss=0.7439 valid_loss=1.1897 lr=0.060000

===== Running part_c2_dropout_early_stopping =====
Part C Direction 2：Regularization，使用 Dropout，并启用 early stopping。
[part_c2_dropout_early_stopping] epoch 1/5 train_acc=0.9066 valid_acc=0.8962 train_loss=1.4656 valid_loss=1.6956 lr=0.060000
[part_c2_dropout_early_stopping] epoch 2/5 train_acc=0.9217 valid_acc=0.9075 train_loss=0.9385 valid_loss=1.1647 lr=0.060000
[part_c2_dropout_early_stopping] epoch 3/5 train_acc=0.9263 valid_acc=0.9096 train_loss=0.7075 valid_loss=0.9452 lr=0.060000
[part_c2_dropout_early_stopping] epoch 4/5 train_acc=0.9267 valid_acc=0.9071 train_loss=0.5759 valid_loss=0.7921 lr=0.060000
[part_c2_dropout_early_stopping] epoch 5/5 train_acc=0.9301 valid_acc=0.9109 train_loss=0.4770 valid_loss=0.6605 lr=0.060000

===== Running part_c3_data_augmentation =====
Part C Direction 3：Data Augmentation，对训练图片做小旋转和小平移。
[part_c3_data_augmentation] epoch 1/5 train_acc=0.1026 valid_acc=0.1017 train_loss=2.3020 valid_loss=2.3021 lr=0.030000
[part_c3_data_augmentation] epoch 2/5 train_acc=0.0980 valid_acc=0.0949 train_loss=2.3168 valid_loss=2.3166 lr=0.030000
[part_c3_data_augmentation] epoch 3/5 train_acc=0.0980 valid_acc=0.0949 train_loss=2.3031 valid_loss=2.3033 lr=0.030000
[part_c3_data_augmentation] epoch 4/5 train_acc=0.0994 valid_acc=0.0988 train_loss=2.3671 valid_loss=2.3679 lr=0.030000
[part_c3_data_augmentation] epoch 5/5 train_acc=0.0994 valid_acc=0.0988 train_loss=2.3424 valid_loss=2.3436 lr=0.030000