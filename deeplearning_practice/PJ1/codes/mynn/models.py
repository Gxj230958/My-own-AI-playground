from .op import * # 导入 op.py 中定义的各种层和操作，例如 Linear、ReLU 等。
import numpy as np # 这里主要用于 reshape、保存形状等数组操作。
import pickle # 用于保存和加载模型参数的库

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None, dropout_p=0.0):
        super().__init__()
        self.size_list = size_list # e.g. [784, 256, 128, 10]，代表输入层 784 个神经元，第一个隐藏层 256 个神经元，第二个隐藏层 128 个神经元，输出层 10 个神经元。
        self.act_func = act_func # e.g. 'ReLU'，代表隐藏层的激活函数是 ReLU。输出层不需要激活函数。
        self.lambda_list = lambda_list # e.g. [0.01, 0.01, 0.01]，代表每个隐藏层的 L2 正则化系数。
        self.dropout_p = dropout_p # dropout_p > 0 时，会在隐藏层激活函数后加入 Dropout，用来做正则化实验。
        self.layers = []

        if size_list is not None and act_func is not None:
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1]) # 创建线性层，输入维度是 size_list[i]，输出维度是 size_list[i + 1]
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)
                    if dropout_p > 0:
                        # Dropout 只加在隐藏层后面，不加在最后输出层后面。
                        # 输出层要保留完整 logits，交给 softmax + cross entropy 处理。
                        self.layers.append(Dropout(drop_prob=dropout_p))

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs) # 按顺序把输入 X 传给每一层，得到输出 outputs，最后返回 outputs 作为模型的输出。
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers): # 按照反向顺序把 loss_grad 传回每一层，得到每层的梯度 grads，最后返回 grads 作为输入 X 的梯度。
            grads = layer.backward(grads)
        return grads

    def train(self):
        # 进入训练模式：主要影响 Dropout。
        self.training = True
        for layer in self.layers:
            layer.train()
        return self

    def eval(self):
        # 进入评估模式：验证和测试时关闭 Dropout 的随机丢弃。
        self.training = False
        for layer in self.layers:
            layer.eval()
        return self

    def load_model(self, param_list): # 这个方法用于从 pickle 文件中恢复 MLP 参数。
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]
        # 旧版保存格式是 [size_list, act_func, layer_param1, layer_param2, ...]。
        # 中间版本格式是 [size_list, act_func, lambda_list, layer_param1, ...]。
        # 新版格式是 [size_list, act_func, lambda_list, dropout_p, layer_param1, ...]。
        # 这里同时兼容两种格式，避免以前保存的模型读不出来。
        if len(param_list) > 2 and isinstance(param_list[2], dict):
            self.lambda_list = None
            self.dropout_p = 0.0
            param_start = 2
        elif len(param_list) > 3 and not isinstance(param_list[3], dict):
            self.lambda_list = param_list[2]
            self.dropout_p = param_list[3]
            param_start = 4
        else:
            self.lambda_list = param_list[2]
            self.dropout_p = 0.0
            param_start = 3

        self.layers = []
        for i in range(len(self.size_list) - 1):
            layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
            saved_layer = param_list[param_start + i] # 每个可训练层保存一个字典，里面有 W、b、weight_decay、lambda。
            layer.W = saved_layer['W']
            layer.b = saved_layer['b']
            layer.params['W'] = layer.W
            layer.params['b'] = layer.b
            layer.weight_decay = saved_layer['weight_decay']
            layer.weight_decay_lambda = saved_layer['lambda']
            if self.act_func == 'Logistic':
                raise NotImplementedError
            elif self.act_func == 'ReLU':
                layer_f = ReLU()
            self.layers.append(layer)
            if i < len(self.size_list) - 2: # 最后一个线性层后面不需要激活函数，所以只在 i < len(self.size_list) - 2 时添加激活层。
                self.layers.append(layer_f) # self.layers 是一个列表，其中的层顺序应该和 forward 中一致，线性层后面跟着激活层。
                if self.dropout_p > 0:
                    self.layers.append(Dropout(drop_prob=self.dropout_p))
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func, self.lambda_list, self.dropout_p]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f) # 把 param_list 保存到 save_path 文件中，供以后加载模型参数使用。param_list 包含了模型结构信息和每个可训练层的参数信息。
        

class Model_CNN(Layer):
    """
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """
    def __init__(self, input_shape=(1, 28, 28), num_classes=10, lambda_list=None, dropout_p=0.0):
        super().__init__()
        # Part B 新手提示：
        # 这里要像 Model_MLP 一样，创建 self.layers，并把各层按顺序放进去。
        # 一个简单、适合 MNIST 的 CNN 可以先设计成：
        # conv2D(1, 8, kernel_size=3, stride=2, padding=1)
        # ReLU()
        # conv2D(8, 16, kernel_size=3, stride=2, padding=1)
        # ReLU()
        # Linear(16 * 7 * 7, 10)
        #
        # 为什么是 16 * 7 * 7？
        # 输入是 [batch, 1, 28, 28]。
        # 第一次 stride=2 后大约变成 [batch, 8, 14, 14]。
        # 第二次 stride=2 后大约变成 [batch, 16, 7, 7]。
        # 进入 Linear 前，需要把它展平成 [batch, 16 * 7 * 7]。
        #
        # 初学时建议先做这个最小可用版本，不急着加池化、dropout 或更深结构。
        self.input_shape = input_shape # MNIST 灰度图默认是 [通道数, 高, 宽] = [1, 28, 28]。
        self.num_classes = num_classes # MNIST 一共有 10 类数字，所以默认输出维度是 10。
        self.lambda_list = lambda_list # 如果想给可训练层加 weight decay，可以传入每层的正则化强度。
        self.dropout_p = dropout_p # 如果大于 0，会在卷积特征进入 Linear 之前加 Dropout。
        self.flatten_shape = None # forward 里做 Flatten 之前的形状会存在这里，backward 需要用它 reshape 回去。
        self.flatten_layer_index = None # 记录在哪一个 Linear 层前做了 Flatten，反向传播时才能在正确位置还原形状。

        # 下面是一个很小的 CNN：
        # 第 1 个卷积层：输入 1 个通道，输出 8 个通道，stride=2 会把 28x28 变成 14x14。
        # 第 2 个卷积层：输入 8 个通道，输出 16 个通道，stride=2 会把 14x14 变成 7x7。
        # 最后把 [batch, 16, 7, 7] 展平成 [batch, 16*7*7]，接一个 Linear 输出 10 类 logits。
        self.layers = [
            conv2D(in_channels=input_shape[0], out_channels=8, kernel_size=3, stride=2, padding=1),
            ReLU(),
            conv2D(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1),
            ReLU()
        ]
        if dropout_p > 0:
            # 这里的 Dropout 作用在 4 维卷积特征图上，它会随机丢弃一部分特征值。
            # 后面 forward 遇到 Linear 时仍然会先 Flatten，所以维度不会出问题。
            self.layers.append(Dropout(drop_prob=dropout_p))
        self.layers.append(Linear(in_dim=16 * 7 * 7, out_dim=num_classes))

        if lambda_list is not None:
            # self.layers 里只有 conv2D 和 Linear 有参数，ReLU 没有参数。
            # 这里按可训练层的顺序应用 lambda_list：
            # 第 0 个值给第一层 conv2D，第 1 个值给第二层 conv2D，第 2 个值给最后的 Linear。
            optimizable_layers = [layer for layer in self.layers if layer.optimizable]
            for layer, lambda_value in zip(optimizable_layers, lambda_list):
                layer.weight_decay = True
                layer.weight_decay_lambda = lambda_value
         

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        # Part B 新手提示：
        # CNN 的输入应该是 [batch, 1, 28, 28]，不是 MLP 用的 [batch, 784]。
        # 你可以按顺序遍历 self.layers。
        #
        # 需要特别处理 Flatten：
        # - 前面的 conv2D 和 ReLU 都可以直接 layer(outputs)。
        # - 在进入最后一个 Linear 之前，把卷积输出 reshape 成 [batch, -1]。
        # - 同时缓存 reshape 前的形状，例如 self.flatten_shape，
        #   因为 backward 时要把 Linear 传回来的梯度 reshape 回去。
        #
        # 如果你暂时不想写单独的 Flatten 类，可以直接在 Model_CNN.forward 里完成 reshape。
        outputs = X

        # 为了让你调试方便，这里同时支持两种输入：
        # 1. CNN 推荐输入：[batch, 1, 28, 28]。
        # 2. 如果你暂时沿用 test_train.py 里 MLP 的展平输入 [batch, 784]，
        #    这里会自动 reshape 回 [batch, 1, 28, 28]。
        if outputs.ndim == 2:
            expected_dim = np.prod(self.input_shape)
            assert outputs.shape[1] == expected_dim, 'Flattened CNN input must have shape [batch, 1*28*28].'
            outputs = outputs.reshape(outputs.shape[0], *self.input_shape)

        self.flatten_shape = None
        self.flatten_layer_index = None

        for i, layer in enumerate(self.layers):
            # conv2D 和 ReLU 都处理 4 维数据：[batch, channels, H, W]。
            # Linear 只能处理 2 维数据：[batch, feature_dim]。
            # 所以第一次遇到 Linear 且 outputs 还是 4 维时，要先展平。
            if isinstance(layer, Linear) and outputs.ndim > 2:
                self.flatten_shape = outputs.shape
                self.flatten_layer_index = i
                outputs = outputs.reshape(outputs.shape[0], -1)
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        # Part B 新手提示：
        # 反向传播顺序和 forward 相反，和 Model_MLP.backward 的思路一样。
        #
        # 如果 forward 里在 Linear 前做过 Flatten，那么 backward 中：
        # 1. 先让最后的 Linear 接收 loss_grad，返回形状为 [batch, 16 * 7 * 7] 的梯度。
        # 2. 再把这个梯度 reshape 回 self.flatten_shape，例如 [batch, 16, 7, 7]。
        # 3. 然后继续按反向顺序传给 ReLU、conv2D 等层。
        #
        # 最后返回传到输入图片上的梯度即可；训练时通常不直接使用它，但接口要保持完整。
        grads = loss_grad
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            grads = layer.backward(grads)

            # forward 中在某个 Linear 前把 4 维卷积输出展平成了 2 维。
            # backward 时，Linear 返回的梯度也是 2 维，
            # 但前面的 ReLU/conv2D 需要 4 维梯度，所以这里要 reshape 回去。
            if i == self.flatten_layer_index:
                grads = grads.reshape(self.flatten_shape)
        return grads

    def train(self):
        # 进入训练模式：Dropout 会随机丢弃特征。
        self.training = True
        for layer in self.layers:
            layer.train()
        return self

    def eval(self):
        # 进入评估模式：Dropout 原样输出，保证验证/测试结果稳定。
        self.training = False
        for layer in self.layers:
            layer.eval()
        return self
    
    def load_model(self, param_list):
        # Part B 新手提示：
        # 这个方法用于从 pickle 文件中恢复 CNN 参数。
        # 可以参考 Model_MLP.load_model 的写法：
        # 1. 先读取保存的结构信息或参数列表。
        # 2. 重新创建相同结构的 self.layers。
        # 3. 对每个可训练层，把保存的 W、b 放回 layer.params。
        #
        # 初次实现 CNN 时，可以先把 forward/backward/train 跑通，
        # 再回来补 save_model/load_model。
        with open(param_list, 'rb') as f:
            saved_model = pickle.load(f)

        self.input_shape = tuple(saved_model['input_shape'])
        self.num_classes = saved_model['num_classes']
        self.lambda_list = saved_model.get('lambda_list', None)
        self.dropout_p = saved_model.get('dropout_p', 0.0)
        self.flatten_shape = None
        self.flatten_layer_index = None
        self.layers = []

        for layer_info in saved_model['layers']:
            layer_type = layer_info['type']

            if layer_type == 'conv2D':
                layer = conv2D(
                    in_channels=layer_info['in_channels'],
                    out_channels=layer_info['out_channels'],
                    kernel_size=layer_info['kernel_size'],
                    stride=layer_info['stride'],
                    padding=layer_info['padding'],
                    weight_decay=layer_info['weight_decay'],
                    weight_decay_lambda=layer_info['lambda']
                )
                layer.params['W'] = layer_info['W']
                layer.params['b'] = layer_info['b']
            elif layer_type == 'Linear':
                layer = Linear(
                    in_dim=layer_info['in_dim'],
                    out_dim=layer_info['out_dim'],
                    weight_decay=layer_info['weight_decay'],
                    weight_decay_lambda=layer_info['lambda']
                )
                layer.W = layer_info['W']
                layer.b = layer_info['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
            elif layer_type == 'ReLU':
                layer = ReLU()
            elif layer_type == 'Dropout':
                layer = Dropout(drop_prob=layer_info['drop_prob'])
            else:
                raise ValueError(f'Unknown layer type: {layer_type}')

            self.layers.append(layer)
        
    def save_model(self, save_path):
        # Part B 新手提示：
        # 这个方法用于保存 CNN 参数，方便 test_model.py 或报告复现实验。
        # 可以参考 Model_MLP.save_model：
        # 1. 准备一个列表，记录模型结构和每个可训练层的参数。
        # 2. 遍历 self.layers，只保存 layer.optimizable == True 的层。
        # 3. 用 pickle.dump 写入 save_path。
        #
        # 注意不要把数据集或很大的中间结果保存进模型文件。
        saved_model = {
            'model_type': 'Model_CNN',
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'lambda_list': self.lambda_list,
            'dropout_p': self.dropout_p,
            'layers': []
        }

        for layer in self.layers:
            if isinstance(layer, conv2D):
                saved_model['layers'].append({
                    'type': 'conv2D',
                    'in_channels': layer.in_channels,
                    'out_channels': layer.out_channels,
                    'kernel_size': layer.kernel_size,
                    'stride': layer.stride,
                    'padding': layer.padding,
                    'W': layer.params['W'],
                    'b': layer.params['b'],
                    'weight_decay': layer.weight_decay,
                    'lambda': layer.weight_decay_lambda
                })
            elif isinstance(layer, Linear):
                saved_model['layers'].append({
                    'type': 'Linear',
                    'in_dim': layer.params['W'].shape[0],
                    'out_dim': layer.params['W'].shape[1],
                    'W': layer.params['W'],
                    'b': layer.params['b'],
                    'weight_decay': layer.weight_decay,
                    'lambda': layer.weight_decay_lambda
                })
            elif isinstance(layer, ReLU):
                saved_model['layers'].append({'type': 'ReLU'})
            elif isinstance(layer, Dropout):
                saved_model['layers'].append({
                    'type': 'Dropout',
                    'drop_prob': layer.drop_prob
                })

        with open(save_path, 'wb') as f:
            pickle.dump(saved_model, f)
