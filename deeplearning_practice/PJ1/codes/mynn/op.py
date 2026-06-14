from abc import abstractmethod
import numpy as np

def kaiming_init(size):
    """
    He/Kaiming 初始化，适合 ReLU 网络。

    原始 starter code 默认用 np.random.normal，也就是标准差约为 1。
    对卷积层和较宽的 Linear 层来说，这个尺度太大，容易让 logits 被某个类别的随机偏置主导，
    CNN 就可能出现“几乎全预测同一类”的现象。
    """
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) == 4:
        fan_in = size[1] * size[2] * size[3]
    else:
        fan_in = np.prod(size[:-1])
    return np.random.randn(*size) * np.sqrt(2.0 / fan_in)


def zeros_init(size):
    """偏置通常初始化为 0，比随机大偏置更稳定。"""
    return np.zeros(size)

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
        # training 用来区分“训练模式”和“评估模式”。
        # 例如 Dropout 训练时要随机丢弃神经元，但验证/测试时不能随机丢弃。
        # Linear、ReLU、conv2D 这类层不太受 training 影响，但统一放在父类里更方便。
        self.training = True

    def train(self):
        # 进入训练模式。返回 self 是为了允许 layer.train() 之后继续链式使用。
        self.training = True
        return self

    def eval(self):
        # 进入评估模式。验证集和测试集前向传播时应该使用 eval 模式。
        self.training = False
        return self
    
    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=kaiming_init, bias_initialize_method=zeros_init, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = bias_initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        # Part A 新手提示：
        # 1. 这一层做的是线性变换，数学形式是 Y = X @ W + b。
        #    X 的形状是 [batch_size, in_dim]，
        #    W 的形状是 [in_dim, out_dim]，
        #    b 的形状是 [1, out_dim]，会自动广播到每个样本。
        # 2. 反向传播时需要用到本次前向传播的输入，所以这里要把 X 存到 self.input。
        # 3. 建议使用 self.params['W'] 和 self.params['b'] 参与计算，
        #    因为优化器更新的是 self.params 里的参数。
        # 4. 写完后先检查输出形状是否是 [batch_size, out_dim]。
        self.input = X
        output = X @ self.params['W'] + self.params['b']
        return output

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        # Part A 新手提示：
        # 这里传入的 grad 是后一层传回来的 dL/dY，形状是 [batch_size, out_dim]。
        # 你需要同时完成三件事：
        # 1. 计算参数 W 的梯度：dW = X.T @ grad，形状应为 [in_dim, out_dim]。
        # 2. 计算参数 b 的梯度：db = grad 在 batch 维度求和，
        #    形状建议保持为 [1, out_dim]，方便和 b 对齐。
        # 3. 计算传给前一层的梯度：dX = grad @ W.T，
        #    返回值形状应为 [batch_size, in_dim]。本层的in_dim是前一层的out_dim
        # 把 dW 和 db 分别存入 self.grads['W']、self.grads['b']。
        # 如果做形状检查，最常见的错误一般出在转置或求和维度。
        self.grads['W'] = self.input.T @ grad
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)
        dX = grad @ self.params['W'].T
        return dX
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=kaiming_init, bias_initialize_method=zeros_init, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        # Part B 新手提示：
        # 这里要保存卷积层的基本超参数：
        # in_channels、out_channels、kernel_size、stride、padding。
        # 权重 W 的形状使用 [out_channels, in_channels, kernel_size, kernel_size]。
        # 偏置 b 的形状使用 [1, out_channels, 1, 1]，这样可以广播到输出特征图。
        # 也要准备：
        # - self.params = {'W': ..., 'b': ...}
        # - self.grads = {'W': None, 'b': None}
        # - self.input，用来缓存 forward 的输入，给 backward 使用。
        # - self.optimizable 保持为 True，因为卷积层有可训练参数。
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.params = {'W': initialize_method(size=(out_channels, in_channels, kernel_size, kernel_size)), 'b': bias_initialize_method(size=(1, out_channels, 1, 1))}
        # 这里b的形状是size=(1, out_channels, 1, 1)，是因为每个输出通道对应一个偏置值，这样在卷积计算时可以自动广播到每个空间位置。
        self.grads = {'W': None, 'b': None}
        self.input = None
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda
        # 以上这些参数都用self.方法存起来是因为后面forward和backward都要用到。不然forward核backward都要重新定义这些参数就麻烦了。


    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [1, out, in, k, k]
        no padding
        """
        # Part B 新手提示：
        # 1. 先缓存输入 X，因为 backward 需要知道每个卷积窗口来自哪里。
        # 2. 如果 padding > 0，需要先在 H、W 两个维度周围补 0。
        # 3. 输出高宽的公式：
        #    new_H = (H + 2 * padding - kernel_size) // stride + 1
        #    new_W = (W + 2 * padding - kernel_size) // stride + 1
        # 4. 最直接、最适合初学者的写法是四层循环：
        #    遍历 batch -> 遍历 out_channel -> 遍历 new_H -> 遍历 new_W。
        # 5. 对每个输出位置，取输入中的一个 patch：
        #    patch 形状是 [in_channels, kernel_size, kernel_size]。
        #    然后计算 sum(patch * 当前输出通道对应的卷积核) + bias。
        # 6. 先保证结果正确，不用急着优化速度。
        self.input = X
        batch_size, _, H, W = X.shape
        if self.padding > 0:
            # X_padded 形状是 [batch, channels, H + 2*padding, W + 2*padding]，后续计算时用它代替 X
            X_padded = np.pad(X, ((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)), mode='constant', constant_values=0)
        else:
            X_padded = X

        # 速度优化版：
        # 之前的四层 for 循环非常直观，但 full MNIST 上会慢到数小时。
        # 这里仍然是“自己实现卷积”，没有调用深度学习库，只是用 NumPy 把所有 patch 一次性取出来。
        #
        # sliding_window_view 会生成所有 k*k 小窗口：
        # X_windows 形状为 [batch, in_channels, all_H, all_W, kernel_size, kernel_size]。
        # 再用 ::stride 取出真正参与卷积的位置。
        X_windows = np.lib.stride_tricks.sliding_window_view(
            X_padded,
            (self.kernel_size, self.kernel_size),
            axis=(2, 3)
        )
        X_windows = X_windows[:, :, ::self.stride, ::self.stride, :, :]

        # einsum 的含义：
        # b c h w p q 是输入 patch；
        # o c p q 是第 o 个输出通道的卷积核；
        # 它们在 c、p、q 上相乘求和，得到 b o h w。
        output = np.einsum('bchwpq,ocpq->bohw', X_windows, self.params['W'])
        output = output + self.params['b']
        return output

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        # Part B 新手提示：
        # grads 是后一层传来的 dL/dY，形状和 conv2D.forward 的输出一致。
        # 你需要计算三类梯度：
        # 1. dW：每个输出位置的梯度值乘以该位置 forward 时用过的输入 patch，
        #    对 batch 和所有空间位置累加。
        # 2. db：对 grads 在 batch、高、宽三个维度求和，
        #    结果形状建议保持为 [1, out_channels, 1, 1]。
        # 3. dX：把每个输出位置的梯度乘以对应卷积核，再加回该 patch 对应的输入区域。
        # 如果 forward 使用了 padding，backward 可以先计算 dX_padded，
        # 最后再裁剪回原始输入 X 的形状。
        # 写完后重点检查：
        # - self.grads['W'] 的形状是否和 W 一样；
        # - self.grads['b'] 的形状是否和 b 一样；
        # - 返回的 dX 形状是否和原始输入 X 一样。
        batch_size, _, H, W = self.input.shape
        _, _, new_H, new_W = grads.shape
        if self.padding > 0:
            X_padded = np.pad(self.input, ((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)), mode='constant', constant_values=0)
        else:
            X_padded = self.input

        X_windows = np.lib.stride_tricks.sliding_window_view(
            X_padded,
            (self.kernel_size, self.kernel_size),
            axis=(2, 3)
        )
        X_windows = X_windows[:, :, ::self.stride, ::self.stride, :, :]

        # dW 仍然来自“输出梯度 * forward 时用过的 patch”，只是一次性对 batch/h/w 求和。
        dW = np.einsum('bohw,bchwpq->ocpq', grads, X_windows)

        # b 被加到同一个输出通道的所有 batch 和空间位置上，所以对这些维度求和。
        db = np.sum(grads, axis=(0, 2, 3), keepdims=True)

        dX_padded = np.zeros_like(X_padded)
        # dX 的计算比 dW 稍微绕一点：
        # 对卷积核中的每个相对位置 (kh, kw)，把该位置权重乘上输出梯度，
        # 再加回它曾经覆盖过的输入位置。
        # 这里仍保留 kernel_size^2 个小循环；对于 3x3 卷积只有 9 次，速度已经很快。
        for kh in range(self.kernel_size):
            for kw in range(self.kernel_size):
                dX_padded[
                    :,
                    :,
                    kh:kh + self.stride * new_H:self.stride,
                    kw:kw + self.stride * new_W:self.stride
                ] += np.einsum('bohw,oc->bchw', grads, self.params['W'][:, :, kh, kw])

        if self.padding > 0:
            dX = dX_padded[:, :, self.padding:-self.padding, self.padding:-self.padding] # 如果有padding，裁剪回原始输入的形状。切片索引self.padding:-self.padding 从第 padding 个开始，到倒数第 padding 个结束
        else:
            dX = dX_padded
        # 存入本层的self.grads，并返回传给前一层的梯度dX
        self.grads['W'] = dW
        self.grads['b'] = db
        return dX
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.maximum(X, 0)
        # np.where(condition, x, y) 是一个元素级的函数，对于输入数组的每个元素，如果满足 condition 就返回 x，否则返回 y。等价于np.maximum(X, 0)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        # 输入和传入的梯度形状应该一样, assert语句是用来检查条件是否满足的，如果不满足就会抛出一个AssertionError异常。这里检查输入和传入的梯度形状是否一样，是为了确保后续的计算不会因为形状不匹配而出错。
        output = np.where(self.input < 0, 0, grads)
        # ReLU的梯度是一个掩码，当输入小于0时，输出梯度为0；当输入大于等于0时，输出梯度等于传入的grads，用np.where函数很合适。
        # 也可以先计算一个掩码mask = (self.input >= 0).astype(float)，然后output = grads * mask (这里astype加不加都行，因为会隐式把bool当成数值)。
        return output

class Dropout(Layer):
    """
    Dropout regularization layer.
    训练时随机把一部分神经元输出变成 0，用来减轻过拟合；测试时直接原样输出。
    """
    def __init__(self, drop_prob=0.5) -> None:
        super().__init__()
        assert 0 <= drop_prob < 1, 'drop_prob should be in [0, 1).'
        self.drop_prob = drop_prob
        self.mask = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        # 如果不是训练模式，Dropout 不应该随机丢弃任何值。
        # 这样验证集/测试集每次预测才是稳定的。
        if (not self.training) or self.drop_prob == 0:
            self.mask = np.ones_like(X)
            return X

        # inverted dropout 写法：
        # 训练时保留下来的神经元除以 (1 - drop_prob)，保证输出的期望大小基本不变。
        keep_prob = 1 - self.drop_prob
        self.mask = (np.random.rand(*X.shape) < keep_prob) / keep_prob
        return X * self.mask

    def backward(self, grads):
        # forward 时被 mask 置 0 的位置，backward 时梯度也应该是 0。
        # forward 时被保留且放大的位置，backward 也乘同一个 mask。
        assert self.mask is not None, 'Dropout.backward should be called after Dropout.forward.'
        return grads * self.mask

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        # Part A 新手提示：
        # 这个 loss 层本身没有需要优化器更新的参数，所以可以设置 self.optimizable = False。
        # 你需要保存：
        # - self.model：backward 时要调用 model.backward，把梯度继续传回整个网络。
        # - self.max_classes：类别数量，MNIST 一般是 10。
        # - self.has_softmax：默认 True，表示 forward 里先做 softmax。
        # - self.probs / self.labels / self.grads 等缓存变量，方便 backward 使用。
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.probs = None
        self.labels = None
        self.grads = None
        self.optimizable = False

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        # / ---- your codes here ----/
        # Part A 新手提示：
        # predicts 是模型输出的 logits，还不是概率。
        # 1. 如果 self.has_softmax 为 True，先调用下面已经给好的 softmax(predicts)，
        #    得到 probs，形状仍是 [batch_size, D]。
        # 2. labels 是每个样本的真实类别编号，例如 [3, 0, 9, ...]。
        #    你要取出每个样本真实类别对应的概率。
        # 3. 交叉熵公式：
        #    loss = -mean(log(真实类别概率))
        # 4. 为了避免 log(0)，可以在概率里加一个很小的 eps，例如 1e-12。
        # 5. 记得缓存 probs 和 labels，因为 backward 要用。
        # 写完后 loss 应该是一个标量，不是一个数组。
        if self.has_softmax:
            self.probs = softmax(predicts)
        else:
            self.probs = predicts
        batch_size = predicts.shape[0] # batch_size 是样本数量，等于 predicts 的第一维大小。后续计算时需要用它来生成正确类别概率的索引，以及计算平均损失。
        eps = 1e-12
        correct_class_probs = self.probs[np.arange(batch_size), labels]
        # 取出每个样本真实类别对应的概率，np.arange(batch_size)生成一个从0到batch_size-1的数组，作为行索引，labels作为列索引
        # self.probs[np.arange(batch_size), labels]的语法是 NumPy 的高级索引，用来从 probs 数组中取出每个样本对应真实类别的概率值。probs 的形状是 [batch_size, D]，labels 的形状是 [batch_size, ]，其中 labels[i] 是第 i 个样本的真实类别编号。通过 np.arange(batch_size) 生成一个行索引数组 [0, 1, 2, ..., batch_size-1]，配合 labels 作为列索引，就可以得到一个一维数组 correct_class_probs，其中 correct_class_probs[i] 就是第 i 个样本的真实类别对应的概率值。这些概率值后续会用于计算交叉熵损失。
        # self.probs[np.arange(batch_size), labels] 的形状是 [batch_size, ]，每个元素是对应样本真实类别的概率值。后续计算交叉熵损失时需要用到这个数组。
        loss = -np.mean(np.log(correct_class_probs + eps)) # 计算交叉熵损失，先对正确类别的概率加上一个小的eps防止log(0)，然后取负号，最后对所有样本求平均
        self.labels = labels
        return loss
    
    def backward(self):
        # first compute the grads from the loss to the input
        # / ---- your codes here ----/
        # Part A 新手提示：
        # softmax 和 cross entropy 合在一起时，梯度可以写得很简洁：
        # grads = probs - one_hot(labels)
        # grads = grads / batch_size
        # 其中 one_hot(labels) 的形状要和 probs 一样，都是 [batch_size, max_classes]。
        # 最后把这个梯度保存到 self.grads。
        batch_size = self.probs.shape[0]
        one_hot_labels = np.zeros_like(self.probs)
        one_hot_labels[np.arange(batch_size), self.labels] = 1
        self.grads = (self.probs - one_hot_labels) / batch_size

        # 注意：这个函数下面已经调用了 self.model.backward(self.grads)，
        # 所以你只需要先把 self.grads 算正确，整张网络的反向传播会继续发生。
        # Then send the grads to model for back propagation
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    def __init__(self, lambda_ = 1e-8) -> None:
        super().__init__()
        self.lambda_ = lambda_ # L2正则化的强度，控制权重衰减的程度。
        self.optimizable = False # L2正则化层本身没有需要优化器更新的参数，所以设置为 False

    def forward(self, model):
        # 这个类不是训练主流程必须用的，因为 optimizer.SGD 已经支持 weight_decay。
        # 这里保留一个显式 L2 loss，方便写报告或做实验时理解：
        # L2 loss = 0.5 * lambda * sum(W^2)
        # 通常只正则化 W，不正则化 b。
        reg_loss = 0.0
        for layer in model.layers:
            if layer.optimizable and 'W' in layer.params:
                reg_loss += 0.5 * self.lambda_ * np.sum(layer.params['W'] ** 2)
        return reg_loss

    def backward(self, model):
        # 如果你选择“把 L2 当成 loss 的一部分”，就可以调用这个函数，
        # 它会把 lambda * W 加到每个可训练层的 W 梯度上。
        # 注意：如果 optimizer 里已经用了 weight_decay，就不要再调用这里，
        # 否则相当于重复做了两次 L2 正则化。
        for layer in model.layers:
            if layer.optimizable and 'W' in layer.params and layer.grads['W'] is not None:
                layer.grads['W'] += self.lambda_ * layer.params['W']
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max) # 减去 x_max 是为了数值稳定，防止指数函数的输入过大导致溢出。这样做不会改变 softmax 的输出，因为 softmax 是对每个样本的所有类别同时进行归一化的，减去一个常数不会改变相对大小。
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition
