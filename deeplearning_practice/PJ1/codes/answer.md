# op.py 相关问题解释

## 0. 这段 `Layer` 代码整体是什么意思？

你问的是这段：

```python
from abc import abstractmethod

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass
```

我们一点一点拆开。

### `from abc import abstractmethod`

这是 Python 的导入语法。

```python
from abc import abstractmethod
```

意思是：从 Python 自带的 `abc` 模块里，导入 `abstractmethod` 这个工具。

`abc` 是 `Abstract Base Classes` 的缩写，意思是“抽象基类”。  
你可以暂时把它理解成：

```text
专门用来写“父类模板”的工具。
```

在神经网络里，`Layer` 就是一个父类模板。  
它规定：所有层最好都有 `forward` 和 `backward`。

### `class Layer():`

这是定义一个类，名字叫 `Layer`。

类可以理解成“对象的模板”。  
这里的 `Layer` 不是某一个具体层，而是所有层共同的模板。

比如后面的：

```python
class Linear(Layer):
```

意思是：`Linear` 是一种 `Layer`。

```python
class ReLU(Layer):
```

意思是：`ReLU` 也是一种 `Layer`。

这样写的好处是：所有层都可以有统一接口，比如都能调用 `forward` 和 `backward`。

### `def __init__(self) -> None:`

这是定义类的初始化函数。

当你创建一个对象时，比如：

```python
layer = Layer()
```

Python 会自动调用：

```python
__init__
```

这里的：

```python
def __init__(self) -> None:
```

可以分成几部分看：

```python
def
```

表示“定义函数”。

```python
__init__
```

是 Python 规定好的特殊名字，表示“初始化函数”。

```python
self
```

表示“当前这个对象自己”。  
比如你创建了：

```python
layer1 = Layer()
layer2 = Layer()
```

那么 `layer1` 和 `layer2` 各自都有自己的 `self.optimizable`。

```python
-> None
```

是类型提示，意思是：

```text
这个函数不返回东西。
```

它只是提示给人或编辑器看的，不写也能运行：

```python
def __init__(self):
```

也可以。

### `self.optimizable = True`

这行是在给当前对象添加一个属性：

```python
self.optimizable
```

值是：

```python
True
```

在这个项目里，它表示：

```text
这一层有没有需要优化器更新的参数。
```

比如：

- `Linear` 有 `W` 和 `b`，所以 `optimizable = True`；
- `conv2D` 有卷积核和偏置，所以也是 `True`；
- `ReLU` 没有参数，所以会改成 `False`；
- `Loss` 层通常也没有参数，所以也可以设成 `False`。

优化器里会用它判断要不要更新这一层：

```python
if layer.optimizable == True:
    ...
```

### `@abstractmethod`

这一行是“装饰器”语法。

装饰器就是写在函数上面、用来给函数加额外含义的东西。

```python
@abstractmethod
def forward(self):
    pass
```

意思是：

```text
forward 是一个抽象方法。
```

“抽象方法”可以理解成：

```text
父类只规定必须有这个方法，但父类自己不写具体内容，具体内容交给子类实现。
```

也就是说，`Layer` 只是告诉大家：

```text
所有神经网络层都应该有 forward。
所有神经网络层都应该有 backward。
```

但 `Layer` 自己不知道 forward 应该怎么算。  
因为：

- `Linear.forward` 是矩阵乘法；
- `ReLU.forward` 是取 `max(x, 0)`；
- `conv2D.forward` 是卷积。

它们的具体实现都不一样。

所以父类 `Layer` 这里只写一个空壳。

### `def forward(self):`

这是定义一个方法，名字叫 `forward`。

它代表前向传播。

在神经网络里：

```text
forward：输入 X，算输出。
backward：拿到后一层传回来的梯度，继续往前传。
```

父类 `Layer` 里的 `forward` 只是规定“应该有这个函数”。  
真正的实现写在子类里，比如：

```python
class Linear(Layer):
    def forward(self, X):
        ...
```

### `pass`

`pass` 的意思是：

```text
这里暂时什么都不做。
```

Python 不允许函数体完全空着。  
如果你写：

```python
def forward(self):
```

然后下面什么都不写，会语法错误。

所以用：

```python
pass
```

占一个位置，表示“这个地方以后再写”或者“这里故意为空”。

在这个项目里，`Layer.forward` 和 `Layer.backward` 只是模板，所以用 `pass`。

### 这段代码在项目里的作用

这段代码的作用可以总结成：

```text
定义一个所有神经网络层共同遵守的模板。
```

它规定：

1. 每一层默认是可以优化的：`self.optimizable = True`。
2. 每一层都应该有 `forward`。
3. 每一层都应该有 `backward`。

然后具体层来继承它：

```python
class Linear(Layer):
    ...

class ReLU(Layer):
    ...

class conv2D(Layer):
    ...
```

这样模型就可以统一地写：

```python
for layer in self.layers:
    outputs = layer(outputs)
```

以及：

```python
for layer in reversed(self.layers):
    grads = layer.backward(grads)
```

不用管这一层到底是 `Linear`、`ReLU` 还是 `conv2D`。  
只要它们都遵守 `Layer` 规定的接口，就能一起工作。

## 1. `super().__init__()` 是什么？

`super()` 的意思是“找到父类”。  
`super().__init__()` 的意思是“调用父类的初始化函数”。

比如：

```python
class Layer:
    def __init__(self):
        self.optimizable = True

class Linear(Layer):
    def __init__(self):
        super().__init__()
```

`Linear` 继承了 `Layer`，所以 `Linear` 是子类，`Layer` 是父类。

当你创建：

```python
layer = Linear()
```

程序会先进入 `Linear.__init__()`。  
里面的：

```python
super().__init__()
```

会去执行 `Layer.__init__()`，于是 `self.optimizable = True` 就被设置好了。

如果不写 `super().__init__()`，父类 `Layer` 里准备好的属性就不会自动出现。

在这个项目里，`optimizer.py` 会检查：

```python
if layer.optimizable == True:
```

所以大多数有参数的层最好调用父类初始化，保证 `optimizable` 这个属性存在。

## 2. `class Layer()` 是什么写法？

这是 Python 定义“类”的语法。

```python
class Layer():
    ...
```

意思是：定义一个名字叫 `Layer` 的类。

类可以理解成“模板”。  
比如 `Layer` 是神经网络层的通用模板，`Linear`、`ReLU`、`conv2D` 都是具体的层。

严格来说，下面两种写法在这里几乎等价：

```python
class Layer():
    ...
```

```python
class Layer:
    ...
```

如果括号里写东西，比如：

```python
class Linear(Layer):
    ...
```

意思就是：`Linear` 继承自 `Layer`。  
也就是 `Linear` 是一种特殊的 `Layer`。

## 3. 为什么 CNN 反向传播时 `dW` 要用“梯度值 × forward 时用过的 patch”，并对 batch 和空间位置累加？

先看卷积前向传播的一个输出位置。

某个输出值可以写成：

```text
out[i, oc, h, w] = sum(patch * W[oc]) + b[oc]
```

其中：

- `patch` 是输入图片中被卷积核盖住的小块；
- `W[oc]` 是第 `oc` 个输出通道的卷积核；
- `out[i, oc, h, w]` 是第 `i` 张图片、第 `oc` 个输出通道、位置 `(h, w)` 的输出。

现在我们要问：`W[oc]` 里的某个参数变大一点，会让这个输出怎么变化？

因为：

```text
out = patch_1 * W_1 + patch_2 * W_2 + ...
```

所以对某个 `W_k` 来说：

```text
d out / d W_k = patch_k
```

而后一层传回来的：

```text
grads[i, oc, h, w]
```

是：

```text
d loss / d out[i, oc, h, w]
```

根据链式法则：

```text
d loss / d W_k
= d loss / d out * d out / d W_k
= grads[i, oc, h, w] * patch_k
```

所以每个输出位置都会给 `W` 贡献一份：

```python
patch * grads[i, oc, h, w]
```

为什么要累加？  
因为同一个卷积核 `W[oc]` 会被重复用在：

- 一个 batch 里的每张图片；
- 每张图片的很多空间位置；
- 同一个输出通道的所有位置。

同一个参数参与了很多次计算，loss 对它的总梯度就是所有贡献加起来。

所以代码里会写成类似：

```python
dW[oc] += patch * grads[i, oc, h, w]
```

## 4. 为什么 `db = grad` 在 batch 维度求和？

先看线性层：

```text
Y = X @ W + b
```

其中 `b` 的形状是：

```python
[1, out_dim]
```

但是输出 `Y` 的形状是：

```python
[batch_size, out_dim]
```

这说明同一个 `b` 被加到了 batch 里的每一个样本上。

举个小例子，假设 batch 里有 3 个样本：

```text
y1 = x1 @ W + b
y2 = x2 @ W + b
y3 = x3 @ W + b
```

同一个 `b` 同时影响了 `y1`、`y2`、`y3`。  
所以 loss 对 `b` 的总梯度，要把这 3 个样本对 `b` 的贡献加起来。

这就是：

```python
self.grads['b'] = np.sum(grad, axis=0, keepdims=True)
```

逐段解释：

```python
grad
```

形状是：

```python
[batch_size, out_dim]
```

`axis=0` 表示沿着第 0 维求和，也就是沿着 batch 维度求和。

比如：

```text
grad =
[
  [g11, g12, g13],
  [g21, g22, g23],
  [g31, g32, g33],
]
```

执行：

```python
np.sum(grad, axis=0)
```

得到：

```text
[g11 + g21 + g31, g12 + g22 + g32, g13 + g23 + g33]
```

如果加上：

```python
keepdims=True
```

形状会从：

```python
[out_dim]
```

保持成：

```python
[1, out_dim]
```

这样就和 `b` 的形状一致。

## 5. `conv2D.backward` 的循环到底在干什么？

可以用一句话理解：

forward 时，一个输出位置由“一个输入 patch”和“一个卷积核”算出来；  
backward 时，就把这个输出位置的梯度，分别分给它用过的输入 patch、卷积核和偏置。

代码里的四层循环通常是：

```python
for i in range(batch_size):
    for oc in range(out_channels):
        for h in range(new_H):
            for w in range(new_W):
```

含义是：逐个处理每一个输出格子。

对于一个固定的输出格子：

```python
grads[i, oc, h, w]
```

它表示：

```text
loss 对这个输出格子的梯度
```

然后做三件事。

第一，找到 forward 时这个输出格子使用的输入区域：

```python
h_start = h * stride
w_start = w * stride
patch = X_padded[i, :, h_start:h_start+k, w_start:w_start+k]
```

第二，更新卷积核梯度：

```python
dW[oc] += patch * grads[i, oc, h, w]
```

因为这个输出格子是由 `patch * W[oc]` 算出来的，所以 `W[oc]` 的梯度和 `patch` 有关。

第三，更新偏置梯度：

```python
db[0, oc, 0, 0] += grads[i, oc, h, w]
```

因为这个输出格子里直接加了 `b[oc]`，所以 `b[oc]` 对输出的导数是 1。  
因此偏置只需要把对应输出位置的梯度加起来。

第四，更新输入梯度：

```python
dX_padded[i, :, h_start:h_start+k, w_start:w_start+k] += W[oc] * grads[i, oc, h, w]
```

因为这个输入 patch 里的每个像素都参与了这个输出格子的计算。  
所以这个输出格子的梯度要按卷积核权重分回输入 patch。

为什么这里也是 `+=`？  
因为一个输入像素可能被多个卷积窗口用到。  
它对 loss 的总影响要把所有窗口传回来的贡献累加。

## 6. 后面要用到的属性必须在 `__init__` 阶段声明好吗？

不是绝对必须。

Python 允许你在任何方法里第一次创建属性。  
比如下面这样是可以的：

```python
def forward(self, X):
    self.input = X
```

即使 `__init__` 里没有写：

```python
self.input = None
```

也不会语法错误。

但是，通常建议在 `__init__` 里先写：

```python
self.input = None
```

原因有三个：

第一，别人一看 `__init__`，就知道这个类有哪些重要属性。  
这对读代码很有帮助。

第二，可以避免某些顺序错误。  
比如如果你还没调用 `forward`，就直接调用 `backward`，那么 `self.input` 还不存在，会报：

```text
AttributeError
```

如果在 `__init__` 里写过 `self.input = None`，报错会更容易理解，调试也更方便。

第三，这是一种常见代码习惯。  
虽然 Python 不强制要求，但神经网络层一般会在 `__init__` 里列出重要缓存和参数。

所以结论是：

- 可以不提前声明；
- 但推荐提前声明；
- 参数、梯度、缓存这类重要属性最好在 `__init__` 里写清楚。

## 7. `assert self.input.shape == grads.shape` 是什么？

`assert` 是 Python 的检查语句。

```python
assert 条件
```

意思是：如果条件为真，程序继续运行；如果条件为假，程序立刻报错。

例如：

```python
assert 3 > 1
```

没问题。

但：

```python
assert 3 < 1
```

会报：

```text
AssertionError
```

在 ReLU 里：

```python
assert self.input.shape == grads.shape
```

意思是检查：

```text
ReLU forward 的输入形状
```

是否等于：

```text
后一层传回来的梯度形状
```

为什么要检查？  
因为 ReLU 是逐元素操作：

```text
output = max(input, 0)
```

它不会改变数组形状。  
所以 backward 传回来的梯度应该和 forward 的输入形状完全一样。

如果形状不一样，说明前面某层的 backward 写错了，或者 reshape 没处理好。

## 8. `def __call__` 是什么？

`__call__` 是 Python 的特殊方法。

如果一个类定义了：

```python
def __call__(self, X):
    return self.forward(X)
```

那么这个对象就可以像函数一样被调用。

比如本来你要写：

```python
output = layer.forward(X)
```

有了 `__call__` 之后，可以写成：

```python
output = layer(X)
```

这就是为什么 `models.py` 里可以写：

```python
outputs = layer(outputs)
```

这里的 `layer(outputs)` 实际上会自动调用：

```python
layer.__call__(outputs)
```

然后 `__call__` 里面再调用：

```python
self.forward(outputs)
```

所以 `__call__` 的作用主要是让神经网络层写起来更像数学函数。

## 9. `L2Regularization` 的 `lambda_` 后面为什么要加 `_`？

因为 `lambda` 是 Python 的关键字，不能当变量名或参数名。

Python 里有一种匿名函数语法：

```python
f = lambda x: x + 1
```

所以如果你写：

```python
def __init__(self, lambda=1e-8):
```

Python 会直接语法错误。

解决办法就是换个名字。  
常见写法是在后面加一个下划线：

```python
lambda_
```

这表示：

```text
我本来想叫它 lambda，但 lambda 是关键字，所以加个下划线避开。
```

这种写法在 Python 里很常见，比如：

```python
class_
from_
id_
```

都是为了避开关键字或内置名字。

## 10. 这次检查 `op.py` 时顺手修正的点

这次主要修了三个真实问题：

1. `L2Regularization.__init__(self, lambda=...)` 会语法错误，改成了 `lambda_`。
2. `conv2D.__init__` 里补了 `super().__init__()`，并保存了 `weight_decay` 和 `weight_decay_lambda`，否则 backward 或 optimizer 可能访问不到这些属性。
3. `Linear.backward` 和 `conv2D.backward` 里去掉了手动加 `weight_decay_lambda * W` 的部分，因为当前 `optimizer.SGD.step()` 已经在做 weight decay。如果两边都加，会重复正则化。
