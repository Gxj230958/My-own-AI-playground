from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
    
    def step(self):
        # SGD 的核心公式：
        # 参数 = 参数 - 学习率 * 参数梯度
        #
        # 如果某层开启了 weight_decay，就额外做一次权重衰减。
        # 这里为了更符合常见做法，只对 W 做 weight decay，不对 b 做。
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.weight_decay and key == 'W':
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    layer.params[key] = layer.params[key] - self.init_lr * layer.grads[key]


class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu):
        super().__init__(init_lr, model)
        self.mu = mu
        # velocity 保存“历史梯度方向”的指数滑动平均。
        # 形象理解：普通 SGD 每一步只看当前坡度；
        # Momentum SGD 会带一点“惯性”，让连续方向的更新更稳定。
        self.velocity = []
        self._init_velocity()

    def _init_velocity(self):
        # 按照 model.layers 的顺序，为每个可训练参数准备一个同形状的 0 数组。
        # 例如 Linear 有 W、b，就分别准备 velocity['W'] 和 velocity['b']。
        self.velocity = []
        for layer in self.model.layers:
            if layer.optimizable:
                layer_velocity = {}
                for key, value in layer.params.items():
                    layer_velocity[key] = np.zeros_like(value)
                self.velocity.append(layer_velocity)
            else:
                self.velocity.append(None)
    
    def step(self):
        # 如果模型结构后来发生变化，velocity 的长度可能不匹配。
        # 这种情况通常出现在重新 load_model 之后，所以这里做一次保险检查。
        if len(self.velocity) != len(self.model.layers):
            self._init_velocity()

        for layer_idx, layer in enumerate(self.model.layers):
            if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.weight_decay and key == 'W':
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)

                    # 一种常见 Momentum 写法：
                    # v = mu * v - lr * grad
                    # param = param + v
                    # 当连续几步梯度方向相近时，v 会积累起来，更新更快；
                    # 当梯度来回震荡时，不同方向会互相抵消，更新更稳。
                    self.velocity[layer_idx][key] = (
                        self.mu * self.velocity[layer_idx][key]
                        - self.init_lr * layer.grads[key]
                    )
                    layer.params[key] = layer.params[key] + self.velocity[layer_idx][key]
