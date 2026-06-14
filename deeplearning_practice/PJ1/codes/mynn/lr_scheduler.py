from abc import abstractmethod
import numpy as np

class scheduler():
    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer
        self.step_count = 0
    
    @abstractmethod
    def step(self):
        pass


class StepLR(scheduler):
    def __init__(self, optimizer, step_size=30, gamma=0.1) -> None:
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def step(self) -> None:
        # 每调用 step 一次，就认为训练又前进了一个 iteration。
        # 当累计次数达到 step_size，就把学习率乘以 gamma。
        self.step_count += 1
        if self.step_count >= self.step_size:
            self.optimizer.init_lr *= self.gamma
            self.step_count = 0

class MultiStepLR(scheduler):
    def __init__(self, optimizer, milestones, gamma=0.1) -> None:
        super().__init__(optimizer)
        # milestones 是一个列表，表示在哪些 step 之后降低学习率。
        # 例如 [800, 2400, 4000] 表示第 800、2400、4000 次更新后 lr *= gamma。
        self.milestones = set(milestones)
        self.gamma = gamma

    def step(self) -> None:
        self.step_count += 1
        if self.step_count in self.milestones:
            self.optimizer.init_lr *= self.gamma

class ExponentialLR(scheduler):
    def __init__(self, optimizer, gamma=0.99) -> None:
        super().__init__(optimizer)
        self.gamma = gamma

    def step(self) -> None:
        # 指数衰减：每一步都把学习率乘以 gamma。
        # gamma 通常接近 1，例如 0.99 或 0.995。
        self.step_count += 1
        self.optimizer.init_lr *= self.gamma
