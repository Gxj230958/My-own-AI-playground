# 用 Tianshou 全方位提升 PyTorch 与 RL Infra 工程能力的学习指南

> 适用场景：你已经把 Tianshou 克隆到本地，希望把它当成一个真实、高质量、可运行、可修改、可验证的强化学习工程训练场，而不只是“读一遍源码”。

## 0. 学习目标与基本假设

### 目标

通过系统学习本仓库，你要逐步获得以下能力：

1. 理解一个现代 PyTorch 强化学习库如何拆分职责：数据容器、环境并行、采样、回放缓冲区、策略、算法、训练器、日志、评估、测试。
2. 掌握 RL infra 的核心工程问题：数据流、并行环境、批处理、on-policy/off-policy/offline 差异、checkpoint、日志、可复现性、测试。
3. 提升 PyTorch 工程实践：`nn.Module` 组织、优化器工厂、分布对象、设备迁移、张量与 NumPy 边界、类型标注、测试驱动修改。
4. 形成“读源码 -> 跑实验 -> 加断点 -> 改一小处 -> 写测试 -> 复盘设计”的工程闭环。

### 我对本仓库的判断

Tianshou v2 的核心设计变化是把“如何行动”的 `Policy` 和“如何学习”的 `Algorithm` 明确分开。这个仓库非常适合作为 RL infra 学习对象，因为它同时具备：

- 低层 procedural API，适合研究和实现新算法。
- 高层 high-level API，适合学习实验配置、工厂模式、builder 模式和可复现实验组织。
- 完整测试，包含基础组件测试和算法训练测试。
- 类型标注、ruff、mypy、pytest、poetry、文档构建等真实开源工程配套设施。

### 学习原则

1. 不从算法论文开始硬啃。先把数据流跑通，再理解公式。
2. 不一次性读完整仓库。按“最短闭环”切片学习。
3. 不只看源码。每读一个模块，都要运行一个测试或写一个最小实验验证理解。
4. 不迷信高级 API。先学低层 API 的对象关系，再看高层 API 如何封装。
5. 不追求改大功能。每次只做一个可验证的小改动。

## 1. 仓库地图：先知道每块代码负责什么

### 顶层目录

| 路径 | 作用 | 学习价值 |
|---|---|---|
| `tianshou/` | 核心库源码 | 主战场 |
| `tianshou/data/` | `Batch`、`Collector`、`ReplayBuffer`、统计对象 | RL infra 的数据层 |
| `tianshou/env/` | 向量化环境、worker、Atari wrapper、Gym wrapper | 并行采样和环境抽象 |
| `tianshou/algorithm/` | 算法基类与 DQN/PPO/SAC 等实现 | RL 算法与 PyTorch 更新逻辑 |
| `tianshou/utils/net/` | 常用 actor/critic/net 组件 | PyTorch 网络工程 |
| `tianshou/trainer.py` | on-policy/off-policy/offline 训练循环 | 训练编排 |
| `tianshou/highlevel/` | 高层实验 API、配置、工厂、builder | 大型项目 API 设计 |
| `examples/` | 可运行示例 | 从使用者角度理解库 |
| `test/` | 单元测试与训练测试 | 最好的行为说明书 |
| `docs/` | 用户指南、deep dives、开发者指南 | 概念入口和设计说明 |
| `pyproject.toml` | 依赖、lint、type-check、测试命令 | 工程规范入口 |

### 首先要读的文档

按顺序读：

1. `README.md`
   - 关注 Tianshou v2 的设计变化。
   - 关注 low-level API 与 high-level API 的区别。
2. `docs/01_user_guide/00_training_process.md`
   - 建立 RL 训练流程的对象地图。
3. `docs/01_user_guide/02_core_abstractions.md`
   - 重点读 `Algorithm`、`Policy`、`Collector`、`Trainer`、`Batch`、`Buffer`。
4. `docs/05_developer_guide/developer_guide.md`
   - 了解 poetry、ruff、mypy、pytest、文档构建和高层 API 扩展方式。
5. `docs/02_deep_dives/`
   - 之后逐个读 notebook：`L1_Batch`、`L2_Buffer`、`L3_Environments`、`L5_Collector`、`L6_MARL`。

## 2. 环境准备：让仓库先能跑

### 推荐安装

本项目使用 Python 3.11 和 Poetry。建议先用较小依赖启动，不要一开始安装所有 extras。

```bash
conda create -n tianshou-dev python=3.11
conda activate tianshou-dev
poetry install --with dev --extras "classic_control argparse"
```

如果你只想先读基础组件，不跑环境实验，可以先：

```bash
poetry install --with dev
```

后续需要 MuJoCo、Atari、Box2D 时再加对应 extras。

### 常用命令

```bash
poe lint
poe type-check
poe test-reduced
pytest test/base/test_batch.py -q
pytest test/base/test_buffer.py -q
pytest test/base/test_collector.py -q
pytest test/discrete/test_dqn.py -q
```

### 学习时的运行策略

1. 先跑基础组件测试：`Batch`、`ReplayBuffer`、`Collector`。
2. 再跑一个轻量算法测试：DQN on CartPole。
3. 最后跑 PPO/SAC 或 MuJoCo 相关示例。

不要一开始跑全量测试。全量测试会消耗很多时间，而且不利于定位你当前正在学习的概念。

## 3. 总体学习路线

建议分成 8 个阶段，每个阶段都有“读源码、跑验证、做练习、复盘问题”。

| 阶段 | 主题 | 主要路径 | 产出 |
|---|---|---|---|
| 1 | 建立最小训练闭环 | `examples/discrete/discrete_dqn.py` | 能画出 DQN 数据流 |
| 2 | 数据容器 Batch | `tianshou/data/batch.py`、`test/base/test_batch.py` | 写出 10 个 Batch 小实验 |
| 3 | ReplayBuffer | `tianshou/data/buffer/`、`test/base/test_buffer.py` | 理解环形缓冲区、采样、episode 边界 |
| 4 | Collector 与向量环境 | `tianshou/data/collector.py`、`tianshou/env/` | 能解释 `collect(n_step)` 的内部循环 |
| 5 | Policy/Algorithm 分离 | `tianshou/algorithm/algorithm_base.py` | 能说清 action 与 update 的职责边界 |
| 6 | DQN/PPO/SAC 三条算法线 | `modelfree/dqn.py`、`ppo.py`、`sac.py` | 能对比 off-policy/on-policy/actor-critic |
| 7 | Trainer 与实验编排 | `tianshou/trainer.py`、`tianshou/highlevel/` | 能自定义训练循环参数 |
| 8 | 工程化贡献训练 | `test/`、`pyproject.toml`、`docs/` | 做一个小 PR 级别改动 |

## 4. 阶段 1：从一个 DQN 示例跑通完整对象关系

### 阅读入口

先读：

- `examples/discrete/discrete_dqn.py`
- `test/discrete/test_dqn.py`

这两个文件都围绕 CartPole DQN，但侧重点不同：

- `examples/discrete/discrete_dqn.py` 更适合理解用户如何使用库。
- `test/discrete/test_dqn.py` 更适合理解工程上如何验证训练结果、seed、日志、buffer、collector、stop function。

### 你要画出的数据流

```text
Gymnasium Env
  -> DummyVectorEnv
  -> Collector
  -> VectorReplayBuffer
  -> batch sample
  -> DQN._preprocess_batch
  -> DQN._update_with_batch
  -> DiscreteQLearningPolicy / Net
  -> Trainer loop
  -> logger / save_best_fn / stop_fn
```

### 关键对象

| 对象 | 位置 | 问自己 |
|---|---|---|
| `Net` | `tianshou/utils/net/common.py` | 输入 obs 后输出什么形状？ |
| `DiscreteQLearningPolicy` | `tianshou/algorithm/modelfree/dqn.py` | epsilon-greedy 在哪里加？ |
| `DQN` | `tianshou/algorithm/modelfree/dqn.py` | target Q 怎么算？loss 怎么算？ |
| `Collector` | `tianshou/data/collector.py` | policy 输出动作后，数据如何进入 buffer？ |
| `VectorReplayBuffer` | `tianshou/data/buffer/vecbuf.py` | 多环境数据如何组织？ |
| `OffPolicyTrainerParams` | `tianshou/trainer.py` | off-policy 训练每步采样和更新比例如何控制？ |

### 验收标准

你能不用看代码，口头解释：

1. 为什么 DQN 需要 replay buffer，而 PPO 通常不复用旧数据。
2. `eps_training` 和 `eps_inference` 为什么不同。
3. `collection_step_num_env_steps` 与 `update_step_num_gradient_steps_per_sample` 如何影响训练节奏。
4. `training_collector` 和 `test_collector` 为什么要分开。

### 小练习

1. 把 DQN 的 hidden sizes 从 `[128, 128, 128]` 改成 `[64, 64]`，跑一次看是否仍能达到阈值。
2. 把 `eps_train` 改成 `0.0`，观察探索不足的影响。
3. 把 `target_update_freq` 改得很小和很大，记录训练波动。
4. 在 `train_fn` 中打印 epoch/env_step/eps，确认 trainer 回调时机。

## 5. 阶段 2：深入 Batch，理解 Tianshou 的数据语言

### 为什么先学 Batch

`Batch` 是 Tianshou 内部几乎所有组件的共同数据格式。你理解了 `Batch`，再读 collector、buffer、policy forward、algorithm update 会轻松很多。

### 阅读路径

1. `docs/02_deep_dives/L1_Batch.ipynb`
2. `tianshou/data/batch.py`
3. `tianshou/data/types.py`
4. `test/base/test_batch.py`

### 重点理解

1. `Batch` 同时支持 attribute-style 和 dict-style 访问。
2. `Batch` 可以嵌套。
3. `Batch` 支持 slice、cat、stack、split。
4. `Batch` 能在 NumPy 与 PyTorch 之间转换。
5. Tianshou 用 `BatchProtocol` 和 `tianshou/data/types.py` 约束“这个 batch 应该有哪些字段”。

### 必做实验

新建一个临时 notebook 或 Python 文件，手写以下实验：

```python
import numpy as np
import torch
from tianshou.data import Batch

b = Batch(obs=np.arange(6).reshape(3, 2), act=np.array([0, 1, 0]))
print(b[0])
print(b[:2])
print(b.to_torch(dtype=torch.float32))
print(Batch.cat([b[:1], b[1:]]))
```

然后继续实验：

1. 嵌套 `Batch`。
2. 对含有 torch distribution 的 `Batch` 切片。
3. 对空字段和不同 shape 的字段做 `cat`/`stack`。
4. 用 `to_numpy_()` 和 `to_torch_()` 比较原地转换与非原地转换。

### 读测试的方法

不要从头到尾机械读 `test/base/test_batch.py`。按能力切片读：

| 测试类/函数 | 学什么 |
|---|---|
| `test_batch` | 基础构造和访问 |
| `test_batch_cat_and_stack` | 批处理拼接语义 |
| `TestBatchConversions` | NumPy/PyTorch 转换边界 |
| `TestAssignment` | 局部赋值与缺失字段 |
| `TestSlicing` | 切片后 shape 和类型 |

### 验收标准

你能解释：

1. 为什么 RL 框架需要一个比 dict 更强的动态数据容器。
2. `Batch` 的第一维为什么通常表示 batch size。
3. `BatchProtocol` 在静态类型检查中解决什么问题。
4. 什么时候应该用 `to_torch()`，什么时候应该保持 NumPy。

## 6. 阶段 3：ReplayBuffer，理解经验数据如何存储与采样

### 阅读路径

1. `docs/02_deep_dives/L2_Buffer.ipynb`
2. `tianshou/data/buffer/buffer_base.py`
3. `tianshou/data/buffer/vecbuf.py`
4. `tianshou/data/buffer/prio.py`
5. `tianshou/data/buffer/her.py`
6. `test/base/test_buffer.py`

### 核心问题

ReplayBuffer 不是简单 list。你要重点理解：

1. 固定容量与环形覆盖。
2. episode 边界如何保存。
3. `prev()` / `next()` 如何沿时间关系移动。
4. `sample_indices()` 与 `sample()` 的关系。
5. 多环境 buffer 如何分区。
6. PER 如何用优先级影响采样。
7. HER 如何改写 goal 和 reward。

### 最小实验

用一个很小的 buffer 手工加入 transition，观察覆盖：

```python
from tianshou.data import Batch, ReplayBuffer

buf = ReplayBuffer(size=3)
for i in range(5):
    buf.add(Batch(obs=i, act=i, rew=float(i), terminated=False, truncated=False, obs_next=i + 1, info={}))
    print("i =", i, "len =", len(buf), "obs =", buf.obs)
```

你要观察：

1. 第 4、5 次 add 后旧数据如何被覆盖。
2. `len(buf)` 和底层存储容量不是一回事。
3. `buf.sample(2)` 返回的是 `(batch, indices)`。

### 进阶实验

1. 构造 terminated=True 的 transition，看 `next()` 如何处理 episode 边界。
2. 用 `VectorReplayBuffer` 对两个环境分别加入数据，看 buffer indices 如何分布。
3. 跑 `test_prioritized_replaybuffer`，理解 PER 的权重更新。
4. 读 `tianshou/data/utils/segtree.py`，理解 PER 为什么需要 segment tree。

### 验收标准

你能解释：

1. 为什么 off-policy 算法可以从旧数据中学习。
2. 为什么 on-policy 算法收集后通常会清空或不长期复用 buffer。
3. n-step return 为什么需要 buffer 的时间结构。
4. 多环境采样时如何避免不同环境 episode 串在一起。

## 7. 阶段 4：Collector 与向量环境，理解采样系统

### 阅读路径

1. `docs/02_deep_dives/L3_Environments.ipynb`
2. `docs/02_deep_dives/L5_Collector.ipynb`
3. `tianshou/data/collector.py`
4. `tianshou/env/venvs.py`
5. `tianshou/env/worker/`
6. `test/base/test_collector.py`

### Collector 做了什么

`Collector.collect()` 是环境交互的调度器，核心循环大致是：

```text
当前 obs
  -> policy.forward / algorithm.forward
  -> map action
  -> env.step(action)
  -> 组织 obs_next, rew, terminated, truncated, info
  -> 写入 replay buffer
  -> 处理 done 环境的 reset
  -> 更新统计信息
  -> 达到 n_step 或 n_episode 后返回 CollectStats
```

### 重点断点

建议在这些位置加断点：

1. `BaseCollector.collect`
2. `Collector._collect`
3. `Collector._compute_action_policy_hidden`
4. `ReplayBuffer.add`
5. `Policy.compute_action`
6. 某个具体 policy 的 `forward`

### 你要观察的变量

| 变量 | 观察点 |
|---|---|
| `self.data.obs` | 当前环境观测 |
| `act` | policy 原始动作 |
| `mapped_action` | 映射到环境动作空间后的动作 |
| `rew` | 环境奖励 |
| `terminated` / `truncated` | episode 结束原因 |
| `info` | Gymnasium 返回的额外信息 |
| `ready_env_ids` | 哪些向量环境本轮有效 |
| `collect_stats` | 采样统计如何累积 |

### 同步与异步环境

重点对比：

- `DummyVectorEnv`
- `SubprocVectorEnv`
- `RayVectorEnv`
- `AsyncCollector`

你暂时不需要深入 Ray，但要理解同步与异步采样的差异：

1. 同步环境等待所有 env step 完成。
2. 异步环境哪个 env 先完成就先处理哪个。
3. 异步采样吞吐更好，但统计、顺序、reset 逻辑更复杂。

### 小练习

1. 用 `DummyVectorEnv` 跑 1、4、16 个 CartPole 环境，比较 collect speed。
2. 在 `Collector` 上挂一个 `StepHook`，额外记录动作分布。
3. 在 `EpisodeRolloutHookMCReturn` 上加断点，看 episode 结束时如何计算 Monte Carlo return。
4. 手写一个很小的自定义 Gymnasium 环境，接入 Tianshou collector。

### 验收标准

你能解释：

1. `n_step` 和 `n_episode` 两种 collect 模式的区别。
2. 为什么 collector 要同时处理 policy hidden state。
3. 为什么测试 collector 通常不需要 replay buffer。
4. 为什么向量环境能提升采样吞吐。

## 8. 阶段 5：Policy 与 Algorithm，理解 v2 的核心设计

### 阅读路径

1. `docs/01_user_guide/02_core_abstractions.md`
2. `tianshou/algorithm/algorithm_base.py`
3. `tianshou/algorithm/modelfree/dqn.py`
4. `tianshou/algorithm/modelfree/reinforce.py`
5. `tianshou/algorithm/modelfree/a2c.py`
6. `tianshou/algorithm/modelfree/ppo.py`

### 核心分工

| 抽象 | 职责 |
|---|---|
| `Policy` | 给定 observation/batch，输出 action 或 action distribution |
| `Algorithm` | 使用 batch 更新 policy/network 参数 |
| `OnPolicyAlgorithm` | 当前策略收集的数据用于当前更新 |
| `OffPolicyAlgorithm` | 从 replay buffer 采样历史数据更新 |
| `OfflineAlgorithm` | 从固定数据集更新，不与环境交互 |

### Policy 重点方法

| 方法 | 作用 |
|---|---|
| `forward` | 批量推理入口，训练和采样时高频调用 |
| `compute_action` | 单样本推理便利方法 |
| `map_action` | 将模型输出映射到环境动作空间 |
| `map_action_inverse` | 反向映射动作 |
| `add_exploration_noise` | 训练探索噪声 |

### Algorithm 重点方法

| 方法 | 作用 |
|---|---|
| `_preprocess_batch` | 更新前加工 batch，例如 n-step return、GAE |
| `_update_with_batch` | 真正做 loss、backward、optimizer step |
| `_postprocess_batch` | 更新后处理，例如 PER 权重 |
| `update` | 对外统一更新入口 |
| `create_trainer` | 根据算法类型创建 trainer |
| `run_training` | 用 trainer params 直接启动训练 |

### 必须看懂的一条链路：DQN

按顺序读：

1. `DiscreteQLearningPolicy.forward`
2. `DiscreteQLearningPolicy.add_exploration_noise`
3. `QLearningOffPolicyAlgorithm._preprocess_batch`
4. `DQN._target_q`
5. `DQN._update_with_batch`

你要确认：

1. Q 网络输出 shape 是什么。
2. epsilon-greedy 在哪里发生。
3. target network 是否启用，何时更新。
4. TD target 如何计算。
5. loss 如何 backward。

### 必须看懂的一条链路：PPO

按顺序读：

1. `examples/mujoco/mujoco_ppo.py`
2. `ProbabilisticActorPolicy`
3. `A2C._preprocess_batch`
4. `PPO._preprocess_batch`
5. `PPO._update_with_batch`

你要确认：

1. actor 输出如何变成 `torch.distributions.Distribution`。
2. GAE 在哪里计算。
3. old log prob 如何用于 ratio。
4. clipping objective 如何实现。
5. value loss、entropy loss、policy loss 如何组合。

### 验收标准

你能解释：

1. 为什么 `Policy` 和 `Algorithm` 都继承 `torch.nn.Module`。
2. 为什么 Tianshou v2 要把 `Policy` 与 `Algorithm` 分离。
3. DQN、PPO、SAC 的 update step 最大差异是什么。
4. `_preprocess_batch` 和 `_update_with_batch` 分离对扩展算法有什么好处。

## 9. 阶段 6：PyTorch 网络层，学习 actor/critic 的工程写法

### 阅读路径

1. `tianshou/utils/net/common.py`
2. `tianshou/utils/net/discrete.py`
3. `tianshou/utils/net/continuous.py`
4. `examples/discrete/discrete_dqn.py`
5. `examples/mujoco/mujoco_ppo.py`
6. `examples/mujoco/mujoco_sac.py`

### 重点模块

| 模块 | 学什么 |
|---|---|
| `Net` | 通用 MLP / preprocess net |
| `MLP` | 层构造、activation、输出维度 |
| `DiscreteActor` | 离散动作 actor |
| `DiscreteCritic` | 离散 Q 网络 |
| `ContinuousActorDeterministic` | DDPG/TD3 风格 actor |
| `ContinuousActorProbabilistic` | PPO/SAC 风格 actor |
| `ContinuousCritic` | 连续控制 critic |
| `ActorCritic` | actor/critic 组合 |
| `EnsembleLinear` | REDQ/ensemble 风格网络 |
| `NoisyLinear` | Rainbow 相关探索 |

### PyTorch 学习重点

1. `nn.Module` 如何组合子模块。
2. 网络的输入输出 shape 如何适配 Gymnasium space。
3. device 是如何传递的。
4. `torch.distributions` 如何表达 stochastic policy。
5. continuous action 的 tanh/clip/scaling 如何处理。
6. actor 与 critic 是否共享 preprocess network。
7. target network 如何保存和更新。

### 小练习

1. 给 `Net` 替换 activation，例如 ReLU -> Tanh，观察 PPO/DQN 的影响。
2. 手写一个很小的 custom network，接入 DQN。
3. 在 actor forward 中打印 tensor shape，确认 batch 维。
4. 对比 deterministic actor 与 probabilistic actor 的输出差异。
5. 用 `torchinfo` 或手写统计函数查看参数量。

### 验收标准

你能解释：

1. 为什么 actor 输出不一定就是环境动作。
2. 为什么 PPO actor 通常输出 distribution 参数。
3. 为什么 SAC 要修正 tanh-squashed Gaussian 的 log probability。
4. 为什么 target network 更新不能参与梯度。

## 10. 阶段 7：Trainer，理解训练循环如何被工程化

### 阅读路径

1. `tianshou/trainer.py`
2. `test/base/test_policy.py`
3. `test/discrete/test_dqn.py`
4. `test/continuous/test_ppo.py`
5. `docs/01_user_guide/00_training_process.md`

### Trainer 的职责

Trainer 不应该知道 DQN/PPO 的公式细节。它负责：

1. epoch 管理。
2. train collect。
3. update step 调度。
4. test collect。
5. logger 写入。
6. stop function 判断。
7. save best/checkpoint。
8. 训练统计聚合。

### 三类 Trainer

| Trainer | 适用算法 | 典型行为 |
|---|---|---|
| `OffPolicyTrainer` | DQN、DDPG、TD3、SAC | 边采样边从 replay buffer 多次更新 |
| `OnPolicyTrainer` | REINFORCE、A2C、PPO、TRPO | 收集一批新数据后更新，通常不长期复用 |
| `OfflineTrainer` | BCQ、CQL、TD3+BC | 不采环境，只从固定 buffer 更新 |

### 关键参数

| 参数 | 理解方式 |
|---|---|
| `max_epochs` | 最外层训练轮数 |
| `epoch_num_steps` | 每个 epoch 目标环境步数 |
| `collection_step_num_env_steps` | 每次 collect 收集多少环境步 |
| `update_step_num_gradient_steps_per_sample` | off-policy 中每个样本对应多少梯度步 |
| `update_step_num_repetitions` | on-policy 中同一批数据重复训练几轮 |
| `batch_size` | 每次 update sample 的 batch 大小 |
| `test_step_num_episodes` | 测试 episode 数 |
| `training_fn` | 训练过程回调，例如 epsilon annealing |
| `stop_fn` | 提前停止条件 |
| `save_best_fn` | 保存最佳模型 |

### 小练习

1. 在 `OffPolicyTrainer._training_step` 加断点，看 collect 和 update 的先后关系。
2. 把 DQN 的 `update_per_step` 从 `0.1` 改成 `1.0`，看训练速度和稳定性变化。
3. 在 `save_best_fn` 中保存更多状态，例如 optimizer、epoch、env_step。
4. 修改 `test_in_training`，观察训练中测试行为差异。

### 验收标准

你能解释：

1. trainer 为什么不直接依赖某个具体算法类。
2. on-policy 与 off-policy trainer 在 update 节奏上的差异。
3. callback 比继承 trainer 更适合哪些自定义需求。
4. logger 和 checkpoint 为什么属于 trainer 编排层，而不是 algorithm 公式层。

## 11. 阶段 8：高层 API，学习大型库如何封装易用接口

### 阅读路径

1. `examples/discrete/discrete_dqn_hl.py`
2. `tianshou/highlevel/experiment.py`
3. `tianshou/highlevel/algorithm.py`
4. `tianshou/highlevel/config.py`
5. `tianshou/highlevel/params/algorithm_params.py`
6. `docs/05_developer_guide/developer_guide.md`

### 高层 API 要解决的问题

低层 API 灵活，但用户需要手工创建：

- env factory
- vector env
- network
- actor/critic
- optimizer
- algorithm
- collector
- trainer params
- logger
- persistence

高层 API 把这些对象创建过程封装成 builder/factory/config。

### 重点设计模式

| 模式 | 在哪里 | 学什么 |
|---|---|---|
| Builder | `ExperimentBuilder` | 链式配置用户体验 |
| Factory | `AlgorithmFactory`、`EnvFactory` | 延迟创建依赖环境信息的对象 |
| Dataclass params | `DQNParams` 等 | 配置对象如何组织 |
| Mixins | high-level builder 和 params | 复用配置能力 |
| Param transformer | params 内部 | 高层参数如何转成低层构造参数 |

### 小练习

1. 对比 `examples/discrete/discrete_dqn.py` 和 `examples/discrete/discrete_dqn_hl.py`，列出高层 API 隐藏了哪些对象。
2. 给 high-level DQN 示例增加一个不同 hidden size 配置。
3. 找到 `DQNExperimentBuilder` 如何最终创建低层 `DQN`。
4. 尝试给一个已有 high-level builder 增加一个轻量的 `with_*` 方法。

### 验收标准

你能解释：

1. 为什么 high-level API 需要 factory，而不是一开始就创建对象。
2. 为什么高层配置更适合实验复现。
3. 哪些需求应该用低层 API，哪些需求应该用高层 API。
4. 扩展一个新算法时，需要改 params、factory、builder、exports 哪些位置。

## 12. 阶段 9：测试系统，把测试当作行为文档

### 阅读路径

按顺序读测试：

1. `test/base/test_batch.py`
2. `test/base/test_buffer.py`
3. `test/base/test_collector.py`
4. `test/base/test_policy.py`
5. `test/discrete/test_dqn.py`
6. `test/continuous/test_ppo.py`
7. `test/offline/`
8. `test/modelbased/`
9. `test/pettingzoo/`

### 测试类型

| 类型 | 例子 | 学什么 |
|---|---|---|
| 纯单元测试 | `test_batch.py` | 数据结构边界条件 |
| 组件集成测试 | `test_collector.py` | env/policy/buffer 协作 |
| 算法训练测试 | `test_dqn.py`、`test_ppo.py` | 真实训练能否达到阈值 |
| determinism 测试 | `test_*_determinism` | 行为快照与可复现 |
| 多环境测试 | pettingzoo、modelbased | 扩展场景 |

### 如何用测试学习

每读一个测试，都回答：

1. 这个测试在保护什么行为？
2. 如果删掉这段实现，测试会怎么失败？
3. 测试的输入最小吗？
4. 测试是否覆盖边界条件？
5. 如果我要改相关源码，应该新增哪个测试？

### 小练习

1. 给 `Batch` 新增一个你认为合理的边界行为测试，不一定要提交。
2. 故意改坏 `ReplayBuffer.add` 的一个细节，观察哪些测试失败，再还原。
3. 在 DQN 测试里降低 epoch，观察是否更容易不稳定。
4. 阅读 determinism 测试说明，但不要轻易改 snapshot。

### 验收标准

你能解释：

1. 为什么 RL 库需要训练级别测试。
2. 为什么 determinism 在 RL 中很难但很重要。
3. 单元测试和训练测试分别适合保护什么。
4. 修改算法行为时应该如何设计测试。

## 13. 推荐的 6 周学习计划

### 第 1 周：跑通与定位

目标：能运行基础测试和 DQN 示例，理解顶层对象关系。

任务：

1. 完成环境安装。
2. 跑 `pytest test/base/test_batch.py -q`。
3. 跑 `pytest test/base/test_buffer.py -q`。
4. 跑 `examples/discrete/discrete_dqn.py` 或 `test/discrete/test_dqn.py`。
5. 画出 DQN 训练数据流图。

产出：

- 一张手绘或 Markdown 数据流图。
- 一份“我还不懂的问题列表”。

### 第 2 周：数据层

目标：吃透 `Batch` 和 `ReplayBuffer`。

任务：

1. 读 `L1_Batch` 和 `L2_Buffer`。
2. 阅读并运行 batch/buffer 测试。
3. 手写 10 个 Batch 实验。
4. 手写一个 ReplayBuffer 覆盖实验。

产出：

- 一份 `Batch` 操作速查表。
- 一份 replay buffer 索引机制笔记。

### 第 3 周：采样层

目标：理解 Collector 与 VectorEnv。

任务：

1. 读 `collector.py`。
2. 跑 `test/base/test_collector.py`。
3. 对 1/4/16 个环境比较采样速度。
4. 写一个极简自定义 Gymnasium env 接入 collector。

产出：

- 一张 `Collector.collect()` 时序图。
- 一个最小自定义环境 demo。

### 第 4 周：算法层

目标：理解 DQN 和 PPO 的 update 过程。

任务：

1. 深读 `algorithm_base.py`。
2. 深读 `dqn.py`。
3. 深读 `ppo.py`。
4. 对 DQN/PPO 的 `_preprocess_batch` 和 `_update_with_batch` 做对照笔记。

产出：

- DQN update 公式与代码映射表。
- PPO loss 与代码映射表。

### 第 5 周：训练编排与高层 API

目标：理解 trainer 和 high-level API 的工程设计。

任务：

1. 读 `trainer.py`。
2. 对比 low-level DQN 和 high-level DQN 示例。
3. 跟踪 `DQNExperimentBuilder.build_and_run()` 到低层算法创建。
4. 改一个 trainer 参数，观察训练行为。

产出：

- `Trainer` 状态机笔记。
- high-level API 对象创建链路图。

### 第 6 周：做一个小 PR 级别项目

目标：用真实工程流程完成一个可验证改动。

可选项目：

1. 给已有测试补一个边界测试。
2. 给某个示例增加一个小配置参数。
3. 给文档补充一个更清晰的例子。
4. 实现一个轻量 wrapper 或 hook。
5. 给 high-level API 增加一个已有低层能力的暴露方式。

要求：

1. 改动少。
2. 有测试或可运行示例。
3. 跑 lint 或对应 pytest。
4. 写清楚变更原因和验证结果。

## 14. 三条专题路线

如果你想按兴趣切入，可以选一条专题路线。

### 路线 A：PyTorch 强化学习实现

顺序：

1. `tianshou/utils/net/common.py`
2. `tianshou/utils/net/discrete.py`
3. `tianshou/utils/net/continuous.py`
4. `tianshou/algorithm/modelfree/dqn.py`
5. `tianshou/algorithm/modelfree/a2c.py`
6. `tianshou/algorithm/modelfree/ppo.py`
7. `tianshou/algorithm/modelfree/sac.py`

重点问题：

1. network 输出如何转成 action。
2. distribution 如何参与 loss。
3. actor/critic 参数如何组织。
4. optimizer 和 lr scheduler 如何封装。
5. target network 如何更新。

### 路线 B：RL Infra 数据与采样

顺序：

1. `tianshou/data/batch.py`
2. `tianshou/data/buffer/buffer_base.py`
3. `tianshou/data/buffer/vecbuf.py`
4. `tianshou/data/collector.py`
5. `tianshou/env/venvs.py`
6. `tianshou/env/worker/`

重点问题：

1. 数据从 env 到 buffer 的路径。
2. 多环境并行如何影响 batch shape。
3. done/reset 逻辑如何处理。
4. 采样吞吐与训练吞吐如何平衡。
5. 异步 collector 为什么复杂。

### 路线 C：大型库工程设计

顺序：

1. `pyproject.toml`
2. `docs/05_developer_guide/developer_guide.md`
3. `tianshou/algorithm/algorithm_base.py`
4. `tianshou/trainer.py`
5. `tianshou/highlevel/`
6. `test/`

重点问题：

1. 抽象边界如何划分。
2. 类型系统如何约束动态数据。
3. builder/factory 如何降低用户复杂度。
4. 测试如何保护算法行为。
5. 文档如何和代码结构对应。

## 15. 建议的调试方法

### 使用断点

最值得断的地方：

1. `Collector._collect`
2. `ReplayBuffer.add`
3. `ReplayBuffer.sample`
4. `Policy.compute_action`
5. 具体 policy 的 `forward`
6. `Algorithm.update`
7. 具体算法的 `_preprocess_batch`
8. 具体算法的 `_update_with_batch`
9. `Trainer.execute_epoch`
10. `Trainer._training_step`

### 每次断点观察四类信息

1. shape：obs、act、rew、done、logits、loss。
2. dtype/device：NumPy 还是 torch，CPU 还是 GPU。
3. control flow：现在是 collect、update 还是 test。
4. state：policy 是否 training，是否 within training step，buffer 长度是多少。

### 推荐打印模板

```python
print("obs", type(batch.obs), getattr(batch.obs, "shape", None))
print("act", type(batch.act), getattr(batch.act, "shape", None))
print("rew", type(batch.rew), getattr(batch.rew, "shape", None))
print("device", next(self.parameters()).device)
```

调试完成后要删掉临时打印，避免污染 diff。

## 16. 读源码时的问题清单

每读一个类，按这个模板写笔记：

```text
类名：
所在文件：
它解决的问题：
它不负责的问题：
主要输入：
主要输出：
关键状态：
最重要的 3 个方法：
它被谁调用：
它调用谁：
对应测试：
我不懂的问题：
```

每读一个函数，按这个模板写笔记：

```text
函数名：
前置条件：
输入 shape/type：
输出 shape/type：
副作用：
异常/边界：
核心算法：
测试覆盖：
```

## 17. 从源码到论文公式的映射方法

学习算法实现时，不要直接把论文公式和代码硬配。建议三步走：

1. 先找数据来源：这个 loss 用的字段来自 buffer、policy output、还是 preprocess。
2. 再找张量 shape：每个 tensor 的 batch 维、action 维、ensemble 维是什么。
3. 最后映射公式：把代码中的变量对应到论文符号。

### DQN 映射重点

| 概念 | 代码位置 |
|---|---|
| Q(s, a) | `DiscreteQLearningPolicy.forward` / model output |
| target Q | `DQN._target_q` |
| n-step return | `QLearningOffPolicyAlgorithm._preprocess_batch` |
| TD loss | `DQN._update_with_batch` |
| target network update | `QLearningOffPolicyAlgorithm._periodically_update_lagged_network_weights` |

### PPO 映射重点

| 概念 | 代码位置 |
|---|---|
| stochastic policy | `ProbabilisticActorPolicy` |
| GAE | `A2C._preprocess_batch` / `Algorithm.compute_episodic_return` |
| ratio | `PPO._update_with_batch` |
| clipped objective | `PPO._update_with_batch` |
| entropy bonus | `PPO._update_with_batch` |

### SAC 映射重点

| 概念 | 代码位置 |
|---|---|
| tanh Gaussian policy | `SACPolicy.forward` |
| log prob correction | `correct_log_prob_gaussian_tanh` |
| twin critics | `SAC` constructor and update |
| alpha | `FixedAlpha` / `AutoAlpha` |
| target Q | `SAC._target_q_compute_value` |

## 18. 工程实践训练：如何做一次小改动

### 推荐流程

1. 写下假设：我要改什么行为，为什么。
2. 找到最小相关测试。
3. 如果是 bug，先写失败测试。
4. 改最少的代码。
5. 跑相关测试。
6. 跑 lint 或 type-check。
7. 写变更说明。

### 不推荐的做法

1. 一边读一边大范围重构。
2. 顺手修改格式。
3. 在理解不足时改公共抽象。
4. 为单一需求加复杂配置。
5. 不跑测试就相信训练结果。

### 适合新手的改动

1. 给 `Batch` 或 `ReplayBuffer` 增加边界测试。
2. 给示例增加更清晰的日志输出。
3. 给 docs 补一个最小示例。
4. 给 high-level 示例加参数。
5. 给 collector hook 写一个更小的示例测试。

### 更进阶的改动

1. 新增一个轻量 exploration noise。
2. 给某个 algorithm 暴露一个已有内部参数。
3. 给 high-level API 支持一个低层已有算法配置。
4. 新增一个小型 wrapper。
5. 给 offline 数据加载补充一个格式转换工具。

## 19. 你应该重点积累的工程能力

### PyTorch 能力

1. `nn.Module` 的组合与状态保存。
2. actor/critic 网络拆分。
3. `torch.distributions` 的使用。
4. loss/backward/optimizer step 的组织。
5. target network 和 Polyak update。
6. gradient clipping。
7. device 和 dtype 管理。
8. NumPy 与 torch 的转换边界。

### RL Infra 能力

1. 环境抽象与 Gymnasium 接口。
2. 向量化环境。
3. 同步/异步采样。
4. replay buffer。
5. PER/HER/n-step/GAE。
6. on-policy/off-policy/offline 训练节奏。
7. logging/checkpoint/evaluation。
8. reproducibility 与 determinism。

### 软件工程能力

1. 类型标注和 Protocol。
2. dataclass 参数对象。
3. factory/builder/mixin。
4. 单元测试和集成测试。
5. lint/type-check/test 的本地工作流。
6. 文档与源码一致性。
7. 小步改动和行为验证。
8. API 兼容性意识。

## 20. 常见卡点与处理方式

### 卡点 1：Batch 太动态，看不懂字段从哪来

处理：

1. 搜索字段名，例如 `rg "returns|adv|act|obs_next" tianshou test examples`。
2. 先看 `tianshou/data/types.py` 中的 Protocol。
3. 加断点看 batch 的 keys。
4. 从测试中找最小构造例子。

### 卡点 2：Collector 控制流太长

处理：

1. 只跟一个单环境、短 episode。
2. 先忽略 AsyncCollector。
3. 在 `env.step` 前后断点。
4. 记录每轮 obs/act/rew/done。

### 卡点 3：算法 loss 看不懂

处理：

1. 先打印所有 tensor shape。
2. 找 `_preprocess_batch` 里新增了哪些字段。
3. 对照测试或示例中的超参数。
4. 最后再回到论文公式。

### 卡点 4：训练结果不稳定

处理：

1. 固定 seed。
2. 降低并行环境数量，方便调试。
3. 缩短训练只验证代码路径，不急着看最终 reward。
4. 查看日志和 loss。
5. 用已有测试的默认参数作为稳定 baseline。

### 卡点 5：high-level API 链路太绕

处理：

1. 先用 low-level 示例理解对象。
2. 对比 high-level 示例隐藏了哪些对象。
3. 从 `build_and_run` 往下追。
4. 只追 DQN 一个算法，不要同时追多个 builder。

## 21. 最小项目建议

### 项目 1：自定义环境 + DQN

目标：掌握 Gymnasium env 到 Tianshou training 的完整接入。

要求：

1. 写一个 1D GridWorld。
2. observation 是当前位置。
3. action 是左右移动。
4. reward 到终点为 1，否则 0 或小惩罚。
5. 用 DQN 训练。
6. 写一个测试确认训练后平均 reward 达标。

你会学到：

- Gymnasium env API。
- discrete action policy。
- collector 和 replay buffer。
- stop_fn 和 test collector。

### 项目 2：给 DQN 加一个学习率调度实验

目标：理解 optimizer factory 与 trainer 参数。

要求：

1. 在 DQN 示例中接入 LR scheduler。
2. 记录不同 epoch 的 lr。
3. 比较固定 lr 与线性衰减。
4. 保持改动局部，不改算法核心。

你会学到：

- optimizer abstraction。
- trainer schedule。
- logging。

### 项目 3：实现一个自定义 Collector Hook

目标：理解采样过程扩展点。

要求：

1. 写一个 `StepHook`。
2. 记录每一步 action 或 action distribution。
3. 在测试中确认 hook 被调用。
4. 不改 collector 主循环。

你会学到：

- hook 设计。
- Batch 扩展字段。
- 测试采样副作用。

### 项目 4：从 low-level DQN 迁移到 high-level DQN

目标：理解 API 封装层。

要求：

1. 用 low-level 写 DQN。
2. 用 high-level 写等价配置。
3. 对比两者创建了哪些对象。
4. 记录 high-level API 的优缺点。

你会学到：

- builder/factory。
- 实验配置。
- API 易用性设计。

## 22. 每次学习后的复盘模板

```text
日期：
今天读的文件：
今天跑的命令：
通过的测试：
失败的测试：
我理解的新概念：
我确认的源码链路：
我仍然困惑的问题：
下一步最小行动：
```

建议把复盘放在你自己的学习笔记里，而不是直接提交到项目仓库。

## 23. 建议的阅读顺序清单

### 第一轮：只建立地图

1. `README.md`
2. `docs/01_user_guide/00_training_process.md`
3. `docs/01_user_guide/02_core_abstractions.md`
4. `examples/discrete/discrete_dqn.py`
5. `test/discrete/test_dqn.py`

### 第二轮：数据层

1. `docs/02_deep_dives/L1_Batch.ipynb`
2. `tianshou/data/batch.py`
3. `test/base/test_batch.py`
4. `docs/02_deep_dives/L2_Buffer.ipynb`
5. `tianshou/data/buffer/buffer_base.py`
6. `test/base/test_buffer.py`

### 第三轮：采样和训练循环

1. `docs/02_deep_dives/L3_Environments.ipynb`
2. `docs/02_deep_dives/L5_Collector.ipynb`
3. `tianshou/data/collector.py`
4. `test/base/test_collector.py`
5. `tianshou/trainer.py`

### 第四轮：算法实现

1. `tianshou/algorithm/algorithm_base.py`
2. `tianshou/algorithm/modelfree/dqn.py`
3. `tianshou/algorithm/modelfree/a2c.py`
4. `tianshou/algorithm/modelfree/ppo.py`
5. `tianshou/algorithm/modelfree/sac.py`

### 第五轮：网络和高层 API

1. `tianshou/utils/net/common.py`
2. `tianshou/utils/net/discrete.py`
3. `tianshou/utils/net/continuous.py`
4. `examples/discrete/discrete_dqn_hl.py`
5. `tianshou/highlevel/experiment.py`
6. `tianshou/highlevel/algorithm.py`
7. `tianshou/highlevel/params/algorithm_params.py`

## 24. 最终验收：什么时候算真正学到位

当你能完成下面这些任务，说明你已经从“看过源码”进入了“具备工程能力”的阶段：

1. 不看示例，独立搭建一个 CartPole DQN 训练脚本。
2. 能解释一条 transition 从 env 到 replay buffer 再到 loss 的完整路径。
3. 能给 `Batch`、`ReplayBuffer`、`Collector` 的边界行为写测试。
4. 能在 DQN/PPO/SAC 任意一个算法中定位 loss 计算。
5. 能修改一个超参数调度或 hook，并验证行为。
6. 能判断一个需求应该改 low-level API、high-level API、trainer、collector 还是 algorithm。
7. 能跑相关测试，并根据失败栈定位问题。
8. 能写出一个小而清晰的 PR 描述：改了什么、为什么、如何验证。

## 25. 一句话路线图

先跑 DQN 建立全局闭环；再吃透 `Batch` 和 `ReplayBuffer`；然后拆 `Collector` 和 `Trainer`；接着深入 `Policy`/`Algorithm` 和 DQN/PPO/SAC；最后用 tests/high-level API/docs 做小型贡献训练。这样学 Tianshou，收获的不只是 RL 算法实现，而是一整套 PyTorch 强化学习基础设施的工程思维。
