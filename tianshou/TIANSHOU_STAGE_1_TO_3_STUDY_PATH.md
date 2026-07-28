# Tianshou 阶段 1-3 新手精读路线

> 目标不变：通过真实运行、阅读、断点和小练习，理解 PyTorch 强化学习项目的训练闭环、数据容器、经验回放与采样系统；最终能独立沿着一条数据流定位代码，而不是只记住名词。

## 开始前：你只需要掌握到什么程度

这一轮不要求你推导 DQN 公式，不要求理解 PPO/SAC，也不要求改算法。你只需要先回答四个问题：

1. 环境返回的观测如何变成动作？
2. 一步交互如何变成 `(obs, act, rew, obs_next, done)`？
3. 这些经验如何存下来并随机抽出一批？
4. 为什么训练环境、测试环境、采样、学习要由不同对象负责？

看到源码中 `[学习导读-阶段1]`、`[学习导读-阶段2]`、`[学习导读-阶段3]` 的地方时停下来读。可用命令快速定位：

```powershell
rg -n "学习导读" examples tianshou
```

## 准备与规则

### 最小运行环境

建议使用项目声明的 Python 3.11。只学习前三阶段时，classic control 即可：

```powershell
poetry install --with dev --extras "classic_control"
poetry run pytest test/base/test_batch.py -q
```

如果没有 Poetry，先不要急着安装全部环境。先确认当前 Python 能导入 `tianshou`、`torch` 和 `gymnasium`，再处理依赖问题。

### 每次学习的固定节奏

每个小节都按同一顺序进行：

1. 先运行，不改代码，确认基线行为存在。
2. 只读指定文件和指定函数，不跳进所有 import。
3. 加断点或临时观察变量，确认数据类型与 shape。
4. 完成一个小练习。
5. 用“验收问题”闭环；答不上来就回到上一步。

临时 `print` 只用于本地理解，完成后删掉。不要为了学习而改动算法逻辑。

## 阶段 1：跑通 DQN 的最小训练闭环

预计时间：2 到 3 天。

### 本阶段结束时的能力

你能画出下图，并能说清每个箭头传递的是什么：

```text
CartPole 环境
  -> DummyVectorEnv
  -> Collector
  -> VectorReplayBuffer
  -> DQN 更新
  -> TensorBoard / 测试 collector
```

### 第 1 步：认识 CartPole，不先陷入算法

阅读 `examples/discrete/discrete_dqn.py` 的开头和 `main()` 参数。

CartPole 的 observation 是 4 个连续数值；action 是离散的两个选择。此处的网络并不直接输出“推左/推右”，而是输出两个动作各自的 Q 值，选择 Q 值更高的动作。

先运行：

```powershell
poetry run python examples/discrete/discrete_dqn.py
```

若你只想检查依赖和导入，不想开始较长训练，先运行：

```powershell
poetry run pytest test/discrete/test_dqn.py -q
```

验收问题：`batch_size=64`、`num_training_envs=10`、`buffer_size=20000` 分别表示什么？它们绝不是同一种“数量”。

### 第 2 步：从环境空间推导网络形状

在示例中依次看：`gym.make`、`SpaceInfo.from_env`、`Net(...)`。

写在笔记里：

| 名称 | CartPole 中的意义 | 常见 shape |
|---|---|---|
| observation | 当前杆和小车状态 | 单环境 `[4]`；10 个环境 `[10, 4]` |
| action | 左/右选择 | 单环境标量；10 个环境 `[10]` |
| Q values | 两个动作的价值预测 | 单环境 `[2]`；batch `[B, 2]` |

断点建议：在 `net = Net(...)` 后观察 `state_shape` 和 `action_shape`。

验收问题：为什么网络输出维度应由 `action_space` 推导，而不是硬编码为 2？

### 第 3 步：区分 Policy 与 Algorithm

继续读 `DiscreteQLearningPolicy(...)` 和 `ts.algorithm.DQN(...)` 的构造部分。

你的心智模型应当是：

| 对象 | 只负责什么 | 不负责什么 |
|---|---|---|
| `DiscreteQLearningPolicy` | observation -> 动作；epsilon-greedy 探索 | 计算 TD loss、更新参数 |
| `DQN` | 用经验计算 target、loss、反向传播 | 管理环境 reset |
| `Collector` | 调用 policy、推进环境、收集 transition | 决定 DQN 公式 |
| `OffPolicyTrainer` | 调度 collect/update/test | 直接计算网络输出 |

练习：把这四个对象画成四个方框，在箭头上标明 `obs`、`act`、`transition`、`batch`。

验收问题：当你把 epsilon 改为 0 时，改变的是“动作选择”还是“loss 公式”？

### 第 4 步：先跟一条经验，不跟全部训练

在 `training_collector = ...` 与 `algorithm.run_training(...)` 之间建立联系。观察 Collector 的输入：算法、环境、buffer。

一条经验的最小形式是：

```text
obs -> act -> env.step(act) -> rew, obs_next, terminated, truncated
```

`terminated` 表示任务自然结束，`truncated` 常表示时间上限等外部截断。两者合并成 `done` 后决定 episode 边界。

断点顺序：

1. `Collector._compute_action_policy_hidden`
2. `Collector._collect` 中的 `env.step`
3. `ReplayBuffer.add`

第一次只观察一个环境更轻松：临时把 `num_training_envs` 和 `num_test_envs` 都改为 `1`，本地运行完后再还原。不要提交该参数改动。

验收问题：为什么 test collector 不传 replay buffer？

### 第 5 步：理解 off-policy 的训练节奏

读 `OffPolicyTrainerParams(...)`，重点只看：

| 参数 | 用新手语言理解 |
|---|---|
| `collection_step_num_env_steps` | 每次先从环境新拿多少条经验 |
| `batch_size` | 每次学习随机取多少条旧经验 |
| `update_step_num_gradient_steps_per_sample` | 收集一条经验，平均安排多少次梯度更新 |
| `test_step_num_episodes` | 每次评估跑多少完整局 |

练习：只改一个参数并记录结果。推荐把 `eps_train` 设为 `0.0`，观察训练是否更不稳定。每次实验只改一个变量。

### 阶段 1 交付物与验收

完成一页笔记，包含：

1. 一张 DQN 对象图。
2. 一条 transition 的字段说明。
3. 四个对象的职责表。
4. 一个自己实际改过的超参数及现象。

必须能回答：为什么 DQN 可以从旧经验随机抽样，而不是只用最新一步？

## 阶段 2：Batch 与 ReplayBuffer，理解“经验数据层”

预计时间：3 到 4 天。

### 本阶段结束时的能力

你能用 `Batch` 构造一批 transition，能解释环形 buffer 为什么覆盖旧数据，能说出 `sample()` 返回的两个值为何都重要。

### 第 1 步：先把 Batch 当作“列对齐的表”

阅读顺序：

1. `docs/02_deep_dives/L1_Batch.ipynb`
2. `tianshou/data/batch.py` 中 `Batch.__init__`
3. `Batch.__getitem__`
4. `Batch.to_torch`
5. `test/base/test_batch.py`

不要把 Batch 理解成普通 dict。普通 dict 切片时不知道如何保持字段对齐；`Batch[:2]` 会同时切 `obs`、`act`、`rew` 等所有字段。

在 Python shell 中运行：

```python
import numpy as np
import torch
from tianshou.data import Batch

batch = Batch(
    obs=np.array([[1.0, 2.0], [3.0, 4.0]]),
    act=np.array([0, 1]),
    rew=np.array([0.0, 1.0]),
)
print(batch[0])
print(batch[:1])
print(batch.to_torch(dtype=torch.float32))
```

观察并记录：`batch[0]` 与 `batch[:1]` 的差别；前者的字段更接近“单条”，后者仍保留 batch 维。

### 第 2 步：用测试学习 Batch 边界

按下面顺序运行，不要一次跑整个文件后只看绿色：

```powershell
poetry run pytest test/base/test_batch.py -q -k "test_batch or test_batch_cat_and_stack"
poetry run pytest test/base/test_batch.py -q -k "TestBatchConversions"
poetry run pytest test/base/test_batch.py -q -k "TestSlicing"
```

每次测试前先猜答案。例如：两个 Batch 的字段不同，`cat` 时会发生什么？嵌套 Batch 切片后是否仍对齐？

### 第 3 步：理解 ReplayBuffer 的两个职责

阅读顺序：

1. `docs/02_deep_dives/L2_Buffer.ipynb`
2. `ReplayBuffer.__init__`
3. `ReplayBuffer.add`
4. `ReplayBuffer.prev` 与 `ReplayBuffer.next`
5. `ReplayBuffer.sample_indices` 与 `ReplayBuffer.sample`
6. `test/base/test_buffer.py`

ReplayBuffer 同时做两件事：

1. 像环形队列一样保存有限容量的数据。
2. 保留 episode 边界，使 n-step return 和 frame stack 不会跨越两局游戏。

这里的关键状态：

| 字段 | 作用 |
|---|---|
| `maxsize` | 最大容量 |
| `_insertion_idx` | 下一次写入的位置 |
| `_size` | 当前实际有效条数 |
| `last_index` | 最近写入位置 |
| `_ep_return` / `_ep_len` | 当前 episode 的累计回报和步数 |

### 第 4 步：手动观察环形覆盖

运行下面代码，不要只看最终结果；每加入一条就观察 buffer：

```python
from tianshou.data import Batch, ReplayBuffer

buffer = ReplayBuffer(size=3)
for i in range(5):
    buffer.add(
        Batch(
            obs=i,
            act=i % 2,
            rew=float(i),
            terminated=False,
            truncated=False,
            obs_next=i + 1,
            info={},
        )
    )
    print(i, len(buffer), buffer.obs)
```

问题：第 4 次写入为什么没有让 `len(buffer)` 变成 4？`_insertion_idx` 接下来会指向哪里？

### 第 5 步：理解随机采样与索引

`sample(batch_size)` 返回 `(batch, indices)`：

1. `batch` 是训练时给网络的数据。
2. `indices` 是这些数据在 buffer 中的位置。

后者对优先经验回放很关键：算法根据每条经验的 TD error 更新优先级时，必须知道更新哪几个位置。

运行：

```powershell
poetry run pytest test/base/test_buffer.py -q -k "replaybuffer or prioritized"
```

### 阶段 2 交付物与验收

完成一页笔记，包含：

1. `Batch`、dict、torch tensor 的区别。
2. 环形 buffer 的 3 格写入 5 条数据示意图。
3. `terminated`、`truncated`、`done` 的关系。
4. 为什么 `sample()` 需要返回 indices。

必须能回答：如果完全不记录 episode 边界，n-step return 会出现什么错误？

## 阶段 3：Collector 与向量环境，理解“经验从哪里来”

预计时间：3 到 4 天。

### 本阶段结束时的能力

你能从 `Collector.collect()` 走到 `env.step()` 和 `buffer.add()`，能解释一个并行环境 batch 的第 0 维代表什么，能区分 `n_step` 与 `n_episode`。

### 第 1 步：先理解 Collector 的角色

阅读 `tianshou/data/collector.py` 的：

1. `BaseCollector.collect`
2. `Collector._compute_action_policy_hidden`
3. `Collector._collect` 的 Step 2、3、6

不要尝试一次读完 `_collect`。第一遍只追下面三步：

```text
last_obs_RO
  -> _compute_action_policy_hidden
  -> env.step(act_normalized)
  -> current_step_batch_R
  -> buffer.add(...)
```

符号中的 `R` 是当前活跃环境数；因此 `obs_RO` 的第一维通常就是 R。刚开始学习时，把它读成“有 R 条并行数据”即可，不需要先掌握全部后缀命名。

### 第 2 步：动作的两次形态

在 `_compute_action_policy_hidden` 中观察：

| 名称 | 含义 |
|---|---|
| `act_RA` | policy 生成的原始动作，可能来自 torch tensor |
| `act_normalized_RA` | 映射到环境 action space 后的动作，传给 `env.step` |
| `policy_R` | 需要随经验保存的附加信息 |
| `hidden_state_RH` | RNN policy 的历史状态；普通 MLP 通常为 `None` |

即使 DQN 的离散动作看起来很简单，也请记住这层分离。它让同一个 collector 能服务连续动作、RNN 和不同 action space。

### 第 3 步：在断点中只观察一轮 while loop

设置断点：

1. `Collector._collect` 中调用 `_compute_action_policy_hidden` 前后。
2. `self.env.step(...)` 后。
3. `self.buffer.add(...)` 前后。

观察以下变量，不要一次看全部局部变量：

| 观察顺序 | 变量 | 你要确认什么 |
|---|---|---|
| 1 | `last_obs_RO.shape` | 有几条并行 observation |
| 2 | `act_RA` | policy 是否为每个环境给了一个动作 |
| 3 | `rew_R` / `done_R` | env 返回是否仍按环境维对齐 |
| 4 | `current_step_batch_R.get_keys()` | 一条 transition 包含哪些字段 |
| 5 | `insertion_idx_R` | 每个环境的数据写到了 buffer 的何处 |

### 第 4 步：理解 n_step 与 n_episode

`collect(n_step=...)`：收集指定数量的环境交互步，适合 off-policy 训练。向量环境会尽量并行推进，因此实际条数可能略超出指定值。

`collect(n_episode=...)`：收集指定数量的完整 episode，适合评估平均回报。Collector 需要更细致地处理哪些环境已经完成、哪些环境还应继续。

运行对应测试：

```powershell
poetry run pytest test/base/test_collector.py -q -k "collector"
```

再读 `test_collector_with_vector_env` 和 `TestAsyncCollector` 的测试名，先只理解“它们希望保护什么行为”，不需要立刻深读异步实现。

### 第 5 步：理解训练与测试 collector 的差异

| 对比项 | training collector | test collector |
|---|---|---|
| 是否写 replay buffer | 是 | 通常否 |
| 主要目标 | 生产学习数据 | 衡量当前策略 |
| 是否需要公平稳定 | 需要探索 | 更关注评估设定 |
| 结果用途 | 算法更新 | early stop / best model |

注意：DQN 的测试 policy 仍可能保留少量 epsilon 探索；是否完全贪心取决于 `eps_inference` 设置。

### 阶段 3 交付物与验收

完成一张时序图，至少有：`obs`、`policy`、`act`、`env.step`、`Batch`、`ReplayBuffer.add`、done 后 reset。

必须能回答：为什么 Collector 需要把 `terminated` 和 `truncated` 都保存下来，而不是只保存一个布尔值？

## 三阶段完成后的综合自测

不用打开源码，尝试口头讲 3 分钟：

1. CartPole 的一批 observation 如何经由 policy 产生动作。
2. 环境结果如何被打包为 transition。
3. transition 如何进入 ReplayBuffer。
4. DQN 为什么能从 ReplayBuffer 抽旧经验学习。
5. 为什么多环境会让所有数据的第 0 维变成“环境/样本数”。

若其中任何一项讲不清，回到对应阶段的断点观察，而不是直接进入 PPO。把前三阶段走稳，后续算法部分会轻很多。
