# Tianshou 强化学习新手概念讲解

> 这份文档只服务于阶段 1-3。目标是先建立直觉，不追求数学推导的完整性。读到一个概念时，优先把它和 CartPole、`Batch`、`ReplayBuffer`、`Collector` 联系起来。

## 1. 强化学习到底在做什么

监督学习通常有“题目和标准答案”；强化学习没有直接给出每一步正确动作。智能体只能不断尝试：观察环境、做动作、收到奖励，再调整之后的行为。

最小循环：

```text
观察 obs -> 策略 policy 选择 act -> 环境返回 rew、obs_next、结束标记
```

在 Tianshou 中，这条循环主要由 `Collector` 执行。

## 2. CartPole：一个足够小的例子

CartPole 的任务是让小车移动以保持杆子不倒。

| 概念 | CartPole 中是什么 |
|---|---|
| observation / obs | 小车位置、速度、杆角度、角速度等 4 个数 |
| action / act | 往左推或往右推 |
| reward / rew | 杆还没倒时通常得到正奖励 |
| episode | 从 reset 开始到杆倒或到达时限的一局 |
| policy | 根据 4 个数决定推左还是推右的规则 |

一开始 policy 很笨，所以它会经常失败。训练的含义不是“记住某一局”，而是从许多失败和成功经验中学到更好的动作倾向。

## 3. observation、state 与 action

### Observation 与 state

初学时可以把 observation 当作“智能体看到的状态”。严格来说，环境真实 state 可能包含智能体看不到的信息，observation 只是可见部分。CartPole 不必纠结这个差异；后续在部分可观测环境和 RNN policy 中才会变得重要。

### Action

动作空间常见两类：

| 类型 | 例子 | 网络常见输出 |
|---|---|---|
| 离散动作 | 左/右、上/下、游戏按键 | 每个动作一个分数或概率 |
| 连续动作 | 机械臂关节角度、油门、转向 | 一个或多个连续数值，常带分布参数 |

CartPole 是离散动作，所以 DQN 网络会输出两个 Q 值。注意：Q 值不是动作本身，而是“此时选择该动作有多好”的预测。

## 4. Reward、Return 与 gamma

### Reward

reward 是环境在当前一步给的反馈。它可以稀疏、延迟，也可能设计得不好。reward 不是模型凭空创造的，而是环境或任务设计者定义的。

### Return

Return 是从现在到未来得到的奖励总和。通常会让越远的奖励权重越低：

```text
G_t = r_t + gamma * r_(t+1) + gamma^2 * r_(t+2) + ...
```

`gamma` 在 0 到 1 之间：

| gamma | 直觉 |
|---|---|
| 小 | 更看重眼前奖励 |
| 接近 1 | 更看重长期结果 |

阶段 1 的 DQN 参数 `gamma=0.9` 就是这种“未来折扣”。你暂时只需知道它影响 TD target，不必推导细节。

## 5. Policy：从 observation 到动作的规则

policy 可以是手写规则，也可以是神经网络。在深度强化学习中，policy 通常是 `torch.nn.Module`。

Tianshou v2 把职责分得很清楚：

```text
Policy：如何行动
Algorithm：如何学习并改进 Policy
```

这意味着 Collector 不必知道 DQN 的 loss；它只要调用 policy 取动作。

## 6. Q 值与 DQN

Q 值 `Q(s, a)` 可以粗略理解为：在当前 observation/state `s` 下执行动作 `a`，之后表现会有多好。

对 CartPole：

```text
网络输入：4 个 observation 数值
网络输出：[Q(推左), Q(推右)]
执行动作：选择较大的 Q 值对应的动作
```

训练时，DQN 用实际得到的 reward 和“下一状态的预测”构造一个目标值，再让当前 Q 值靠近它。这个“用旧预测帮助新预测”的思想叫 bootstrap。

### 为什么不总选最大 Q 值

模型刚开始的 Q 值完全不可靠。如果永远选当前最大的一个，可能很早陷入坏选择。因此 DQN 常用 epsilon-greedy：

```text
以 epsilon 的概率随机探索
以 1 - epsilon 的概率选择当前 Q 值最大的动作
```

在示例中，`eps_training` 控制训练时探索，`eps_inference` 控制评估时探索。

## 7. Transition：一条可学习的经验

强化学习最常用的数据单元是一条 transition：

```text
(obs, act, rew, obs_next, terminated, truncated)
```

含义如下：

| 字段 | 问题 |
|---|---|
| `obs` | 我当时看到了什么？ |
| `act` | 我做了什么？ |
| `rew` | 环境立即给了什么反馈？ |
| `obs_next` | 做完后看到了什么？ |
| `terminated` | 任务是否自然终止？ |
| `truncated` | 是否被时间上限等外部规则截断？ |

`done = terminated OR truncated` 常用于标记一局结束，但 Tianshou 仍保留两个原始字段，因为它们的语义不同，某些算法的 value target 会区别处理。

## 8. Batch：把多条 transition 放在一起

`Batch` 是 Tianshou 用来装数据的容器。可以把它想成一张按列存储的表：

```text
obs:  [[...], [...], [...]]
act:  [0, 1, 0]
rew:  [1.0, 1.0, 1.0]
done: [False, False, True]
```

第 0 维表示“有多少条数据”。`Batch[:2]` 会让每一列同时取前两行，保证第 i 条 observation、action、reward 仍属于同一次交互。

### NumPy 与 torch 的分工

| 位置 | 常见数据类型 | 原因 |
|---|---|---|
| 环境/collector/buffer | NumPy | Gymnasium 环境接口通常使用 NumPy |
| 网络与 loss | torch tensor | 需要自动求导和 GPU 加速 |

因此 `Batch.to_torch()` 很重要：它通常是“准备学习数据”的转换，而不是所有数据一产生就转 tensor。

## 9. ReplayBuffer：经验记忆本

ReplayBuffer 把很多 transition 保存起来。DQN 训练时随机从里面抽一批，而不是只学习刚发生的一步。

这样做有三个直觉好处：

1. 旧经验可以被重复利用，数据效率更高。
2. 随机抽样打散相邻时间步的高度相关性，训练更稳定。
3. buffer 让采样和学习解耦：环境可以继续产生经验，网络可以从历史中学习。

### 为什么叫环形 buffer

容量固定，例如只能放 3 条数据：

```text
写入 0, 1, 2 后满了
继续写入 3，会覆盖最旧的 0
继续写入 4，会覆盖最旧的 1
```

这防止训练无限占用内存。Tianshou 用 `_insertion_idx` 记住下一次该覆盖的位置。

### 为什么需要 episode 边界

有些计算需要找“下一步”或“前几步”。如果一局结束后直接把下一局开头当作上一局的后续，就会创造不存在的轨迹。`terminated`/`truncated` 和 buffer 的 `next()`/`prev()` 共同避免这个错误。

## 10. on-policy 与 off-policy

这是算法“能否复用旧经验”的区别：

| 类型 | 核心想法 | 代表算法 |
|---|---|---|
| on-policy | 主要使用当前 policy 刚收集的数据 | PPO、A2C |
| off-policy | 可以复用过去策略留下的数据 | DQN、DDPG、SAC |

DQN 是 off-policy，所以特别依赖 ReplayBuffer。阶段 1 的 `OffPolicyTrainer` 会不断“收集一小批 -> 从 buffer 抽样学习”。

## 11. Collector：强化学习里的“现场记录员”

Collector 不负责发明策略，也不负责计算 loss。它负责把 policy 和 environment 接起来：

```text
当前 obs
  -> policy 给 act
  -> action mapping
  -> env.step
  -> 得到 obs_next、reward、结束标记
  -> 打包成 Batch
  -> 写入 ReplayBuffer
```

这层分离很有工程价值：同一个 Algorithm 可以接不同环境；同一个 Collector 也能服务不同算法。

## 12. 向量环境：一次推进多个环境

`DummyVectorEnv` 把多个独立环境组合成一个批量接口。若有 10 个 CartPole：

```text
obs 的 shape: [10, 4]
act 的 shape: [10]
rew 的 shape: [10]
```

它不是“一个环境有 10 个动作”，而是“10 个独立环境各有一个动作”。并行环境提高采样吞吐，也让 batch shape 和 done/reset 逻辑更复杂。

## 13. terminated 与 truncated

Gymnasium 把 episode 结束拆成两个信号：

| 信号 | 典型含义 |
|---|---|
| `terminated=True` | 任务本身结束，例如杆倒了、到达目标、失败 |
| `truncated=True` | 被时间上限或外部限制停止 |

学习时先牢记：两者都意味着需要 reset，但算法在构造学习目标时有时需要知道“这是不是任务真正终止”。因此不要把原始字段一开始就丢掉。

## 14. 常见误解

### “reward 就是模型的 loss”

不是。reward 来自环境；loss 是算法根据 reward、网络预测和目标值构造的优化信号。

### “Batch 就是 PyTorch 的 batch”

不完全是。Tianshou 的 `Batch` 是通用容器，可以装 NumPy、tensor、嵌套数据和 policy 附加信息；PyTorch tensor 只是其中一种字段值。

### “有 replay buffer 就一定是 DQN”

不是。很多 off-policy 算法使用 replay buffer；差别在于它们如何计算 target 和 loss。

### “测试一定完全没有探索”

不一定。DQN 的 `eps_inference` 可能仍大于 0。评估规则要看具体 policy 配置。

### “并行环境等于多 GPU”

不是。向量环境主要加速环境交互；网络训练是否用 GPU 是另一回事。

## 15. 阶段 1-3 最小词汇表

| 英文 | 中文直觉 |
|---|---|
| environment | 任务世界/模拟器 |
| observation | 智能体看到的信息 |
| action | 智能体做出的选择 |
| reward | 环境即时反馈 |
| episode | 从 reset 到结束的一局 |
| policy | 选择动作的规则 |
| Q value | 动作的长期好坏预测 |
| transition | 一步交互记录 |
| Batch | 多条结构化数据的对齐容器 |
| ReplayBuffer | 保存并随机抽取历史经验的环形记忆 |
| Collector | 驱动交互、生成经验的组件 |
| vectorized environment | 一次推进多个独立环境的接口 |
| gradient update | 根据 loss 调整网络参数的一次学习 |

学到这里就足以开始读 DQN 的训练链路。接下来再进入 PPO 或 SAC 时，模型和公式会更复杂，但这套“数据从哪里来、经过谁、在哪里变形”的基础不会变。
