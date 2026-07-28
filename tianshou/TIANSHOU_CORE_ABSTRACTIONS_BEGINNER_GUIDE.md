# Tianshou 核心抽象：从基本模块到完整 RL 系统的新手讲解

> 阅读对象：刚开始学习 PyTorch、强化学习与工程化实现的读者。
>
> 阅读目标：看懂 Tianshou 如何把一个 RL 系统拆成可独立理解、可替换、可测试的模块，而不是先记住 DQN、PPO 等算法名称。

## 1. 先看全局：六个模块如何协作

你引用的文档说明，可以压缩为下面这一条数据流：

```text
环境给出 obs
    |
    v
Policy 选择 act
    |
    v
Collector 调用 env.step(act)，生成 transition
    |
    v
ReplayBuffer 保存历史 transition
    |
    v
ReplayBuffer.sample() 返回 Batch
    |
    v
Algorithm 用 Batch 计算 loss 并更新 Policy
    |
    v
Trainer 重复安排 collect -> update -> test -> log
```

这不是六个彼此孤立的类，而是一条职责明确的生产线：

| 模块 | 把什么变成什么 | 最关键的问题 |
|---|---|---|
| `Policy` | `obs -> act` | 当前状态下应该做什么动作？ |
| `Collector` | `obs + act -> transition` | 如何安全地与环境交互并记录经验？ |
| `ReplayBuffer` | transition 流 -> 可采样历史数据 | 如何保存、覆盖、索引与抽样经验？ |
| `Batch` | 多个字段 -> 对齐的一批数据 | 如何让 `obs/act/rew/...` 始终属于同一批样本？ |
| `Algorithm` | Batch -> loss -> 新参数 | 如何从经验中学习？ |
| `Trainer` | 各模块 -> 完整实验过程 | 什么时候采样、更新、测试、保存与停止？ |

## 2. 为什么要拆成这些模块

如果把所有逻辑塞进一个 `train()` 函数，会很快遇到问题：换环境会碰算法代码，换算法会碰环境循环，测试时也很难知道哪里坏了。

Tianshou 的拆分让每层只承担一种变化：

```text
换任务环境       -> 主要替换 env
换神经网络       -> 主要替换 policy 内的网络
换学习规则       -> 主要替换 algorithm
换采样方式       -> 主要替换 collector / vector env
换训练计划       -> 主要替换 trainer 参数
```

这就是 RL infra 的核心工程能力：把“经常变化的部分”隔离开，把“所有算法都要做的流程”复用起来。

## 3. 先认识数据：transition 和 Batch

### 3.1 一条 transition

强化学习中的一条经验通常写作：

```text
(obs, act, rew, obs_next, terminated, truncated)
```

| 字段 | 新手解释 | CartPole 示例 |
|---|---|---|
| `obs` | 动作前看到了什么 | 小车位置、速度、杆角度等 4 个数 |
| `act` | 当时做了什么 | 向左或向右推 |
| `rew` | 环境的即时反馈 | 杆没有倒通常获得正奖励 |
| `obs_next` | 动作后看到了什么 | 下一时刻的 4 个数 |
| `terminated` | 任务是否自然结束 | 杆倒了等失败状态 |
| `truncated` | 是否被外部限制截断 | 例如达到时间上限 |

`done = terminated OR truncated` 表示这一局不能继续推进，但保留两个原始字段能让算法区分“真实终止”和“时间截断”。

### 3.2 Batch：让多条经验始终对齐

`Batch` 位于 [batch.py](D:/Code/Projects/AI与数据科学/AI_programs/tianshou/tianshou/data/batch.py:625)。把它先理解为一张按列储存的表：

```text
第 0 条：obs[0], act[0], rew[0], obs_next[0]
第 1 条：obs[1], act[1], rew[1], obs_next[1]
第 2 条：obs[2], act[2], rew[2], obs_next[2]
```

因此 `batch[:2]` 不是只切 observation，而是同时切所有字段，仍然得到两条完整且对齐的经验。

PyTorch 视角：`Batch` 本身不是 tensor；它是容器。环境/Buffer 常保存 NumPy，而算法更新前通过 `Batch.to_torch()` 将数值字段转成对应 device 上的 tensor。这个边界很重要：Gymnasium 使用 NumPy，神经网络反向传播使用 PyTorch。

## 4. Policy：只负责“如何行动”

源码入口：[algorithm_base.py](D:/Code/Projects/AI与数据科学/AI_programs/tianshou/tianshou/algorithm/algorithm_base.py:161)。

### 4.1 Policy 的职责

`Policy` 继承 `torch.nn.Module`，因为内部通常有 actor、Q 网络或其他神经网络参数；但它的公共职责仍然很单纯：给定 observation，输出动作或构造动作所需的中间信息。

| 方法 | 用新手语言理解 |
|---|---|
| `forward(batch, state)` | 批量输入 observation，输出至少包含 `act` 的 Batch |
| `compute_action(obs)` | 单环境调试/部署的便利入口，自动补 batch 维 |
| `map_action(act)` | 把网络原始输出变成环境真正能执行的动作 |
| `add_exploration_noise(act, batch)` | 训练时可覆写为探索机制 |

### 4.2 `forward` 的输入输出契约

`Policy.forward` 是具体策略子类实现的核心方法。它的输入不是裸 tensor，而是含 `obs`、可选 `info` 的 Batch；它的输出也是 Batch。

```text
输入：Batch(obs=[B, observation_dim], info=...)
输出：Batch(act=[B, ...], state=可选, policy=可选)
```

`B` 是 batch size。它既可能是并行环境数，也可能是训练时从 Buffer 抽出的样本数；要靠调用位置判断。

### 4.3 为什么还要 `map_action`

神经网络的输出空间与环境动作空间未必相同：

```text
网络原始输出 [-inf, +inf]
    -> clip 或 tanh
标准化范围 [-1, 1]
    -> scaling
环境所需范围 [action_space.low, action_space.high]
```

CartPole 是离散动作，这一步基本没有数值缩放；MuJoCo 一类连续控制环境则非常依赖它。这个设计让网络本身不需要知道每个环境具体的动作范围。

### 4.4 学习 Policy 时该看什么

先不要读全部子类。按顺序：

1. 读基类的 `compute_action`，理解“单个 obs 如何补 batch 维”。
2. 读 [dqn.py](D:/Code/Projects/AI与数据科学/AI_programs/tianshou/tianshou/algorithm/modelfree/dqn.py:33) 的 `DiscreteQLearningPolicy.forward`。
3. 在 Collector 调用 Policy 的位置打断点，观察 `obs_batch_R.obs.shape` 与 `act_batch_RA.act.shape`。

验收问题：Policy 为什么不直接调用 `env.step()`？答案是：Policy 只描述决策，环境交互是 Collector 的职责；分离后同一策略才能服务训练、测试和部署。

## 5. Collector：把“行动”变成“经验”

源码入口：[collector.py](D:/Code/Projects/AI与数据科学/AI_programs/tianshou/tianshou/data/collector.py:302)。

### 5.1 Collector 的职责

Collector 持有 Policy、环境和 Buffer，反复执行：

```text
last_obs
  -> policy 得到 act
  -> env.step(act)
  -> 得到 obs_next、rew、terminated、truncated、info
  -> 打包为 Batch
  -> buffer.add(batch)
```

它不计算 loss，也不调用 `optimizer.step()`。因此同一个 Collector 能服务 DQN、PPO、随机策略或你自己的新策略。

### 5.2 关键方法

| 方法 | 职责 |
|---|---|
| `BaseCollector.collect` | 公开入口，规定按 `n_step` 或 `n_episode` 收集 |
| `Collector._compute_action_policy_hidden` | 组织输入 Batch，调用 policy，映射动作，传递 RNN hidden state |
| `Collector._collect` | 执行循环：取动作、step、统计、写 Buffer、done 后 reset |

### 5.3 向量环境与后缀字母

Tianshou 频繁使用向量环境。若同时运行 `R` 个环境，变量名中的 `R` 常表示第 0 维长度为当前活跃环境数：

```text
last_obs_RO: [R, observation_dim]
act_RA:      [R, action_dim] 或 [R]
rew_R:       [R]
done_R:      [R]
```

第一次阅读时只记住：第 0 维对应“同时推进的环境/样本”。不用马上掌握全部下标命名。

### 5.4 为什么训练和测试各有一个 Collector

训练 Collector 的经验会写入 Buffer，供 Algorithm 更新。测试 Collector 的职责只是衡量当前策略，通常不需要保存经验。这样可避免评估数据污染训练，也能独立配置训练与评估环境数量。

## 6. ReplayBuffer：把经验流变成可学习数据

源码入口：[buffer_base.py](D:/Code/Projects/AI与数据科学/AI_programs/tianshou/tianshou/data/buffer/buffer_base.py:27)。

### 6.1 Buffer 的两项工作

1. 用固定容量的环形队列保存 transition。
2. 保存 episode 边界和时间邻接关系，让算法能正确计算 n-step return 等量。

关键状态：

| 字段 | 作用 |
|---|---|
| `maxsize` | 最大容量 |
| `_insertion_idx` | 下一条经验要写入的位置 |
| `_size` | 当前有效经验数量，最大不超过 `maxsize` |
| `_meta` | 真正保存 `obs/act/rew/...` 字段的底层 Batch |
| `_ep_return` / `_ep_len` | 当前 episode 的累计回报与步数 |

### 6.2 为什么是环形队列

假设容量是 3：

```text
加入 A, B, C：buffer 已满
加入 D：覆盖最旧的 A
加入 E：覆盖最旧的 B
```

这控制内存成本。RL 不需要永久保存全部历史，只需要足够多、足够多样的近期经验。

### 6.3 `add`、`sample` 与 `indices`

| 方法 | 做什么 |
|---|---|
| `add(batch)` | 写入 transition，更新 done、episode 长度、回报和写指针 |
| `sample_indices(batch_size)` | 只决定抽哪些位置 |
| `sample(batch_size)` | 返回 `(Batch, indices)` |
| `next(index)` | 找同一 episode 的下一条经验；不会越过 done |

为什么 `sample` 还返回 `indices`？优先经验回放等扩展需要把 TD error 写回“原来抽到的那几条经验”，所以必须知道它们在 Buffer 中的位置。

## 7. Algorithm：把 Batch 变成参数更新

源码入口：[algorithm_base.py](D:/Code/Projects/AI与数据科学/AI_programs/tianshou/tianshou/algorithm/algorithm_base.py:437)。

### 7.1 Algorithm 的职责边界

Algorithm 继承 `torch.nn.Module`，因为它除了 Policy 外也可能持有 critic、target network、optimizer、学习率调度器等训练状态。

它不推进环境，也不管理 epoch。它只关心：从 Buffer 拿到一批经验后，如何计算并执行一次学习更新。

### 7.2 通用更新骨架

`Algorithm._update` 是理解框架最重要的方法之一：

```text
buffer.sample(sample_size)
    -> batch, indices
_preprocess_batch(batch, buffer, indices)
    -> 加 returns / advantage / target Q 等字段
_update_with_batch(batch)
    -> forward -> loss -> backward -> optimizer.step
_postprocess_batch(batch, buffer, indices)
    -> 例如更新优先经验回放权重
lr_scheduler.step()
```

具体算法只需要实现自己真正不同的部分：

| 算法类别 | 常在预处理阶段增加什么 | 常在更新阶段做什么 |
|---|---|---|
| DQN | n-step TD target / `returns` | Q loss 与 target network 更新 |
| PPO | return、advantage、old log prob | clipped policy loss、value loss、entropy |
| SAC | Q target、动作分布相关量 | actor loss、双 critic loss、温度参数 |

这是一种很好的工程设计：把固定流程放在基类，把数学差异放在子类。

### 7.3 先理解 on-policy / off-policy 的工程差异

| 类型 | 数据来源 | Trainer 更新节奏 |
|---|---|---|
| off-policy，如 DQN | 长期 ReplayBuffer | 收集少量新数据后，从历史中随机抽样多次更新 |
| on-policy，如 PPO | 当前策略刚收集的数据 | 收集一批后更新，通常不长期复用旧轨迹 |
| offline RL | 固定数据 Buffer | 不与环境交互，只不断采样更新 |

不要先背公式。先问：该算法的 Buffer 是否长期保留？每收集一次数据后更新几次？这两个问题决定了 Trainer 和 Buffer 的使用方式。

## 8. Trainer：把组件变成一次完整实验

源码入口：[trainer.py](D:/Code/Projects/AI与数据科学/AI_programs/tianshou/tianshou/trainer.py:356)。

### 8.1 Trainer 的职责

Trainer 把 Algorithm、Collector、日志器和训练参数组织起来：

```text
reset
  -> execute_epoch
       -> 多次 _training_step
       -> checkpoint
       -> test collector 评估
       -> 保存 best model / early stop
  -> 下一个 epoch
```

`Trainer` 基类不包含 DQN loss。不同学习范式把 `_training_step` 留给 `OffPolicyTrainer`、`OnPolicyTrainer`、`OfflineTrainer` 去实现。

### 8.2 你需要看懂的三个方法

| 方法 | 作用 |
|---|---|
| `run` | 最高层训练入口，reset 后循环 epoch |
| `execute_epoch` | 一个 epoch 的调度：training step、日志、checkpoint、测试 |
| `_training_step` | 子类扩展点，决定 collect/update 的实际节奏 |

### 8.3 工程上为什么需要 Trainer

把日志、测试、提前停止、保存最佳模型放到 Trainer，而不放在 Algorithm 中，有三个好处：

1. 算法实现可以只关注数学更新，不被实验管理细节淹没。
2. 不同算法可以共享 checkpoint、TensorBoard、early stop 的逻辑。
3. 测试更加容易：既可测试 DQN 的更新，也可单独测试训练编排。

## 9. 一次 DQN 训练到底发生了什么

把前面六个模块连起来，DQN 的一次训练 step 可读成：

```text
1. Trainer 调用 OffPolicyTrainer._training_step
2. Collector.collect(n_step=...)
3. Policy.forward(obs Batch) 产生动作
4. Collector 调用 env.step，并把 transition 写入 ReplayBuffer
5. Trainer 调用 DQN.update(buffer, batch_size)
6. Algorithm._update 从 ReplayBuffer.sample 得到 Batch
7. DQN 预处理 batch，计算 TD target
8. DQN 的具体更新函数计算 loss，PyTorch 执行 backward/optimizer.step
9. Trainer 记录 loss，定期用 test Collector 评估
```

你不必现在理解第 7、8 步所有公式；但必须理解第 1 到 6 步的数据由谁拥有、在哪里变形。这正是 RL infra 的基础。

## 10. 从 PyTorch 角度学习这个架构

| PyTorch 概念 | 在 Tianshou 中的位置 | 为什么重要 |
|---|---|---|
| `nn.Module` | `Policy`、`Algorithm` | 参数注册、`.train()`/`.eval()`、`state_dict()` |
| tensor shape | Policy/网络输入输出、训练 Batch | shape 错误是 RL 实现最常见 bug 之一 |
| device / dtype | `Batch.to_torch()`、网络参数 | 环境数据常是 NumPy，训练数据必须与模型 device 一致 |
| optimizer | Algorithm 的 optimizer wrapper | 统一管理 `zero_grad`、`backward`、step、梯度裁剪与状态保存 |
| `torch.no_grad()` | target value 计算等 | 防止目标网络分支意外建立梯度图 |
| `state_dict` | Policy/Algorithm/optimizer | checkpoint 必须同时考虑模型与优化器状态 |

新手最值得养成的习惯：每次追一条数据，都写下它的 `type`、`shape`、`dtype`、`device`。比盯着公式更早发现问题。

## 11. 从 RL infra 角度学习这个架构

| 工程问题 | Tianshou 的回答 |
|---|---|
| 如何并行收集经验？ | VectorEnv + Collector |
| 如何让多字段数据保持对齐？ | Batch |
| 如何复用旧经验？ | ReplayBuffer |
| 如何支持多种学习范式？ | OnPolicy/OffPolicy/Offline Algorithm 与 Trainer 子类 |
| 如何避免算法与实验管理耦合？ | Algorithm 和 Trainer 分离 |
| 如何验证行为没有退化？ | `test/base/` 的组件测试与算法训练测试 |

这张表可以作为你读任何 RL 项目时的检查清单。即使将来你读 Stable-Baselines3、RLlib 或自己写项目，也能用同样的问题定位模块。

## 12. 推荐学习顺序：六次短学习，不要一次通读

每次 45 到 90 分钟。每完成一次，写 5 行自己的总结再进入下一次。

### 第 1 次：数据先行

1. 运行 `poetry run pytest test/base/test_batch.py -q`。
2. 读 `Batch.__init__`、`__getitem__`、`to_torch`。
3. 手写一个含 `obs/act/rew` 的 Batch，尝试 `batch[0]`、`batch[:2]`。

验收：解释为什么 `Batch[:2]` 不会让 observation 与 reward 错位。

### 第 2 次：经验记忆

1. 运行 `poetry run pytest test/base/test_buffer.py -q`。
2. 读 `ReplayBuffer.__init__`、`add`、`sample`、`next`。
3. 用容量为 3 的 Buffer 连续写 5 条 transition，记录 `_insertion_idx`。

验收：解释为什么 buffer 满后不会无限增长，以及为什么 `next()` 不能跨 episode。

### 第 3 次：动作选择

1. 读 `Policy.compute_action` 与 `map_action`。
2. 读 DQN 的 `DiscreteQLearningPolicy.forward`。
3. 在 `Collector._compute_action_policy_hidden` 打断点。

验收：指出从 NumPy observation 到环境动作的每一次转换。

### 第 4 次：采样循环

1. 运行 `poetry run pytest test/base/test_collector.py -q`。
2. 只读 `Collector._collect` 的 Step 2、3、6。
3. 观察 `current_step_batch_R.get_keys()`。

验收：口头讲清一条 transition 从 env 到 buffer 的路径。

### 第 5 次：学习更新

1. 读 `Algorithm._update`。
2. 读 `DQN._preprocess_batch` 与 `DQN._update_with_batch`。
3. 画出 `sample -> returns -> loss -> optimizer.step`。

验收：说明为什么 Algorithm 需要 `indices`，以及为什么预处理与具体 loss 分开。

### 第 6 次：训练编排

1. 读 `Trainer.run`、`Trainer.execute_epoch`。
2. 读 `OffPolicyTrainer._training_step`。
3. 回到 `examples/discrete/discrete_dqn.py`，把每个对象连到对应基类。

验收：解释为什么 `Trainer` 不应该直接包含 DQN loss。

## 13. 断点清单：每次只观察四件事

推荐断点顺序：

1. `Policy.compute_action` 或具体 `Policy.forward`
2. `Collector._compute_action_policy_hidden`
3. `Collector._collect` 的 `env.step`
4. `ReplayBuffer.add`
5. `ReplayBuffer.sample`
6. `Algorithm._update`
7. 具体算法的 `_update_with_batch`
8. `Trainer.execute_epoch`

每个断点只看：

```text
它是谁调用的？
输入的 type/shape/device 是什么？
输出增加或删除了哪些字段？
它修改了什么长期状态？
```

不要一上来观察全部变量；那会让你淹没在对象细节中。

## 14. 常见误解

### “Policy 就是整个算法”

不是。Policy 是行动规则；Algorithm 是学习规则。DQN 的 Q 网络可以位于 Policy 中，但 DQN 的 target、loss、optimizer 调度属于 Algorithm。

### “ReplayBuffer 是数据库”

它更像专为 RL 设计的内存数据结构：不仅保存字段，还要维护环形覆盖、episode 边界和时间相邻关系。

### “Trainer 就是一个 for 循环，所以不重要”

Trainer 是实验可复现性的核心：它决定采样频率、更新频率、测试集、日志、checkpoint 与提前停止。RL 结果不稳定时，很多问题就出在这里的参数与节奏上。

### “Batch 只是为了方便写代码”

Batch 是数据契约。它让 Collector、Buffer、Policy 和 Algorithm 可以交换同一种结构化数据，而不用把每个字段作为长参数列表传来传去。

## 15. 你完成这一轮后的标准

不看源码，用自己的话讲清：

1. `Policy`、`Algorithm` 的区别。
2. `Collector` 为什么同时依赖 Policy、Env 和 Buffer。
3. Buffer 抽样后为什么是 `Batch + indices`。
4. Algorithm 的三段式更新骨架。
5. Trainer 为什么负责流程但不负责 loss。

讲清这五件事后再深入 DQN/PPO/SAC。此时你不只是会“运行一个 RL 脚本”，而是已经掌握了理解 RL 工程项目的骨架。
