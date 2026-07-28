import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

import tianshou as ts
from tianshou.algorithm.modelfree.dqn import DiscreteQLearningPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import CollectStats
from tianshou.trainer import OffPolicyTrainerParams
from tianshou.utils.net.common import Net
from tianshou.utils.space_info import SpaceInfo


def main() -> None:
    # [学习导读-阶段1] CartPole 是离散动作的入门环境：每一时刻观察 4 个数，
    # DQN 从两个动作的 Q 值中选择一个。先用它理解完整训练链路，再学习复杂环境。
    task = "CartPole-v1"
    # `batch_size` 是一次梯度更新使用的旧经验数量；它不是并行环境数量。
    lr, epoch, batch_size = 1e-3, 10, 64
    # 训练和测试环境必须分开：训练环境用于产生经验，测试环境只负责公平评估当前策略。
    num_training_envs, num_test_envs = 10, 100
    # n_step 用于构造更长的 TD 目标；target_freq 控制目标网络多久同步一次。
    gamma, n_step, target_freq = 0.9, 3, 320
    buffer_size = 20000
    eps_train, eps_test = 0.1, 0.05
    epoch_num_steps, collection_step_num_env_steps = 10000, 10

    # [学习导读-阶段1] logger 不参与学习；它只是把 reward、loss、速度等训练证据写到磁盘。
    logger = ts.utils.TensorboardLogger(SummaryWriter("log/dqn"))  # TensorBoard is supported!
    # For other loggers, see https://tianshou.readthedocs.io/en/master/tutorials/logger.html

    # [学习导读-阶段1] `DummyVectorEnv` 把多个独立环境包装成一个“批量环境”。
    # 一次 collector.step 会同时推进这些环境，并把第 0 维作为环境/样本维度。
    # Create the environments
    # You can also try SubprocVectorEnv, which will use parallelization
    training_envs = ts.env.DummyVectorEnv(
        [lambda: gym.make(task) for _ in range(num_training_envs)]
    )
    test_envs = ts.env.DummyVectorEnv([lambda: gym.make(task) for _ in range(num_test_envs)])

    # [学习导读-阶段1] 先创建单个 env 读取 observation/action space；网络的输入和输出维度
    # 不应该靠手写常量猜测，而应由环境空间推导。
    # Create the network and optimizer
    # Note: You can easily define other networks.
    # See https://tianshou.readthedocs.io/en/master/01_tutorials/00_dqn.html#build-the-network
    env = gym.make(task, render_mode="human")
    assert isinstance(env.action_space, gym.spaces.Discrete)
    space_info = SpaceInfo.from_env(env)
    state_shape = space_info.observation_info.obs_shape
    action_shape = space_info.action_info.action_shape
    # `Net` 输出每个离散动作的 Q 值，shape 通常为 [batch_size, action_num]。
    net = Net(state_shape=state_shape, action_shape=action_shape, hidden_sizes=[128, 128, 128])
    # 工厂把“怎样创建优化器”交给 Algorithm；此处还没有真正调用 optimizer.step()。
    optim = AdamOptimizerFactory(lr=lr)

    # Policy 只回答“当前观测应采取什么动作”，并封装 epsilon-greedy 探索。
    policy = DiscreteQLearningPolicy(
        model=net,
        action_space=env.action_space,
        eps_training=eps_train,
        eps_inference=eps_test,
    )
    # Algorithm 负责“怎样用经验学习”：计算 TD target、loss、反向传播和目标网络更新。
    algorithm = ts.algorithm.DQN(
        policy=policy,
        optim=optim,
        gamma=gamma,
        n_step_return_horizon=n_step,
        target_update_freq=target_freq,
    )
    # [学习导读-阶段1] Collector 是 policy 和 env 之间的调度器：
    # obs -> policy action -> env.step -> transition -> replay buffer。
    training_collector = ts.data.Collector[CollectStats](
        algorithm,
        training_envs,
        # 每个并行环境保留自己的轨迹边界，避免不同 episode 的 transition 被错误拼接。
        ts.data.VectorReplayBuffer(buffer_size, num_training_envs),
        exploration_noise=True,
    )
    # 测试不写 replay buffer；测试数据不应该被拿来训练。
    test_collector = ts.data.Collector[CollectStats](
        algorithm,
        test_envs,
        exploration_noise=True,
    )  # because DQN uses epsilon-greedy method

    def stop_fn(mean_rewards: float) -> bool:
        # [学习导读-阶段1] 训练停止依赖“多条测试 episode 的平均回报”，
        # 而不是某一次偶然成功的轨迹。
        if env.spec:
            if not env.spec.reward_threshold:
                return False
            else:
                return mean_rewards >= env.spec.reward_threshold
        return False

    # [学习导读-阶段1] OffPolicyTrainer 的节奏是：收集少量新经验 -> 从历史 buffer
    # 随机采样 batch -> 做梯度更新 -> 定期测试。DQN 可以重复利用旧经验，所以称 off-policy。
    result = algorithm.run_training(
        OffPolicyTrainerParams(
            training_collector=training_collector,
            test_collector=test_collector,
            max_epochs=epoch,
            epoch_num_steps=epoch_num_steps,
            collection_step_num_env_steps=collection_step_num_env_steps,
            test_step_num_episodes=num_test_envs,
            batch_size=batch_size,
            # 此设置表示平均每收集 1 个环境步做 0.1 次梯度更新；它决定“采样”和“学习”的比例。
            update_step_num_gradient_steps_per_sample=1 / collection_step_num_env_steps,
            stop_fn=stop_fn,
            logger=logger,
            test_in_training=True,
        )
    )
    print(f"Finished training in {result.timing.total_time} seconds")

    # 最后单独创建一个环境观看策略。这里仍保留 DQN 的 epsilon-greedy 行为，
    # 因此它展示的是“带探索的测试表现”，不是完全贪心策略。
    # watch performance
    collector = ts.data.Collector[CollectStats](algorithm, env, exploration_noise=True)
    collector.collect(n_episode=100, render=1 / 35)


if __name__ == "__main__":
    main()
