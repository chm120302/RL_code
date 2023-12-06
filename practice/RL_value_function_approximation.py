import torch
from torch.utils import data

import RL_env
import numpy as np
import matplotlib.pyplot as plt
import time
from net import *


class Solve:
    def __init__(self, env: RL_env.GridEnv):
        self.gama = 0.9
        self.env = env
        self.action_space_size = env.action_space_size
        self.state_space_size = env.size ** 2
        self.reward_space_size = len(env.reward_list)
        self.reward_list = env.reward_list
        self.state_value = np.ones(shape=self.state_space_size)
        self.qvalue = np.ones(shape=(self.state_space_size, self.action_space_size))
        self.mean_policy = np.ones(shape=(self.state_space_size, self.action_space_size)) / self.action_space_size
        self.policy = self.random_epsilon_greedy_policy(1)  # exploring policy (mean policy)

    def show_policy_(self):
        for state in range(self.state_space_size):
            for action in range(self.action_space_size):
                policy = self.policy[state, action]
                self.env.render_.draw_action(pos=self.env.state2pos(state),
                                             toward=policy * 0.4 * self.env.pos_bias[action],
                                             radius=policy * 0.1)

    def show_state_value(self, state_value, y_offset=0.2):
        for state in range(self.state_space_size):
            self.env.render_.write_word(pos=self.env.state2pos(state), word=str(round(state_value[state], 1)),
                                        y_offset=y_offset,
                                        size_discount=0.7)

    def random_epsilon_greedy_policy(self, epsilon):
        policy = np.zeros(shape=(self.state_space_size, self.action_space_size))
        for state in range(self.state_space_size):
            action_ = np.random.choice(range(self.action_space_size))
            for action in range(self.action_space_size):
                if action == action_:
                    policy[state, action] = 1 - epsilon / self.action_space_size * (self.action_space_size - 1)
                else:
                    policy[state, action] = epsilon / self.action_space_size
        return policy

    def gfv(self, fourier: bool, state: int, ord: int) -> np.ndarray:
        """
        get_feature_vector with state value
        :param fourier: 是否使用傅里叶特征函数
        :param state: 状态
        :param ord: 特征函数最高阶次数/傅里叶q(对应书)
        :return: 代入state后的计算结果
        """

        if state < 0 or state >= self.state_space_size:
            raise ValueError("Invalid state value")
        # np.arrays 注意y和x是反着的 python中的矩阵是按行排列的
        y, x = self.env.state2pos(state) + (1, 1)
        feature_vector = []
        if fourier:  # use fourier function to approximate
            # normalization x, y in [0,1]
            x_normalized = x / self.env.size
            y_normalized = y / self.env.size
            for i in range(ord + 1):
                for j in range(ord + 1):
                    feature_vector.append(np.cos(np.pi * (i * x_normalized + j * y_normalized)))

        else:  # use polynomials function to approximate
            # normalization x, y in [-1,1]
            x_normalized = (x - (self.env.size - 1) * 0.5) / (self.env.size - 1)
            y_normalized = (y - (self.env.size - 1) * 0.5) / (self.env.size - 1)
            for i in range(ord + 1):
                for j in range(i + 1):
                    feature_vector.append(y_normalized ** (ord - i) * x_normalized ** j)

        return np.array(feature_vector)

    def calculate_qValue(self, state, action, state_value):
        qvalue = 0
        for index in range(self.reward_space_size):
            qvalue += self.reward_list[index] * self.env.Rsa[state, action, index]
        for next_state in range(self.state_space_size):
            qvalue += self.gama * self.env.Psa[state, action, next_state] * state_value[next_state]
        return qvalue

    def policy_evaluation(self, policy, threshold=0.001, step=10):
        state_value_k = np.ones(self.state_space_size)
        state_value = np.zeros(self.state_space_size)
        while np.linalg.norm(state_value_k - state_value, ord=1) > threshold:
            # 每轮将所有state都更新一次，迭代n次，每个state就一共迭代n次
            state_value = state_value_k.copy()
            for state in range(self.state_space_size):
                # state-value function = policy * action-value function
                value = 0
                for action in range(self.action_space_size):
                    value += policy[state, action] * self.calculate_qValue(state, action, state_value_k.copy())
                state_value_k[state] = value
        return state_value_k

    def obtain_episode(self, start_state, start_action, start_policy, length):
        self.env.agent_location = self.env.state2pos(start_state)
        episode = []
        next_action = start_action
        next_state = start_state
        while length > 0:
            length -= 1
            state = next_state
            action = next_action
            _, reward, done, _ = self.env.step(action)
            next_state = self.env.pos2state(self.env.agent_location)
            next_action = np.random.choice(np.arange(len(start_policy[next_state])), p=start_policy[next_state])
            episode.append({"state": state, "action": action, "reward": reward, "next state": next_state,
                            "next action": next_action})
        return episode

    def td_state_value_with_function_approximation(self, learning_rate=0.0015, episode_length=100000, fourier=True,
                                                   ord=3):
        self.state_value = self.policy_evaluation(policy=self.policy)
        start_state = np.random.randint(self.state_space_size)
        start_action = np.random.choice(np.arange(self.action_space_size), p=self.policy[start_state])
        episode = self.obtain_episode(start_state, start_action, self.policy, episode_length)
        if fourier:
            dim = (ord + 1) ** 2
        else:
            dim = np.arange(ord + 2).sum()
        #  default_rng: 随机数生成器(random number generator)，从高斯分布中采样，个数为dim，形成列向量
        w = np.random.default_rng().normal(size=dim)
        state_value_approximation = np.zeros(self.state_space_size)
        rmse = []
        for step in range(episode_length):
            reward = episode[step]['reward']
            next_state = episode[step]['next state']
            state = episode[step]['state']
            td_target = reward + self.gama * np.dot(self.gfv(fourier, next_state, ord), w)
            td_error = td_target - np.dot(self.gfv(fourier, state, ord), w)
            #  linear case
            gradient = self.gfv(fourier, state, ord)
            w += learning_rate * td_error * gradient
            # for state in range(self.state_space_size):
            state_value_approximation[state] = np.dot(self.gfv(fourier, state, ord), w)
            rmse.append(np.sqrt(np.mean((self.state_value - state_value_approximation) ** 2)))

        fig_rmse = plt.figure(figsize=(8, 6))  # 设置图形的尺寸，宽度为8，高度为6
        ax_rmse = fig_rmse.add_subplot(111)
        # 绘制 rmse 图像
        ax_rmse.plot(rmse)
        ax_rmse.set_title('RMSE')
        ax_rmse.set_xlabel('Epoch')
        ax_rmse.set_ylabel('RMSE')
        plt.show()
        return state_value_approximation

    def gfv_a(self, fourier: bool, state: int, action: int, ord: int) -> np.ndarray:
        feature_vector = []
        y, x = self.env.state2pos(state) + (1, 1)
        if fourier:
            x_normalized = x / self.env.size
            y_normalized = y / self.env.size
            action_normalized = action / self.action_space_size
            for i in range(ord + 1):
                for j in range(ord + 1):
                    for k in range(ord + 1):
                        feature_vector.append(
                            np.cos(np.pi * (i * x_normalized + j * y_normalized + k * action_normalized)))
        else:
            x_normalized = (x - (self.env.size - 1) * 0.5) / (self.env.size - 1)
            y_normalized = (y - (self.env.size - 1) * 0.5) / (self.env.size - 1)
            action_normalized = (action - (self.action_space_size - 1) * 0.5) / (self.action_space_size - 1)
            for i in range(ord + 1):
                for j in range(i + 1):
                    for k in range(j + 1):
                        feature_vector.append(
                            y_normalized ** (ord - i) * x_normalized ** (i - j) + action_normalized * k)
        return np.array(feature_vector)

    def sarsa_with_function_approximation(self, learning_rate=0.0015, epsilon=0.1, num_episodes=10000, fourier=True,
                                          ord=6):
        if fourier:
            dim = (ord + 1) ** 3
        else:
            dim = 0  # 不会计算
        w = np.random.default_rng().normal(size=dim)
        origin_iterations = num_episodes
        q_value_approximation = np.zeros(shape=(self.state_space_size, self.action_space_size))
        episode_length_list = []
        total_rewards_list = []
        self.policy = self.random_epsilon_greedy_policy(1)
        while num_episodes > 0:
            num_episodes -= 1
            start_state = 0
            start_action = np.random.choice(np.arange(self.action_space_size), p=self.policy[start_state])
            episode_length = 0
            total_reward = 0
            while start_state != self.env.pos2state(self.env.target_location):
                episode_length += 1
                episode = self.obtain_episode(start_state, start_action, self.policy, 1)
                reward = episode[0]['reward']
                next_state = episode[0]['next state']
                next_action = episode[0]['next action']
                total_reward += reward
                td_target = reward + self.gama * np.dot(self.gfv_a(fourier, next_state, next_action, ord), w)
                td_error = td_target - np.dot(self.gfv_a(fourier, start_state, start_action, ord), w)
                gradient = self.gfv_a(fourier, start_state, start_action, ord)
                w += learning_rate * td_error * gradient
                q_value_approximation[start_state, start_action] = np.dot(
                    self.gfv_a(fourier, start_state, start_action, ord), w)
                qvalue_opt = q_value_approximation[start_state].max()
                action_opt = q_value_approximation[start_state].tolist().index(qvalue_opt)
                for a in range(self.action_space_size):
                    if a == action_opt:
                        self.policy[start_state, a] = 1 - (
                                self.action_space_size - 1) / self.action_space_size * epsilon
                    else:
                        self.policy[start_state, a] = 1 / self.action_space_size * epsilon
                start_state = next_state
                start_action = next_action

            total_rewards_list.append(total_reward)
            episode_length_list.append(episode_length)
            print("episode={},length={}, rewards={}".format(origin_iterations - num_episodes, episode_length,
                                                            total_reward))

        fig = plt.figure(figsize=(10, 10))
        ax1 = fig.add_subplot(211)
        ax1.plot(episode_length_list)
        ax1.set_xlabel('episode index')
        ax1.set_ylabel('episode length')

        fig_rw = plt.figure(figsize=(10, 10))
        ax2 = fig_rw.add_subplot(212)
        ax2.plot(total_rewards_list)
        ax2.set_xlabel('episode index')
        ax2.set_ylabel('total rewards')
        plt.show()
        return q_value_approximation

    def q_learning_with_function_approximation(self, learning_rate=0.0015, epsilon=0.1, num_episodes=500, fourier=True,
                                               ord=5):
        if fourier:
            dim = (ord + 1) ** 3
        else:
            dim = 0  # 不会计算
        w = np.random.default_rng().normal(size=dim)
        origin_iterations = num_episodes
        q_value_approximation = np.zeros(shape=(self.state_space_size, self.action_space_size))
        total_reward_list = []
        episode_length_list = []
        self.policy = self.random_epsilon_greedy_policy(1)

        while num_episodes > 0:
            num_episodes -= 1
            start_state = 0
            start_action = np.random.choice(np.arange(self.action_space_size), p=self.policy[start_state])
            episode_length = 0
            total_rewards = 0
            while start_state != self.env.pos2state(self.env.target_location):
                episode_length += 1
                episode = self.obtain_episode(start_state, start_action, self.policy, 1)
                reward = episode[0]['reward']
                next_state = episode[0]['next state']
                next_action = episode[0]['next action']
                total_rewards += reward
                q_approx_next_state_max = np.dot(self.gfv_a(fourier, next_state, next_action, ord), w)
                for action in range(self.action_space_size):
                    q = np.dot(self.gfv_a(fourier, next_state, action, ord), w)
                    if q > q_approx_next_state_max:
                        q_approx_next_state_max = q
                td_target = reward + self.gama * q_approx_next_state_max
                td_error = td_target - np.dot(self.gfv_a(fourier, start_state, start_action, ord), w)
                gradient = self.gfv_a(fourier, start_state, start_action, ord)
                w += learning_rate * td_error * gradient
                q_value_approximation[start_state, start_action] = np.dot(
                    self.gfv_a(fourier, start_state, start_action, ord), w)
                qvalue_opt = q_value_approximation[start_state].max()
                action_opt = q_value_approximation[start_state].tolist().index(qvalue_opt)
                for a in range(self.action_space_size):
                    if a == action_opt:
                        self.policy[start_state, a] = 1 - (
                                self.action_space_size - 1) / self.action_space_size * epsilon
                    else:
                        self.policy[start_state, a] = 1 / self.action_space_size * epsilon
                start_state = next_state
                start_action = np.random.choice(np.arange(self.action_space_size), p=self.policy[start_state])

            total_reward_list.append(total_rewards)
            episode_length_list.append(episode_length)
            print("episode={},length={},total rewards={}".format(origin_iterations - num_episodes, episode_length,
                                                                 total_rewards))

        fig = plt.figure(figsize=(10, 10))
        ax1 = fig.add_subplot(211)
        ax1.plot(episode_length_list)
        ax1.set_xlabel('episode index')
        ax1.set_ylabel('episode length')

        fig_rw = plt.figure(figsize=(10, 10))
        ax2 = fig_rw.add_subplot(212)
        ax2.plot(total_reward_list)
        ax2.set_xlabel('episode index')
        ax2.set_ylabel('total rewards')
        plt.show()
        return q_value_approximation

    def get_data_iter(self, episode, batch_size=64, is_train=True):
        """

        :param episode: 输入一条生成的episode
        :param batch_size: 规定dataLoader中的batch_size
        :param is_train: 规定dataLoader中的shuffle
        :return: 返回pytorch数据迭代器
        """
        reward = []
        state_action = []
        next_state = []
        for step in range(len(episode)):
            reward.append(episode[step]['reward'])
            y, x = self.env.state2pos(episode[step]['state'])
            action = episode[step]['action']
            state_action.append((y, x, action))
            y, x = self.env.state2pos(episode[step]['next state'])
            next_state.append((y, x))
        # reshape(-1, 1)变成一列，reshape(1, -1)变成一行
        reward = torch.tensor(reward).reshape(-1, 1)
        state_action = torch.tensor(state_action)
        next_state = torch.tensor(next_state)
        data_arrays = (state_action, reward, next_state)
        dataset = data.TensorDataset(*data_arrays)
        return data.DataLoader(dataset, batch_size, shuffle=is_train, drop_last=False)

    def dqn(self, learning_rate=0.0015, episode_length=5000, epochs=600, batch_size=100, update_step=10):
        # 这种程序不适合用gpu，还没cpu快，gpu 花费时间：232  cpu花费时间：148
        # if torch.cuda.is_available():
        #     device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
        #     print("Running on the GPU")
        # else:
        #     device = torch.device("cpu")
        #     print("Running on the CPU")

        # 针对模型的.to方法是inplace的, 针对tensor的.to方法不是inplace的，因此如果只是写成input_ids.to(device)，数据并不会被放到gpu中。
        q_net = QNET()
        # q_net.to(device)
        target_net = QNET()
        # target_net.to(device)
        target_net.load_state_dict(q_net.state_dict())  # 保持参数相同
        optimizer = torch.optim.SGD(q_net.parameters(), lr=learning_rate)  # 定义优化器
        loss = torch.nn.MSELoss()  # 平方损失函数
        # 从(0, 0)位置开始探索
        episode = self.obtain_episode(0, 0, self.random_epsilon_greedy_policy(1), episode_length)
        data_iter = self.get_data_iter(episode, batch_size)
        q_value_approximation = np.zeros(shape=(self.state_space_size, self.action_space_size))
        update_q_index = 0  # 确定是否更新target net
        rmse_list = []  # state value error
        loss_list = []
        self.state_value = np.zeros(shape=self.state_space_size)
        # state_value = self.state_value.copy()
        for epoch in range(epochs):
            loss_value = 0
            for state_action, reward, next_state in data_iter:
                # state_action, reward, next_state = state_action.to(device), reward.to(device), next_state.to(device)
                #  每次迭代载入batch_size大小的数据
                update_q_index += 1
                q_approx_value = q_net(state_action)  # 得到的是一个100*1的列向量q
                q_approx_target = torch.empty((batch_size, 0))  # 定义空的张量[]，若为(batch_size, 1)会自动赋值
                # q_approx_target = q_approx_target.to(device)
                for action in range(self.action_space_size):
                    # torch.full在大小为(100,1)的列向量中填充action的值
                    # torch.cat进行张量拼接，dim=0按行拼接，dim=1按列拼接
                    action_tensor = torch.full((batch_size, 1), action)
                    # action_tensor = action_tensor.to(device)
                    s_a = torch.cat((next_state, action_tensor), dim=1)
                    # s_a = s_a.to(device)
                    q_approx_target = torch.cat((q_approx_target, target_net(s_a)), dim=1)
                #  torch.max(input, dim, keepdim=False)
                #  dim = 0 寻找每一列的最大值，dim = 1寻找每一行的最大值
                #  keepdim 是否需要保持输出的维度与输入一样
                #  https://blog.csdn.net/zylooooooooong/article/details/112576268
                #  torch.max()[0]只返回最大值的value
                #  torch.max()[1]返回最大值对应的索引
                q_opt = torch.max(q_approx_target, dim=1, keepdim=True)[0]  # 得到100*1的列向量q_opt
                # q_opt = q_opt.to(device)
                y_target = reward + self.gama * q_opt
                # 计算损失函数
                l = loss(q_approx_value, y_target)
                loss_value += l
                # 开始优化，将梯度置为0(PyTorch中默认梯度会累积)
                optimizer.zero_grad()
                # 执行反向传播计算梯度
                l.backward()
                # 并通过优化器更新模型参数
                optimizer.step()

                if update_q_index % update_step == 0 and update_q_index != 0:
                    target_net.load_state_dict(q_net.state_dict())
            loss_list.append(float(loss_value))
            print("loss:{},epoch:{}".format(loss_value, epoch))

            self.policy = np.zeros(shape=(self.state_space_size, self.action_space_size))
            state_value = self.state_value.copy()
            self.state_value = np.zeros(shape=self.state_space_size)
            for s in range(self.state_space_size):
                y, x = self.env.state2pos(s)
                for a in range(self.action_space_size):
                    s_a_tensor = torch.tensor((y, x, a)).reshape(-1, 3)
                    # s_a_tensor = s_a_tensor.to(device)
                    q_value_approximation[s, a] = float(q_net(s_a_tensor))
                q_star_index = q_value_approximation[s].argmax()
                self.policy[s, q_star_index] = 1
                self.state_value[s] = q_value_approximation[s, q_star_index]
            rmse_list.append(np.sqrt(np.mean((state_value - self.state_value) ** 2)))
        self.show_policy()

        fig = plt.figure(figsize=(8, 12))
        ax_rmse = fig.add_subplot(211)
        ax_rmse.plot(rmse_list)
        ax_rmse.set_title('RMSE')
        ax_rmse.set_xlabel('Epoch')
        ax_rmse.set_ylabel('RMSE')

        ax_loss = fig.add_subplot(212)
        ax_loss.plot(loss_list)
        ax_loss.set_title('loss')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        plt.show()

    def show_policy(self):
        print("policy is: ")
        for state in range(self.state_space_size):
            for action in range(self.action_space_size):
                print(self.policy[state, action], end='  ')
            print("")


if __name__ == "__main__":


    env = RL_env.GridEnv(size=5, target=[2, 3],
                         forbidden=[[2, 2], [2, 1], [1, 1], [3, 3], [1, 4], [1, 3]], render_mode='')
    solver = Solve(env)

    start_time = time.time()
    num_episodes = 1000
    # solver.td_state_value_with_function_approximation()
    # solver.sarsa_with_function_approximation()
    # solver.q_learning_with_function_approximation()

    # 最后一个epoch的loss收到0.04
    # 增加隐藏层，loss和rsme都下降了
    solver.dqn()
    end_time = time.time()
    cost_time = end_time - start_time
    print("cost_time:{}".format(round(cost_time, 2)))
    x = np.arange(num_episodes)
    print(len(env.render_.trajectory))
    solver.show_policy_()  # solver.env.render()
    solver.show_state_value(solver.state_value, y_offset=0.25)
    solver.env.render()
