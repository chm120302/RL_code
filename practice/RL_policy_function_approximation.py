import time

import numpy as np
import torch
from net import *
import matplotlib.pyplot as plt
import RL_env


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
        self.policy = self.mean_policy  # exploring policy (mean policy)

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

    def obtain_episode_by_policyNet(self, policyNet, start_state, start_action, max_length):
        self.env.agent_location = self.env.state2pos(start_state)
        episode = []
        next_state = start_state
        next_action = start_action
        done = False
        while not done:
            if max_length < 0:
                break
            max_length -= 1
            start_state = next_state
            start_action = next_action
            _, reward, done, _ = self.env.step(start_action)
            next_state = self.env.pos2state(self.env.agent_location)
            y, x = self.env.state2pos(next_state)
            prb = policyNet(torch.tensor((y, x)).reshape(-1, 2))
            # 从计算图中脱离出来，返回一个新的tensor，新的tensor和原tensor共享数据内存，（这也就意味着修改一个tensor的值，另外一个也会改变），
            # 但是不涉及梯度计算。在从tensor转换成为numpy的时候，如果转换前面的tensor在计算图里面（requires_grad = True），
            # 那么这个时候只能先进行detach操作才能转换成为numpy
            next_action = np.random.choice(np.arange(self.action_space_size), p=np.squeeze(prb.detach().numpy()))
            episode.append({"state": start_state, "action": start_action, "reward": reward, "next_state": next_state,
                            "next_action": next_action})
        return done, episode

    def reinforce(self, learning_rate=0.000009, epochs=1000, max_episode_length=1000):
        """
        训练中发现的一些问题：
        1.学习率不宜过大，过大会导致梯度消失/爆炸，从而引起下一次episode采样的长度过长，一旦有次问题等于白训练（训练结果为：全部保持原地不动）
        2.epoch并不是越多越好，训练次数过多也容易造成以上后果
        3.同样的lr和epoch训练结果也不一样，个人觉得原因在于采样数据不同（没有好的数据算法白搭）
        4.没有办法解释为什么结果不如dqn(神经网络是个坑）
        """
        policyNet = PolicyNet()
        optimizer = torch.optim.Adam(policyNet.parameters(), lr=learning_rate)
        loss_list = []
        for epoch in range(epochs):
            # 寻找从位置(0, 0)到target的路径
            pro = policyNet(torch.tensor((0, 0)).reshape(-1, 2))[0]
            start_action = np.random.choice(np.arange(self.action_space_size), p=pro.detach().numpy())
            done, episode = self.obtain_episode_by_policyNet(policyNet, 0, start_action, max_episode_length)

            if not done:
                continue

            discounted_rewards = []
            for step in range(len(episode)):
                Gt = 0
                pw = 0
                for k in range(step, len(episode), 1):
                    Gt = Gt + self.gama ** pw * episode[k]['reward']
                    pw = pw + 1
                discounted_rewards.append(Gt)
            discounted_rewards = torch.tensor(discounted_rewards)
            # 折扣奖励是标准化的（即减去均值并除以周期中所有奖励的标准差）这样保持了训练的稳定性。
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)  # normalize discounted rewards


            g = 0  # 采用MC得到g用来表示qt(st, at)
            loss_value = []
            for step in range(len(episode)):
                state = episode[step]['state']
                action = episode[step]['action']
                y, x = self.env.state2pos(state)
                pro = policyNet(torch.tensor((y, x)).reshape(-1, 2))[0]
                log_pro = torch.log(pro[action])
                loss = -log_pro * discounted_rewards[step]
                loss_value.append(loss)
            #loss_list.append(sum(loss_value))
            #loss_value1 = torch.Tensor(loss_value)
            loss_value2 = torch.stack(loss_value).sum()
            #loss_value1 = torch.sum(loss_value1)
            loss_list.append(loss_value2.item())
            optimizer.zero_grad()
            #loss_value1.requires_grad_(True)
            loss_value2.backward()  # 通过反向传播更新网络参数
            optimizer.step()
            print("epoch:{}, episode_length:{}, loss value:{}".format(epoch, len(episode), loss_value2))


        for s in range(self.state_space_size):
            y, x = self.env.state2pos(s) / self.env.size
            prb = policyNet(torch.tensor((y, x)).reshape(-1, 2))[0].detach().numpy()
            p_max = prb.max()
            a_max = prb.tolist().index(p_max)
            #self.policy[s, :] = prb.copy()
            for a in range(self.action_space_size):
                if a == a_max:
                    self.policy[s, a] = 1
                else:
                    self.policy[s, a] = 0
        self.show_policy()

        fig = plt.figure(figsize=(10, 10))
        ax1 = fig.add_subplot(211)
        ax1.plot(loss_list)
        ax1.set_xlabel('epoch index')
        ax1.set_ylabel('loss very epoch')
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
    solver.reinforce()
    end_time = time.time()
    cost_time = end_time - start_time
    print("cost_time:{}".format(round(cost_time, 2)))
    print(len(env.render_.trajectory))
    solver.show_policy_()  # solver.env.render()
    solver.show_state_value(solver.state_value, y_offset=0.25)
    solver.env.render()
