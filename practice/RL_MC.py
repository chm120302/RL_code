import random
import time

from practice import RL_env
import numpy as np


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
        # self.policy = self.mean_policy.copy()
        self.policy = self.random_greed_policy()

    def random_greed_policy(self):
        policy = np.zeros(shape=(self.state_space_size, self.action_space_size))
        for state in range(self.state_space_size):
            action = np.random.choice(range(self.action_space_size))
            policy[state, action] = 1
        return policy

    '''
    对于所有的(s,a), 以每个(s,a)作为开始状态进行episode的构建
    构建的episode举例如下：
    (s0, a1), (s1, a2), (s1, a3), (s0, a0), (s5, a3), (s0, a1), (s1, a3).....
    (s1, a3), (s0, a2), (s3, a4), (s0, a1), (s1, a2), (s1, a3), (s0, a2).....
    (s1, a2), (s2, a3), (s0, a1), (s2, a2), (s5, a0), (s4, a1), (s3, a3).....
    遍历(s,a)空间，倒序遍历episode中的元素
    1. first visit
    对于(s,a)来说，
      当目前的episode中含有(s,a)
        若e_state(0) = state and e_action(0) = action, 那么g为所有(s,a)的总和
        否则，采用gList保存每次e_state(i) = state and e_action(i)=action时的g，最后令g = gList[-1]
      否则跳过不计数
    最后将所有episodes的g加起来求平均
    e.g. (s0, a1) g1 = 7, g2 = 4, g3 = 5
    2. every visit
    对于(s,a)来说
      当目前的episode中含有(s,a)
        采用gList保存每次e_state(i) = state and e_action(i)=action时的g， 令g = mean(gList)
      否则跳过不计数
    最后将所有episodes的g加起来求平均
      
      
    
    
    '''

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

    def mc_basic(self, length=30, steps=10):
        origin_steps = steps
        while steps > 0:
            steps -= 1
            print("the {}th iteration:".format(origin_steps - steps))
            for state in range(self.state_space_size):
                for action in range(self.action_space_size):
                    # 这里可以选择生成多条episode取平均
                    episode = self.obtain_episode(state, action, self.policy, length)
                    g = 0
                    for step in range(len(episode) - 1, -1, -1):
                        g = episode[step]['reward'] + self.gama * g
                    self.qvalue[state][action] = g
                qvalue_opt = self.qvalue[state].max()
                action_opt = self.qvalue[state].tolist().index(qvalue_opt)
                self.policy[state] = np.zeros(shape=self.action_space_size)
                self.policy[state, action_opt] = 1

            self.show_policy()
            self.show_q_value()

        print("origin iterations: ", origin_steps)
        print("iterations: ", origin_steps - steps)

    def mc_exploring_starts(self, length=10, threshold=0.001, steps=500):
        # 平均策略
        policy = np.ones(shape=(self.state_space_size, self.action_space_size)) / self.action_space_size
        qvalue = np.zeros(shape=(self.state_space_size, self.action_space_size))
        returns: list = [[[0 for row in range(1)] for col in range(5)] for block in range(25)]
        origin_steps = steps
        while np.linalg.norm(qvalue - self.qvalue, ord=1) > threshold and steps > 0:
            steps -= 1
            policy = self.policy.copy()
            qvalue = self.qvalue.copy()
            episodes = []
            for state in range(self.state_space_size):
                for action in range(self.action_space_size):
                    episode = self.obtain_episode(state, action, self.policy, length)
                    episodes.append(episode)

            for epi in episodes:
                # 存储每轮episode中每个(s,a)首次访问的g
                gList = np.zeros(shape=(self.state_space_size, self.action_space_size), dtype=float)
                g = 0
                for step in range(len(epi) - 1, -1, -1):
                    state = epi[step]['state']
                    action = epi[step]['action']
                    reward = epi[step]['reward']
                    g = reward + self.gama * g
                    # 从后往前遍历过程中若遇到(s,a)就更新它的值，确保最后得到的是first visit
                    gList[state][action] = g
                for state in range(self.state_space_size):
                    for action in range(self.action_space_size):
                        if gList[state][action] != 0.0:
                            returns[state][action].append(gList[state][action])
            for state in range(self.state_space_size):
                for action in range(self.action_space_size):
                    self.qvalue[state, action] = np.array(returns[state][action]).mean()
                qvalue_opt = self.qvalue[state].max()
                action_opt = self.qvalue[state].tolist().index(qvalue_opt)

                self.policy[state] = np.zeros(shape=self.action_space_size).copy()
                self.policy[state, action_opt] = 1
                # self.qvalue = qvalue

            self.show_policy()
            self.show_q_value()

            # print(np.linalg.norm(policy - self.policy, ord=1))
        print("origin iterations: ", origin_steps)
        print("iterations: ", origin_steps - steps)

    def mc_epsilon_greedy(self, epsilon=0, length=1000, threshold=0.001, steps=100):
        # 这里注意epsilon为1时进行充分的探索，需要较大的length, 若epsilon较小则探索程度减少，确定性增加，需要length较小
        policy = np.ones(shape=(self.state_space_size, self.action_space_size)) / self.action_space_size
        qvalue = np.random.random(size=(self.state_space_size, self.action_space_size))
        returns = [[[0 for row in range(1)] for col in range(5)] for block in range(25)]
        origin_steps = steps
        # while np.linalg.norm(qvalue - self.qvalue, ord=1) > threshold and steps > 0:
        while steps > 0:
            steps -= 1
            qvalue = self.qvalue.copy()
            policy = self.policy.copy()
            episodes = []
            # for state in range(self.state_space_size):
            # for action in range(self.action_space_size):
            # episode = self.obtain_episode(state, action, self.policy, length)
            # episodes.append(episode)
            episodes = []
            for state in range(self.state_space_size):
                for action in range(self.action_space_size):
                    episode = self.obtain_episode(state, action, self.policy, length)
                    episodes.append(episode)

            for episode in episodes:
                g = 0
                # every visit 这里不需要gList，因为每次访问到(s,a)时，g值都会被计算进去
                for step in range(len(episode) - 1, -1, -1):
                    reward = episode[step]['reward']
                    e_state = episode[step]['state']
                    e_action = episode[step]['action']
                    g = reward + self.gama * g
                    # every visit
                    returns[e_state][e_action].append(g)
            for state in range(self.state_space_size):
                for action in range(self.action_space_size):
                    self.qvalue[state][action] = np.array(returns[state][action]).mean()
                qvalue_opt = self.qvalue[state].max()
                action_opt = self.qvalue[state].tolist().index(qvalue_opt)
                for ac in range(self.action_space_size):
                    if ac == action_opt:
                        self.policy[state, ac] = 1 - (self.action_space_size - 1) / self.action_space_size * epsilon
                    else:
                        self.policy[state, ac] = 1 / self.action_space_size * epsilon
            self.show_policy()
            self.show_q_value()

            print(np.linalg.norm(policy - self.policy, ord=1))
        print("origin iterations: ", origin_steps)
        print("iterations:", origin_steps - steps)

    def show_policy(self):
        print("policy is: ")
        for state in range(self.state_space_size):
            for action in range(self.action_space_size):
                print(self.policy[state, action], end='  ')
            print("")

    def show_q_value(self):
        print("q value is: ")
        for state in range(self.state_space_size):
            for action in range(self.action_space_size):
                print(self.qvalue[state][action], end='  ')
            print("")

    def show_policy_(self):
        for state in range(self.state_space_size):
            for action in range(self.action_space_size):
                policy = self.policy[state, action]
                self.env.render_.draw_action(pos=self.env.state2pos(state),
                                             toward=policy * 0.4 * self.env.pos_bias[action],
                                             radius=policy * 0.1)

    def show_state_value_(self, state_value, y_offset=0.2):
        for state in range(self.state_space_size):
            self.env.render_.write_word(pos=self.env.state2pos(state), word=str(round(state_value[state], 1)),
                                        y_offset=y_offset,
                                        size_discount=0.7)


if __name__ == "__main__":
    env = RL_env.GridEnv(size=5, target=[2, 3],
                         forbidden=[[2, 2], [2, 1], [1, 1], [3, 3], [1, 4], [1, 3]], render_mode='')
    solver = Solve(env)

    start_time = time.time()
    # 得到结果的误差率：
    # mc_basic:3/25    mc_exploring_starts: 6/25  mc_epsilon: 5/25
    # solver.mc_basic()
    solver.mc_exploring_starts()
    # solver.mc_epsilon_greedy()
    end_time = time.time()
    cost_time = end_time - start_time
    print("cost_time:{}".format(round(cost_time, 2)))
    print(len(env.render_.trajectory))
    solver.show_policy_()  # solver.env.render()
    solver.show_state_value_(solver.state_value, y_offset=0.25)
    solver.env.render()
