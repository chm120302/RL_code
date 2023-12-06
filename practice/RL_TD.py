from practice import RL_env
import numpy as np
import time
import utils


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

    def random_greed_policy(self):
        policy = np.zeros(shape=(self.state_space_size, self.action_space_size))
        for state in range(self.state_space_size):
            action = np.random.choice(range(self.action_space_size))
            policy[state, action] = 1
        return policy

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

    def obtain_n_step_exp(self, start_state, start_action, start_policy, n):
        self.env.agent_location = self.env.state2pos(start_state)
        exp = []
        next_action = start_action
        next_state = start_state
        while n > 0:
            n -= 1
            state = next_state
            action = next_action
            _, reward, done, _ = self.env.step(action)
            next_state = self.env.pos2state(self.env.agent_location)
            next_action = np.random.choice(np.arange(len(start_policy[next_state])), p=start_policy[next_state])
            exp.append({"start_state": state, "start_action": action, "reward": reward, "next_state": next_state,
                        "next_action": next_action})
        return exp

    def sarsa(self, alpha=0.01, epsilon=0.01, n=1, if_expected=False, num_episodes=1000):
        """
        :param if_expected: if the TD target is expected sarsa
        :param alpha: learning rate
        :param epsilon:
        :param n: the length of steps, when n is 1, this is sarsa otherwise is n-steps sarsa
        :param num_episodes: the number of samples
        :return:
        """
        self.policy = self.random_epsilon_greedy_policy(epsilon)
        origin_num_episodes = num_episodes
        rewards_list = []
        episode_length_list = []
        while num_episodes > 0:
            num_episodes -= 1
            total_rewards = 0
            episode_length = 0
            self.env.reset()
            start_state = (origin_num_episodes - num_episodes - 1) % self.state_space_size
            start_action = np.random.choice(np.arange(len(self.policy[start_state])), p=self.policy[start_state])
            next_state = start_state
            next_action = start_action
            while start_state != self.env.pos2state(self.env.target_location):
                exp = self.obtain_n_step_exp(start_state, start_action, self.policy, n)
                episode_length += n
                td_target = 0
                if not if_expected:
                    for step in range(n):
                        reward = exp[step]['reward']
                        total_rewards += reward
                        td_target += self.gama ** step * reward
                        if step == n - 1:
                            state_n = exp[step]['next_state']
                            action_n = exp[step]['next_action']
                            next_state = state_n
                            next_action = action_n
                            td_target += self.gama ** n * self.qvalue[state_n, action_n]
                else:
                    reward = exp[n - 1]['reward']
                    state_n = exp[n - 1]['next_state']
                    action_n = exp[n - 1]['next_action']
                    td_target += reward
                    expected_q = 0
                    for action in range(self.action_space_size):
                        expected_q += self.policy[state_n, action] * self.qvalue[state_n, action]
                    next_state = state_n
                    next_action = action_n
                    td_target += self.gama * expected_q

                td_error = self.qvalue[start_state, start_action] - td_target
                self.qvalue[start_state, start_action] = self.qvalue[start_state, start_action] - alpha * td_error
                qvalue_opt = self.qvalue[start_state].max()
                action_opt = self.qvalue[start_state].tolist().index(qvalue_opt)
                for action in range(self.action_space_size):
                    if action == action_opt:
                        self.policy[start_state, action] = 1 - (
                                self.action_space_size - 1) / self.action_space_size * epsilon
                    else:
                        self.policy[start_state, action] = 1 / self.action_space_size * epsilon

                start_state = next_state
                start_action = next_action
            rewards_list.append(total_rewards)
            episode_length_list.append(episode_length)
            self.show_policy()
            self.show_q_value()

        print("reward for every episode:")
        print(rewards_list)
        print("episode length for every episode:")
        print(episode_length_list)
        return rewards_list, episode_length_list

    def q_learning_on_policy(self, alpha=0.01, epsilon=0.01, num_episodes=1000):
        self.policy = self.random_epsilon_greedy_policy(epsilon)
        origin_num_episodes = num_episodes
        rewards_list = []
        episode_length_list = []
        while num_episodes > 0:
            num_episodes -= 1
            total_rewards = 0
            episode_length = 0
            self.env.reset()
            start_state = (origin_num_episodes - num_episodes - 1) % self.state_space_size
            start_action = np.random.choice(np.arange(len(self.policy[start_state])), p=self.policy[start_state])
            next_state = start_state
            next_action = start_action
            while start_state != self.env.pos2state(self.env.target_location):
                exp = self.obtain_n_step_exp(start_state, start_action, self.policy, 1)
                episode_length += 1
                td_target = 0

                for step in range(1):
                    reward = exp[step]['reward']
                    total_rewards += reward
                    td_target += self.gama ** step * reward
                    next_state = exp[step]['next_state']
                    next_action = exp[step]['next_action']

                td_target += self.gama * self.qvalue[next_state].max()
                td_error = self.qvalue[start_state, start_action] - td_target
                self.qvalue[start_state, start_action] = self.qvalue[start_state, start_action] - alpha * td_error
                qvalue_opt = self.qvalue[start_state].max()
                action_opt = self.qvalue[start_state].tolist().index(qvalue_opt)
                for action in range(self.action_space_size):
                    if action == action_opt:
                        self.policy[start_state, action] = 1 - (
                                self.action_space_size - 1) / self.action_space_size * epsilon
                    else:
                        self.policy[start_state, action] = 1 / self.action_space_size * epsilon

                start_state = next_state
                start_action = next_action
            rewards_list.append(total_rewards)
            episode_length_list.append(episode_length)
            self.show_policy()
            self.show_q_value()

        print("reward for every episode:")
        print(rewards_list)
        print("episode length for every episode:")
        print(episode_length_list)
        return rewards_list, episode_length_list

    def q_learning_off_policy(self, alpha=0.5, epsilon=0.1, num_episodes=10, episode_length=1000000):
        # 这是每个state都训练，也可以选定一个state训练
        # 结果不太好，policy看起来有点痴呆
        self.policy = self.random_epsilon_greedy_policy(epsilon)
        start_state = self.env.pos2state(self.env.agent_location)
        start_action = np.random.choice(np.arange(len(self.policy[start_state])), p=self.policy[start_state])
        episode = self.obtain_n_step_exp(start_state, start_action, self.policy, episode_length)

        for step in range(episode_length):
            state = episode[step]['start_state']
            action = episode[step]['start_action']
            reward = episode[step]['reward']
            next_state = episode[step]['next_state']
            td_target = reward + self.gama * self.qvalue[next_state].max()
            td_error = self.qvalue[state, action] - td_target
            self.qvalue[state, action] = self.qvalue[state, action] - alpha * td_error
            qvalue_opt = self.qvalue[state].max()
            action_opt = self.qvalue[state].tolist().index(qvalue_opt)
            for action_ in range(self.action_space_size):
                if action_ == action_opt:
                    self.policy[state, action_] = 1
                else:
                    self.policy[state, action_] = 0
        self.show_policy()
        self.show_q_value()

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


if __name__ == "__main__":
    env = RL_env.GridEnv(size=5, target=[2, 3],
                         forbidden=[[2, 2], [2, 1], [1, 1], [3, 3], [1, 4], [1, 3]], render_mode='')
    solver = Solve(env)

    start_time = time.time()
    num_episodes = 1000
    # rewards_list, episode_length_list = solver.sarsa(num_episodes=num_episodes)
    solver.q_learning_off_policy()
    end_time = time.time()
    cost_time = end_time - start_time
    print("cost_time:{}".format(round(cost_time, 2)))
    print(len(env.render_.trajectory))
    solver.show_policy_()  # solver.env.render()
    solver.show_state_value(solver.state_value, y_offset=0.25)
    solver.env.render()

