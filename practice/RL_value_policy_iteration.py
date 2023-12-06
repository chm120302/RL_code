import RL_env
import numpy as np
import time


class Solve:
    def __init__(self, env: RL_env.GridEnv):
        self.gama = 0.9
        self.env = env
        self.action_space_size = env.action_space_size
        self.state_space_size = env.size ** 2
        self.reward_space_size = len(env.reward_list)
        self.reward_list = env.reward_list
        self.state_value = np.zeros(shape=self.state_space_size)
        self.qvalue = np.zeros(shape=(self.state_space_size, self.action_space_size))
        self.mean_policy = np.ones(shape=(self.state_space_size, self.action_space_size)) / self.action_space_size
        self.policy = self.random_greed_policy()

    def calculate_qValue(self, state, action, state_value):
        qvalue = 0
        for index in range(self.reward_space_size):
            qvalue += self.reward_list[index] * self.env.Rsa[state, action, index]
        for next_state in range(self.state_space_size):
            qvalue += self.gama * self.env.Psa[state, action, next_state] * state_value[next_state]
        return qvalue

    def random_greed_policy(self):
        policy = np.zeros(shape=(self.state_space_size, self.action_space_size))
        for state in range(self.state_space_size):
            action = np.random.choice(range(self.action_space_size))
            policy[state, action] = 1
        return policy

    def value_iteration(self, threshold=0.001, steps=100):
        # 这里要注意state和state value是不一样的
        # state取值范围:{0, 1,...,8}
        # state_value的值，初始可以随机赋值，在迭代过程中，会把最大的q赋值给state_value
        state_value_k = np.ones(self.state_space_size)
        while np.linalg.norm(state_value_k - self.state_value, ord=1) > threshold and steps > 0:
            steps -= 1
            # value update
            self.state_value = state_value_k.copy()
            self.show_state_value()
            # policy update
            self.policy, state_value_k = self.policy_improvement(state_value_k.copy())
            self.show_policy()
        return steps

    def policy_iteration(self, threshold=0.001, steps=100):
        policy = self.random_greed_policy()
        while np.linalg.norm(policy - self.policy, ord=1) > threshold and steps > 0:
            steps -= 1
            policy = self.policy.copy()
            self.show_policy()
            self.state_value = self.policy_evaluation(self.policy.copy(), threshold, steps)
            self.show_state_value()
            self.policy, _ = self.policy_improvement(self.state_value)
        return steps

    def policy_evaluation(self, policy, threshold=0.001, step=10):
        # 这里不设置step，就是policy iteration, 设置step是truncated policy iteration
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

    def policy_improvement(self, state_value):
        """
        进行策略更新，根据q确定策略和下一次迭代的state_value值(最大q)
        """
        policy = np.zeros(shape=(self.state_space_size, self.action_space_size))
        state_value_k = state_value.copy()
        for state in range(self.state_space_size):
            qValue_list = []
            for action in range(self.action_space_size):
                qValue_list.append(self.calculate_qValue(state, action, state_value.copy()))
            state_value_k[state] = max(qValue_list)
            action_opt = qValue_list.index(max(qValue_list))
            policy[state, action_opt] = 1
        return policy, state_value_k

    def show_policy(self):
        print("policy is: ")
        for state in range(self.state_space_size):
            for action in range(self.action_space_size):
                print(self.policy[state, action], end='  ')
            print("")

    def show_state_value(self):
        print("state value is: ")
        for state in range(self.state_space_size):
            print(self.state_value[state], end='  ')
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
    length = 100
    start_time = time.time()
    # 发现policy_iteration能够更快收敛state_value
    steps = solver.policy_iteration()
    # steps = solver.value_iteration()
    end_time = time.time()
    cost_time = end_time - start_time
    print("cost_time:{}".format(round(cost_time, 2)))
    print("iterations:", length - steps)
    print(len(env.render_.trajectory))
    solver.show_policy_()  # solver.env.render()
    solver.show_state_value_(solver.state_value, y_offset=0.25)
    solver.env.render()

