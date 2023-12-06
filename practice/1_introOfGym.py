import gym
from gym import envs
import torch


if __name__ == '__main__':
    # env = gym.make("FrozenLake-v1")  # 创建环境
    # env = env.unwrapped  # 解封装才能访问状态转移矩阵P
    # env.reset()
    # for _ in range(10):
    # env.render()
    # env.step(env.action_space.sample())  # take a random action
    # env.close()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")







