
from typing import Optional, Union, List, Tuple
import gym
import numpy as np
from gym.core import RenderFrame, ActType, ObsType
import utils
import time

np.random.seed(1)


class GridEnv(gym.Env):
    def __init__(self, size, target, forbidden, render_mode: str):
        """
        注意这里没有定义观察值observation，搭配gym可设置观察值(Box)
        """
        self.agent_location = np.array([0, 0])
        self.size = size
        self.target_location = np.array(target)
        self.forbidden_location = []
        for f in forbidden:
            self.forbidden_location.append(np.array(f))
        self.action_space_size = 5
        self.reward_list = [-10, -10, 1, 0]
        self.render_mode = render_mode
        self.render_ = utils.Render(target=target, forbidden=forbidden, size=size)

        # 定义动作空间每个动作的含义
        self.pos_bias = {
            0: np.array([-1, 0]),  # 向上
            1: np.array([0, 1]),  # 向右
            2: np.array([1, 0]),  # 向下
            3: np.array([0, -1]),  # 向左
            4: np.array([0, 0])  # 不动
        }
        self.Rsa = None
        self.Psa = None
        self.psa_rsa_init()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[ObsType, dict]:
        super().reset(seed=seed)
        self.agent_location = np.array([0, 0])
        observation = self.get_obs()
        info = self.get_obs()
        return observation, info

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        """
        当前的状态信息已经记录在agent_location中， 提供action参数能够生成对应的reward,以及下一个状态(被保存在agent_location中)
        """
        reward = self.reward_list[self.Rsa[self.pos2state(self.agent_location), action].tolist().index(1)]
        direction = self.pos_bias[action]
        # numpy.clip(a, a_min, a_max, out=None) 限定数组大小
        self.agent_location = np.clip(self.agent_location + direction, 0, self.size - 1)
        done = np.array_equal(self.agent_location, self.target_location)
        observation = self.get_obs()
        info = None
        return observation, reward, done, info

    def get_obs(self):
        return {"agent": self.agent_location, "target": self.target_location, "barrier": self.forbidden_location}

    def state2pos(self, state: int) -> np.ndarray:
        row = state // self.size  # 注意//是取整
        col = state % self.size
        return np.array([row, col])

    def pos2state(self, pos: np.ndarray) -> int:
        return pos[0] * self.size + pos[1]

    def psa_rsa_init(self):
        state_size = self.size ** 2
        self.Psa = np.zeros(shape=(state_size, self.action_space_size, state_size), dtype=float)
        self.Rsa = np.zeros(shape=(state_size, self.action_space_size, len(self.reward_list)), dtype=float)

        """
        状态转移规则定义：
        1.当转移后超出边界，那么reward = -1
          当转移后依然在accessible区域，那么reward = 0
          当转移后进入forbidden区域，那么reward = -1
          当转移后进入target区域，那么reward = 1
        2.当采取action超出边界后，返回到原来的state
          当采取action进行forbidden区域是允许的，但是有惩罚
        """
        for state_index in range(state_size):
            for action_index in range(self.action_space_size):
                current_pos = self.state2pos(state_index)
                next_pos = current_pos + self.pos_bias[action_index]
                if next_pos[0] < 0 or next_pos[0] > self.size - 1 or next_pos[1] < 0 or next_pos[1] > self.size - 1:
                    self.Psa[state_index, action_index, state_index] = 1
                    self.Rsa[state_index, action_index, 0] = 1
                else:
                    self.Psa[state_index, action_index, self.pos2state(next_pos)] = 1
                    if arr_in_list(next_pos, self.forbidden_location):
                        self.Rsa[state_index, action_index, 1] = 1
                    elif np.array_equal(next_pos, self.target_location):
                        self.Rsa[state_index, action_index, 2] = 1
                    else:
                        self.Rsa[state_index, action_index, 3] = 1

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        if self.render_mode == "video":
            self.render_.save_video('image/' + str(time.time()))
        self.render_.show_frame(0.3)
        return None

    def close(self):
        pass


def arr_in_list(array, _list) -> bool:
    for arr in _list:
        if np.array_equal(array, arr):
            return True
    return False
