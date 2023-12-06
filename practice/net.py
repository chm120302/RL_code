import torch
import torch.nn as nn


class QNET(nn.Module):
    """
    用于DQN估计q
    """
    #  输入x, y, a输出一个q估计
    def __init__(self, input_dim=3, output_dim=1):
        super(QNET, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=output_dim),

        )

    def forward(self, x):
        x = x.type(torch.float32)
        return self.fc(x)


class PolicyNet(nn.Module):
    """
    用于策略估计的网络
    """
    #  输入x, y输出5个action下的策略估计
    def __init__(self, input_dim=2, output_dim=5):
        super(PolicyNet, self).__init__()
        self.fc = nn.Sequential(

            nn.Linear(in_features=input_dim, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=output_dim),

            #nn.Linear(in_features=input_dim, out_features=100),
            #nn.ReLU(),
            #nn.Linear(in_features=100, out_features=output_dim),
            # 进行归一化
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.type(torch.float32)
        return self.fc(x)

