import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Actor(nn.Module):
    # 演员Actor网络
    def __init__(self, action_dim, state_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 300)
        self.fc2 = nn.Linear(300, action_dim)

        self.ln = nn.LayerNorm(300)   # 层归一化，可以对网络中的每个神经元的输出进行归一化，使得网络中每一层的输出都具有相似的分布。

    def forward(self, s):
        if isinstance(s, np.ndarray):  # isinstance() 函数来判断一个对象是否是一个已知的类型
            s = torch.FloatTensor(s)
        x = self.ln(F.relu(self.fc1(s)))
        out = F.softmax(self.fc2(x), dim=-1)      # 输出为每个动作的概率

        return out


class Critic(nn.Module):
    # 评论家Critic网络，状态值函数
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 300)
        self.fc2 = nn.Linear(300, 1)

        self.ln = nn.LayerNorm(300)

    def forward(self, s):
        if isinstance(s, np.ndarray):
            s = torch.FloatTensor(s)
        x = self.ln(F.relu(self.fc1(s)))
        out = self.fc2(x)

        return out


class Actor_Critic:
    def __init__(self, env):
        self.gamma = 0.99   # 折扣因子
        self.lr_a = 3e-4    # 学习率
        self.lr_c = 5e-4

        self.env = env
        self.action_dim = self.env.action_space.n             # 获取 动作actor的个数
        self.state_dim = self.env.observation_space.shape[0]  # 获取 描述状态state的数据维度

        self.actor = Actor(self.action_dim, self.state_dim)   # 创建演员actor网络
        self.critic = Critic(self.state_dim)                  # 创建评论家critic网络

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

        self.loss = nn.MSELoss()   # 损失函数：均方误差

    def get_action(self, s):
        # 根据当前状态选择动作action
        a = self.actor(s)       # 得到动作的离散概率分布
        dist = Categorical(a)   # 使用概率创建Categorical对象
        action = dist.sample()             # 从分布中采样一个随机动作
        log_prob = dist.log_prob(action)   # 计算给定动作的对数概率

        return action.detach().numpy(), log_prob

    def learn(self, log_prob, s, s_, rew):
        # 更新两个网络
        # 输入当前状态s,下一个状态s_,以及反馈reward
        v = self.critic(s)
        v_ = self.critic(s_)

        critic_loss = self.loss(self.gamma * v_ + rew, v)  # Critic网络损失函数
        self.critic_optim.zero_grad()    # 梯度清零
        critic_loss.backward()
        self.critic_optim.step()

        td = self.gamma * v_ + rew - v          # 计算TD误差
        loss_actor = -log_prob * td.detach()    # Actor网络损失函数
        self.actor_optim.zero_grad()            # 梯度清零
        loss_actor.backward()
        self.actor_optim.step()


def my_test(model, env):
    s = env.reset()
    env.render()
    done = False  # 记录当前回合游戏是否结束
    ep_r = 0  # 游戏当前回合总回报
    while not done:
        # 通过model对当前状态做出行动
        env.render()
        time.sleep(0.)
        a, *_ = model.get_action(s)

        # 获得在做出a行动后的状态和反馈
        s_, rew, done, *_ = env.step(a)

        # 计算当前reward
        ep_r += rew
        s = s_
    print(f"test_reward:{ep_r}")
    return ep_r


