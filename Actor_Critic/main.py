import gym
from model import Actor_Critic
from model import my_test
import matplotlib.pyplot as plt

if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    #  智能体需要在两个动作中进行选择——向左移动还是向右移动——使得小车上的杆能够保持直立
    #  在倒立摆任务中，每一个时间步的奖励均为 + 1，但是一旦小车偏离中心超过4.8个单位或者杆的倾斜超过15度，任务就会终止。
    #  我们的目标是使得该任务能够尽可能地运行得更久，以便获得更多的收益。
    model = Actor_Critic(env)  # 实例化Actor_Critic算法类
    reward = []
    for episode in range(200):
        s = env.reset()  # 重置环境，并获取环境状态
        # env.render()   # 界面可视化
        done = False     # 记录当前回合游戏是否结束
        ep_r = 0         # 游戏当前回合总回报
        while not done:
            # 通过Actor_Critic算法对当前状态做出行动
            a, log_prob = model.get_action(s)

            # 获得在做出a行动后的状态和反馈
            s_, rew, done, *_ = env.step(a)

            # 计算当前reward
            ep_r += rew

            # 训练模型，更新两个网络
            model.learn(log_prob, s, s_, rew)

            # 更新状态
            s = s_
        reward.append(ep_r)
        # print(f"episode:{episode} ep_r:{ep_r}")
    print(reward)
    plt.plot(reward)
    plt.show()
    my_test(model, env)
