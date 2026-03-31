import os
import time

import torch

from VRP_PPO_Model import Agentppo, Memory
from creat_vrp import creat_data, reward, reward1
from collections import OrderedDict
from collections import namedtuple
from itertools import product
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# device = torch.device('cpu')
n_nodes = 51


def rollout(model, dataset, batch_size, steps):
    """
    在给定模型和数据集的情况下，对数据集进行批量评估并返回总成本
    """
    # 评估模式，不训练网络
    model.eval()

    def eval_model_bat(bat):
        # 不需要计算梯度
        with torch.no_grad():
            cost, _ = model.act(bat, 0, steps, batch_size, True, False)

            cost = reward1(bat.x, cost.detach(), n_nodes)
        return cost.cpu()
    totall_cost = torch.cat([eval_model_bat(bat.to(device))for bat in dataset], 0)
    return totall_cost


class TrainPPO:
    def __init__(self, steps, greedy, lr, input_node_dim, hidden_node_dim, input_edge_dim,
                 hidden_edge_dim, epoch=40, batch_size=32, conv_laysers=3, entropy_value=0.01, eps_clip=0.2,
                 timestep=4, ppo_epoch=2):
        """
        steps: 每个episode中的最大步数
        greedy: 在选择动作时是否使用贪婪策略
        lr: 学习率
        input_node_dim: 输入节点的维度
        hidden_node_dim: 隐藏节点的维度
        input_edge_dim: 输入边的维度
        hidden_edge_dim: 隐藏边的维度
        epoch: 训练的总轮数
        batch_size: 每个训练批次中的样本数量
        conv_layers: 卷积层的数量
        entropy_value: PPO算法中的熵项系数
        eps_clip: PPO算法中的优势函数裁剪范围
        timestep: PPO算法中的更新步长
        ppo_epoch: PPO算法中的优化器迭代次数
        """
        self.steps = steps
        self.greedy = greedy
        self.batch_size = batch_size
        self.update_timestep = timestep
        self.epoch = epoch
        self.memory = Memory()
        self.agent = Agentppo(steps, greedy, lr, input_node_dim, hidden_node_dim,
                              input_edge_dim, hidden_edge_dim, ppo_epoch, batch_size,
                              conv_laysers, entropy_value, eps_clip)

    def run_train(self, data_loader, batch_size, valid_loder):
        """
        data_loader: 训练数据集的数据加载器
        batch_size：每个训练批次中的样本数量
        valid_loder：用于验证的数据加载器
        """
        memory = Memory()
        # self.agent.old_polic（旧的策略网络）移动到设备（device）上
        self.agent.old_polic.to(device)
        # initWeights(self.agent.old_polic)
        # initWeights(self.agent.policy)
        folder = 'vrp-{}-GAT'.format(n_nodes)
        filename = '20201125'
        filepath = os.path.join(folder, filename)

        '''path = os.path.join(filepath,'%s' % 3)
        if os.path.exists(path):
            path1 = os.path.join(path, 'actor.pt')
            self.agent.old_polic.load_state_dict(torch.load(path1, device))'''

        costs = []
        for i in range(self.epoch):
            print('old_epoch:', i, '***************************************')
            self.agent.old_polic.train()
            times, losses, rewards2, critic_rewards = [], [], [], []
            # 开始的时间
            epoch_start = time.time()
            start = epoch_start
            # 在每个训练批次中，从data_loader中获取批次数据
            for batch_idx, batch in enumerate(data_loader):

                x, attr, capcity, demand = batch.x, batch.edge_attr, batch.capcity, batch.demand
                # print(x.size(),index.size(),attr.size())
                # 通过view函数对输入数据进行预处理，将输入的节点、边属性、容量和需求调整为正确的形状
                x, attr, capcity, demand = x.view(batch_size, n_nodes, 2), attr.view(batch_size, n_nodes*n_nodes, 1),\
                                           capcity.view(batch_size, 1), demand.view(batch_size, n_nodes, 1)
                batch = batch.to(device)
                # 使用self.agent.old_polic对批次数据执行动作选择，得到动作和对应的对数概率
                actions, log_p = self.agent.old_polic.act(batch, 0, self.steps, batch_size, self.greedy, False)
                # 根据选择的动作计算奖励
                rewards = reward1(batch.x, actions, n_nodes)
                # 将数据转移到CPU上
                actions = actions.to(torch.device('cpu')).detach()
                log_p = log_p.to(torch.device('cpu')).detach()
                rewards = rewards.to(torch.device('cpu')).detach()

                # print(actions.size(),log_p.size(),entropy.size())
                # 将动作、对数概率和奖励存储到memory中
                for i_batch in range(self.batch_size):
                    memory.input_x.append(x[i_batch])
                    # memory.input_index.append(index[i_batch])
                    memory.input_attr.append(attr[i_batch])
                    memory.actions.append(actions[i_batch])
                    memory.log_probs.append(log_p[i_batch])
                    memory.rewards.append(rewards[i_batch])
                    memory.capcity.append(capcity[i_batch])
                    memory.demand.append(demand[i_batch])
                # 每当达到指定的更新步长（self.update_timestep）时，调用self.agent.update方法进行模型的更新，并重置memory
                if (batch_idx+1) % self.update_timestep == 0:
                    self.agent.update(memory, i)
                    memory.def_memory()
                rewards2.append(torch.mean(rewards.detach()).item())
                time_Space = 100
                if (batch_idx+1) % time_Space == 0:
                    # 每当达到指定的时间间隔（time_Space）时，打印出当前的批次数、奖励平均值和所花费的时间。
                    end = time.time()
                    times.append(end - start)
                    start = end
                    # 计算最近 time_Space 个批次的平均奖励 mean_reward
                    mean_reward = np.mean(rewards2[-time_Space:])
                    print('  Batch %d/%d, reward: %2.3f,took: %2.4fs' %
                          (batch_idx, len(data_loader), mean_reward,
                           times[-1]))
            # 使用函数 rollout 对模型 self.agent.policy 在验证集 valid_loder 上进行推断，得到每个样本的代价（cost）
            cost = rollout(self.agent.policy, valid_loder, batch_size, self.steps)
            cost = cost.mean()
            # 记录每个 epoch 的平均代价
            costs.append(cost.item())
            print('Problem:TSP''%s' % n_nodes, '/ Average distance:', cost.item())
            print(costs)
            # 在每个 epoch 结束后将模型参数保存到指定的文件中，以便后续的模型加载和使用。每个epoch的模型参数会保存在不同的文件夹中，方便进行模型版本管理和回溯
            epoch_dir = os.path.join(filepath, '%s' % i)
            if not os.path.exists(epoch_dir):
                os.makedirs(epoch_dir)
            save_path = os.path.join(epoch_dir, 'actor.pt')
            torch.save(self.agent.old_polic.state_dict(), save_path)


def train():
    class RunBuilder():
        # RunBuilder 类中定义了一个静态方法 get_runs(params)，该方法接受一个参数params，该参数是一个字典，其中包含了不同的参数及其取值范围
        @staticmethod
        def get_runs(params):
            Run = namedtuple('Run', params.keys())
            runs = []
            for v in product(*params.values()):
                runs.append(Run(*v))
            return runs

    params = OrderedDict(
        lr=[3e-4],
        hidden_node_dim=[128],
        hidden_edge_dim=[16],
        epoch=[100],
        batch_size=[512],
        conv_laysers=[4],
        entropy_value=[0.01],
        eps_clip=[0.2],
        timestep=[1],
        ppo_epoch=[3],
        data_size=[512000],
        valid_size=[10000]
    )
    # 调用 RunBuilder.get_runs(params) 方法生成不同参数组合的列表 runs，每个元素是一个命名元组Run，包含了不同参数的取值组合
    runs = RunBuilder.get_runs(params)

    for lr, hidden_node_dim, hidden_edge_dim, epoch, batch_size, conv_laysers, entropy_value, eps_clip, timestep, \
        ppo_epoch, data_size, valid_size in runs:
        print('lr', 'batch_size', 'hidden_node_dim', 'hidden_edge_dim', 'conv_laysers', 'epoch,batch_size',
              'entropy_value', 'eps_clip', 'timestep:', 'data_size', 'valid_size', lr, hidden_node_dim,
              hidden_edge_dim, epoch, batch_size, conv_laysers,
              entropy_value, eps_clip, timestep, data_size, valid_size)
        # 使用 creat_data() 函数创建训练集和验证集的数据加载器data_loder和valid_loder
        data_loder = creat_data(n_nodes, data_size, batch_size)
        valid_loder = creat_data(n_nodes, valid_size, batch_size)
        print('DATA CREATED/Problem size:', n_nodes)
        trainppo = TrainPPO(n_nodes*2, False, lr, 3, hidden_node_dim, 1, hidden_edge_dim,
                            epoch, batch_size, conv_laysers, entropy_value, eps_clip, timestep, ppo_epoch)
        trainppo.run_train(data_loder, batch_size, valid_loder)


train()
