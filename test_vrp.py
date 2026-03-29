import argparse
import os
import sys
import time

import numpy as np
import torch
from torch_geometric.data import Data, DataLoader

from creat_vrp import reward1
from VRP_Actor import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 权重放在 trained/{n_node}/actor.pt（与手动从 VRP_Rollout_train 保存结果复制至此的路径一致）
TRAINED_FOLDER = 'trained'


def trained_actor_path(n_node):
    return os.path.join(TRAINED_FOLDER, str(n_node), 'actor.pt')


def rollout(model, dataset, n_nodes):
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            # VRP_Actor.Model: forward(datas, n_steps, greedy=False, T=1)
            cost, _ = model(bat, n_nodes * 2, True)
            cost = reward1(bat.x, cost.detach(), n_nodes)
        return cost.cpu()

    totall_cost = torch.cat([eval_model_bat(bat.to(device)) for bat in dataset], 0)
    return totall_cost


def evaliuate(valid_loder, n_node):
    """从 trained/{n_node}/actor.pt 加载 VRP_Rollout_train 训练得到的权重（需为 VRP_Actor 结构）。"""
    agent = Model(3, 128, 1, 16, conv_laysers=4).to(device)

    path1 = trained_actor_path(n_node)
    agent.load_state_dict(torch.load(path1, map_location=device))
    print('Loaded:', path1)

    cost = rollout(agent, valid_loder, n_node)
    cost = cost.mean()
    print('Problem:TSP''%s' % n_node, '/ Average distance:', cost.item())

    cost1 = cost.min()

    return cost, cost1


def run(n_node):
    datas = []

    if n_node == 21 or n_node == 51 or n_node == 101 or n_node == 151:
        ckpt = trained_actor_path(n_node)
        if not os.path.exists(ckpt):
            print('Warning: checkpoint not found:', ckpt)
            sys.exit(1)
        node_ = np.loadtxt('./test_data/vrp{}_test_data.csv'.format(n_node - 1), dtype=np.float, delimiter=',')
        demand_ = np.loadtxt('./test_data/vrp{}_demand.csv'.format(n_node - 1), dtype=np.float, delimiter=',')
        capcity_ = np.loadtxt('./test_data/vrp{}_capcity.csv'.format(n_node - 1), dtype=np.float, delimiter=',')
        batch_size = 128
        print(n_node)
    else:
        print('Please enter 21, 51, 101 or 151')
        return
    node_ = node_.reshape(-1, n_node, 2)

    def c_dist(x1, x2):
        return ((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2) ** 0.5

    data_size = node_.shape[0]

    edges = np.zeros((data_size, n_node, n_node, 1))
    for k, data in enumerate(node_):
        for i, (x1, y1) in enumerate(data):
            for j, (x2, y2) in enumerate(data):
                d = c_dist((x1, y1), (x2, y2))
                edges[k][i][j][0] = d
    edges_ = edges.reshape(data_size, -1, 1)

    edges_index = []
    for i in range(n_node):
        for j in range(n_node):
            edges_index.append([i, j])
    edges_index = torch.LongTensor(edges_index)
    edges_index = edges_index.transpose(dim0=0, dim1=1)
    for i in range(data_size):
        data = Data(x=torch.from_numpy(node_[i]).float(), edge_index=edges_index,
                    edge_attr=torch.from_numpy(edges_[i]).float(),
                    demand=torch.tensor(demand_[i]).unsqueeze(-1).float(),
                    capcity=torch.tensor(capcity_[i]).unsqueeze(-1).float())
        datas.append(data)

    print('Data Loaded')
    start_time = time.time()
    dl = DataLoader(datas, batch_size=batch_size)
    evaliuate(dl, n_node)
    end_time = time.time()
    execution_time = end_time - start_time
    print("代码运行时间：", execution_time, "秒")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VRP 评估（trained 目录下的 VRP_Actor 权重）')
    parser.add_argument('--n_node', type=int, required=True, choices=[21, 51, 101, 151],
                        help='节点数，需与 trained 子目录名及 test_data 一致')
    args = parser.parse_args()
    run(args.n_node)
