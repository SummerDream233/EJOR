import datetime
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data, DataLoader
from tqdm import tqdm


def creat_instance(num, n_nodes=100, random_seed=None):
    """
    这段代码定义了一个函数create_instance，用于生成一个包含指定数量节点、随机位置和随机需求、随机距离矩阵、固定容量的TSP问题实例。
    函数的输入参数包括：
    num: 节点数量（包括一个仓库节点）
    n_nodes: 数据集中包含的节点数量，默认为100
    random_seed: 随机种子，用于生成随机数据，默认为None
    函数的输出包括：
    datas: 一个包含所有节点坐标的列表，其长度为节点数量
    edges: 一个包含所有节点之间距离的矩阵，其形状为(n_nodes*n_nodes, 1)，表示将矩阵展平后的一维向量
    demand: 一个包含每个节点的需求的列表，其长度为节点数量
    capcity: 固定容量，表示所有车辆的容量均为该值
    """

    # 设置种子，产生随机数
    if random_seed is None:
        random_seed = np.random.randint(123456789)
    np.random.seed(random_seed)

    def random_tsp(n_nodes, random_seed=None):

        data = np.random.uniform(0, 1, (n_nodes, 2))
        return data
    # 数据的数量为n_nodes * 2,值在(0,1)之间
    datas = random_tsp(n_nodes)

    def c_dist(x1, x2):
        # 二维欧式距离
        return ((x1[0]-x2[0])**2+(x1[1]-x2[1])**2)**0.5
    # edges = torch.zeros(n_nodes,n_nodes)
    # edges中值都为零，(n_nodes,n_nodes,1)后面的1表示以列的方式进行排列
    edges = np.zeros((n_nodes, n_nodes, 1))
    # 存储两节点之间的距离
    for i, (x1, y1) in enumerate(datas):
        for j, (x2, y2) in enumerate(datas):
            d = c_dist((x1, y1), (x2, y2))
            edges[i][j][0] = d
    edges = edges.reshape(-1, 1)
    CAPACITIES = {
        10: 2.,
        20: 3.,
        50: 4.,
        100: 5.,
        150: 6.
    }
    # 除去一个仓库节点其他节点的需求，值在(1-9)之间
    demand = np.random.randint(1, 10, size=(n_nodes-1))  # Demand, uniform integer 1 ... 9
    # 需求的值在(0-0.9)之间变化
    demand = np.array(demand)/10
    # 在数组的第一个位置插入0值表示其为仓库，其需求也为0
    demand = np.insert(demand, 0, 0.)
    # 不同节点其容量也不同
    capcity = CAPACITIES[n_nodes-1]
    return datas, edges, demand, capcity  # demand(num,node) capcity(num)


'''
a, s, d, f = creat_instance(2, 21)
print(a.shape, s.shape, d.shape, f)  # (21, 2) (441, 1) (22,) 3.0
'''


def creat_data(n_nodes, num_samples=10000, batch_size=32):
    """
    这段代码是用来创建数据的。传入的参数包括节点数，样本数和批量大小。首先生成所有可能的边，
    然后循环创建指定数量的样本，每个样本都包括节点、边、需求和容量等特征，并将它们存储在一个列表中。
    节点和边都是用numpy数组表示的，并将其转换为torch张量以便操作。
    """
    edges_index = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            edges_index.append([i, j])
    edges_index = torch.LongTensor(edges_index)
    edges_index = edges_index.transpose(dim0=0, dim1=1)

    datas = []

    for i in range(num_samples):
        node, edge, demand, capcity = creat_instance(num_samples, n_nodes)
        data = Data(x=torch.from_numpy(node).float(),
                    edge_index=edges_index,
                    edge_attr=torch.from_numpy(edge).float(),
                    demand=torch.tensor(demand).unsqueeze(-1).float(),
                    capcity=torch.tensor(capcity).unsqueeze(-1).float())
        datas.append(data)
        if (i + 1) % 50000 == 0:
            print("Generated %d samples, current time: %s" % ((i + 1), datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    # print(datas)
    dl = DataLoader(datas, batch_size=batch_size)
    return dl

def reward1(static, tour_indices, n_nodes):
    """
    这段代码定义了一个函数 reward1，接受三个参数：static、tour_indices 和 n_nodes。
    函数首先将 static 转换为形状为 (batch_size, n_nodes, 2) 的张量。其中，n_nodes 是指节点数，static 的第二个维度为 2(表示坐标维度)，表示每个节点的坐标。
    接着，tour_indices 用于提取出 static 张量中代表旅行顺序的点。具体来说，idx = tour_indices.unsqueeze(1).expand(-1, static.size(1), -1)
    将 tour_indices 从 (batch_size, n_nodes) 扩展为 (batch_size, n_nodes, n_nodes)，以便可以使用 torch.gather 函数从 static 中提取出对应的点。
    tour 张量表示提取后的点，形状为 (batch_size, n_nodes, n_nodes, 2)，其中最后一个维度表示坐标。
    为了形成一个完整的旅行路线，从起点到终点，函数将 start = static.data[:, :, 0].unsqueeze(1) 和 start 与 tour 进行拼接，
    得到形状为 (batch_size, n_nodes+1, 2) 的张量 y。
    最后，函数计算旅行路线的长度。具体来说，它计算了每个点之间的欧几里得距离，即 tour_len = torch.sqrt(torch.sum(torch.pow(y[:, :-1] - y[:, 1:], 2), dim=2))。
    最后，它将每个样本的距离加和，并返回得到的张量 tour_len.sum(1)。
    """
    static = static.reshape(-1, n_nodes, 2)

    static = static.transpose(2, 1)

    idx = tour_indices.unsqueeze(1).expand(-1, static.size(1), -1)

    tour = torch.gather(static, 2, idx).permute(0, 2, 1)
    # print(tour.shape,tour[0])
    # print(idx.shape,idx[0])
    # Make a full tour by returning to the start
    start = static.data[:, :, 0].unsqueeze(1)
    y = torch.cat((start, tour, start), dim=1)

    # 每个连续点之间的欧氏距离
    tour_len = torch.sqrt(torch.sum(torch.pow(y[:, :-1] - y[:, 1:], 2), dim=2))
    # print(tour_len.sum(1))
    return tour_len.sum(1).detach()

