
import os
import numpy as np
import torch
from creat_vrp import reward1

from torch_geometric.data import Data, DataLoader
from VRP_Actor import Model
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_nodes = 101


def discrete_cmap(N, base_cmap=None):
    """
    N：离散颜色的数量。
    base_cmap：基础颜色映射（可选参数），默认为None
    该离散颜色映射将不同的数据点或类别以不同的颜色展示出来
    """
    # 获取基础颜色映射，存储在base变量中
    base = plt.cm.get_cmap(base_cmap)
    # np.linspace方法在0到1之间均匀分布生成长度为N的数组，表示颜色映射中每个离散点对应的位置
    color_list = base(np.linspace(0, 1, N))
    # 命名规则为基础颜色映射的名称加上离散颜色的数量
    cmap_name = base.name + str(N)
    # base.from_list方法创建新的颜色映射，传入颜色列表、颜色映射名称和离散颜色的数量
    return base.from_list(cmap_name, color_list, N)


def plot_vehicle_routes(data, route, ax1, Greedy, markersize=5, visualize_demands=False, demand_scale=1, round_demand=False):
    # 图标的字体样式为 Times New Roman，字体大小为 10
    plt.rc('font', family='Times New Roman', size=10)

    routes = [r[r != 0] for r in np.split(route.cpu().numpy(), np.where(route.cpu().numpy() == 0)[0]) if (r != 0).any()]
    depot = data.x[0].cpu().numpy()
    locs = data.x[1:].cpu().numpy()
    demands = data.demand.cpu().numpy()*10
    demands = demands[1:]

    capacity = data.capcity*10
    # 绘制仓库节点的标记，使用黑色方块表示，大小为 markersize * 4
    x_dep, y_dep = depot
    ax1.plot(x_dep, y_dep, 'sk', markersize=markersize * 4)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    legend = ax1.legend(loc='upper center')
    # len(routes) + 2为离散颜色的数量，而'nipy_spectral' 是基础的连续色彩映射
    cmap = discrete_cmap(len(routes) + 2, 'nipy_spectral')
    dem_rects = []  # 存储需求矩形的对象
    used_rects = []  # 存储已使用容量矩形的对象
    cap_rects = []  # 存储容量矩形的对象
    qvs = []  # 存储箭头对象
    total_dist = 0  # 存储总距离
    for veh_number, r in enumerate(routes):
        # 根据车辆的序号确定颜色
        color = cmap(len(routes) - veh_number)  # Invert to have in rainbow order
        # 获取当前路径上每个节点的需求
        route_demands = demands[r - 1]
        # 获取当前路径上每个节点的坐标
        coords = locs[r - 1, :]
        # 将坐标转置为 x 坐标和 y 坐标
        xs, ys = coords.transpose()
        # 计算当前路径的总需求量
        total_route_demand = sum(route_demands)
        # assert total_route_demand <= capacity
        if not visualize_demands:
            ax1.plot(xs, ys, 'o', mfc=color, markersize=markersize, markeredgewidth=0.0)

        dist = 0
        x_prev, y_prev = x_dep, y_dep  # 起始点坐标为仓库的坐标
        cum_demand = 0  # 初始化累积需求为0
        for (x, y), d in zip(coords, route_demands):  # （x,y）为坐标，d为该坐标的需求
            # 计算当前节点与前一个节点之间的距离，并累加到路径长度中
            dist += np.sqrt((x - x_prev) ** 2 + (y - y_prev) ** 2)
            # 在当前节点位置添加一个矩形，表示容量的矩形
            cap_rects.append(Rectangle((x, y), 0.01, 0.1))
            # 在当前节点位置添加一个矩形，表示已使用的容量的矩形
            used_rects.append(Rectangle((x, y), 0.01, 0.1 * total_route_demand / capacity))
            # 在当前节点位置添加一个矩形，表示需求的矩形
            dem_rects.append(Rectangle((x, y + 0.1 * cum_demand / capacity), 0.01, 0.1 * d / capacity))

            x_prev, y_prev = x, y
            cum_demand += d
        dist += np.sqrt((x_dep - x_prev) ** 2 + (y_dep - y_prev) ** 2)
        total_dist += dist
        xs = np.insert(xs, 0, x_dep)
        xs = np.insert(xs, xs.size, x_dep)
        ys = np.insert(ys, 0, y_dep)
        ys = np.insert(ys, ys.size, y_dep)
        qv = ax1.quiver(
            xs[:-1],
            ys[:-1],
            xs[1:] - xs[:-1],
            ys[1:] - ys[:-1],
            scale_units='xy',
            angles='xy',
            scale=1,
            color=color,
            label='R{}, N({}), C {} / {}, D {:.2f}'.format(
                veh_number,
                len(r),
                int(total_route_demand) if round_demand else total_route_demand,
                int(capacity) if round_demand else capacity,
                dist
            )
        )

        qvs.append(qv)
    if Greedy:
        ax1.set_title('Greedy,{} routes, total carbon emission {:.2f}g'.format(len(routes), total_dist * 10), family='Times New Roman', size=20)
    else:
        ax1.set_title('Sampling1280,{} routes, total carbon emission {:.2f}g'.format(len(routes), total_dist * 10), family='Times New Roman', size=20)

    ax1.legend(handles=qvs)
    plt.legend(loc=1)
    pc_cap = PatchCollection(cap_rects, facecolor='whitesmoke', alpha=1.0, edgecolor='lightgray')
    pc_used = PatchCollection(used_rects, facecolor='lightgray', alpha=1.0, edgecolor='lightgray')
    pc_dem = PatchCollection(dem_rects, facecolor='black', alpha=1.0, edgecolor='black')

    if visualize_demands:
        ax1.add_collection(pc_cap)
        ax1.add_collection(pc_used)
        ax1.add_collection(pc_dem)
    plt.show()
    # plt.savefig("./temp{}.png".format(54), dpi=600, bbox_inches='tight')


def vrp_matplotlib(Greedy=True):
    node_ = np.loadtxt('./test_data/vrp100_test_data.csv', dtype=float, delimiter=',')
    demand_ = np.loadtxt('./test_data/vrp100_demand.csv', dtype=float, delimiter=',')
    capcity_ = np.loadtxt('./test_data/vrp100_capcity.csv', dtype=float, delimiter=',')
    node_, demand_ = node_.reshape(-1, n_nodes, 2), demand_.reshape(-1, n_nodes)
    data_size = node_.shape[0]

    x = np.random.randint(1, data_size)
    # Calculate the distance matrix
    edges = np.zeros((n_nodes, n_nodes, 1))

    def c_dist(x1, x2):
        return ((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2) ** 0.5
    for i, (x1, y1) in enumerate(node_[x]):
        for j, (x2, y2) in enumerate(node_[x]):
            d = c_dist((x1, y1), (x2, y2))
            edges[i][j][0] = d
    edges_ = edges.reshape(-1, 1)

    edges_index = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            edges_index.append([i, j])
    edges_index = torch.LongTensor(edges_index)
    edges_index = edges_index.transpose(dim0=0, dim1=1)

    datas = []
    data = Data(x=torch.from_numpy(node_[x]).float(), edge_index=edges_index, edge_attr=torch.from_numpy(edges_).float(),
                demand=torch.tensor(demand_[x]).unsqueeze(-1).float(),
                capcity=torch.tensor(capcity_[x]).unsqueeze(-1).float())
    datas.append(data)

    data_loder = DataLoader(datas, batch_size=1)

    agent = Model(3, 128, 1, 16, conv_laysers=4).to(device)
    agent.to(device)
    folder = 'trained'
    filepath = os.path.join(folder, '%s' % n_nodes)

    if os.path.exists(filepath):
        path1 = os.path.join(filepath, 'actor.pt')
        agent.load_state_dict(torch.load(path1, device))
    if Greedy:
        batch = next(iter(data_loder))
        batch.to(device)
        agent.eval()
        # -------------------------------------------------------------------------------------------Greedy
        with torch.no_grad():
            tour, _ = agent(batch, n_nodes * 2, True)
            # cost = reward1(batch.x, tour.detach(), n_nodes)
            # print(cost)
            # print(tour)
    # -------------------------------------------------------------------------------------------sampling1280
    else:
        datas_ = []
        batch_size1 = 128  # sampling batch_size
        for y in range(1280):
            data = Data(x=torch.from_numpy(node_[x]).float(), edge_index=edges_index,
                        edge_attr=torch.from_numpy(edges_).float(),
                        demand=torch.tensor(demand_[x]).unsqueeze(-1).float(),
                        capcity=torch.tensor(capcity_[x]).unsqueeze(-1).float())
            datas_.append(data)
        dl = DataLoader(datas_, batch_size=batch_size1)

        min_tour = []
        min_cost = 100
        T = 1.2  # Temperature hyperparameters
        for batch in dl:
            with torch.no_grad():
                batch.to(device)
                tour1, _ = agent(batch, n_nodes * 2, False, T)
                cost = reward1(batch.x, tour1.detach(), n_nodes)

                id = np.array(cost.cpu()).argmin()
                m_cost = np.array(cost.cpu()).min()
                tour1 = tour1.reshape(batch_size1, -1)
                if m_cost < min_cost:
                    min_cost = m_cost
                    min_tour = tour1[id]

        tour = min_tour.unsqueeze(-2)

    # --------------------------------------------------------------------------------------------
    for i, (data, tour) in enumerate(zip(data_loder, tour)):
        if Greedy:
            print(data.x, data.demand, tour)
            fig, ax = plt.subplots(figsize=(10, 10))
            plot_vehicle_routes(data, tour, ax, Greedy, visualize_demands=False, demand_scale=50, round_demand=True)
        else:
            fig, ax = plt.subplots(figsize=(10, 10))
            plot_vehicle_routes(data, tour, ax, Greedy, visualize_demands=False, demand_scale=50, round_demand=True)

# True:Greedy decoding / False:sampling1280


if __name__ == '__main__':
    vrp_matplotlib(Greedy=True)

