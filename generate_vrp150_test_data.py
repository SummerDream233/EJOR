"""
生成与现有 test_data（vrp20/vrp50/vrp100）相同格式的 vrp150 测试集文件。

数据分布与 creat_vrp.creat_instance 一致（与训练/验证随机实例同分布）：
  - 节点坐标 Uniform(0,1)^2
  - 客户需求 Uniform{1,...,9}/10，仓库为 0
  - 车辆容量与 creat_vrp.CAPACITIES[n_nodes-1] 一致（151 节点 -> 键 150 -> 6.0）

说明（与论文表述对应）：
  - 验证集与随机测试集可用同一分布生成 10,000 个实例；本脚本默认 num_samples=10000。
  - TSPLIB / CVRPLIB 等公开 benchmark 需自行从库中下载，不由本脚本生成。

本脚本仅依赖 numpy（不 import creat_vrp，避免拉起 torch_geometric）。

用法:
  python generate_vrp150_test_data.py
  python generate_vrp150_test_data.py --num_samples 10000 --seed 20250329 --out_dir ./test_data
"""

import argparse
import os

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kw: x  # noqa: E731


# 与 creat_vrp.creat_instance 中 CAPACITIES 保持一致
CAPACITIES = {
    10: 2.0,
    20: 3.0,
    50: 4.0,
    100: 5.0,
    150: 6.0,
}


def creat_instance_numpy(n_nodes, random_seed):
    """与 creat_vrp.creat_instance 同分布；仅返回坐标与需求、容量（test_vrp 会从坐标重算边）。"""
    np.random.seed(int(random_seed))
    datas = np.random.uniform(0.0, 1.0, (n_nodes, 2))
    demand = np.random.randint(1, 10, size=(n_nodes - 1))
    demand = np.asarray(demand, dtype=np.float64) / 10.0
    demand = np.insert(demand, 0, 0.0)
    capcity = float(CAPACITIES[n_nodes - 1])
    return datas, demand, capcity


def main():
    parser = argparse.ArgumentParser(description="生成 vrp150_* CSV（与 creat_instance / test_vrp 一致）")
    parser.add_argument("--n_nodes", type=int, default=151, help="节点数（含 depot），151 -> 文件名前缀 vrp150")
    parser.add_argument("--num_samples", type=int, default=10000, help="实例数量")
    parser.add_argument("--seed", type=int, default=20250329, help="首个实例随机种子，第 i 个实例用 seed+i")
    parser.add_argument("--out_dir", type=str, default="./test_data", help="输出目录")
    args = parser.parse_args()

    n_nodes = args.n_nodes
    n_samples = args.num_samples
    if n_nodes - 1 not in CAPACITIES:
        raise ValueError(
            "n_nodes-1 必须在 CAPACITIES 中，当前 n_nodes=%s；请在脚本中补充 CAPACITIES。" % n_nodes
        )

    prefix = n_nodes - 1
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    path_coords = os.path.join(out_dir, "vrp{}_test_data.csv".format(prefix))
    path_demand = os.path.join(out_dir, "vrp{}_demand.csv".format(prefix))
    path_cap = os.path.join(out_dir, "vrp{}_capcity.csv".format(prefix))

    all_coords = np.zeros((n_samples * n_nodes, 2), dtype=np.float64)
    all_demands = np.zeros((n_samples, n_nodes), dtype=np.float64)
    all_caps = np.zeros(n_samples, dtype=np.float64)

    for i in tqdm(range(n_samples), desc="instances"):
        node, demand, capcity = creat_instance_numpy(n_nodes, args.seed + i)
        all_coords[i * n_nodes : (i + 1) * n_nodes] = node
        all_demands[i] = demand
        all_caps[i] = capcity

    np.savetxt(path_coords, all_coords, fmt="%.6f", delimiter=",")
    np.savetxt(path_demand, all_demands, fmt="%.6f", delimiter=",")
    np.savetxt(path_cap, all_caps, fmt="%.6f")

    print("Written:")
    print(" ", path_coords, "rows =", all_coords.shape[0], "(= num_samples * n_nodes)")
    print(" ", path_demand, "shape =", all_demands.shape)
    print(" ", path_cap, "rows =", all_caps.shape[0])
    print("Sample capacity:", all_caps[0], "(CAPACITIES[%d]=%s)" % (n_nodes - 1, CAPACITIES[n_nodes - 1]))


if __name__ == "__main__":
    main()
