import torch

from scipy.stats import ttest_rel
import copy

from creat_vrp import reward1

from torch.nn import DataParallel
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_inner_model(model):
    # 无论是否已经进行了并行处理，确保能够正确地获取模型的内部模型
    return model.module if isinstance(model, DataParallel) else model


def rollout(model, dataset, n_nodes):

    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _ = model(bat, n_nodes*2, True)
            cost = reward1(bat.x, cost.detach(), n_nodes)
        return cost.cpu()
    totall_cost = torch.cat([eval_model_bat(bat.to(device))for bat in dataset], 0)
    return totall_cost


class RolloutBaseline():
    def __init__(self, model,  dataset, n_nodes=50, epoch=0):
        super(RolloutBaseline, self).__init__()
        """
        model: 模型对象，用于进行预测和推断
        dataset: 数据集对象，用于提供数据进行训练或评估
        n_nodes: 节点数目，默认为50
        epoch: 当前的训练轮数，默认为0
        """
        self.n_nodes = n_nodes
        self.dataset = dataset
        self._update_model(model, epoch)

    def _update_model(self, model, epoch, dataset=None):
        # 对输入的模型 model 进行深拷贝，以确保每次更新的模型是独立的副本
        self.model = copy.deepcopy(model)
        # 使用深拷贝后的模型进行 rollout 操作，计算 baseline 值，将 baseline 值转换为 NumPy 数组，并保存到bl_vals属性中
        self.bl_vals = rollout(self.model, self.dataset, n_nodes=self.n_nodes).cpu().numpy()
        # 计算baseline值的平均值，并保存到mean属性中
        self.mean = self.bl_vals.mean()
        # 更新当前的训练轮数 epoch
        self.epoch = epoch

    def eval(self, x, n_nodes):
        # 用于评估模型在输入数据上的性能
        with torch.no_grad():
            tour, _ = self.model(x, n_nodes, True)
            v = reward1(x.x, tour.detach(), n_nodes)

        # There is no loss
        # 返回旅行路径的奖励值作为模型的评估结果
        return v

    def epoch_callback(self, model, epoch):
        """
        model: 当前训练轮次的候选模型
        epoch: 当前训练轮次的编号
        """
        # 在评价数据集上评价候选模型
        print("Evaluating candidate model on evaluation dataset")
        # 使用候选模型 model 对评估数据集 self.dataset 进行推断，计算候选模型在评估数据集上的奖励值
        candidate_vals = rollout(model, self.dataset, self.n_nodes).cpu().numpy()
        # 计算候选模型的奖励值的平均值 candidate_mean
        candidate_mean = candidate_vals.mean()
        # 打印输出当前轮次的候选模型的奖励平均值、基准模型的奖励值平均值以及二者之间的差异
        print("Epoch {} candidate mean {}, baseline epoch {} mean {}, difference {}".format(
            epoch, candidate_mean, self.epoch, self.mean, candidate_mean - self.mean))
        # 如果候选模型的奖励值平均值比基准模型的奖励值平均值低，进一步进行统计显著性检验
        if candidate_mean - self.mean < 0:
            # Calc p value
            # 计算候选模型和基准模型的奖励值之间的T统计量和p值
            t, p = ttest_rel(candidate_vals, self.bl_vals)
            # 根据p值计算单侧的p值（one-sided）
            p_val = p / 2  # one-sided
            assert t < 0, "T-statistic should be negative"
            print("p-value: {}".format(p_val))
            # 如果p值小于 0.05，则更新基准模型为候选模型
            if p_val < 0.05:
                print('Update baseline')
                self._update_model(model, epoch)

    def state_dict(self):
        # 通过调用state_dict方法，可以获得对象状态的字典表示，这对于使用torch.save和torch.load等序列化技术保存和加载对象的状态非常有用
        return {
            'model': self.model,
            'dataset': self.dataset,
            'epoch': self.epoch
        }

    def load_state_dict(self, state_dict):
        """
        通过调用load_state_dict方法，并传递相应的状态字典作为参数，可以加载模型的状态并更新当前对象的状态，以便继续使用或进行进一步操作
        """
        # 创建一个副本模型load_model，使用copy.deepcopy()函数复制当前对象的模型
        load_model = copy.deepcopy(self.model)
        # 使用get_inner_model()函数获取load_model和state_dict['model']中的内部模型
        # 调用load_state_dict()方法，将state_dict['model']的状态字典加载到load_model的内部模型中
        get_inner_model(load_model).load_state_dict(get_inner_model(state_dict['model']).state_dict())
        # 调用_update_model()方法，使用加载后的模型load_model、状态字典中的epoch和dataset来更新当前对象的状态
        self._update_model(load_model, state_dict['epoch'], state_dict['dataset'])
