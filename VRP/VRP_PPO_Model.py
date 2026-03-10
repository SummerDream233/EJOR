import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
import math
from torch.distributions.categorical import Categorical
from torch.optim.lr_scheduler import LambdaLR
import time
from vrpUpdate import update_mask, update_state
# from PPORolloutBaselin import RolloutBaseline
from sklearn.preprocessing import MinMaxScaler

INIT = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
max_grad_norm = 2

n_nodes = 21


# device = torch.device('cpu')


class GatConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_channels, negative_slope=0.2, dropout=0):
        """
        这段代码定义了一个GAT（Graph Attention Network）卷积层，该层继承自PyTorch的MessagePassing类，实现了message passing算法，用于图卷积神经网络中的信息传递和特征提取。

        在初始化函数中，首先定义了该卷积层的输入和输出通道数、负斜率、dropout等参数。接着，通过定义两个全连接层，分别用于线性变换和计算注意力系数。
        其中，fc是一个线性变换，将输入特征变换到指定的输出通道数。attn是一个全连接层，将节点i与节点j之间的特征和边的特征拼接在一起，经过线性变换和LeakyReLU激活函数之后，得到注意力系数。
        最后，如果初始化参数INIT为True，则对权重进行正交初始化和偏置初始化，这可以提高模型的训练效果。
        """
        super(GatConv, self).__init__(aggr='add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.fc = nn.Linear(in_channels, out_channels)
        self.attn = nn.Linear(2 * out_channels + edge_channels, out_channels)
        # INIT 是一个布尔值，用来表示是否要进行参数初始化。这个变量可能在程序的其它地方被设置为 False，以免不必要的初始化操作。
        if INIT:
            for name, p in self.named_parameters():
                if 'weight' in name:
                    if len(p.size()) >= 2:
                        # 正交初始化
                        nn.init.orthogonal_(p, gain=1)
                elif 'bias' in name:
                    # 初始化为0
                    nn.init.constant_(p, 0)

    def forward(self, x, edge_index, edge_attr, size=None):
        """
        这段代码定义了GAT层的前向传播过程，输入参数为节点特征x、边的索引edge_index和边的属性edge_attr，以及size（如果需要）。
        在前向传播过程中，先对节点特征x进行一次线性变换，然后调用MessagePassing类中的propagate方法对图进行消息传递，并返回传递后得到的新节点特征x。
        其中propagate方法的参数edge_index和edge_attr代表图中的边索引和边属性，x代表节点特征，size代表传递的消息形状。
        """
        x = self.fc(x)
        return self.propagate(edge_index, size=size, x=x, edge_attr=edge_attr)

    def message(self, edge_index_i, x_i, x_j, size_i, edge_attr):
        """
        这段代码定义了 message 函数，是 GAT 中非常重要的一步，用于计算每个节点和其邻居节点之间的信息传递。
        在该函数中，首先将每个节点及其邻居节点的特征向量 x_i, x_j 和边的属性 edge_attr 连接起来，通过全连接层 self.attn 计算出注意力系数 alpha。
        注意力系数是一个实数值，代表了节点 j 对节点 i 的重要程度，而 alpha 通过 softmax 函数归一化后，可以看作是概率分布，用于对节点 i 的邻居节点进行加权平均。
        该函数最终返回的是节点 j 的特征向量 x_j 乘以注意力系数 alpha。
        """
        x = torch.cat([x_i, x_j, edge_attr], dim=-1)
        alpha = self.attn(x)  # 注意力系数
        alpha = F.leaky_relu(alpha, self.negative_slope)  # 乘上一个小的负斜率（negative_slope）可以避免输出为0
        alpha = softmax(alpha, edge_index_i, size_i)

        # Sample attention coefficients stochastically.
        # 随机采样注意力系数
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha

    def update(self, aggr_out):
        """
        这段代码是定义了一个update方法，用于将汇聚操作的结果aggr_out作为新节点表示，更新图中每个节点的特征。
        在这个GAT卷积层中，aggr='add'表示将来自相邻节点的信息相加，所以在这里update方法返回的结果就是aggr_out
        """
        return aggr_out


class Encoder(nn.Module):
    def __init__(self, input_node_dim, hidden_node_dim, input_edge_dim, hidden_edge_dim, conv_layers=3, n_heads=4):
        """
        模块将节点和边缘特征的维度(input_node_dim, input_edge_dim)和隐藏节点和边缘特征的期望输出维度(hidden_node_dim, hidden_edge_dim)作为输入。
        它还为卷积层的数量(conv_layers)和注意头的数量(n_heads)接受可选参数。
        __init__函数通过定义编码过程中使用的层来初始化模块，包括用于节点特征的线性层，用于节点和边缘特征的批处理规范化层，以及用于边缘特征的线性层。它
        还初始化要使用的GatConv卷积层的列表，并设置它们的输入和输出维度。

        如果INIT全局变量设置为True，则if INIT块初始化模块参数的权重。它初始化权重参数为正交，偏置参数为常数零。
        """

        super(Encoder, self).__init__()
        self.hidden_node_dim = hidden_node_dim
        self.fc_node = nn.Linear(input_node_dim, hidden_node_dim)
        # 批量归一化
        self.bn = nn.BatchNorm1d(hidden_node_dim)
        self.be = nn.BatchNorm1d(hidden_edge_dim)
        self.fc_edge = nn.Linear(input_edge_dim, hidden_edge_dim)  # 1-16
        # self.bn = nn.ModuleList([nn.BatchNorm1d(hidden_node_dim) for i in range(conv_layers)])
        # self.convs = nn.ModuleList([GatConv(hidden_node_dim, hidden_node_dim, hidden_edge_dim) for i in range(n_heads)])
        self.convs1 = nn.ModuleList(
            [GatConv(hidden_node_dim, hidden_node_dim, hidden_edge_dim) for i in range(conv_layers)])

        # self.convs = nn.ModuleList([GATConv(hidden_node_dim, hidden_node_dim) for i in range(conv_layers)])
        if INIT:
            for name, p in self.named_parameters():
                if 'weight' in name:
                    if len(p.size()) >= 2:
                        nn.init.orthogonal_(p, gain=1)
                elif 'bias' in name:
                    nn.init.constant_(p, 0)

    def forward(self, data):
        """
        这是Encoder类的forward方法。它接受一个数据对象作为输入，该对象包含要编码的图的特征和连通性信息。

        该方法首先沿着最后一个维度将节点特征(data.x)与需求特征(data.demand)连接起来。
        然后，它对连接的特征应用线性转换(self.fc_node)和批处理规范化(self.bn)。类似地，它对边缘特征(data.edge_attr)应用线性变换(self.fc_edge)和批处理规范化(self.be)。

        接下来，它使用边缘特征(edge_attr)和连接信息(data.edge_index)将图注意网络(GAT)卷积(self.convs1)的conv_layers数应用到节点特征(x)上。
        在每次迭代中，将前一次迭代的输出添加到当前迭代的输出中，以允许迭代之间的信息流。

        最后，输出节点特征(x)被重塑为具有维度(batch_size， -1, self.hidden_node_dim)，其中第二个维度对应于每个图中的节点数量。
        输出张量表示图的编码表示，其中每一行对应一个节点，每一列对应一个特征。
        """
        batch_size = data.num_graphs
        # print(batch_size)
        # edge_attr = data.edge_attr

        x = torch.cat([data.x, data.demand], -1)
        x = self.fc_node(x)
        x = self.bn(x)
        edge_attr = self.fc_edge(data.edge_attr)
        edge_attr = self.be(edge_attr)
        for conv in self.convs1:
            # x = conv(x,data.edge_index)
            x1 = conv(x, data.edge_index, edge_attr)
            x = x + x1  # 残差连接

        x = x.reshape((batch_size, -1, self.hidden_node_dim))

        return x


class Attention1(nn.Module):
    def __init__(self, n_heads, cat, input_dim, hidden_dim, attn_dropout=0.1, dropout=0):
        """
        在初始化函数__init__()中，定义了模型的各个参数和网络层。
        其中，n_heads表示多头注意力机制中的头数；cat表示输入数据是沿着什么维度拼接起来的，input_dim表示输入数据的维度；
        hidden_dim表示隐藏层的维度，attn_dropout和dropout分别表示在attention和全连接层中的dropout比例。
        """
        super(Attention1, self).__init__()

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.head_dim = self.hidden_dim / self.n_heads
        self.norm = 1 / math.sqrt(self.head_dim)

        self.w = nn.Linear(input_dim * cat, hidden_dim, bias=False)
        self.k = nn.Linear(input_dim, hidden_dim, bias=False)
        self.v = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=False)
        if INIT:
            for name, p in self.named_parameters():
                if 'weight' in name:
                    if len(p.size()) >= 2:
                        nn.init.orthogonal_(p, gain=1)
                elif 'bias' in name:
                    nn.init.constant_(p, 0)

    def forward(self, state_t, context, mask):
        """
        在forward()函数中，定义了模型的前向计算流程，首先使用全连接层 self.w 对输入进行线性变换，
        然后将结果重塑成(batch_size, 1, n_heads, head_dim)的形状，其中batch_size表示数据的批次大小。
        然后，使用全连接层self.k和self.v分别对上下文信息进行线性变换，得到(batch_size, n_nodes, n_heads, head_dim)的形状。
        接着，将Q、K、V分别转置，然后使用torch.matmul()函数计算出compatibility张量，即各头注意力机制下的得分张量。
        通过在这个得分张量上使用softmax函数，得到各个节点被选中的概率，然后再将这个概率张量乘以V张量，得到最后的输出结果
        """
        '''
        :param state_t: (batch_size,1,input_dim*3(GAT_embeding,fist_node,end_node))
        :param context: （batch_size,n_nodes,input_dim）
        :param mask: selected nodes  (batch_size,n_nodes)
        :return:
        '''
        batch_size, n_nodes, input_dim = context.size()
        Q = self.w(state_t).view(batch_size, 1, self.n_heads, -1)
        K = self.k(context).view(batch_size, n_nodes, self.n_heads, -1)
        V = self.v(context).view(batch_size, n_nodes, self.n_heads, -1)
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
        # Q 和 K的点积，再除以根号d
        compatibility = self.norm * torch.matmul(Q, K.transpose(2,
                                                                3))  # (batch_size,n_heads,1,hidden_dim)*(batch_size,n_heads,hidden_dim,n_nodes)
        compatibility = compatibility.squeeze(2)  # (batch_size,n_heads,n_nodes)
        mask = mask.unsqueeze(1).expand_as(compatibility)
        u_i = compatibility.masked_fill(mask.bool(), float("-inf"))

        scores = F.softmax(u_i, dim=-1)  # (batch_size,n_heads,n_nodes)
        scores = scores.unsqueeze(2)
        out_put = torch.matmul(scores, V)  # (batch_size,n_heads,1,n_nodes )*(batch_size,n_heads,n_nodes,head_dim)
        out_put = out_put.squeeze(2).view(batch_size, self.hidden_dim)  # （batch_size,n_heads,hidden_dim）
        out_put = self.fc(out_put)

        return out_put  # (batch_size,hidden_dim)


class ProbAttention(nn.Module):
    def __init__(self, n_heads, input_dim, hidden_dim):
        super(ProbAttention, self).__init__()
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mhalayer = Attention1(n_heads, 1, input_dim, hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.ReLU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.k = nn.Linear(input_dim, hidden_dim, bias=False)

        if INIT:
            for name, p in self.named_parameters():
                if 'weight' in name:
                    if len(p.size()) >= 2:
                        nn.init.orthogonal_(p, gain=1)
                elif 'bias' in name:
                    nn.init.constant_(p, 0)

    def forward(self, state_t, context, mask):
        x = self.mhalayer(state_t, context, mask)
        x = self.norm1(x + state_t)
        ffn_out = self.ffn(x)
        x = self.norm2(ffn_out + x)

        batch_size, n_nodes, input_dim = context.size()
        Q = x.view(batch_size, 1, -1)
        K = self.k(context).view(batch_size, n_nodes, -1)
        compatibility = torch.matmul(Q, K.transpose(1, 2))  # (batch_size,1,n_nodes)
        compatibility = compatibility.squeeze(1)
        x = torch.tanh(compatibility)
        x = x * 10
        x = x.masked_fill(mask.bool(), float("-inf"))
        scores = F.softmax(x, dim=-1)
        return scores


class Decoder1(nn.Module):
    def __init__(self, input_dim, hidden_dim, device, init=True):
        super(Decoder1, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.init = init

        self.prob = ProbAttention(8, input_dim, hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        if self.init:
            self._initialize_parameters()

    def forward(self, encoder_inputs, pool, actions_old, capacity, demand, n_steps, batch_size, greedy=False, _action=False):
        mask1 = encoder_inputs.new_zeros((encoder_inputs.size(0), encoder_inputs.size(1)))
        mask = encoder_inputs.new_zeros((encoder_inputs.size(0), encoder_inputs.size(1)))

        dynamic_capacity = capacity.view(encoder_inputs.size(0), -1)
        demands = demand.view(encoder_inputs.size(0), encoder_inputs.size(1))
        index = torch.zeros(encoder_inputs.size(0)).to(self.device).long()

        if _action:
            actions_old = actions_old.reshape(batch_size, -1)
            entropys = []
            old_actions_probs = []

            for i in range(n_steps):
                if not mask1[:, 1:].eq(0).any():
                    break
                if i == 0:
                    _input = encoder_inputs[:, 0, :]  # depot

                decoder_input = torch.cat([_input, dynamic_capacity], -1)
                decoder_input = self.layer_norm1(decoder_input)

                p = self.prob(decoder_input, encoder_inputs, mask)

                dist = Categorical(p)
                old_actions_prob = dist.log_prob(actions_old[:, i])
                entropy = dist.entropy()
                is_done = (mask1[:, 1:].sum(1) >= (encoder_inputs.size(1) - 1)).float()
                old_actions_prob = old_actions_prob * (1. - is_done)
                entropy = entropy * (1. - is_done)

                entropys.append(entropy.unsqueeze(1))
                old_actions_probs.append(old_actions_prob.unsqueeze(1))

                dynamic_capacity = update_state(demands, dynamic_capacity, actions_old[:, i].unsqueeze(-1), capacity[0].item())
                mask, mask1 = update_mask(demands, dynamic_capacity, actions_old[:, i].unsqueeze(-1), mask1, i)

                _input = torch.gather(encoder_inputs, 1, actions_old[:, i].unsqueeze(-1).unsqueeze(-1).expand(encoder_inputs.size(0), -1, encoder_inputs.size(2))).squeeze(1)

            entropys = torch.cat(entropys, dim=1)
            old_actions_probs = torch.cat(old_actions_probs, dim=1)
            num_e = entropys.ne(0).float().sum(1)
            entropy = entropys.sum(1) / num_e
            old_actions_probs = old_actions_probs.sum(dim=1)

            return 0, 0, entropy, old_actions_probs
        else:
            log_ps = []
            actions = []

            for i in range(n_steps):
                if not mask1[:, 1:].eq(0).any():
                    break
                if i == 0:
                    _input = encoder_inputs[:, 0, :]  # depot

                decoder_input = torch.cat([_input, dynamic_capacity], -1)
                decoder_input = self.layer_norm1(decoder_input)

                p = self.prob(decoder_input, encoder_inputs, mask)

                dist = Categorical(p)
                if greedy:
                    _, index = p.max(dim=-1)
                else:
                    index = dist.sample()

                actions.append(index.data.unsqueeze(1))
                log_p = dist.log_prob(index)
                is_done = (mask1[:, 1:].sum(1) >= (encoder_inputs.size(1) - 1)).float()
                log_p = log_p * (1. - is_done)

                log_ps.append(log_p.unsqueeze(1))

                dynamic_capacity = update_state(demands, dynamic_capacity, index.unsqueeze(-1), capacity[0].item())
                mask, mask1 = update_mask(demands, dynamic_capacity, index.unsqueeze(-1), mask1, i)

                _input = torch.gather(encoder_inputs, 1, index.unsqueeze(-1).unsqueeze(-1).expand(encoder_inputs.size(0), -1, encoder_inputs.size(2))).squeeze(1)

            log_ps = torch.cat(log_ps, dim=1)
            actions = torch.cat(actions, dim=1)

            log_p = log_ps.sum(dim=1)

            return actions, log_p, 0, 0

    def _initialize_parameters(self):
        for name, p in self.named_parameters():
            if 'weight' in name:
                if len(p.size()) >= 2:
                    nn.init.orthogonal_(p, gain=1)
            elif 'bias' in name:
                nn.init.constant_(p, 0)


class Model(nn.Module):
    def __init__(self, input_node_dim, hidden_node_dim, input_edge_dim, hidden_edge_dim, conv_laysers):
        """
        接受五个参数：input_node_dim、hidden_node_dim、input_edge_dim、hidden_edge_dim 和 conv_layers。
        这些参数分别表示节点特征的输入维度、节点特征的隐藏维度、边特征的输入维度、边特征的隐藏维度和卷积层的数量。

        在构造函数中，首先创建了一个名为 encoder 的 Encoder 类实例，Encoder 是节点和边特征的编码器，将输入的节点和边特征转换为隐藏表示。
        然后创建了一个名为 decoder 的 Decoder1 类实例，Decoder1 是模型的解码器，将隐藏表示和其他信息映射为输出。
        """
        super(Model, self).__init__()
        self.encoder = Encoder(input_node_dim, hidden_node_dim, input_edge_dim, hidden_edge_dim, conv_laysers)
        self.decoder = Decoder1(hidden_node_dim, hidden_node_dim, device)

    def forward(self, datas, actions_old, n_steps, batch_size, greedy, _action):
        """
        forward 方法接受五个参数：datas、actions_old、n_steps、batch_size、greedy 和 _action。
        datas 表示输入的数据，actions_old 表示上一步的输出，n_steps 表示需要模拟的步数，batch_size 表示批次大小，
        greedy 表示是否使用贪心策略选择输出，_action 是一个特殊的参数，表示当前选择的输出。

        在 forward 方法中，首先使用 encoder 对输入的 datas 进行编码，得到一个形状为 (batch, seq_len, hidden_node_dim) 的张量 x，
        其中 batch 表示批次大小，seq_len 表示序列长度，hidden_node_dim 表示节点特征的隐藏维度。
        接着对 x 进行平均池化得到一个形状为 (batch, hidden_node_dim) 的张量 pooled，表示整个序列的平均隐藏表示。

        然后获取 datas 中的需求 demand 和容量 capacity，将它们作为参数传递给 decoder。
        decoder 使用 x、pooled、actions_old、capacity、demand 和其他参数计算输出 actions，
        并返回输出、对数概率、熵和分布等信息以及 x。最后 forward 方法返回这些信息。
        """
        x = self.encoder(datas)  # (batch,seq_len,hidden_node_dim)
        pooled = x.mean(dim=1)
        demand = datas.demand
        capcity = datas.capcity

        actions, log_p, entropy, dists = self.decoder(x, pooled, actions_old, capcity, demand, n_steps, batch_size,
                                                      greedy, _action)
        return actions, log_p, entropy, dists, x


class Critic(nn.Module):
    """Estimates the problem complexity.

    This is a basic module that just looks at the log-probabilities predicted by
    the encoder + decoder, and returns an estimate of complexity
    """

    def __init__(self, hidden_node_dim):
        """
        这段代码定义了一个评估模型的类，叫做Critic，用于估计问题的复杂度。
        该类定义了三个1D卷积层，其中第一个卷积层的输入大小为隐藏节点的维度，输出大小为20，卷积核大小为1，第二个和第三个卷积层的输入输出大小都为20和1，卷积核大小都为1。
        """
        super(Critic, self).__init__()

        self.fc1 = nn.Conv1d(hidden_node_dim, 20, kernel_size=1)
        self.fc2 = nn.Conv1d(20, 20, kernel_size=1)
        self.fc3 = nn.Conv1d(20, 1, kernel_size=1)
        '''if INIT:
            for name, p in self.named_parameters():
                if 'weight' in name:
                    if len(p.size()) >= 2:
                        nn.init.orthogonal_(p, gain=1)
                elif 'bias' in name:
                    nn.init.constant_(p, 0)'''

    def forward(self, x):
        """
        在forward方法中，将输入的x张量转置后，通过三个卷积层计算输出，并将输出张量进行求和和压缩，最终输出评估值value。
        """
        x1 = x.transpose(2, 1)
        output = F.relu(self.fc1(x1))
        output = F.relu(self.fc2(output))
        value = self.fc3(output).sum(dim=2).squeeze(-1)
        return value


class Actor_critic(nn.Module):
    """
    Actor用于学习产生行动的策略，Critic用于评估当前状态的价值函数。
    该模型输入包括节点特征和边特征，使用Encoder对图数据进行编码得到一个隐藏向量。
    Actor从隐藏向量中解码产生行动和对应的概率分布，同时记录下来当前状态的负对数似然和熵。
    负对数似然指的是在给定当前状态下，采取某个动作的概率的相反数的对数，负对数似然的目的就是为了衡量当前采取的动作的概率，如果当前采取的动作的概率越大，那么它的负对数似然就越小。
    熵是指在给定当前状态下，所有可能的动作的概率的加权平均的相反数的对数。熵可以看做是对当前策略的不确定性的度量，使得策略在最大化期望回报的同时，也能尽可能地保持多样性。
    Critic则基于隐藏向量计算当前状态的价值。
    """
    def __init__(self, input_node_dim, hidden_node_dim, input_edge_dim, hidden_edge_dim, conv_laysers):
        super(Actor_critic, self).__init__()
        self.actor = Model(input_node_dim, hidden_node_dim, input_edge_dim, hidden_edge_dim, conv_laysers)
        self.critic = Critic(hidden_node_dim)

    def act(self, datas, actions, steps, batch_size, greedy, _action):
        """
        datas: 状态数据，包含当前任务的节点和边信息；
        actions: 上一步选择的动作，用于对状态转移进行记录；
        steps: 当前已经执行的步数；
        batch_size: 批大小；
        greedy: 是否使用贪心策略，如果为 True，则返回对应的动作；否则，返回动作概率分布；
        _action: 贪心策略使用的随机数。
        在方法内部，act 方法会调用 actor模型的 forward方法，获取模型在给定状态下选择动作actions的对数概率 log_p。然后，它将动作actions和log_p返回给调用者。
        """
        actions, log_p, _, _, _ = self.actor(datas, actions, steps, batch_size, greedy, _action)

        return actions, log_p

    def evaluate(self, datas, actions, steps, batch_size, greedy, _action):
        """
        datas：状态数据
        actions：上一步选择的动作，用于对状态转移进行记录；
        steps：执行的步数
        batch_size：批次大小
        greedy：是否使用贪心策略
        _action：动作
        函数通过调用 actor 对象（包括 Actor 部分和 Critic 部分）来得到当前状态 x 的负对数似然old_log_p和熵entropy，并使用Critic部分得到当前状态的价值 value。
        然后，函数将三个结果返回。
        """
        _, _, entropy, old_log_p, x = self.actor(datas, actions, steps, batch_size, greedy, _action)

        value = self.critic(x)

        return entropy, old_log_p, value


class Memory:
    def __init__(self):
        self.input_x = []  # 输入的节点
        # self.input_index = []
        self.input_attr = []  # 输入的边
        self.actions = []  # 动作
        self.rewards = []  # 奖励
        self.log_probs = []  # 概率
        self.capcity = []  # 容量
        self.demand = []  # 需求

    def def_memory(self):
        self.input_x.clear()
        # self.input_index.clear()
        self.input_attr.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.capcity.clear()
        self.demand.clear()


class Agentppo:
    def __init__(self, steps, greedy, lr, input_node_dim, hidden_node_dim, input_edge_dim, hidden_edge_dim, epoch=1,
                 batch_size=32, conv_laysers=3, entropy_value=0.2, eps_clip=0.2):
        """
        steps:执行多少个时间步长（也就是决策）
        greedy:表示是否使用贪心策略
        lr:学习率
        batch_size:训练次数
        conv_layers:使用的卷积层数量
        entropy_value:熵的权重
        eps_clip:重要性采样比率的截断阈值
        """
        # 当前模型
        self.policy = Actor_critic(input_node_dim, hidden_node_dim, input_edge_dim, hidden_edge_dim, conv_laysers)
        # 旧模型
        self.old_polic = Actor_critic(input_node_dim, hidden_node_dim, input_edge_dim, hidden_edge_dim,
                                      conv_laysers)
        self.old_polic.load_state_dict(self.policy.state_dict())
        # Adam优化器
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.MseLoss = nn.MSELoss()
        self.batch_size = batch_size  # 表示批次大小
        self.epoch = epoch  # 训练轮数
        self.steps = steps
        self.entropy_value = entropy_value
        self.eps_clip = eps_clip
        self.greedy = greedy
        self._action = True
        self.conv_layers = conv_laysers
        self.input_node_dim = input_node_dim
        self.input_edge_dim = input_edge_dim
        self.hidden_node_dim = hidden_node_dim
        self.hidden_edge_dim = hidden_edge_dim
        self.batch_idx = 1
        self.times, self.losses, self.rewards, self.critic_rewards = [], [], [], []

    def adv_normalize(self, adv):
        """
        这个函数实现了一个简单的优势归一化，将传入的优势值进行标准化处理。
        在深度强化学习中，优势值用来表示一个状态或动作的价值相对于平均值的变化程度。
        由于优势值可能具有很大的差异，因此将其标准化可以使优化过程更加稳定。
        通常，这种标准化方法是将每个优势值减去其平均值，然后除以标准差。这个函数的实现方式与此相同，只不过加了一个很小的数以避免分母为零的情况。
        """
        std = adv.std()
        assert std != 0. and not torch.isnan(std), 'Need nonzero std'
        n_advs = (adv - adv.mean()) / (adv.std() + 1e-8)
        return n_advs

    def value_loss_gae(self, val_targ, old_vs, value_od, clip_val):
        """
        val_targ：目标值函数的估计值；
        old_vs：旧的值函数估计值；
        value_od：新的值函数估计值；
        clip_val：用于限制值函数估计值变化的阈值。
        """
        """
        1.计算出通过GAE(Generalized Advantage Estimation)得到的Advantage（优势值）估计值：adv = val_targ - old_vs
        2.对Advantage做标准化处理，得到标准化后的Advantage：adv = self.adv_normalize(adv)
        3.使用标准化后的Advantage计算值函数损失：mse = self.MseLoss(old_vs, val_targ + adv)
        4.限制值函数估计值变化范围在[-clip_val, clip_val]内，以减小估计值变化过大对优化的影响，
        得到最终的值函数损失：val_loss_mat = torch.max(val_loss_mat_unclipped, val_loss_mat_clipped)，
        其中，val_loss_mat_unclipped = self.MseLoss(old_vs, val_targ + adv)，
        val_loss_mat_clipped = self.MseLoss(old_vs, vs_clipped)，
        vs_clipped = old_vs + torch.clamp(old_vs - value_od, -clip_val, +clip_val)。
        """
        # vs_clipped是通过对旧值函数（old value function）和当前值函数（value function）的差值进行剪切（clipping）操作得到的
        # 剪切操作使用了torch.clamp函数，将差值限制在-clip_val和+clip_val之间
        vs_clipped = old_vs + torch.clamp(old_vs - value_od, -clip_val, +clip_val)
        val_loss_mat_unclipped = self.MseLoss(old_vs, val_targ)
        val_loss_mat_clipped = self.MseLoss(vs_clipped, val_targ)

        # 选择了未剪切和剪切后的值函数的均方误差中较大的值，作为最终的值函数损失val_loss_mat
        val_loss_mat = torch.max(val_loss_mat_unclipped, val_loss_mat_clipped)

        mse = val_loss_mat

        return mse

    def update(self, memory, epoch):
        """
        这是一个使用 PyTorch 实现的 PPO（Proximal Policy Optimization，近端策略优化）算法的训练函数，用于更新神经网络策略模型。主要包括以下几个步骤：
        将之前的经验池 memory 中保存的数据提取出来，包括输入 x、边的属性、动作、奖励、对数概率等，并存储到一个 Data 类的列表 datas 中，用于后续的批量训练。
        构造一个 DataLoader 对象，将数据分成多个 batch 进行训练。
        根据当前的 epoch 和学习率衰减函数，设置当前学习率。
        对于每个 batch，使用策略模型 policy 计算出动作的概率分布、价值函数和熵，并计算出 PPO 损失函数。
        计算出 loss 后使用反向传播算法进行参数更新。
        对于每个 epoch，记录下训练时间、平均损失、平均奖励和平均 critic 奖励等指标，用于后续的可视化和分析。
        在训练完成后，将当前策略模型 policy 的参数复制到旧的策略模型 old_policy 中，用于下一轮训练时的重要性采样。
        """
        old_input_x = torch.stack(memory.input_x)
        # old_input_index = torch.stack(memory.input_index)
        old_input_attr = torch.stack(memory.input_attr)
        old_demand = torch.stack(memory.demand)
        old_capcity = torch.stack(memory.capcity)

        old_action = torch.stack(memory.actions)
        old_rewards = torch.stack(memory.rewards).unsqueeze(-1)
        old_log_probs = torch.stack(memory.log_probs).unsqueeze(-1)

        datas = []
        edges_index = []
        for i in range(n_nodes):
            for j in range(n_nodes):
                edges_index.append([i, j])
        edges_index = torch.LongTensor(edges_index)
        edges_index = edges_index.transpose(dim0=0, dim1=1)
        for i in range(old_input_x.size(0)):
            data = Data(
                x=old_input_x[i],
                edge_index=edges_index,
                edge_attr=old_input_attr[i],
                actions=old_action[i],
                rewards=old_rewards[i],
                log_probs=old_log_probs[i],
                demand=old_demand[i],
                capcity=old_capcity[i]
            )
            datas.append(data)
        # print(np.array(datas).shape)
        self.policy.to(device)
        data_loader = DataLoader(datas, batch_size=self.batch_size, shuffle=False)
        # 学习率退火
        scheduler = LambdaLR(self.optimizer, lr_lambda=lambda f: 0.96 ** epoch)
        value_buffer = 0

        for i in range(self.epoch):

            self.policy.train()
            epoch_start = time.time()
            start = epoch_start
            self.times, self.losses, self.rewards, self.critic_rewards = [], [], [], []

            for batch_idx, batch in enumerate(data_loader):
                self.batch_idx += 1
                batch = batch.to(device)
                # 当前策略网络（self.policy）对输入数据进行评估，得到动作的对数概率log_probs、值函数的预测值value和策略网络的熵entropy
                entropy, log_probs, value = self.policy.evaluate(batch, batch.actions, self.steps, self.batch_size,
                                                                 self.greedy, self._action)
                # advangtage function

                # base_reward = self.adv_normalize(base_reward)
                rewar = batch.rewards
                rewar = self.adv_normalize(rewar)
                # rewar = rewar/torch.max(rewar)
                # Value function clipping
                # 计算价值函数的均方误差损失 mse_loss，衡量预测值与真实回报之间的差距
                mse_loss = self.MseLoss(rewar, value)
                # 比率 ratios，即当前策略下动作的概率与旧策略下动作的概率的比值
                ratios = torch.exp(log_probs - batch.log_probs)

                # norm advantages
                advantages = rewar - value.detach()

                # advantages = self.adv_normalize(advantages)
                # PPO loss
                # 第一个项 surr1 是比率乘以优势函数
                surr1 = ratios * advantages
                # 第二个项 surr2 是对比率进行剪切操作后乘以优势函数
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                # total loss
                loss = torch.min(surr1, surr2) + 0.5 * mse_loss - self.entropy_value * entropy
                # 清零优化器的梯度
                self.optimizer.zero_grad()
                # 反向传播计算梯度
                loss.mean().backward()
                # 对梯度进行裁剪，以防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)
                # 更新参数模型
                self.optimizer.step()
                # 更新学习率
                scheduler.step()
                # 记录当前批次的回报和损失，用于后续统计和监控
                self.rewards.append(torch.mean(rewar.detach()).item())
                self.losses.append(torch.mean(loss.detach()).item())
                # print(epoch,self.optimizer.param_groups[0]['lr'])
        # 将旧策略网络的参数更新为当前策略网络的参数
        self.old_polic.load_state_dict(self.policy.state_dict())




if __name__ == '__main__':
    raise Exception('Cannot be called from main')
