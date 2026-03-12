import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
import math
from torch.distributions.categorical import Categorical
from vrpUpdate import update_mask, update_state
INIT = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
max_grad_norm = 2
n_nodes = 51


# device = torch.device('cpu')



class GatConv(MessagePassing):
    """
    实现了图神经网络中的消息传递机制。在消息传递过程中，每个节点将与其相邻节点的特征进行聚合和组合，生成新的节点表示
    """
    def __init__(self, in_channels, out_channels, edge_channels,
                 negative_slope=0.2, dropout=0):
        """
        fc：一个线性层，用于将输入节点的特征从in_channels维度映射到out_channels维度
        attn：一个线性层，用于计算节点之间的注意力权重。它将节点的特征、邻接节点的特征以及边的特征作为输入，并输出注意力权重
        negative_slope：负斜率参数，用于控制LeakyReLU激活函数的斜率
        dropout：用于控制在计算注意力权重时进行的dropout操作的概率
        """
        super(GatConv, self).__init__(aggr='add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.fc = nn.Linear(in_channels, out_channels)
        # 2 * out_channels表示将两个节点的特征拼接起来，而edge_channels表示边的特征维度
        self.attn = nn.Linear(2 * out_channels + edge_channels, out_channels)
        if INIT:
            # 模型初始化
            for name, p in self.named_parameters():
                if 'weight' in name:
                    if len(p.size()) >= 2:
                        # 正交初始化方法（nn.init.orthogonal_）对参数进行初始化
                        nn.init.orthogonal_(p, gain=1)
                # bias的参数，就将其初始化为常数0（nn.init.constant_）。
                elif 'bias' in name:
                    nn.init.constant_(p, 0)

    def forward(self, x, edge_index, edge_attr, size=None):
        # 全连接层（self.fc）对输入节点特征x进行线性变换
        x = self.fc(x)
        return self.propagate(edge_index, size=size, x=x, edge_attr=edge_attr)

    def message(self, edge_index_i, x_i, x_j, size_i, edge_attr):
        """
        edge_index_i：输入的边索引，表示从节点i到节点j的边的索引
        x_i：源节点i的特征
        x_j：目标节点j的特征
        size_i：节点i所属的图的大小
        edge_attr：边的属性
        """
        # 拼接得到一个新的特征向量x
        x = torch.cat([x_i, x_j, edge_attr], dim=-1)
        # 通过线性变换self.attn(x)将特征向量x映射到注意力系数
        alpha = self.attn(x)
        # 注意力系数经过LeakyReLU激活函数进行非线性变换
        alpha = F.leaky_relu(alpha, self.negative_slope)
        # 使用softmax函数对注意力系数进行归一化，以获得标准化的边权重
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)

        # Sample attention coefficients stochastically.
        # 为了增加模型的鲁棒性，使用dropout函数对注意力系数进行随机抑制，以防止过拟合
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha

    def update(self, aggr_out):
        # 直接返回聚合后的消息aggr_out作为节点的更新结果
        return aggr_out


class Encoder(nn.Module):
    def __init__(self, input_node_dim, hidden_node_dim, input_edge_dim, hidden_edge_dim, conv_layers=3, n_heads=4):
        """
        input_node_dim：输入节点特征的维度
        hidden_node_dim：隐藏节点特征的维度，也是编码后的节点特征的维度
        input_edge_dim：输入边特征的维度
        hidden_edge_dim：隐藏边特征的维度，也是编码后的边特征的维度
        conv_layers：GATConv图卷积层的数量，默认为3
        n_heads：GATConv图卷积层中的头数，默认为4

        Encoder的前向传播过程通过多次的图卷积操作，在图数据中传递和更新信息，提取图的结构和特征，并生成更丰富、更具表达力的节点特征表示。
        这些节点特征可以用于后续的任务，如图的分类、节点的预测等
        """
        super(Encoder, self).__init__()
        self.hidden_node_dim = hidden_node_dim
        self.fc_node = nn.Linear(input_node_dim, hidden_node_dim)
        # 批量归一化
        self.bn = nn.BatchNorm1d(hidden_node_dim)
        self.be = nn.BatchNorm1d(hidden_edge_dim)
        self.fc_edge = nn.Linear(input_edge_dim, hidden_edge_dim)  # 1-16
        # 创建了一个由3个GATConv图卷积层组成的列表convs1，充分提取节点的特征信息
        self.convs1 = nn.ModuleList(
            [GatConv(hidden_node_dim, hidden_node_dim, hidden_edge_dim) for i in range(conv_layers)])
        if INIT:
            for name, p in self.named_parameters():
                if 'weight' in name:
                    if len(p.size()) >= 2:
                        nn.init.orthogonal_(p, gain=1)
                elif 'bias' in name:
                    nn.init.constant_(p, 0)

    def forward(self, data):
        batch_size = data.num_graphs
        # print(batch_size)
        # edge_attr = data.edge_attr
        # 首先将节点特征和需求特征拼接起来
        x = torch.cat([data.x, data.demand], -1)
        # 经过全连接层self.fc_node进行线性变换
        x = self.fc_node(x)
        # 批归一化层self.bn进行归一化操作
        x = self.bn(x)
        # 边特征也经过了线性变换，通过全连接层self.fc_edge对边特征进行映射，然后经过批归一化层self.be进行归一化操作
        edge_attr = self.fc_edge(data.edge_attr)
        edge_attr = self.be(edge_attr)
        for conv in self.convs1:
            # x = conv(x,data.edge_index)
            # 每个GATConv图卷积层，对节点特征进行多次的信息传递和特征更新
            x1 = conv(x, data.edge_index, edge_attr)
            # 每个图卷积层的输出被添加到节点特征x上，实现了信息的累积和更新(残差连接)
            x = x + x1

        x = x.reshape((batch_size, -1, self.hidden_node_dim))

        return x


class Attention1(nn.Module):
    def __init__(self, n_heads, cat, input_dim, hidden_dim, attn_dropout=0.1, dropout=0):
        super(Attention1, self).__init__()
        """
        n_heads表示注意力头的数量
        input_dim表示输入特征的维度
        hidden_dim表示隐藏层的维度
        head_dim表示每个注意力头的维度，由隐藏层维度除以注意力头的数量得到
        norm表示归一化因子，为了使注意力权重归一化，它的值为每个注意力头的维度的倒数的平方根
        """
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.head_dim = self.hidden_dim / self.n_heads
        self.norm = 1 / math.sqrt(self.head_dim)
        """
        w用于将输入特征与上下文进行组合，输入维度为input_dim * cat（其中cat表示输入特征的拼接维度），输出维度为hidden_dim
        k用于计算键（key）的表示，输入维度为input_dim，输出维度为hidden_dim
        v用于计算值（value）的表示，输入维度为input_dim，输出维度为hidden_dim
        fc是最后的全连接层，用于将注意力头的结果进行合并和降维，输入维度为hidden_dim，输出维度为hidden_dim
        """
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
        '''
        state_t：当前时间步的状态数据，形状为(batch_size, 1, input_dim * 3)，其中input_dim表示输入特征的维度，* 3表示由三个部分拼接而成
        context：上下文信息，形状为(batch_size, n_nodes, input_dim)，其中n_nodes表示节点数量，input_dim表示输入特征的维度
        mask：选定的节点掩码，形状为(batch_size, n_nodes)，用于指示哪些节点是有效的
        :param state_t: (batch_size,1,input_dim*3(GATembeding,fist_node,end_node))
        :param context: （batch_size,n_nodes,input_dim）
        :param mask: selected nodes  (batch_size,n_nodes)
        :return:
        '''
        batch_size, n_nodes, input_dim = context.size()
        # 首先，通过线性层w将state_t与上下文进行组合，得到查询（Query）Q，形状为(batch_size, 1, n_heads, hidden_dim)
        Q = self.w(state_t).view(batch_size, 1, self.n_heads, -1)

        # 通过线性层k和v分别计算上下文的键（Key）K和值（Value）V，并将它们进行维度变换，得到K和V的形状为(batch_size, n_nodes, n_heads, hidden_dim)
        K = self.k(context).view(batch_size, n_nodes, self.n_heads, -1)
        V = self.v(context).view(batch_size, n_nodes, self.n_heads, -1)

        # 将Q、K、V的维度进行转置，得到形状为(batch_size, n_heads, 1, hidden_dim)的Q，以及形状为(batch_size, n_heads, n_nodes, hidden_dim)的K和V
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)

        # (batch_size,n_heads,1,hidden_dim)*(batch_size,n_heads,hidden_dim,n_nodes)
        # 通过点积注意力机制计算查询与键之间的相似度，得到注意力权重compatibility，形状为(batch_size, n_heads, n_nodes)
        compatibility = self.norm * torch.matmul(Q, K.transpose(2, 3))
        compatibility = compatibility.squeeze(2)  # (batch_size,n_heads,n_nodes)

        # 对于无效的节点，通过将掩码应用到注意力权重compatibility中，将其对应位置的权重设置为负无穷大
        mask = mask.unsqueeze(1).expand_as(compatibility)
        u_i = compatibility.masked_fill(mask.bool(), float("-inf"))

        # 将scores与值V进行加权求和，得到输出out_put，形状为(batch_size, n_heads, hidden_dim)
        scores = F.softmax(u_i, dim=-1)  # (batch_size,n_heads,n_nodes)
        scores = scores.unsqueeze(2)
        out_put = torch.matmul(scores, V)  # (batch_size,n_heads,1,n_nodes )*(batch_size,n_heads,n_nodes,head_dim)
        out_put = out_put.squeeze(2).view(batch_size, self.hidden_dim)  # （batch_size,n_heads,hidden_dim）

        # 将输出out_put进行维度变换和降维，得到形状为(batch_size, hidden_dim)的最终输出结果
        out_put = self.fc(out_put)

        return out_put  # (batch_size,hidden_dim)


class ProbAttention(nn.Module):
    def __init__(self, n_heads, input_dim, hidden_dim):
        super(ProbAttention, self).__init__()
        """
        n_heads:注意力头的数量
        input_dim:输入特征的维度
        hidden_dim:隐藏特征的维度
        """
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.norm = 1 / math.sqrt(hidden_dim)
        self.k = nn.Linear(input_dim, hidden_dim, bias=False)
        self.mhalayer = Attention1(n_heads, 1, input_dim, hidden_dim)
        if INIT:
            for name, p in self.named_parameters():
                if 'weight' in name:
                    if len(p.size()) >= 2:
                        nn.init.orthogonal_(p, gain=1)
                elif 'bias' in name:
                    nn.init.constant_(p, 0)

    def forward(self, state_t, context, mask, T):
        '''
        :param state_t: (batch_size,1,input_dim*3(GATembeding,fist_node,end_node))
        :param context: （batch_size,n_nodes,input_dim）
        :param mask: selected nodes  (batch_size,n_nodes)
        :return:softmax_score
        '''
        # MHA层（self.mhalayer）对state_t和context进行注意力计算，MHA层将state_t作为查询（Q），context作为键（K）和值（V），计算得到注意力表示x
        x = self.mhalayer(state_t, context, mask)

        # 对context进行线性变换得到键（K），然后计算查询（Q）与键（K）的相似度，得到compatibility。通过缩放因子self.norm对compatibility进行归一化
        batch_size, n_nodes, input_dim = context.size()
        Q = x.view(batch_size, 1, -1)
        K = self.k(context).view(batch_size, n_nodes, -1)
        compatibility = self.norm * torch.matmul(Q, K.transpose(1, 2))  # (batch_size,1,n_nodes)
        compatibility = compatibility.squeeze(1)

        # compatibility进行非线性转换，使用tanh函数，并乘以10进行缩放
        x = torch.tanh(compatibility)
        x = x * 10

        # mask进行填充操作，将被选中的节点对应的得分设置为负无穷大，以排除无效节点的影响
        x = x.masked_fill(mask.bool(), float("-inf"))
        # 到最终的softmax得分(scores)
        scores = F.softmax(x/T, dim=-1)
        return scores


class Decoder1(nn.Module):
    """
    使用ProbAttention模块计算注意力得分来引导解码过程，并利用全连接层对得分和其他输入特征进行线性变换和组合，从而生成最终的输出
    """
    def __init__(self, input_dim, hidden_dim):
        super(Decoder1, self).__init__()

        super(Decoder1, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.prob = ProbAttention(8, input_dim, hidden_dim)

        self.fc = nn.Linear(hidden_dim+1, hidden_dim, bias=False)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # self._input = nn.Parameter(torch.Tensor(2 * hidden_dim))
        # self._input.data.uniform_(-1, 1)
        if INIT:
            for name, p in self.named_parameters():
                if 'weight' in name:
                    if len(p.size()) >= 2:
                        nn.init.orthogonal_(p, gain=1)
                elif 'bias' in name:
                    nn.init.constant_(p, 0)

    def forward(self, encoder_inputs, pool, capcity, demand, n_steps, T, greedy=False):

        mask1 = encoder_inputs.new_zeros((encoder_inputs.size(0), encoder_inputs.size(1)))
        mask = encoder_inputs.new_zeros((encoder_inputs.size(0), encoder_inputs.size(1)))

        dynamic_capcity = capcity.view(encoder_inputs.size(0), -1)  # bat_size
        demands = demand.view(encoder_inputs.size(0), encoder_inputs.size(1))  # （batch_size,seq_len）
        index = torch.zeros(encoder_inputs.size(0)).to(device).long()

        log_ps = []
        actions = []

        for i in range(n_steps):
            # 在每个步骤中，首先检查是否所有样本的mask1中除第一列外的元素都不为零。如果是，则跳出循环
            # 对于第一个步骤（i==0），从encoder_inputs中选择第一个节点（depot）作为_input
            if not mask1[:, 1:].eq(0).any():
                break
            if i == 0:
                _input = encoder_inputs[:, 0, :]  # depot

            # -----------------------------------------------------------------------------pool+cat(first_node,current_node)
            decoder_input = torch.cat([_input, dynamic_capcity], -1)
            decoder_input = self.fc(decoder_input)
            pool = self.fc1(pool)
            decoder_input = decoder_input + pool
            # -----------------------------------------------------------------------------cat(pool,first_node,current_node)
            '''decoder_input = torch.cat([pool,_input,dynamic_capcity], dim=-1)
            decoder_input  = self.fc(decoder_input)'''
            # -----------------------------------------------------------------------------------------------------------
            if i == 0:
                # 调用update_mask函数更新mask和mask1，以根据初始需求和容量更新掩码
                mask, mask1 = update_mask(demands, dynamic_capcity, index.unsqueeze(-1), mask1, i)
            p = self.prob(decoder_input, encoder_inputs, mask, T)
            dist = Categorical(p)

            # 根据greedy参数的值，如果为True，则选择概率最大的动作索引作为index；否则，从概率分布dist中采样一个动作索引作为index
            if greedy:
                _, index = p.max(dim=-1)
            else:
                index = dist.sample()

            # 将选择的动作索引index添加到actions列表中，并计算该动作的对数概率log_p
            actions.append(index.data.unsqueeze(1))
            log_p = dist.log_prob(index)

            # 同时，计算一个指示是否完成的标志is_done，如果在mask1中除第一列外的元素之和大于等于(encoder_inputs.size(1) - 1)，则认为完成
            is_done = (mask1[:, 1:].sum(1) >= (encoder_inputs.size(1) - 1)).float()
            log_p = log_p * (1. - is_done)

            log_ps.append(log_p.unsqueeze(1))

            # 通过调用update_state函数，根据选择的动作索引index更新动态容量dynamic_capcity
            dynamic_capcity = update_state(demands, dynamic_capcity, index.unsqueeze(-1), capcity[0].item())

            # 再次调用update_mask函数，根据更新后的动态容量dynamic_capcity和索引index，更新mask和mask1
            mask, mask1 = update_mask(demands, dynamic_capcity, index.unsqueeze(-1), mask1, i)

            # 根据选择的动作索引index从encoder_inputs中选择对应的节点作为下一个输入_input，用于下一步的循环
            _input = torch.gather(encoder_inputs, 1,
                                  index.unsqueeze(-1).unsqueeze(-1).expand(encoder_inputs.size(0), -1,
                                                                           encoder_inputs.size(2))).squeeze(1)
        log_ps = torch.cat(log_ps, dim=1)
        actions = torch.cat(actions, dim=1)

        # 使用log_ps.sum(dim=1)对log_ps张量按照维度1进行求和，得到每个样本的总对数概率log_p
        log_p = log_ps.sum(dim=1)

        return actions, log_p


class Model(nn.Module):
    def __init__(self, input_node_dim, hidden_node_dim, input_edge_dim, hidden_edge_dim, conv_laysers):
        super(Model, self).__init__()
        self.encoder = Encoder(input_node_dim, hidden_node_dim, input_edge_dim, hidden_edge_dim, conv_laysers)
        self.decoder = Decoder1(hidden_node_dim, hidden_node_dim)

    def forward(self, datas,  n_steps, greedy=False, T=1):
        x = self.encoder(datas)  # (batch,seq_len,hidden_node_dim)
        # 池化操作，得到最终的图嵌入（Graph Embedding）
        pooled = x.mean(dim=1)
        demand = datas.demand
        capcity = datas.capcity

        actions, log_p = self.decoder(x, pooled, capcity, demand, n_steps, T, greedy)
        return actions, log_p
