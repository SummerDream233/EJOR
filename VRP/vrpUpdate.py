import torch
import time


def update_state(demand, dynamic_capcity, selected, c=20):  # dynamic_capcity(num,1)
    """
    demand：需求信息，形状为(batch_size, seq_len)，表示每个样本的需求情况
    dynamic_capcity：动态容量信息，形状为(batch_size, 1)，表示每个样本的容量情况
    selected：选择的动作，形状为(batch_size, 1)，表示每个样本选择的动作
    """
    depot = selected.squeeze(-1).eq(0)  # Is there a group to access the depot
    # 使用torch.gather(demand, 1, selected)根据选中的动作从需求信息中获取当前选择的需求情况
    current_demand = torch.gather(demand, 1, selected)
    # 将当前需求从容量中减去，更新容量状态
    dynamic_capcity = dynamic_capcity-current_demand
    # 如果有选择到起始位置的动作（depot中有任意为True的元素），将对应位置的容量重新设置为初始值c
    if depot.any():
        dynamic_capcity[depot.nonzero().squeeze()] = c
    # 返回更新后的容量状态
    return dynamic_capcity.detach()  # (bach_size,1)


def update_mask(demand, capcity, selected, mask, i):
    """
    demand：需求信息，形状为(batch_size, seq_len)，表示每个样本的需求情况
    capcity：容量信息，形状为(batch_size, 1)，表示每个样本的容量情况
    selected：选择的动作，形状为(batch_size, 1)，表示每个样本选择的动作
    mask：当前的掩码信息，形状为(batch_size, seq_len)
    i：当前步数
    用于环境状态的更新，根据选择的动作更新掩码信息，以便在后续的计算中使用更新后的掩码
    """
    # If there is a route to select a depot, mask the depot, otherwise it will not mask the depot
    go_depot = selected.squeeze(-1).eq(0)
    # print(go_depot.nonzero().squeeze())
    # visit = selected.ne(0)
    # 使用scatter方法将选中的动作在掩码中标记为1，更新mask1
    mask1 = mask.scatter(1, selected.expand(mask.size(0), -1), 1)
    # 如果存在至少一个样本的动作没有选择到起始位置（(~go_depot).any()为True），将对应位置的第一个元素（对应起始位置）标记为0，以确保起始位置不被重复选择
    if (~go_depot).any():
        mask1[(~go_depot).nonzero(), 0] = 0
    # 如果当前步数i+1超过了需求信息的长度，说明已经完成了所有的动作选择，将掩码中的所有样本的第一个元素（对应起始位置）标记为0，以确保之后不再选择起始位置
    if i+1 > demand.size(1):
        is_done = (mask1[:, 1:].sum(1) >= (demand.size(1) - 1)).float()
        combined = is_done.gt(0)
        mask1[combined.nonzero(), 0] = 0
        '''for i in range(demand.size(0)):
            if not mask1[i,1:].eq(0).any():
                mask1[i,0] = 0'''
    # 根据需求信息和容量信息比较得到的布尔张量a，将其与mask1相加，得到最终的掩码mask
    a = demand > capcity
    mask = a + mask1

    return mask.detach(), mask1.detach()
