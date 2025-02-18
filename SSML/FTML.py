import torch

import utills
import torch.nn as nn
import torch.optim as optim
from model import *


class Meta(nn.Module):
    """
    Meta Learner
    """

    def __init__(self, FLAGS):
        """

        :param FLAGS:
        """
        super(Meta, self).__init__()

        self.lr = FLAGS.lr
        self.FLAGS = FLAGS
        self.net = ConstGATModel(FLAGS)

    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm / counter

    def forward(self, *args):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        origin_weight = self.net.parameters()
        departure, driver_id, weekday, = args[0:3]
        start_id, end_id, mid_start_id, = args[3:6]
        all_link_feature, all_real, all_flow, all_linkdistance,mask = args[6:11]
        all_num, all_mid_num, all_re_num = args[-7:-4]
        label_spt = args[-4:-2]
        label_qry = args[-2]
        loss_func = args[-1]
        all_loss = 0
        pr_loss = 0
        er_loss = 0
        y, y_tr, mid_rep = self.net(departure, driver_id, weekday, start_id, end_id, all_real, all_flow, all_linkdistance, all_link_feature, mask, all_mid_num=all_mid_num)
        # 1. run the i-th task and compute loss for k=0
        if loss_func == mape:
            pr_loss = loss_func(y_tr, label_spt[1].float())
        else:
            pr_loss = loss_func(y_tr, label_spt[1].float())
            loss = loss_func(y, label_spt[0].float())
            pr_MPIW = torch.mean(y_tr[:, 2] - y_tr[:, 0])
        # 训练模型后执行此代码
        all_loss += pr_loss
        all_loss += loss
        all_loss +=  pr_MPIW
        grads = torch.autograd.grad(pr_loss, self.net.parameters(), retain_graph=True, allow_unused=True)
        fast_weights = list(map(lambda p: p[1] - self.lr * p[0] if p[0] is not None else p[1], zip(grads, self.net.parameters())))  # 更新网络参数
        self.update_params(self.net, fast_weights)
        re_link_feature, re_real, re_flow, re_linkdistance, re_mask = args[11:16]
        y = self.net(departure, driver_id, weekday, mid_start_id, end_id, re_real, re_flow, re_linkdistance, re_link_feature, re_mask, all_re_num=all_re_num, mid_rep=mid_rep)
        if loss_func == mape:
            er_loss = loss_func(y, label_qry.float(), all_re_num)
        else:
            er_loss = loss_func(y, label_qry.float())
        all_loss += er_loss

        self.update_params(self.net, origin_weight)  # 参数还原

        return all_loss, pr_loss, er_loss

    # @staticmethod
    # def gather_learnable_params(model_params):
    #     """ Gather the learnable parameters in the models. """
    #     return [params for params in model_params if params.requires_grad]

    @staticmethod
    def update_params(model, fastweight):
        """ Update the learnable parameters with gradients. """
        for params, fast in zip(model.parameters(), fastweight):
            params.data = fast


