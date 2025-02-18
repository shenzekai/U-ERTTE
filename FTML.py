from utills.metrics import *
import torch.nn as nn
from models import ConSTGAT, MetaERTTE, SSML, WDR, WDR_LC, MLPTTE


class FTML(nn.Module):
    """
    Meta Learner
    """

    def __init__(self, FLAGS):
        """

        :param FLAGS:
        """
        super(FTML, self).__init__()

        self.lr = FLAGS.lr
        if FLAGS.model == 'ConSTGAT':
            self.net = ConSTGAT.ConstGATModel(FLAGS)
        elif FLAGS.model == 'SSML':
            self.net = SSML.ConstGATModel(FLAGS)
        elif FLAGS.model == 'MetaER-TTE':
            self.net = MetaERTTE.ConstGATModel(FLAGS)
        elif FLAGS.model == 'WDR':
            self.net = WDR.WDR(FLAGS)
        elif FLAGS.model == 'WDR_LC':
            self.net = WDR_LC.WDR(FLAGS)
        elif FLAGS.model == 'MLPTTE':
            self.net = MLPTTE.MLPTTE(FLAGS)
        else:
            raise ValueError('Model not defined')

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
        all_loss = 0
        label_spt, label_qry, loss_func = args[-4:-2], args[-2], args[-1]
        if isinstance(self.net, (WDR.WDR, WDR_LC.WDR)):
            wide_index, wide_value, deep_category, deep_real = args[0:4]
            all_link_feature, all_real, all_flow, all_linkdistance = args[4:8]
            all_num, all_mid_num, all_re_num = args[-7:-4]
            # Pre-training predict the full travel time and the traveled travel time and corresponding confidence interval
            y, y_tr = self.net(wide_index, wide_value, deep_category, deep_real, all_link_feature, all_num,
                               all_flow, all_linkdistance, all_real, all_mid_num)
            pr_loss = self.compute_and_update_pr_loss(loss_func, y_tr, y, label_spt)
            all_loss += pr_loss
            # Meta-training update params
            grads = torch.autograd.grad(pr_loss, self.net.parameters(), retain_graph=True, allow_unused=True)
            fast_weights = list(map(lambda p: p[1] - self.lr * p[0] if p[0] is not None else p[1], zip(grads, self.net.parameters())))  # 更新网络参数
            self.update_params(self.net, fast_weights)
            # Re-training predict the remaining travel time
            re_link_feature, re_real, re_flow, re_linkdistance = args[8:12]
            y, target = self.net(wide_index, wide_value, deep_category, deep_real, re_link_feature, all_re_num,\
                                 re_flow, re_linkdistance, re_real, re_target=label_qry)
            er_loss = loss_func(y, target.float())
            all_loss += er_loss
        elif isinstance(self.net, SSML.ConstGATModel):
            departure, driver_id, weekday, = args[0:3]
            start_id, end_id, mid_start_id, = args[3:6]
            all_link_feature, all_real, all_flow, all_linkdistance, mask = args[6:11]
            all_num, all_mid_num, all_re_num = args[-7:-4]
            y, y_tr, mid_rep = self.net(departure, driver_id, weekday, start_id, end_id, all_real, all_flow,
                                        all_linkdistance, all_link_feature, mask, all_mid_num=all_mid_num)
            pr_loss = self.compute_and_update_pr_loss(loss_func, y_tr, y, label_spt)
            all_loss += pr_loss
            grads = torch.autograd.grad(pr_loss, self.net.parameters(), retain_graph=True, allow_unused=True)
            fast_weights = list(map(lambda p: p[1] - self.lr * p[0] if p[0] is not None else p[1], zip(grads, self.net.parameters())))  # 更新网络参数
            self.update_params(self.net, fast_weights)
            re_link_feature, re_real, re_flow, re_linkdistance, re_mask = args[11:16]
            y = self.net(departure, driver_id, weekday, mid_start_id, end_id, re_real, re_flow, re_linkdistance, re_link_feature, re_mask, all_re_num=all_re_num, mid_rep=mid_rep)
            er_loss = loss_func(y, label_qry.float())
            all_loss += er_loss
        elif isinstance(self.net, (MetaERTTE.ConstGATModel, ConSTGAT.ConstGATModel)):
            departure, driver_id, weekday, = args[0:3]
            start_id, end_id, mid_start_id, = args[3:6]
            all_link_feature, all_real, all_flow, all_linkdistance, mask = args[6:11]
            all_num, all_mid_num, all_re_num = args[-7:-4]
            y, y_tr = self.net(departure, driver_id, weekday, start_id, end_id, all_real, all_flow, all_linkdistance,
                                all_link_feature, mask, all_mid_num=all_mid_num)
            pr_loss = self.compute_and_update_pr_loss(loss_func, y_tr, y, label_spt)
            all_loss += pr_loss
            grads = torch.autograd.grad(pr_loss, self.net.parameters(), retain_graph=True, allow_unused=True)
            fast_weights = list(map(lambda p: p[1] - self.lr * p[0] if p[0] is not None else p[1], zip(grads, self.net.parameters())))  # 更新网络参数
            self.update_params(self.net, fast_weights)
            re_link_feature, re_real, re_flow, re_linkdistance, re_mask = args[11:16]
            y = self.net(departure, driver_id, weekday, mid_start_id, end_id, re_real, re_flow, re_linkdistance,
                         re_link_feature, re_mask, all_re_num=all_re_num)
            er_loss = loss_func(y, label_qry.float())
            all_loss += er_loss
        elif isinstance(self.net, MLPTTE.MLPTTE):
            all_link_feature, all_real, all_flow, all_linkdistance = args[0:4]
            all_num, all_mid_num, all_re_num = args[-7:-4]
            y, y_tr = self.net(all_link_feature, all_flow, all_linkdistance, all_real, all_num, all_mid_num)
            pr_loss = self.compute_and_update_pr_loss(loss_func, y_tr, y, label_spt)
            all_loss += pr_loss
            grads = torch.autograd.grad(pr_loss, self.net.parameters(), retain_graph=True, allow_unused=True)
            fast_weights = list(map(lambda p: p[1] - self.lr * p[0] if p[0] is not None else p[1],
                                    zip(grads, self.net.parameters())))  # 更新网络参数
            self.update_params(self.net, fast_weights)
            re_link_feature, re_real, re_flow, re_linkdistance = args[4:8]
            y = self.net(re_link_feature, re_flow, re_linkdistance, re_real, all_re_num)
            er_loss = loss_func(y, label_qry.float())
            all_loss += er_loss
        else:
            raise ValueError('Invalid model type')

        self.update_params(self.net, origin_weight)  # 参数还原

        return all_loss, pr_loss, er_loss

    @staticmethod
    def update_params(model, fastweight):
        """ Update the learnable parameters with gradients. """
        for params, fast in zip(model.parameters(), fastweight):
            params.data = fast

    @staticmethod
    def compute_and_update_pr_loss(loss_func, y_tr, y, label_spt, pr_loss_weight=1.0):
        tr_loss = loss_func(y_tr, label_spt[1].float())
        loss = loss_func(y, label_spt[0].float())
        tr_MPIW = torch.mean(y_tr[:, 2] - y_tr[:, 0])
        pr_loss = tr_loss + tr_MPIW * pr_loss_weight + loss
        return pr_loss
