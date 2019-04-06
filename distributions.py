import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import scipy.stats as stats
import numpy as np

from utils import AddBias, init, init_normc_
from torch.distributions import LowRankMultivariateNormal

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

FixedCategorical = torch.distributions.Categorical

old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self: old_sample(self).unsqueeze(-1)

log_prob_cat = FixedCategorical.log_prob
FixedCategorical.log_probs = lambda self, actions: log_prob_cat(self, actions.squeeze(-1)).unsqueeze(-1)

FixedCategorical.mode = lambda self: self.probs.argmax(dim=1, keepdim=True)

FixedNormal = torch.distributions.Normal
FixedMulti = torch.distributions.LowRankMultivariateNormal
log_prob_normal = FixedNormal.log_prob
log_prob_multi = FixedMulti.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(self, actions).sum(-1, keepdim=True)
FixedMulti.log_probs = lambda self, actions: log_prob_multi(self, actions).sum(-1, keepdim=True)

entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: entropy(self).sum(-1)

FixedNormal.mode = lambda self: self.mean
from torch.distributions import MultivariateNormal


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)



MIRROR_FACT = np.array([[0.2, 0.45, 0.6, 0.08, 0.15, -0.2, -0.45, -0.6,
                         -0.08, -0.15],
                        [0.2, 0.17, 0.4, 0.2, 0.2, -0.2, -0.17, -0.4,
                         -0.2, -0.2],
                        [-0.15, 0.275, 0.2, -0.075, 0.3, 0.15, -0.275, -0.2,
                         0.075, -0.3],
                        [-0.3, 0.05, -0.05, -0.1, 0.05, 0.3, -0.05, 0.05,
                         0.1, -0.05],
                        [-0.15, 0.3, 0.2, 0.1, -0.05, 0.15, -0.3, -0.2,
                         -0.1, 0.05],
                        [-0.1, 0.55, 0.25, 0.1, -0.1, 0.1, -0.55, -0.25,
                         -0.1, 0.1],
                        [0.8, -0.15, -0.1, -0.1, -0.25, -0.8, 0.15, 0.1,
                         0.1, 0.25],
                        [0.2, 0.1, 0.1, 0.2, 0.4, -0.2, -0.1, -0.1,
                         -0.2, -0.4],
                        [-0.25, 0.2, 0.2, 0.2, 0.7, 0.25, -0.2, -0.2,
                         -0.2, -0.7],
                        [-0.2, -0.45, -0.6, -0.08, -0.15, 0.2, 0.45, 0.6,
                         0.08, 0.15],
                        [-0.2, -0.17, -0.4, -0.2, -0.2, 0.2, 0.17, 0.4,
                         0.2, 0.2],
                        [0.15, -0.275, -0.2, 0.075, -0.3, -0.15, 0.275, 0.2,
                         -0.075, 0.3],
                        [0.3, -0.05, 0.05, 0.1, -0.05, -0.3, 0.05, -0.05,
                         -0.1, 0.05],
                        [0.15, -0.3, -0.2, -0.1, 0.05, -0.15, 0.3, 0.2,
                         0.1, -0.05],
                        [0.1, -0.55, -0.25, -0.1, 0.1, -0.1, 0.55, 0.25,
                         0.1, -0.1],
                        [-0.8, 0.15, 0.1, 0.1, 0.25, 0.8, -0.15, -0.1,
                         -0.1, -0.25],
                        [-0.2, -0.1, -0.1, -0.2, -0.4, 0.2, 0.1, 0.1,
                         0.2, 0.4],
                        [0.25, -0.2, -0.2, -0.2, -0.7, -0.25, 0.2, 0.2,
                         0.2, 0.7]])


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m,
                                    init_normc_,
                                    lambda x: nn.init.constant_(x, 0))

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs, bias=True))
        # self.fc_mean.weight.data = torch.from_numpy(MIRROR_FACT).float()
        self.logstd = AddBias(torch.zeros(num_outputs))
        self.synlogstd = AddBias(torch.ones(num_inputs)*0)
        self.opto_probs = AddBias(torch.ones(1) * 0.01)
        self.syn_probs = 1.0
        # self.opto_probs = nn.Parameter(data=torch.ones(1) * 0.5, requires_grad=True).cuda()

    @staticmethod
    def init_normc_(weight, gain=1):
        weight.normal_(0, 1)
        # TODO: Flag this
        weight.abs_()
        weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))

    @staticmethod
    def init(module, weight_init, bias_init, gain=1):

        weight_init(module.weight.data, gain=gain)
        try:
            bias_init(module.bias.data)
        except AttributeError:
            print("Module is missing bias..")
        return module

    def eval_actions(self, xs, actions, syns, qs):
        # dist = self(x)
        log_probs = torch.zeros(qs.shape[0])
        zeros = torch.zeros([1])
        if xs.is_cuda:
            zeros = zeros.cuda()
        q_dist = torch.distributions.bernoulli.Bernoulli(self.syn_probs)
        for i in range(qs.shape[0]):
            q = qs[i:i + 1]
            x = xs[i:i + 1]
            action = actions[i:i + 1]
            syn = syns[i:i + 1]
            log_probs_q = q_dist.log_prob(q)
            log_probs[i] = log_probs_q
            log_probs[i] = 0
            if q == 0:
                action_mean = self.fc_mean(x)
                #  An ugly hack for my KFAC implementation.
                zeros = torch.zeros(action_mean.size())
                if x.is_cuda:
                    zeros = zeros.cuda()

                action_logstd = self.logstd(zeros)
                f = FixedNormal(action_mean, action_logstd.exp())
                log_probs[i] = log_probs[i] + f.log_probs(action).flatten()
            else:
                zeros = torch.zeros(x.size())
                if x.is_cuda:
                    zeros = zeros.cuda()
                syn_logstd = self.synlogstd(zeros)
                f = FixedNormal(x, syn_logstd.exp())
                log_probs[i] = log_probs[i] + f.log_probs(syn).flatten()
            # return f.log_probs(action)#, f.entropy().mean()
        if x.is_cuda:
            log_probs = log_probs.cuda()
        return log_probs

    def forward(self, x):
        # TODO: put q on the gradient path....
        # this can be done by multiplying the log probabliteis together, as tjhats the legit one...
        # q_dist = torch.distributions.Normal(self.opto_mean, 1.0)
        zeros = torch.zeros([1])
        if x.is_cuda:
            zeros = zeros.cuda()
        q_dist = torch.distributions.bernoulli.Bernoulli(self.syn_probs)
        q = q_dist.sample()
        if q == 0:
            action_mean = self.fc_mean(x)
            zeros = torch.zeros(action_mean.size())
            if x.is_cuda:
                zeros = zeros.cuda()
            synergy = x
            action_logstd = self.logstd(zeros)
            f = FixedNormal(action_mean, action_logstd.exp())
            action = f.sample()
            log_probs = f.log_probs(action)
        else:
            zeros = torch.zeros(x.size())
            if x.is_cuda:
                zeros = zeros.cuda()
            syn_logstd = self.synlogstd(zeros)
            f = FixedNormal(x, syn_logstd.exp())
            synergy = f.sample()
            log_probs = f.log_probs(synergy)
            # action =
            action = self.fc_mean(synergy)
        return action, synergy, q, log_probs + q_dist.log_prob(q)
