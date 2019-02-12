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


class MonteGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(MonteGaussian, self).__init__()

        init_ = lambda m: init(m, init_normc_,
                               lambda x: nn.init.constant_(x, 0))
        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs, bias=False))
        self.logstd = AddBias(torch.ones(num_inputs) * 0.6)
        self.action_dict = {}
        self.action_flag = False
        self.counter = 0

    @staticmethod
    def tcov(m, y=None):
        if y is not None:
            m = torch.cat((m, y), dim=0)
        m_exp = torch.mean(m, dim=1)
        x = m - m_exp[:, None]
        cov = 1 / (x.size(1) - 1) * x.mm(x.t())
        return cov

    def eval_actions(self, xs, actions):
        liklihoods = []
        for i in range(xs.size()[0]):
            liklihoods.append(self.eval_action(xs[i], actions[i]))
        return torch.cat([l.view(1) for l in liklihoods])

    def eval_action(self, x, action):
        # mvn_mean = self.fc_mean(x)
        # ws = self.fc_mean.weight*self.logstd._bias.data.exp().view(7)
        # mvn_cov = torch.matmul(ws, ws.transpose(0,1))
        # norm_cov = torch.diag(mvn_cov)

        zeros = torch.zeros(torch.Size((1, 7)))
        # hack = torch.zeros((3500))
        if x.is_cuda:
            zeros = zeros.cuda()
            # hack = hack.cuda()
        action_logstd = self.logstd(zeros)
        normy = FixedNormal(x, action_logstd.exp())
        samples = normy.sample(torch.Size([1000]))
        res = self.fc_mean(samples)
        cov = self.tcov(res[:, 0, :].t())
        mvn_mean = self.fc_mean(x)
        mvn = MultivariateNormal(mvn_mean, cov + torch.eye(18).cuda() * 0.01)
        liklihood = mvn.log_prob(action)
        # ws = self.fc_mean.weight * self.logstd._bias.data.exp().view(7)
        # lr = FixedMulti(loc=mvn_mean, cov_diag=(torch.ones(18) * 0.00001).cuda(), cov_factor=ws)
        # mvn_cov = torch.matmul(ws, ws.transpose(0, 1))
        # norm_cov = torch.diag(mvn_cov)
        # normy = FixedNormal(mvn_mean, torch.sqrt(norm_cov))
        # action = normy.sample()
        # action = lr.sample()
        # liklihood = lr.log_prob(action)

        return liklihood
        # numpy_action = action.detach().cpu().numpy()
        # with torch.no_grad():
        #     samples = self.fc_mean(normy.sample(hack.size())).detach().cpu().numpy()
        #     liklihoods = np.zeros((samples.shape[1]))
        #     for i in range(samples.shape[1]):
        #         gkde = stats.gaussian_kde(samples[:,i,:].flatten())
        #         liklihood = gkde.logpdf(numpy_action[i,0])
        #         liklihoods[i] = liklihood
        #     liklihood = torch.from_numpy(liklihoods).float().cuda()
        # return liklihood

    def get_liklihood(self, action, dist):
        hack = torch.zeros((100))
        if action.is_cuda:
            hack = hack.cuda()
        samples = dist.sample(hack.size())
        propagrations = self.fc_mean(samples)

    def forward(self, x):

        # zeros = torch.zeros(x.size())

        zeros = torch.zeros(torch.Size((1, 7)))
        # hack = torch.zeros((3500))
        if x.is_cuda:
            zeros = zeros.cuda()
            # hack = hack.cuda()
        action_logstd = self.logstd(zeros)
        normy = FixedNormal(x, action_logstd.exp())
        samples = normy.sample(torch.Size([1000]))
        res = self.fc_mean(samples)
        cov = self.tcov(res[:, 0, :].t())
        mvn_mean = self.fc_mean(x)
        mvn = MultivariateNormal(mvn_mean, cov + torch.eye(18).cuda() * 0.01)
        action = mvn.sample()
        liklihood = mvn.log_prob(action)
        # hack = torch.zeros((3500))
        # if x.is_cuda:
        #     zeros = zeros.cuda()
        #     hack = hack.cuda()
        # action_logstd = self.logstd(zeros)

        # mvn_mean = self.fc_mean(x)
        # ws = self.fc_mean.weight * self.logstd._bias.data.exp().view(7)
        # lr = FixedMulti(loc=mvn_mean, cov_diag=(torch.ones(18) * 0.00001).cuda(), cov_factor=ws)
        # mvn_cov = torch.matmul(ws, ws.transpose(0, 1))
        # norm_cov = torch.diag(mvn_cov)
        # normy = FixedNormal(x, action_logstd.exp())
        # v = normy.sample()
        # action= self.fc_mean(v)
        # action = normy.sample()
        # action = lr.sample()
        # liklihood = self.eval_action(x, action)

        # normy = FixedNormal(x, action_logstd.exp())

        # sample = normy.sample()
        # action = self.fc_mean(sample)
        # liklihood = self.eval_action(x, action)
        # act_str = str(action.detach().cpu().numpy())
        # numpy_mean = action.detach().cpu().numpy()
        # with torch.no_grad():
        #     samples = self.fc_mean(normy.sample(hack.size())).detach().cpu().numpy()
        #     liklihoods = np.zeros((samples.shape[1]))
        #     for i in range(samples.shape[1]):
        #         gkde = stats.gaussian_kde(samples[:,i,:].flatten())
        #         liklihood = gkde.logpdf(numpy_mean[i,0])
        #         liklihoods[i] = liklihood
        #     liklihood = torch.from_numpy(liklihoods).float().cuda()

        return action, action, liklihood


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: self.init(m,
                                    self.init_normc_,
                                    lambda x: nn.init.constant_(x, 0))

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs, bias=False))
        self.logstd = AddBias(torch.zeros(num_outputs))
        self.synlogstd = AddBias(torch.zeros(num_inputs))
        self.opto_probs = AddBias(torch.ones(1)*0.01)
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
            q = qs[i:i+1]
            x = xs[i:i+1]
            action = actions[i:i+1]
            syn = syns[i:i+1]
            log_probs_q = q_dist.log_prob(q)
            log_probs[i] = log_probs_q
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
            action = self.fc_mean(synergy)
        return action, synergy, q, log_probs + q_dist.log_prob(q)
