import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from distributions import Categorical, DiagGaussian
from utils import init, init_normc_

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


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}

        if len(obs_shape) == 3:
            self.base = CNNBase(obs_shape[0], **base_kwargs)
        elif len(obs_shape) == 1:
            self.base = MLPBase(obs_shape[0], **base_kwargs)
        else:
            raise NotImplementedError

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            # self.dist = MonteGaussian(self.base.output_size, num_outputs)
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)


        # dist = self.dist(actor_features)
        action, synergy, q, action_log_probs = self.dist(actor_features)
        # print(action)
        # return value, action, action_log_probs, rnn_hxs
        # if deterministic:
        #     action = dist.mode()
        # else:
        #     action = dist.sample()
        #
        # action_log_probs = dist.log_probs(action)
        # dist_entropy = dist.entropy().mean()

        return value, action, synergy, q, action_log_probs, rnn_hxs

    def adjust_synergy(self, syn=1.0):
        if 0.0 <= syn <= 1.0:
            self.dist.syn_probs = syn
        else:
            print("BAD SYNERGY LEVEL ", syn)

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def eval_monte(self, action):
        pass

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, synergy, q):
        """
        Given an action, tell me what the value of this state is and how probably this action is given the inputs
        :param inputs:
        :param rnn_hxs:
        :param masks:
        :param action:
        :return:
        """
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)

        # find the probablity of this action given the new network
        liklihoods = self.dist.eval_actions(actor_features, action, synergy, q)

        # value, action, liklihoods, rnn_hxs = self.act(inputs, rnn_hxs, masks)

        # dist = self.dist(actor_features)
        #
        # action_log_probs = dist.log_probs(action)
        dist_entropy = 0

        return value, liklihoods, dist_entropy, rnn_hxs


class NNBase(nn.Module):

    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRUCell(recurrent_input_size, hidden_size)
            nn.init.orthogonal_(self.gru.weight_ih.data)
            nn.init.orthogonal_(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size
        # return 10
        # return 7

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x = hxs = self.gru(x, hxs * masks)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N, 1)

            outputs = []
            for i in range(T):
                hx = hxs = self.gru(x[i], hxs * masks[i])
                outputs.append(hx)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.stack(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)),
            nn.ReLU()
        )

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m,
                               init_normc_,
                               lambda x: nn.init.constant_(x, 0))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            # nn.Dropout(0.2),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            # # nn.Dropout(0.2),
            # init_(nn.Linear(hidden_size, self.output_size, bias=False)),
            # # init_(nn.Linear(hidden_size, 10)),
            # nn.Tanh()
        )

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)
        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs
