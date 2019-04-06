from osim.env import L2RunEnv
import numpy as np
import json
import torch

HEIGHT_WEIGHT = 1.25
JOINT_WEIGHT = -0.0714
ACTIVATION_WEIGHT = -2.0
FORCE_NORM = 1 / 74430
JOINT_MUL = {
    'knee': 1.5,
    'hip': 2.0,
    'back': 1.0,
    'ankle': 3.0
}

JOINT_MAX = {
    'knee': 0,
    'hip': 0.087,
    'ankle': 0.26,
    'back': 0.174
}

JOINT_MIN = {
    'knee': -1.13,
    'hip': -0.698,
    'ankle': -0.52,
    'back': -0.174
}

ANKLE_MUL = 3.0
KNEE_MUL = 1.5
HIP_MUL = 2.0
TRUNK_MUL = 1.0


class TorchRunner(L2RunEnv):
    def __init__(self, acc=0.03):
        super(TorchRunner, self).__init__(False, acc)
        self.internal_time = 0.0
        self.cycle = 0.0
        self.joints = ['LHF',
                       "LKF",
                       'LAP',
                       'RHF',
                       'RKF',
                       'RAP']
        self.sim_joints_map = {'trunk': 'back',
                               'LHF': 'hip_l',
                               'LKF': 'knee_l',
                               'LAP': 'ankle_l',
                               'RHF': 'hip_r',
                               'RKF': 'knee_r',
                               "RAP": 'ankle_r'}

    def reset(self, project=True):
        obs = super(TorchRunner, self).reset()
        self.internal_time = 0.0
        obs = self.process_observation(obs)
        return torch.from_numpy(np.array(obs))

    @staticmethod
    def gen_penalty(pos, min_val, max_val, multiplier=0.01):
        assert len(pos) == 1
        pos = pos[0]
        if min_val <= pos <= max_val:
            return 0
        elif pos <= min_val:
            return multiplier * abs(pos - min_val)
        else:
            return multiplier * abs(max_val - pos)

    @staticmethod
    def get_ground_state(body_pos):
        """
        :return: ON/OFF of each heel and toe (r/l)
        """
        ground_state = {
            "calcn_r": False,
            "calcn_l": False,
            "toes_r": False,
            "toes_l": False
        }
        for key in ground_state.keys():
            # TODO: fix
            if body_pos[key][1] < 0.01:
                ground_state[key] = True

        return ground_state

    @staticmethod
    def process_observation(obs):
        # return obs[0:32]
        return obs[0:32]

    @staticmethod
    def sigmoid(x):
        return 1/(1 + np.exp(-x))

    def step(self, action, project=True):
        action = action.squeeze().cpu().numpy()
        action = TorchRunner.sigmoid(action)
        total_reward = 0
        for i in range(1):
            obs, reward, done, info = super(TorchRunner, self).step(action)
            obs = self.process_observation(obs)
            joint_pos = self.osim_model.state_desc['joint_pos']
            body_pos = self.osim_model.state_desc['body_pos']
            muscles = self.osim_model.state_desc['muscles']

            # r_h
            pelvis_height = body_pos['pelvis'][1]
            height_reward = min(0.8, pelvis_height)

            # r_j
            joint_punishment = 0
            for joint in ['hip', 'knee', 'ankle']:
                for side in ['l', 'r']:
                    jn = joint
                    joint_punishment += self.gen_penalty(
                        joint_pos[jn + '_' + side],
                        JOINT_MIN[jn],
                        JOINT_MAX[jn],
                        JOINT_MUL[jn]
                    )
            joint_punishment += self.gen_penalty(
                joint_pos['back'],
                JOINT_MIN['back'],
                JOINT_MAX['back'],
                JOINT_MUL['back']
            )

            # r_a
            activation_punishment = 0
            # muscles is a dict: https://gist.github.com/clvcooke/d8389724c4e107233d3e6e5fba67aefc
            for muscle in muscles.values():
                activation_punishment += muscle['fiber_force']
            activation_punishment *= FORCE_NORM
            height_reward = np.exp(height_reward) - 1
            # total_reward = height_reward * HEIGHT_WEIGHT + joint_punishment * JOINT_WEIGHT \
            #                + activation_punishment * ACTIVATION_WEIGHT
            total_reward = height_reward * HEIGHT_WEIGHT #  + joint_punishment*JOINT_WEIGHT
            # total_reward += joint_punishment*JOINT_WEIGHT
            # total_reward += (np.exp(height_reward) - 1)
            # total_reward += height_reward * HEIGHT_WEIGHT
        # total_reward += joint_punishment*JOINT_WEIGHT
        # total_reward = -activation_punishment
        return torch.from_numpy(np.expand_dims(np.array(obs), 0)), torch.from_numpy(
            np.expand_dims(np.array(total_reward), 0)), [done], [info]
