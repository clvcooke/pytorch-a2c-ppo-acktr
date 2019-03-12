from osim.env import L2RunEnv
import  numpy as np
import json
import torch


class TorchRunner(L2RunEnv):
    def __init__(self, visualize=True, acc=0.03):
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
        with open('movement_data.json') as fp:
            self.joint_data = json.load(fp)

    def reset(self, project=True):
        obs = super(TorchRunner, self).reset()
        self.internal_time = 0.0
        return torch.from_numpy(np.array(obs))

    def gen_penalty(self, pos, min_val, max_val, multiplier=0.01):
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

    def calculate_cycle(self):
        self.internal_time += 1.0 / 100
        if self.internal_time < 4.63:
            self.cycle = self.internal_time / 4.63
        else:
            self.cycle = (self.internal_time - 4.63) / 1.96 % 1

    def get_goal_angles(self):
        goal_angs = {'trunk': 0.0}
        if self.internal_time < 4.63:
            idx = int(self.internal_time * 100)
        else:
            idx = int(((self.internal_time - 4.63) % 1.96) * 100) + int(4.63 * 100)
        for joint in self.joints:
            goal_angs[joint] = self.joint_data[joint][idx]
        return goal_angs

    def step(self, action, project=True):
        action = action.squeeze().cpu().numpy()
        total_reward = 0
        for i in range(1):
            self.calculate_cycle()

            obs, reward, done, info = super(TorchRunner, self).step(action)

            GOAL_STATE = 'terminal_stance'

            # lets check if the joints are within their range
            joint_pos = self.osim_model.state_desc['joint_pos']
            joint_vel = self.osim_model.state_desc['joint_vel']
            joint_acc = self.osim_model.state_desc['joint_acc']
            body_pos = self.osim_model.state_desc['body_pos']
            muscles = self.osim_model.state_desc['muscles']
            #
            # pose_reward = 0
            # # rewarding progress towards goal state (fixed for now to be terminal stance):
            # ground_state = self.get_ground_state(body_pos)
            #
            # # heels and toes
            # if not ground_state['toes_l']:
            #     pose_reward += 1
            # if ground_state['toes_r']:
            #     pose_reward += 1
            # if ground_state['calcn_l']:
            #     pose_reward += 1
            # if not ground_state['calcn_r']:
            #     pose_reward += 1
            #
            # r = self.gen_penalty(joint_pos['knee_r'], -0.1, 0.1, 1)
            # pose_reward += 1 - r
            #
            # r = self.gen_penalty(joint_pos['knee_l'], -0.1, 0.1, 1)
            # pose_reward += 1 - r
            #
            # r = self.gen_penalty(joint_pos['hip_r'], -0.3, -0.2, 1)
            # pose_reward += 1 - r
            #
            # r = self.gen_penalty(joint_pos['hip_l'], 0.2, 0.3, 1)
            # pose_reward += 1 - r
            #
            # pose_reward = pose_reward/10
            # print("POSE REWARD IS: ", pose_reward)

            joint_punishment = 0
            ankle_min = -0.52
            ankle_max = 0.26
            knee_max = 0.174
            knee_max = 0
            knee_min = -1.13
            hip_max = 0.087
            hip_min = -0.698
            trunk_min = -0.087
            trunk_max = 0.087

            goal_angles = self.get_goal_angles()
            for joint_name, joint_value in goal_angles.items():
                # print(joint_name)
                try:
                    joint_value = joint_value[0]
                except TypeError:
                    pass
                # print(joint_value)
                joint_punishment += self.gen_penalty(joint_pos[self.sim_joints_map[joint_name]],
                                                     joint_value, joint_value)

            # joint_punishment += self.gen_penalty(joint_pos['ankle_r'], ankle_min, ankle_max)
            # joint_punishment += self.gen_penalty(joint_pos['ankle_l'], ankle_min, ankle_max)

            joint_punishment += self.gen_penalty(joint_pos['knee_l'], knee_min, knee_max, multiplier=0.001)
            joint_punishment += self.gen_penalty(joint_pos['knee_r'], knee_min, knee_max, multiplier=0.001)

            # joint_punishment += self.gen_penalty(joint_pos['hip_r'], hip_min, hip_max)
            # joint_punishment += self.gen_penalty(joint_pos['hip_l'], hip_min, hip_max)
            #
            # joint_punishment += self.gen_penalty(joint_pos['ground_pelvis'][0:1], trunk_min, trunk_max, multiplier=0.1)
            # joint_punishment = 0
            # print("GP: ", joints['ground_pelvis'])
            pelvis_height = min(0.8, body_pos['pelvis'][1])

            activation_punishment = 0
            for muscle_names, muscle in muscles.items():
                activation_punishment += muscle['activation'] / (180 * 8)

            # print("pelvis", pelvis_height)

            # print("Joint Punishment is: ", joint_punishment)
            # reward *= 10
            reward = -activation_punishment
            # print("Fwd reward is: ", reward)
            reward += pelvis_height / 10
            # print("non-punished reward is: ", reward)
            reward = reward - joint_punishment
            # print("Total", reward)
            total_reward += reward

        return torch.from_numpy(np.expand_dims(np.array(obs), 0)), torch.from_numpy(
            np.expand_dims(np.array(total_reward), 0)), [done], [info]
