import matplotlib

# matplotlib.use('Qt5Agg')

import copy
import glob
import os

import numpy as np
import torch
import json

import algo
from arguments import get_args
from model import Policy
from storage import RolloutStorage
from osim.env import L2RunEnv
from torchrunner import TorchRunner

args = get_args()
assert args.algo in ['a2c', 'ppo', 'acktr']
if args.recurrent_policy:
    assert args.algo in ['a2c', 'ppo'], \
        'Recurrent policy is not implemented for ACKTR'

num_updates = int(args.num_frames / args.num_steps / args.num_processes)
print("NUM", num_updates)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed + 1)

try:
    os.makedirs(args.log_dir)
except OSError:
    files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)

eval_log_dir = args.log_dir + "_eval"

try:
    os.makedirs(eval_log_dir)
except OSError:
    files = glob.glob(os.path.join(eval_log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)


def envwild():
    # each environment gets its own subprocess and the step happens asyncrhounously
    # we rotate through each environment
    pass


class VectorRunner(TorchRunner):
    def __init__(self):
        super().__init__()


def main():
    import matplotlib.pyplot as plt

    # You probably won't need this if you're embedding things in a tkinter plot...
    plt.ion()
    x = np.linspace(0, 6 * np.pi, 100)
    y = np.sin(x)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    import time
    line1, = ax.plot([0, 1, 2], [0, 1, 1], 'r-')  # Returns a tuple of line objects, thus the comma
    time.sleep(0.01)

    torch.set_num_threads(1)
    args.num_processes = 1
    # device = torch.device("cuda:0" if args.cuda else "cpu")
    device = torch.device("cpu")

    if args.vis:
        from visdom import Visdom
        viz = Visdom(port=args.port)
        win = None

    envs = TorchRunner(acc=0.005)
    ob_shape = envs.reset().shape
    # envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
    #                     args.gamma, args.log_dir, args.add_timestep, device, False)
    #
    actor_critic = Policy(ob_shape, envs.action_space,
                          base_kwargs={'recurrent': args.recurrent_policy})

    # # try to load the previous policy
    # data = torch.load(
    #     r"C:\Users\clvco\URA_F18\pytorch-a2c-ppo-acktr\trained_models\ppo\weight_positiverev_test.pt")
    # # # print(data)
    # actor_critic.load_state_dict(data[0].state_dict())
    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, lr=args.lr,
                               eps=args.eps, alpha=args.alpha,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                         args.value_loss_coef, args.entropy_coef, lr=args.lr,
                         eps=args.eps,
                         max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, acktr=True)
    obs = envs.reset()
    ob_shape = obs.shape
    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              ob_shape, envs.action_space, (agent.actor_critic.base.output_size), (1),
                              actor_critic.recurrent_hidden_state_size)
    print(args.num_processes)
    print(envs.observation_space.shape)
    print(obs.shape)
    print(rollouts.obs[0].shape)
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = list()
    ep_reward = 0
    import tqdm
    start = time.time()
    print(args)
    print(int(args.num_frames) // args.num_steps // args.num_processes)
    print('NUM', num_updates)
    timestep = 0
    ep_ends = []
    for j in range(num_updates):
        if j == 0:
            print("UPDATING SYNERGY")
            actor_critic.adjust_synergy(0.0)
        for step in tqdm.tqdm(range(args.num_steps)):
            # Sample actions
            timestep += 1
            with torch.no_grad():
                value, action, synergy, q, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step],
                    rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)
            ep_reward += reward[0]
            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                       for done_ in done])
            if done[0]:
                obs = envs.reset()
                episode_rewards.append(ep_reward)
                ep_ends.append(timestep)
                ep_reward = 0
            # print(action)
            rollouts.insert(obs, recurrent_hidden_states, action, synergy, q, action_log_prob, value, reward, masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1],
                                                rollouts.recurrent_hidden_states[-1],
                                                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        if j % args.save_interval == 0 and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            # A really ugly way to save a model to CPU
            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()

            save_model = [save_model]
            print("Saving model")
            torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))
            print("Saved model to: ", os.path.join(save_path, args.env_name + ".pt"))

        total_num_steps = (j + 1) * args.num_processes * args.num_steps
        print("update time", print(len(episode_rewards)))
        if True:
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.5f}/{:.5f}, min/max reward {:.5f}/{:.5f}\n".
                    format(j, total_num_steps,
                           int(total_num_steps / (end - start)),
                           len(episode_rewards),
                           np.mean(episode_rewards[-10:]),
                           np.median(episode_rewards[-10:]),
                           np.min(episode_rewards[-10:]),
                           np.max(episode_rewards[-10:]), dist_entropy,
                           value_loss, action_loss))

            import time
            ydata = np.convolve(episode_rewards, np.ones(10) / 10, mode='valid')
            line1.set_xdata(np.arange(0, len(ydata)))
            line1.set_ydata(ydata)
            ax.set_xlim(0, len(ydata))
            ax.set_ylim(min(ydata), max(ydata))
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(0.01)
            # save the returns
            xdata = np.array(ep_ends)
            ret_dir = 'returns_weight_experiments'
            os.makedirs(ret_dir, exist_ok=True)
            ret_path = ret_dir + '/' + args.env_name + '_' + str(args.seed) + '.npy'
            ep_path = ret_dir + '/' + "x_data-" + args.env_name + '_' + str(args.seed) + '.npy'
            np.save(ret_path, np.array(np.array(episode_rewards)))
            np.save(ep_path, ep_ends)
            # plt.plot(range(len(episode_rewards)), episode_rewards)

        # if (args.eval_interval is not None
        #         and len(episode_rewards) > 1
        #         and j % args.eval_interval == 0):
        # eval_envs = make_vec_envs(
        #     args.env_name, args.seed + args.num_processes, args.num_processes,
        #     args.gamma, eval_log_dir, args.add_timestep, device, True)

        # vec_norm = get_vec_normalize(eval_envs)
        # if vec_norm is not None:
        #     vec_norm.eval()
        #     vec_norm.ob_rms = get_vec_normalize(envs).ob_rms
        #
        # eval_episode_rewards = []
        #
        # obs = eval_envs.reset()
        # eval_recurrent_hidden_states = torch.zeros(args.num_processes,
        #                                            actor_critic.recurrent_hidden_state_size, device=device)
        # eval_masks = torch.zeros(args.num_processes, 1, device=device)

        # while len(eval_episode_rewards) < 10:
        #     with torch.no_grad():
        #         _, action, _, eval_recurrent_hidden_states = actor_critic.act(
        #             obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)
        #
        #     # Obser reward and next obs
        #     obs, reward, done, infos = eval_envs.step(action)
        #
        #     eval_masks = torch.FloatTensor([[0.0] if done_ else [1.0]
        #                                     for done_ in done])
        #     for info in infos:
        #         if 'episode' in info.keys():
        #             eval_episode_rewards.append(info['episode']['r'])
        #
        # eval_envs.close()
        #
        # print(" Evaluation using {} episodes: mean reward {:.5f}\n".
        #       format(len(eval_episode_rewards),
        #              np.mean(eval_episode_rewards)))

        # if args.vis and j % args.vis_interval == 0:
        #     try:
        #         # Sometimes monitor doesn't properly flush the outputs
        #         win = visdom_plot(viz, win, args.log_dir, args.env_name,
        #                           args.algo, args.num_frames)
        #     except IOError:
        #         pass


if __name__ == "__main__":
    main()
