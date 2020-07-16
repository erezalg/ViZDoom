#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# E. Culurciello
# August 2017

from __future__ import division
from __future__ import print_function
from vizdoom import *
import itertools as it
from random import sample, randint, random
from time import time, sleep
import numpy as np
import skimage.color, skimage.transform
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from tqdm import trange
import argparse
from trains import Task
import pandas as pd

task = Task.init(project_name='vizdoom',task_name='test')
logger = task.get_logger()

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
# Q-learning settings
parser.add_argument('--learning_rate', default=0.00025)
parser.add_argument('--discount_factor', default=0.99)
parser.add_argument('--epochs',default=20)
parser.add_argument('--learning_steps_per_epoch', default=2000)
parser.add_argument('--replay_memory_size', default=10000)

# NN learning settings
parser.add_argument('--batch_size', default=32)

# Training regime
parser.add_argument('--test_episodes_per_epoch',default=100)

# Other parameters
parser.add_argument('--frame_repeat', default=12)
resolution = (30, 45)
parser.add_argument('--episodes_to_watch', default=10)

parser.add_argument('--model_savefile', default="./model-doom.pth")
parser.add_argument('--save_model', default=True)
parser.add_argument('--load_model', default=False)
parser.add_argument('--skip_learning', default=False)
parser.add_argument('--report_img_every_n_steps',default=1000)
parser.add_argument('--report_depth_automap',default=True)

# Configuration file path
parser.add_argument('--config_file_path', default="../../scenarios/simpler_basic.cfg")
args = parser.parse_args()


# config_file_path = "../../scenarios/rocket_basic.cfg"
# config_file_path = "../../scenarios/basic.cfg"
# Converts and down-samples the input image
def preprocess(img):
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)

    return img


class ReplayMemory:
    def __init__(self, capacity):
        channels = 1
        state_shape = (capacity, channels, resolution[0], resolution[1])
        self.s1 = np.zeros(state_shape, dtype=np.float32)
        self.s2 = np.zeros(state_shape, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.isterminal = np.zeros(capacity, dtype=np.float32)

        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def add_transition(self, s1, action, s2, isterminal, reward):
        self.s1[self.pos, 0, :, :] = s1
        self.a[self.pos] = action
        if not isterminal:
            self.s2[self.pos, 0, :, :] = s2
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, sample_size):
        i = sample(range(0, self.size), sample_size)
        return self.s1[i], self.a[i], self.s2[i], self.isterminal[i], self.r[i]


class Net(nn.Module):
    def __init__(self, available_actions_count):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=6, stride=3)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=2)
        self.fc1 = nn.Linear(192, 128)
        self.fc2 = nn.Linear(128, available_actions_count)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 192)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

criterion = nn.MSELoss()


def learn(s1, target_q):
    s1 = torch.from_numpy(s1)
    target_q = torch.from_numpy(target_q)
    s1, target_q = Variable(s1), Variable(target_q)
    output = model(s1)
    loss = criterion(output, target_q)
    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def get_q_values(state):
    state = torch.from_numpy(state)
    state = Variable(state)
    return model(state)

def get_best_action(state):
    q = get_q_values(state)
    m, index = torch.max(q, 1)
    action = index.data.numpy()[0]
    return action


def learn_from_memory():
    """ Learns from a single transition (making use of replay memory).
    s2 is ignored if s2_isterminal """

    # Get a random minibatch from the replay memory and learns from it.
    if memory.size > args.batch_size:
        s1, a, s2, isterminal, r = memory.get_sample(args.batch_size)

        q = get_q_values(s2).data.numpy()
        q2 = np.max(q, axis=1)
        target_q = get_q_values(s1).data.numpy()
        # target differs from q only for the selected action. The following means:
        # target_Q(s,a) = r + gamma * max Q(s2,_) if isterminal else r
        target_q[np.arange(target_q.shape[0]), a] = r + args.discount_factor * (1 - isterminal) * q2
        return learn(s1, target_q)


def perform_learning_step(epoch):
    """ Makes an action according to eps-greedy policy, observes the result
    (next state, reward) and learns from the transition"""

    def exploration_rate(epoch):
        """# Define exploration rate change over time"""
        start_eps = 1.0
        end_eps = 0.1
        const_eps_epochs = 0.1 * args.epochs  # 10% of learning time
        eps_decay_epochs = 0.6 * args.epochs  # 60% of learning time

        if epoch < const_eps_epochs:
            return start_eps
        elif epoch < eps_decay_epochs:
            # Linear decay
            return start_eps - (epoch - const_eps_epochs) / \
                               (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
        else:
            return end_eps

    s1 = preprocess(game.get_state().screen_buffer)

    # With probability eps make a random action.
    eps = exploration_rate(epoch)
    if random() <= eps:
        a = randint(0, len(actions) - 1)
    else:
        # Choose the best action according to the network.
        s1 = s1.reshape([1, 1, resolution[0], resolution[1]])
        a = get_best_action(s1)
    action_hist[a] += 1
    reward = game.make_action(actions[a], args.frame_repeat)

    isterminal = game.is_episode_finished()
    s2 = preprocess(game.get_state().screen_buffer) if not isterminal else None

    #logger.report_image()
    # Remember the transition that was just experienced.
    memory.add_transition(s1, a, s2, isterminal, reward)

    return learn_from_memory()


# Creates and initializes ViZDoom environment.
def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.GRAY8)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.init()
    #game.set_automap_buffer_enabled(args.report_depth_automap)
    #game.set_depth_buffer_enabled(args.report_depth_automap)
    #game.set_automap_mode(vzd.AutomapMode.NORMAL)
    print("Doom initialized.")
    return game


if __name__ == '__main__':
    # Create Doom instance
    game = initialize_vizdoom(args.config_file_path)

    # Action = which buttons are pressed
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]
    action_hist = [0] * len(actions)
    test_action_hist = [0] * len(actions)
    # Create replay memory which will store the transitions
    memory = ReplayMemory(capacity=args.replay_memory_size)

    if args.load_model:
        print("Loading model from: ", args.model_savefile)
        model = torch.load(args.model_savefile)
    else:
        model = Net(len(actions))
    
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate)

    print("Starting the training!")
    time_start = time()
    loss_arr = []
    episode_end_stat = pd.DataFrame(columns=['ammo', 'dmg', 'X pos', 'Y pos', 'Z pos'])
    task.register_artifact(name='episode end stats',artifact=episode_end_stat)


    if not args.skip_learning:
        for epoch in range(args.epochs):
            print("\nEpoch %d\n-------" % (epoch + 1))

            train_scores = []
            train_episodes_finished = 0
            print("Training...")
            game.new_episode()
            for learning_step in trange(args.learning_steps_per_epoch, leave=False):
                loss = perform_learning_step(epoch)
                loss_arr.append(loss)
                iteration = epoch*args.learning_steps_per_epoch + learning_step
                if loss:
                    logger.report_scalar(title='loss',series='training loss',value=loss,iteration=iteration)
                if(learning_step+1) % args.report_img_every_n_steps == 0:
                    logger.report_image(title='scene',iteration=learning_step,series='screen',
                                        image=game.get_state().screen_buffer)
                    if args.report_depth_automap:
                        logger.report_image(title='scene', iteration=iteration, series='automap',
                                            image=game.get_state().automap_buffer)
                        logger.report_image(title='scene', iteration=iteration, series='depth',
                                            image=game.get_state().depth_buffer)
                if game.is_episode_finished():
                    score = game.get_total_reward()
                    logger.report_scalar(title='score', iteration=train_episodes_finished, series='episode score',
                                        value=score)
                    train_scores.append(score)
                    state = game.get_state()

                    ammo2 = game.get_game_variable(GameVariable.AMMO2)
                    dmg = game.get_game_variable(GameVariable.DAMAGECOUNT)
                    posx = game.get_game_variable(GameVariable.POSITION_X)
                    posy = game.get_game_variable(GameVariable.POSITION_Y)
                    posz = game.get_game_variable(GameVariable.POSITION_Z)

                    #TODO this will be overriden on new epoch
                    episode_end_stat.loc[train_episodes_finished] = [ammo2,dmg,posx,posy,posz]

                    #TODO add tracking to objects
                    #print(Label.object_position_x)
                    #print(Label.object_position_y)
                    #print(Label.object_position_z)
                    game.new_episode()

                    train_episodes_finished += 1


            print("%d training episodes played." % train_episodes_finished)
            logger.report_scalar("Training","training episodes",iteration=epoch+1, value=train_episodes_finished)
            train_scores = np.array(train_scores)

            print("Results: mean: %.1f +/- %.1f," % (train_scores.mean(), train_scores.std()), \
                  "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())
            logger.report_scalar("Max score", "Train", iteration=epoch + 1, value=train_scores.min())
            logger.report_scalar("Min score", "Train", iteration=epoch + 1, value=train_scores.max())
            logger.report_scalar("Mean score", "Train", iteration=epoch + 1, value=train_scores.mean())
            logger.report_scalar("Std score", "Train", iteration=epoch + 1, value=train_scores.std())
            logger.report_histogram(title='train', series='actions', values=action_hist, iteration=epoch+1)
            action_hist = [0] * len(actions)

            print("\nTesting...")
            test_episode = []
            test_scores = []
            for test_episode in trange(args.test_episodes_per_epoch, leave=False):
                game.new_episode()
                while not game.is_episode_finished():
                    state = preprocess(game.get_state().screen_buffer)
                    state = state.reshape([1, 1, resolution[0], resolution[1]])
                    best_action_index = get_best_action(state)
                    test_action_hist[best_action_index] += 1
                    game.make_action(actions[best_action_index], args.frame_repeat)
                r = game.get_total_reward()
                test_scores.append(r)

            test_scores = np.array(test_scores)
            print("Results: mean: %.1f +/- %.1f," % (
                test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(),
                  "max: %.1f" % test_scores.max())
            logger.report_scalar("Max score", "Test", iteration=epoch + 1, value=test_scores.min())
            logger.report_scalar("Min score", "Test", iteration=epoch + 1, value=test_scores.max())
            logger.report_scalar("Mean score", "Test", iteration=epoch + 1, value=test_scores.mean())
            logger.report_scalar("Std score", "Test", iteration=epoch + 1, value=test_scores.std())
            logger.report_histogram(title='test', series='actions', values=test_action_hist, iteration=epoch+1)
            test_action_hist = [0] * len(actions)
            print("Saving the network weigths to:", args.model_savefile)
            torch.save(model, args.model_savefile)

            print("Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0))

    game.close()
    print("======================================")
    print("Training finished. It's time to watch!")

    # Reinitialize the game with window visible
    game.set_window_visible(True)
    game.set_mode(Mode.ASYNC_PLAYER)
    game.init()

    for episode in range(args.episodes_to_watch):
        game.new_episode()
        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer)
            state = state.reshape([1, 1, resolution[0], resolution[1]])
            best_action_index = get_best_action(state)

            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            game.set_action(actions[best_action_index])
            for _ in range(args.frame_repeat):
                game.advance_action()

        # Sleep between episodes
        sleep(1.0)
        score = game.get_total_reward()
        print("Total score: ", score)
        logger.report_scalar(title="Total Score",series="score",iteration=episode,value=score)
