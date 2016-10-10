# An attempt to replicate http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html
# (You can download it from Scihub)

from __future__ import print_function, division
import tensorflow as tf
import numpy as np
import gym
from collections import namedtuple
import random
import os
import h5py
from PIL import Image
from itertools import cycle

FRAMES_PER_STATE=4
FRAME_WIDTH = 84
FRAME_HEIGHT = 110

def reduce_stdev(t):
    m = tf.reduce_mean(t)
    return tf.sqrt(tf.reduce_mean(tf.square(t - m)))

def explained_variance(t, p):
    return 1 - reduce_stdev(t - p) / reduce_stdev(t)

def weight_variable(*shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(*shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def ident_none(f):
    return (lambda x: x) if f is None else f

def lrelu(t):
    return tf.maximum(t, 0.01 * t)

def conv_layer(x, window, out_channels, stride, vlist, nonlin = lrelu):
    nonlin = ident_none(nonlin)
    in_channels = x.get_shape()[3].value
    W_conv = weight_variable(window, window, in_channels, out_channels)
    b_conv = bias_variable(out_channels)
    vlist.append(W_conv)
    vlist.append(b_conv)
    return nonlin(
            tf.nn.conv2d(x, W_conv, strides=[1, stride, stride, 1], padding='SAME') + b_conv)

def product(i):
    ret = 1
    for n in i: 
        ret *= n
    return ret
    
def flatten(x):
    dim = product(d.value for d in x.get_shape()[1:])
    return tf.reshape(x, [-1, dim])

def fcl(x, size, vlist, nonlin = lrelu):
    nonlin = ident_none(nonlin)
    W = weight_variable(x.get_shape()[1].value, size)
    b = bias_variable(size)
    vlist.append(W)
    vlist.append(b)
    return nonlin(tf.matmul(x, W) + b)

def down_sample(s):
    im = Image.fromarray(s)
    im.thumbnail((FRAME_WIDTH, FRAME_HEIGHT))
    bw = im.convert('L')
    arr = np.array(bw.getdata(), dtype=np.uint8)
    return arr.reshape([FRAME_HEIGHT, FRAME_WIDTH])

QNetwork = namedtuple('QNetwork', 'frames qvals vlist')
Transition = namedtuple('Transition', 'begin action reward terminal end')

def count_dead(name, t):
    zero = tf.less_equal(t, 0)
    all_zero = tf.reduce_all(zero, 0)
    zero_as_ones = tf.cast(all_zero, tf.float32)
    tf.scalar_summary('%s_dead' % name, tf.reduce_sum(zero_as_ones))

def make_Qnetwork(num_outputs, count_dead_summaries=False):
    vlist = []
    inputs = tf.placeholder(tf.float32, [None, FRAME_HEIGHT, FRAME_WIDTH, FRAMES_PER_STATE])
    scaled_inputs = inputs / 256.0
    h_conv1 = conv_layer(scaled_inputs, 8, 32, 4, vlist)
    h_conv2 = conv_layer(h_conv1, 4, 64, 2, vlist)
    h_conv3 = conv_layer(h_conv2, 3, 64, 1, vlist)
    h_conv3_flat = flatten(h_conv3)
    h_fc = fcl(h_conv3_flat, 512, vlist)
    Q_vals = fcl(h_fc, num_outputs, vlist, nonlin=None)

    if count_dead_summaries:
        count_dead('conv1', h_conv1)
        count_dead('conv2', h_conv2)
        count_dead('conv3', h_conv3)
        count_dead('fc', h_fc)

    return QNetwork(inputs, Q_vals, vlist)

class QNetworkPair(object):
    def __init__(self, session, discount_rate, num_outputs, global_step):
        self.session = session
        self.global_step = global_step

        target_network = make_Qnetwork(num_outputs)
        training_network = make_Qnetwork(num_outputs, count_dead_summaries=True)
        copy_ops = [target_var.assign(train_var) for target_var, train_var in
                zip(target_network.vlist, training_network.vlist)]
        self.copy_op = tf.group(*copy_ops)

        self.start_frames = training_network.frames

        # Used to choose actions during roll-outs
        self.best_action = tf.argmax(training_network.qvals, dimension=1)

        self.terminal = tf.placeholder(tf.bool, [None])
        not_terminal = tf.logical_not(self.terminal)
        reward_mask = tf.cast(not_terminal, tf.float32)
        max_Q = tf.reduce_max(target_network.qvals, reduction_indices=1)
        self.end_frames = target_network.frames
        self.rewards = tf.placeholder(tf.float32, [None])
        # Terminate expected value sequence if game ended.
        estimated_Qs = max_Q * discount_rate * reward_mask + self.rewards

        tf.scalar_summary('average_Q', tf.reduce_mean(estimated_Qs))

        self.actions_chosen = tf.placeholder(tf.uint8, [None])

        # This is ugly
        actions_hot = tf.one_hot(self.actions_chosen, num_outputs)
        Qs_taken = tf.reduce_sum(training_network.qvals * actions_hot,
                reduction_indices=1)

        # Do not train the target network!
        Q_error = Qs_taken - tf.stop_gradient(estimated_Qs)
        squared_error = tf.square(Q_error)
        tf.scalar_summary('explained_variance', explained_variance(estimated_Qs, Qs_taken))
        # Error clipping
        loss = tf.reduce_sum(tf.sqrt(tf.square(Q_error) + 1))
        tf.scalar_summary('loss', tf.reduce_mean(squared_error))
        self.opt = tf.train.AdamOptimizer().minimize(
                loss, global_step=global_step)

        self.summaries = tf.merge_all_summaries()


    def choose_actions(self, states):
        return self.session.run(self.best_action, {self.start_frames: states})

    def train(self, starts, actions, rewards, terminals, ends):
        _, summaries, step = self.session.run(
                [self.opt, self.summaries, self.global_step],
                {self.start_frames: starts,
                 self.actions_chosen: actions,
                 self.rewards: rewards,
                 self.terminal: terminals,
                 self.end_frames: ends})
        return summaries, step

    def update_targets(self):
        self.session.run(self.copy_op)

class TransitionTable(object):
    def __init__(self, f, prefix, size):
        self.logical_size = size
        self.physical_size = size + 1

        self.starts_var = f.require_dataset('%s_starts' % prefix, (self.physical_size, FRAME_HEIGHT, FRAME_WIDTH, 4), dtype=np.uint8)
        self.starts = np.array(self.starts_var, dtype=np.uint8)
        self.actions_var = f.require_dataset('%s_actions' % prefix, (self.physical_size,), dtype=np.uint8)
        self.actions = np.array(self.actions_var, dtype=np.uint8)
        self.rewards_var = f.require_dataset('%s_rewards' % prefix, (self.physical_size,), dtype=np.int8)
        self.rewards = np.array(self.rewards_var, dtype=np.int8)
        self.terminal_var = f.require_dataset('%s_terminal' % prefix,
            (self.physical_size,), dtype=np.bool8)
        self.terminal = np.array(self.terminal_var, dtype=np.bool8)

        self.write_ind_var = f.require_dataset('%s_write_ind' % prefix, (1,),
                dtype=np.uint32)
        self.full_var = f.require_dataset('%s_full' % prefix, (1,),
                dtype=np.bool)
        self.write_ind = self.write_ind_var[0]
        self.full = self.full_var[0]

    def save(self):
        self.starts_var[:,:,:,:] = self.starts
        self.actions_var[:] = self.actions
        self.rewards_var[:] = self.rewards
        self.terminal_var[:] = self.terminal
        self.write_ind_var[0] = self.write_ind
        self.full_var[0] = self.full

    def count(self):
        if self.full:
            return self.logical_size
        else:
            return max(self.write_ind - 1, 0)

    def insert(self, start, action, reward, terminal):
        self.starts[self.write_ind] = start
        self.actions[self.write_ind] = action
        self.rewards[self.write_ind] = reward
        self.terminal[self.write_ind] = terminal
        self.write_ind += 1
        if self.write_ind == self.physical_size:
            self.full = True
            self.write_ind = 0

    def ignored_index(self):
        return (self.write_ind - 1) % self.physical_size

    def sample(self, n):
        selections = np.random.choice(self.count(), min(n, self.count()), replace=False)
        shifted_selections = [((i+1) % self.physical_size) if i >= self.ignored_index() else i for i in selections]
        end_selections = [(i+1) % self.physical_size for i in shifted_selections]
        return Transition(
                self.starts[shifted_selections],
                self.actions[shifted_selections],
                self.rewards[shifted_selections],
                self.terminal[shifted_selections],
                self.starts[end_selections])

blank_frames = [np.empty([FRAME_HEIGHT, FRAME_WIDTH], dtype=np.uint8)
        for i in range(FRAMES_PER_STATE - 1)]
for b in blank_frames:
    b.fill(0)
         
def reset_env(env):
    return blank_frames + [down_sample(env.reset())]

def stack_frames(frames):
    return np.stack(frames, axis=2)

def sgn(x):
    if x < 0:
        return -1
    if x > 0:
        return 1
    return 0

class Stepper(object):
    def __init__(self, game, frames_same, table=None):
        self.env = gym.make(game)
        # ARGH
        self.env.frameskip = 1
        self.frames = reset_env(self.env)
        self.last_state = stack_frames(self.frames)
        self.frames_same = frames_same
        self.transition_table = table

    def step(self, action=None, render=False):
        if action is None:
            action = self.env.action_space.sample()
        old_state = self.last_state
        total_reward = 0
        done = False
        for i in range(self.frames_same):
            frame, reward, done, info = self.env.step(action)
            if render:
                self.env.render()
            total_reward += sgn(reward)
            if done: 
                break
            self.frames.append(down_sample(frame))
        if done:
            self.frames = reset_env(self.env)
        else:
            self.frames[:-FRAMES_PER_STATE] = []
        self.last_state = stack_frames(self.frames)
        if self.transition_table is not None:
            self.transition_table.insert(old_state, action, total_reward, done)

class TrainingEnvironment(object):
    saves_dir = 'saves'
    epsilon_floor = .1
    epsilon_drop_over = 1024 * 1024
    epsilon = 1
    num_steppers = 16
    study_repeats = 8
    transitions_to_keep = 1024 * 1024
    frames_same = 4
    train_every = 32
    frames_start = 50 * 1024
    update_target_every = 10 * 1024
    total_training_steps = 10 * 1024 * 1024
    discount_rate = 0.96

    def __init__(self, game, save_name, swap_path, **kwargs):
        # you can set arbitrary hyperparameters
        for k, v in kwargs.items():
            if getattr(self, k, None) is None:
                raise ValueError('undefined param %s' % k)
            setattr(self, k, v)
        self.frames_per_training = self.train_every * self.num_steppers
        self.swap_file = h5py.File('%s/%s' % (swap_path, save_name))
        self.tables = [TransitionTable(
            self.swap_file, '%d_' % i, self.transitions_to_keep // self.num_steppers)
            for i in range(self.num_steppers)]
        self.steppers = [Stepper(game, self.frames_same, t) for t in self.tables]
        num_outputs = self.steppers[0].env.action_space.n
        self.session = tf.Session()
        with self.session:
            global_step = tf.Variable(0, name='global_step', trainable=False)
            self.qnetwork = QNetworkPair(self.session, self.discount_rate, num_outputs,
                    global_step)
        self.summary_writer = tf.train.SummaryWriter(
                '%s/%s_summaries' % (self.saves_dir, save_name))
        self.saver = tf.train.Saver()
        self.save_path = '%s/%s.ckpt' % (self.saves_dir, save_name)
        self.stop_file = '%s/%s.stop' % (self.saves_dir, save_name)
        self.save_file = '%s/%s.save' % (self.saves_dir, save_name)
        if os.path.isfile(self.save_path):
            self.saver.restore(self.session, self.save_path)
        else:
            self.session.run(tf.initialize_all_variables())
            self.qnetwork.update_targets()
        self.frames_seen = 0

    def sample_transitions(self):
        samples = [table.sample(self.study_repeats * self.train_every) for table in self.tables if
                table.count() > 0]
        return Transition(*[np.concatenate(arrays) for arrays in zip(*samples)])

    def train(self):
#        begin, action, reward, end
        transitions = self.sample_transitions()
        summaries, step = self.qnetwork.train(*transitions)
        frames_seen = step * self.frames_per_training
        self.summary_writer.add_summary(summaries, frames_seen)

        # Do some occasional stuff. Do it here because we know what step it is
        epsilon_rate = (1 - self.epsilon_floor) / self.epsilon_drop_over
        self.epsilon = max(1 - frames_seen * epsilon_rate, self.epsilon_floor)
        if step % (self.update_target_every // self.frames_per_training) == 0:
            self.qnetwork.update_targets()
        if step % 10000 == 0:
            self.save()
        self.frames_seen = frames_seen

    def save(self):
        self.saver.save(self.session, self.save_path)

    def save_tables(self):
        for t in self.tables:
            t.save()

    def step(self):
        paired_steppers = [(random.random(), s) for s in self.steppers]
        rand_steppers = [s for r, s in paired_steppers if r < self.epsilon]
        determ_steppers = [s for r, s in paired_steppers if r >= self.epsilon]
        for s in rand_steppers:
            s.step()
        if determ_steppers:
            states = [s.last_state for s in determ_steppers]
            actions = self.qnetwork.choose_actions(states)
            for s, a in zip(determ_steppers, actions):
                s.step(a)

    def run(self):
        total_saved = sum(t.count() for t in self.tables)
        saved_remaining = self.frames_start - total_saved
        for i, s in zip(range(saved_remaining), cycle(self.steppers)):
            s.step()
        print('stores primed')
        while self.frames_seen < self.total_training_steps:
            for i in range(self.train_every):
                self.step()
            if os.path.exists(self.stop_file):
                os.remove(self.stop_file)
                break
            self.train()
            if os.path.exists(self.save_file):
                os.remove(self.save_file)
                self.saver.save(self.session, self.save_path)
