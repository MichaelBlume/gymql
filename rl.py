from __future__ import print_function, division
import tensorflow as tf
import numpy as np
import gym
from collections import namedtuple
import random
import os

FRAMES_PER_STATE=4


def weight_variable(*shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def bias_variable(*shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

def ident_none(f):
    return (lambda x: x) if f is None else f

def conv_layer(x, window, out_channels, stride, nonlin = tf.nn.relu):
    nonlin = ident_none(nonlin)
    in_channels = x.get_shape()[3].value
    W_conv = weight_variable(window, window, in_channels, out_channels)
    b_conv = bias_variable(out_channels)
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

def fcl(x, size, nonlin = tf.nn.relu):
    nonlin = ident_none(nonlin)
    W = weight_variable(x.get_shape()[1].value, size)
    b = bias_variable(size)
    return nonlin(tf.matmul(x, W) + b)

def down_sample(s):
    return np.mean(s[::2,::2,:],axis=2).astype(np.uint8)

Transition = namedtuple('Transition', 'begin action reward end')

blank_frames = [np.empty([105, 80], dtype=np.uint8) for i in range(FRAMES_PER_STATE - 1)]
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
    def __init__(self, game, frames_same):
        self.env = gym.make(game)
        self.frames = reset_env(self.env)
        self.last_state = stack_frames(self.frames)
        self.frames_same = frames_same

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
        return Transition(old_state, action, total_reward, self.last_state)

class TrainingEnvironment(object):
    saves_dir = 'saves'
    epsilon = 1
    epsilon_drop = 0.01
    epsilon_drop_every_frames = 100 * 1000
    save_things = False
    num_steppers = 16
    study_repeats = 32
    transitions_to_keep = 1000 * 1000
    discount_rate = 0.95
    frames_same = 3

    def __init__(self, game, save_name, **kwargs):
        # you can set arbitrary hyperparameters
        for k, v in kwargs.items():
            if getattr(self, k, None) is None:
                raise ValueError('undefined param %s' % k)
            setattr(self, k, v)
        self.batch_size = self.num_steppers * self.study_repeats
        self.steppers = [Stepper(game, self.frames_same) for i in range(self.num_steppers)]
        self.num_outputs = self.steppers[0].env.action_space.n
        self.transitions = []
        self.session = tf.Session()
        with self.session:
            self.make_network()
        self.summary_writer = tf.train.SummaryWriter(
                '%s/%s_summaries' % (self.saves_dir, save_name))
        self.saver = tf.train.Saver()
        self.save_path = '%s/%s.ckpt' % (self.saves_dir, save_name)
        if self.save_things and os.path.isfile(self.save_path):
            self.saver.restore(self.session, self.save_path)
        else:
            self.session.run(tf.initialize_all_variables())

    def sample_transitions(self, transitions):
        n = min(len(transitions), self.batch_size)
        return random.sample(transitions, n)

    def make_network(self):
        inputs = tf.placeholder(tf.float32, [None, 105, 80, FRAMES_PER_STATE])
        h_conv1 = conv_layer(inputs, 8, 16, 4)
        h_conv2 = conv_layer(h_conv1, 4, 32, 2)
        h_conv2_flat = flatten(h_conv2)
        h_fc = fcl(h_conv2_flat, 256)
        Q_vals = fcl(h_fc, self.num_outputs, None)
        best_action = tf.argmax(Q_vals, dimension=1)
        actions_taken = tf.placeholder(tf.uint8, [None])

        # This is ugly
        actions_hot = tf.one_hot(actions_taken, self.num_outputs)
        Qs_taken = tf.reduce_sum(Q_vals * actions_hot, reduction_indices=1)

        Qs_observed = tf.placeholder(tf.float32, [None])
        mean_Q = tf.reduce_mean(Qs_observed)
        clipped_mean_Q = tf.maximum(-10.0, (tf.minimum(10.0, mean_Q)))
        tf.scalar_summary('average_Q', clipped_mean_Q)

        loss_squares = tf.square(Qs_observed - Qs_taken)
        loss = tf.reduce_sum(loss_squares)
        mean_loss = tf.reduce_mean(loss_squares)
        tf.scalar_summary('loss', tf.minimum(10.0, mean_loss))
        tf.scalar_summary('batch_size', tf.shape(Qs_observed)[0])
        global_step = tf.Variable(0, name='global_step', trainable=False)
        opt = tf.train.AdamOptimizer().minimize(loss, global_step=global_step)

        self.epsilon_var = tf.placeholder(tf.float32)
        tf.scalar_summary('epsilon', self.epsilon_var)

        self.inputs = inputs
        self.best_action = best_action
        self.actions_taken = actions_taken
        self.Qs_observed = Qs_observed
        self.Qs_taken = Qs_taken
        self.opt = opt
        self.global_step = global_step
        self.max_Q = tf.reduce_max(Q_vals, reduction_indices=1)
        self.summaries = tf.merge_all_summaries()

    def choose_actions(self, states):
        return self.session.run(self.best_action,
                {self.inputs: states})

    def train(self):
#        begin, action, reward, end
        transitions = self.sample_transitions(self.transitions)
        if not transitions:
            # Won't happen
            return
        final_Qs = self.session.run(self.max_Q,
                {self.inputs: [t.end for t in transitions]})
        total_Qs = [q * self.discount_rate + t.reward
                for q, t in zip(final_Qs, transitions)]
        _, summaries, step, predictedQ = self.session.run(
                [self.opt, self.summaries, self.global_step, self.Qs_taken],
                {self.inputs: [t.begin for t in transitions],
                 self.epsilon_var: self.epsilon,
                 self.actions_taken: [t.action for t in transitions],
                 self.Qs_observed: total_Qs})
#       print('final')
#       print(final_Qs[:10])
#       print('total')
#       print(total_Qs[:10])
#       print('predicted')
#       print(predictedQ[:10])
        self.summary_writer.add_summary(summaries, step)

        # Do some occasional stuff. Do it here because we know what step it is
        if step % (self.epsilon_drop_every_frames // self.num_steppers) == 0:
            self.epsilon -= self.epsilon_drop
        if self.save_things and step % 10000 == 0:
            self.saver.save(self.session, self.save_path)

    def step(self):
        paired_steppers = [(random.random(), s) for s in self.steppers]
        rand_steppers = [s for r, s in paired_steppers if r < self.epsilon]
        determ_steppers = [s for r, s in paired_steppers if r >= self.epsilon]
        for s in rand_steppers:
            self.transitions.append(s.step())
        if determ_steppers:
            states = [s.last_state for s in determ_steppers]
            actions = self.choose_actions(states)
            for s, a in zip(determ_steppers, actions):
                self.transitions.append(s.step(a))
        self.train()
        self.transitions[:-self.transitions_to_keep] = []

    def run(self, n):
        for i in range(n):
            self.step()
