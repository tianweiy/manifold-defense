"""
Implementation of attack methods. Running this file as a program will
apply the attack to the model specified by the config file and store
the examples in an .npy file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import torch as ch


def ce(y_, y):
    return tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))


class L2OPAttack:
    def __init__(self, generator, model, epsilon, num_steps, step_size, random_start, loss_func):
        """overpowered attack using max-min optimization"""
        self.model = model
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.rand = random_start
        self.gen = generator

        if loss_func == 'xent':
            loss = model.xent
        elif loss_func == 'cw':
            label_mask = tf.one_hot(model.y_input,
                                    10,
                                    on_value=1.0,
                                    off_value=0.0,
                                    dtype=tf.float32)
            correct_logit = tf.reduce_sum(label_mask * model.pre_softmax, axis=1)
            wrong_logit = tf.reduce_max((1 - label_mask) * model.pre_softmax - 1e4 * label_mask, axis=1)
            loss = -tf.nn.relu(correct_logit - wrong_logit + 50)
        else:
            print('Unknown loss function. Defaulting to cross-entropy')
            loss = model.xent

        self.grad = tf.gradients(loss, model.x_input)[0]

    def perturb(self, sess, net, gan, eps, im_size=784, embed_feats=256, num_images=50, z_lr=5e-3, lambda_lr=1e-4,
             num_steps=1000,
             batch_num=None, ind=None):

        BATCH_SIZE = 1

        batch1 = tf.zeros((num_images, 1, 28, 28))
        batch2 = tf.zeros((num_images, 1, 28, 28))
        is_valid = tf.zeros(num_images)
        EPS = eps
        for i in range(num_images // BATCH_SIZE):
            z1 = tf.Variable(tf.random.uniform([BATCH_SIZE, embed_feats]))
            z2 = tf.Variable(tf.random.uniform([BATCH_SIZE, embed_feats]))

            ones = tf.ones(z1.shape[0])

            lambda_ = tf.Variable(1e0 * tf.ones([z1.shape[0], 1]))

            opt1 = tf.train.GradientDescentOptimizer(z_lr)
            opt2 = tf.train.GradientDescentOptimizer(lambda_lr)

            for j in range(num_steps):
                x1 = gan(z1)
                x2 = gan(z2)
                distance_mat = tf.norm(tf.reshape(x1 - x2, (x1.shape[0], -1)), axis=-1, keep_dims=False) - EPS * ones

                net_res1 = sess.run(net.predictions, feed_dict={self.model.x_input: x1})
                net_res2 = sess.run(net.predictions, feed_dict={self.model.x_input: x2})

                is_adv = tf.cast(1 - (net_res1 == net_res2), tf.float32)
                is_feasible = tf.cast((distance_mat <= 0), tf.float32)
                not_valid = tf.cast(1 - (is_adv * is_feasible), tf.float32)
                if ch.sum(is_adv * is_feasible) == BATCH_SIZE:
                    batch1[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, ...] = x1
                    batch2[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, ...] = x2
                    is_valid[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] = 1.
                    break

                # pre-softmax logits
                net_pre1 = sess.run(net.pre_softmax, feed_dict={self.model.x_input: x1})
                net_pre2 = sess.run(net.pre_softmax, feed_dict={self.model.x_input: x2})

                # calculate loss1, update opt1
                loss1 = (-1.* tf.reduce_sum(ce(net_pre1, net_pre2), reduction_indices=None)*not_valid) +\
                        tf.reduce_sum(lambda_ * distance_mat*not_valid) + 1e-4*tf.reduce_sum(tf.norm(z1, axis=-1)*not_valid) +\
                        1e-4*tf.reduce_sum(tf.norm(z2,axis=-1)*not_valid)/ch.sum(not_valid)

                opt1.minimize(loss1, var_list=[z1, z2])

                loss2 = -1. * tf.reduce_mean(lambda_ * distance_mat * not_valid)
                opt2.minimize(loss2, var_list=[lambda_])

            batch1[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, ...] = x1
            batch2[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, ...] = x2
            is_valid[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] = is_adv * is_feasible

        count = tf.reduce_sum(is_valid)
        print('number of adversarial pairs found:%d\n' % (count))

        return batch1, batch2, is_valid


if __name__ == '__main__':
    import json
    import sys
    import math

    from model import Model

    with open('config.json') as config_file:
        config = json.load(config_file)

    model_file = tf.train.latest_checkpoint(config['model_dir'])
    if model_file is None:
        print('No model found')
        sys.exit()

    model = Model(mode='eval')
    attack = LinfPGDAttack(model,
                           config['epsilon'],
                           config['num_steps'],
                           config['step_size'],
                           config['random_start'],
                           config['loss_func'])
    saver = tf.train.Saver()

    data_path = config['data_path']
    cifar = cifar10_input.CIFAR10Data(data_path)

    with tf.Session() as sess:
        # Restore the checkpoint
        saver.restore(sess, model_file)

        # Iterate over the samples batch-by-batch
        num_eval_examples = config['num_eval_examples']
        eval_batch_size = config['eval_batch_size']
        num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

        x_adv = []  # adv accumulator

        print('Iterating over {} batches'.format(num_batches))

        for ibatch in range(num_batches):
            bstart = ibatch * eval_batch_size
            bend = min(bstart + eval_batch_size, num_eval_examples)
            print('batch size: {}'.format(bend - bstart))

            x_batch = cifar.eval_data.xs[bstart:bend, :]
            y_batch = cifar.eval_data.ys[bstart:bend]

            x_batch_adv = attack.perturb(x_batch, y_batch, sess)

            x_adv.append(x_batch_adv)

        print('Storing examples')
        path = config['store_adv_path']
        x_adv = np.concatenate(x_adv, axis=0)
        np.save(path, x_adv)
        print('Examples stored in {}'.format(path))
