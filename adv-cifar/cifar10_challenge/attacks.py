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

    def perturb(self, x_nat, y, sess):
        """Given a set of examples (x_nat, y), returns a set of adversarial
           examples within epsilon of x_nat in l_infinity norm."""
        if self.rand:
            x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)  # rewrite in tensorflow
            x = np.clip(x, 0, 255)  # ensure valid pixel range   # tf clip_by_value
        else:
            x = np.copy(x_nat)

        for i in range(self.num_steps):
            grad = sess.run(self.grad, feed_dict={self.model.x_input: x,
                                                  self.model.y_input: y})

            x = np.add(x, self.step_size * np.sign(grad), out=x, casting='unsafe')  # tf.math.add

            x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)  #
            x = np.clip(x, 0, 255)  # ensure valid pixel range

        # return x


    def opmaxmin(self, sess, net, gan, eps, im_size=784, embed_feats=256, num_images=50, z_lr=5e-3, lambda_lr=1e-4,
             num_steps=1000,
             batch_num=None, ind=None):
        softmax = ch.nn.Softmax()
        logsoftmax = ch.nn.LogSoftmax()

        BATCH_SIZE = 1

        batch1 = ch.zeros((num_images, 1, 28, 28)).cuda()
        batch2 = ch.zeros((num_images, 1, 28, 28)).cuda()
        is_valid = ch.zeros(num_images).cuda()
        count = 0
        EPS = eps
        for i in range(num_images // BATCH_SIZE):
            z1 = tf.Variable(tf.random.uniform([BATCH_SIZE, embed_feats]))
            z2 = tf.Variable(tf.random.uniform([BATCH_SIZE, embed_feats]))

            ones = tf.ones(z1.shape[0])

            lambda_ = tf.Variable(1e0 * tf.ones([z1.shape[0], 1]))

            total_loss = 0

            opt1 = tf.train.GradientDescentOptimizer(z_lr)
            opt2 = tf.train.GradientDescentOptimizer(lambda_lr)

            for j in range(num_steps):

                x1 = gan(z1)
                x2 = gan(z2)
                distance_mat = tf.norm(tf.reshape(x1 - x2, (x1.shape[0], -1)), axis=-1, keep_dims=False) - EPS * ones

                net_res1 = sess.run(net.predictions, feed_dict={self.model.x_input: x1})
                net_res2 = sess.run(net.predictions, feed_dict={self.model.x_input: x2})

                # print('Cross entropy:%f \t distance=%f \t lambda=%f'%(ce(cla(x1),cla(x2)),distance_mat,lambda_))

                is_adv = 1 - (net_res1 == net_res2).float()
                is_feasible = (distance_mat <= 0).float()
                not_valid = 1 - (is_adv * is_feasible)
                if ch.sum(is_adv * is_feasible) == BATCH_SIZE:
                    #                 ind = (ch.abs(net_res1 - net_res2)*is_valid*is_feasible_mat).argmax(0)
                    batch1[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, ...] = x1
                    batch2[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, ...] = x2
                    is_valid[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] = 1.
                    break

                # pre-softmax logits
                net_pre1 = sess.run(net.pre_softmax, feed_dict={self.model.x_input: x1})
                net_pre2 = sess.run(net.pre_softmax, feed_dict={self.model.x_input: x2})

                # calculate loss1, update opt1
                loss1 = 0
                opt1.minimize(loss1, var_list=[z1, z2])

                loss2 = -1. * tf.reduce_mean(lambda_ * distance_mat * (not_valid))
                opt2.minimize(loss2, var_list=[lambda_])

                # update opt2
            batch1[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, ...] = x1
            batch2[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, ...] = x2
            is_valid[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] = is_adv * is_feasible

        count = ch.sum(is_valid)
        print('number of adversarial pairs found:%d\n' % (count))

        return batch1.detach(), batch2.detach(), is_valid



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
