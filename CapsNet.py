from __future__ import print_function
import numpy as np
import cntk as ct

from CapsLayer import PrimaryCaps, DigitCaps, Length

class CapsNet(object):
    '''
    Capsule Net Architecture (Sara Sabour, et al. 2017, "Dynamic Routing Between Capsules")
    '''

    def model(self, features):
        """
        CapsNet Architecture model generation
        """

        # Layer1: ..."The architecture is shallow with only two convolutional layers and one fully
        # connected layer. Conv1 has 256, 9 × 9 convolution kernels with a stride of 1 and ReLU
        # activation. This layer converts pixel intensities to the activities of local feature
        # detectors that are then used as inputs to the primary capsules."
        conv1 = ct.layers.Convolution2D(num_filters=256, filter_shape=9, strides=1,
                                        pad=False, activation=ct.relu, name='Conv1')(features)

        # Layer2: ... "The second layer (PrimaryCapsules) is a convolutional capsule layer with 32
        # channels of convolutional 8D capsules (i.e. each primary capsule contains 8
        # convolutional units with a 9 × 9 kernel and a stride of 2).
        primary = PrimaryCaps(num_capsules=32, dim_out_vector=8, filter_shape=9, strides=2, pad=False,
                              name='PrimaryCaps')(conv1)

        # Layer 3: .. "The final Layer (DigitCaps) has one 16D capsule per digit class and each of
        # these capsules receives input from all the capsules in the layer below."
        self.digitcaps = DigitCaps(primary, num_capsules=10, dim_out_vector=16, routings=3, name='DigitCaps')

        self.length = Length()(self.digitcaps)

        return self.digitcaps, self.length

    def criterion(self, labels):

        # hyperparameters
        lambda_val = 0.5

        # Margin loss
        left = ct.square(ct.relu(0.9 - self.length))
        right = ct.square(ct.relu(self.length - 0.1))
        left = ct.reshape(left, (-1))
        right = ct.reshape(right, (-1))
        lc = labels * left + lambda_val * (1 - labels) * right

        margin_loss = ct.reduce_sum(lc, axis=0)
        margin_loss = ct.reduce_mean(margin_loss, axis=ct.axis.Axis.default_batch_axis())

        # classification_error
        predict = ct.softmax(self.length, axis=0)
        error = ct.classification_error(ct.reshape(predict, (10)), labels)

        return margin_loss, error
