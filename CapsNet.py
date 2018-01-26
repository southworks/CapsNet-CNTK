from __future__ import print_function
import numpy as np
import cntk as ct

from CapsLayer import PrimaryCaps, DigitCaps, Length, Masking

class CapsNet(object):
    '''
    Capsule Net Architecture (Sara Sabour, et al. 2017, "Dynamic Routing Between Capsules")
    '''

    def __init__(self, features, labels, use_reconstruction=True):
        self.features = features
        self.labels = labels
        self.use_reconstruction = use_reconstruction

    def model(self):
        """
        CapsNet Architecture model generation
        """

        # Layer1: ..."The architecture is shallow with only two convolutional layers and one fully
        # connected layer. Conv1 has 256, 9 × 9 convolution kernels with a stride of 1 and ReLU
        # activation. This layer converts pixel intensities to the activities of local feature
        # detectors that are then used as inputs to the primary capsules."
        conv1 = ct.layers.Convolution2D(num_filters=256, filter_shape=9, strides=1,
                                        pad=False, activation=ct.relu, name='Conv1')(self.features)

        # Layer2: ... "The second layer (PrimaryCapsules) is a convolutional capsule layer with 32
        # channels of convolutional 8D capsules (i.e. each primary capsule contains 8
        # convolutional units with a 9 × 9 kernel and a stride of 2).
        primary = PrimaryCaps(num_capsules=32, dim_out_vector=8, filter_shape=9, strides=2, pad=False,
                              name='PrimaryCaps')(conv1)

        # Layer 3: .. "The final Layer (DigitCaps) has one 16D capsule per digit class and each of
        # these capsules receives input from all the capsules in the layer below."
        self.digitcaps = DigitCaps(primary, num_capsules=10, dim_out_vector=16, routings=3, name='DigitCaps')

        self.length = Length()(self.digitcaps)

        # Model the reconstruction Layer
        self.encoder = None
        if self.use_reconstruction:
            # Ouput shape: [#][160]
            self.masking = Masking()(self.digitcaps, self.labels)

            fc1 = ct.layers.Dense(512, activation=ct.relu)(self.masking)
            fc2 = ct.layers.Dense(1024, activation=ct.relu)(fc1)
            self.encoder = ct.layers.Dense(784, activation=ct.sigmoid)(fc2)

        return self.digitcaps, self.length, self.encoder

    def criterion(self):

        # hyperparameters
        lambda_val = 0.5

        # Margin loss
        left = ct.square(ct.relu(0.9 - self.length))
        right = ct.square(ct.relu(self.length - 0.1))
        left = ct.reshape(left, (-1))
        right = ct.reshape(right, (-1))
        lc = self.labels * left + lambda_val * (1 - self.labels) * right

        margin_loss = ct.reduce_sum(lc, axis=0)
        margin_loss = ct.reduce_mean(margin_loss, axis=ct.axis.Axis.default_batch_axis())

        # classification_error
        predict = ct.softmax(self.length, axis=0)
        error = ct.classification_error(ct.reshape(predict, (10)), self.labels)

        total_loss = margin_loss
        reconstruction_err = 0

        if self.use_reconstruction:
            features = ct.reshape(self.features, shape=(-1,))
            encoder = ct.reshape(self.encoder, shape=(-1,))
            squared = ct.square(encoder - features)
            reconstruction_err = ct.reduce_mean(squared, axis=0)
            reconstruction_err = ct.reduce_mean(reconstruction_err, axis=ct.axis.Axis.default_batch_axis())
            total_loss = margin_loss + (0.0005*784) * reconstruction_err

        return total_loss, error
