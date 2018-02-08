from __future__ import print_function
import numpy as np
import cntk as ct
import os
from collections import OrderedDict

from _cntk_py import set_fixed_random_seed, force_deterministic_algorithms
from utils import *

# Ensure that we always get the same results
np.random.seed(1)
set_fixed_random_seed(1)
force_deterministic_algorithms()

class Main():
    '''
    Capsule Networks MNIST using CNTK v2.3.1
    '''

    # Define model dimensions
    input_dim_model = (1, 28, 28)    # images are 28 x 28 with 1 channel of color (gray)
    output_dim_model = (10,)
    perturbations_dim = (16,)

    input_dim = 28*28
    num_output_classes = 10
    reconstruction_model = None

    def train_and_test(self, reader_train, reader_test, reader_cv, restore_checkpoint=True):
        '''
        Train the model and validate the results

        Args:
            reader_train (:class:`~cntk.io.MinibatchSource`): the dataset reader for training.
            reader_test (:class:`~cntk.io.MinibatchSource`): the dataset reader for evaluation.
            restore_checkpoint (bool, optional): Continue training form latest checkpoint if True (default)

        Returns:
            None
        '''
        from CapsNet import CapsNet

        self.input = ct.input_variable(self.input_dim_model, name='MINST_Input')
        self.labels = ct.input_variable(self.output_dim_model, name='MINST_Labels')
        self.perturbations = ct.input_variable(self.perturbations_dim,  name='Perturbations')

        self.caps_net = CapsNet(self.input/255., self.labels, routings=3, use_reconstruction=True)

        # models
        self.training_model, self.digitcaps_model, self.prediction_model, self.reconstruction_model = self.caps_net.models()
        self.manipulation_model = self.caps_net.manipulation(self.perturbations)

        # loss & error
        loss, error = self.caps_net.criterion()

        # Number of parameters in the network
        # 5. Capsules on MNIST "... CapsNet has 8.2M parameters and 6.8M parameters without the reconstruction subnetwork."
        num_parameters, num_tensors = get_number_of_parameters(self.training_model)
        print("DigitCaps contains {} learneable parameters in {} parameter tensors.".format(num_parameters, num_tensors))

        # Initialize the parameters for the trainer
        minibatch_size = 128
        num_samples_per_sweep = 60000
        num_sweeps_to_train_with = 30

        # Report & Checkpoint frequency
        print_frequency = (4, ct.DataUnit.minibatch)
        checkpoint_frequency = (100, ct.DataUnit.minibatch)
        cross_validation_frequency = (1, ct.DataUnit.minibatch)

        tensorboard_logdir = './tensorboard'

        # Map the data streams to the input and labels.
        self.input_map = {
            self.labels : reader_train.streams.labels,
            self.input  : reader_train.streams.features
        }

        self.test_input_map = {
            self.labels : reader_test.streams.labels,
            self.input  : reader_test.streams.features
        }

        self.cv_input_map = {
            self.labels : reader_cv.streams.labels,
            self.input  : reader_cv.streams.features
        }

        # Instantiate progress writers.
        progress_writers = [ct.logging.ProgressPrinter(
            tag='Training',
            num_epochs=int(num_samples_per_sweep * num_sweeps_to_train_with / minibatch_size / print_frequency[0]))]

        training_progress_output_freq = 1

        if tensorboard_logdir is not None:
            self.tb_printer = ct.logging.TensorBoardProgressWriter(freq=training_progress_output_freq, log_dir=tensorboard_logdir, model=self.training_model)
            progress_writers.append(self.tb_printer)

        # Instantiate the learning rate schedule
        learning_rate_schedule = [0.01] * 30 + [0.007]
        learning_rate_schedule = ct.learning_parameter_schedule(learning_rate_schedule, minibatch_size=minibatch_size, epoch_size=num_samples_per_sweep)

        # Instantiate the trainer object to drive the model training
        learner = ct.adam(
            self.training_model.parameters,
            learning_rate_schedule,
            momentum=[0.9],
            variance_momentum=[0.999],
            gaussian_noise_injection_std_dev=[0.0]
        )
        trainer = ct.Trainer(self.training_model, (loss, error), [learner], progress_writers)

        ct.training_session(
            trainer=trainer,
            mb_source=reader_train,
            mb_size=minibatch_size,
            model_inputs_to_streams=self.input_map,
            max_samples=num_samples_per_sweep * num_sweeps_to_train_with,
            progress_frequency=print_frequency,
            checkpoint_config=ct.CheckpointConfig(
                filename='./checkpoints/checkpoint',
                frequency=checkpoint_frequency,
                restore=restore_checkpoint
            ),
            cv_config=ct.CrossValidationConfig(
                minibatch_size=128,
                minibatch_source=reader_cv,
                frequency=cross_validation_frequency,
                callback=self.cross_validation_callbackfunc,
                max_samples=1024,
                model_inputs_to_streams=self.cv_input_map
            ),
            test_config=ct.TestConfig(
                minibatch_source=reader_test,
                minibatch_size=minibatch_size,
                model_inputs_to_streams=self.test_input_map
            )
        ).train()

        # save models
        self.digitcaps_model.save('./models/digitcaps_model.cntk')
        self.training_model.save('./models/training_model.cntk')
        self.prediction_model.save('./models/prediction_model.cntk')
        if self.reconstruction_model:
            self.reconstruction_model.save('./models/reconstruction_model.cntk')
            self.manipulation_model.save('./models/manipulation_model.cntk')

        print('Done.')

    def cross_validation_callbackfunc(self, index, average_error, cv_num_samples, cv_num_minibatches):

        # Use a 5x5 matrix of images
        minibatch = self.reader_cv.next_minibatch(25)
        source_images = get_stream_by_shape(minibatch, (1, 784)).data.asarray()
        source_labels = get_stream_by_shape(minibatch, (1, 10)).data.asarray()
        decoded_images = self.reconstruction_model.eval({ self.input: np.reshape(source_images, (-1, 1, 28, 28)) })

        # Reconstruction network
        source_img = image_grid(source_images)
        decoded_img = image_grid(decoded_images * 255)

        # the input_variable is required by the write_image c++ implementation
        img_shape = ct.input_variable(shape=(1, 140, 140), dtype=np.float32)
        self.tb_printer.write_image('reconstruction', { img_shape : decoded_img }, index)
        self.tb_printer.write_image('original', { img_shape : source_img }, index)

        # Confidence graphs
        softmax = self.caps_net.predict_class.eval({ self.input: np.reshape(source_images, (-1, 1, 28, 28))  })
        ma1 = np.max(softmax, axis=1)
        std = np.std(ma1)
        mean = np.mean(ma1)

        self.tb_printer.write_value('confidence/mean', mean, index)
        self.tb_printer.write_value('confidence/std', std, index)

        self.tb_printer.flush()
        return True

    def capsule_network(self, data_dir):

        train_file = os.path.join(data_dir, "Train-28x28_cntk_text.txt")
        test_file = os.path.join(data_dir, "Test-28x28_cntk_text.txt")
        self.reader_train = create_reader(train_file, True, self.input_dim, self.num_output_classes)
        self.reader_cv = create_reader(test_file, True, self.input_dim, self.num_output_classes)
        self.reader_test = create_reader(test_file, False, self.input_dim, self.num_output_classes)

        return self.train_and_test(self.reader_train, self.reader_test, self.reader_cv)

if __name__ == '__main__':

    data_dir = os.path.join("data", "MNIST")
    Main().capsule_network(data_dir)
