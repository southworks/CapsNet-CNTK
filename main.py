from __future__ import print_function
import numpy as np
import cntk as ct
import os

from _cntk_py import set_fixed_random_seed, force_deterministic_algorithms
from utils import get_number_of_parameters

# Ensure that we always get the same results
np.random.seed(1)
set_fixed_random_seed(1)
force_deterministic_algorithms()

# Define model dimensions
input_dim_model = (1, 28, 28)    # images are 28 x 28 with 1 channel of color (gray)
output_dim_model = (10)
input_dim = 28*28
num_output_classes = 10

# Read a CTF formatted text using the CTF deserializer from a file
def create_reader(path, is_training, input_dim, num_label_classes):

    ctf = ct.io.CTFDeserializer(path, ct.io.StreamDefs(
          labels=ct.io.StreamDef(field='labels', shape=num_label_classes, is_sparse=False),
          features=ct.io.StreamDef(field='features', shape=input_dim, is_sparse=False)))

    return ct.io.MinibatchSource(ctf,
        randomize = is_training, max_sweeps = ct.io.INFINITELY_REPEAT if is_training else 1)

def train_and_test(reader_train, reader_test, reader_cv, restore_checkpoint=True):
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

    caps_net = CapsNet()
    input = ct.input_variable(input_dim_model, name='MINST_Image_Input')
    digitcaps, length = caps_net.model(input/255.)

    # ct.logging.graph.plot(length, 'images/CapsNetArch.png')

    # evaluation
    predict = ct.softmax(length, axis=0)
    y_hay = ct.argmax(predict, axis=0)

    # loss & error
    labels = ct.input_variable(output_dim_model)
    loss, error = caps_net.criterion(labels)

    # Number of parameters in the network
    num_parameters, num_tensors = get_number_of_parameters(digitcaps)
    print("DigitCaps contains {} learneable parameters in {} parameter tensors.".format(num_parameters, num_tensors))

    # Initialize the parameters for the trainer
    minibatch_size = 128
    num_samples_per_sweep = 60000
    num_sweeps_to_train_with = 30

    # Report & Checkpoint frequency
    print_frequency = (1, ct.DataUnit.minibatch)
    checkpoint_frequency = (100, ct.DataUnit.minibatch)
    cross_validation_frequency = (20, ct.DataUnit.minibatch)

    tensorboard_logdir = './tensorboard'

    # Map the data streams to the input and labels.
    input_map = {
        labels : reader_train.streams.labels,
        input  : reader_train.streams.features
    }

    test_input_map = {
        labels : reader_test.streams.labels,
        input  : reader_test.streams.features
    }

    # Instantiate progress writers.
    progress_writers = [ct.logging.ProgressPrinter(
        tag='Training',
        num_epochs=int(num_samples_per_sweep * num_sweeps_to_train_with / minibatch_size))]

    training_progress_output_freq = 1

    if tensorboard_logdir is not None:
        progress_writers.append(ct.logging.TensorBoardProgressWriter(freq=training_progress_output_freq, log_dir=tensorboard_logdir, model=digitcaps))

    # Instantiate the learning rate schedule
    learning_rate_schedule = [0.01] * 5 + [0.003] * 5 + [0.001]
    learning_rate_schedule = ct.learning_parameter_schedule(learning_rate_schedule, minibatch_size=minibatch_size, epoch_size=num_samples_per_sweep)

    # Instantiate the trainer object to drive the model training
    learner = ct.adam(
        digitcaps.parameters,
        learning_rate_schedule,
        momentum=[0.9],
        variance_momentum=[0.999],
        gaussian_noise_injection_std_dev=[0.0]
    )
    trainer = ct.Trainer(digitcaps, (loss, error), [learner], progress_writers)

    ct.training_session(
        trainer=trainer,
        mb_source=reader_train,
        mb_size=minibatch_size,
        model_inputs_to_streams=input_map,
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
            max_samples=256,
            model_inputs_to_streams=test_input_map
        ),
        test_config=ct.TestConfig(
            minibatch_source=reader_test,
            minibatch_size=minibatch_size,
            model_inputs_to_streams=test_input_map
        )
    ).train()

def capsule_network(data_dir):

    train_file = os.path.join(data_dir, "Train-28x28_cntk_text.txt")
    test_file = os.path.join(data_dir, "Test-28x28_cntk_text.txt")
    reader_train = create_reader(train_file, True, input_dim, num_output_classes)
    reader_cv = create_reader(test_file, True, input_dim, num_output_classes)
    reader_test = create_reader(test_file, False, input_dim, num_output_classes)

    return train_and_test(reader_train, reader_test, reader_cv)

if __name__ == '__main__':

    data_dir = os.path.join("data", "MNIST")
    capsule_network(data_dir)
