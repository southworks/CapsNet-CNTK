# Export network as json
from __future__ import print_function
import numpy as np
import cntk as ct
import os

from utils import image_grid, create_reader, save_image

class Export():
    '''
    Expor Capsule Networks MNIST as
    '''

    # Define model dimensions
    input_dim_model = (1, 28, 28)    # images are 28 x 28 with 1 channel of color (gray)
    output_dim_model = (10,)
    perturbations_dim = (16,)

    input_dim = 28*28
    num_output_classes = 10
    reconstruction_model = None

    def load_models(self):
        '''
        Load all models
        '''
        from CapsNet import CapsNet

        self.input = ct.input_variable(self.input_dim_model, name='MINST_Input')
        self.labels = ct.input_variable(self.output_dim_model, name='MINST_Labels')
        self.perturbations = ct.input_variable(self.perturbations_dim,  name='Perturbations')

        self.caps_net = CapsNet(self.input/255., self.labels, routings=3, use_reconstruction=True)

        # models
        self.training_model, self.prediction_model, self.reconstruction_model = self.caps_net.models()
        self.manipulation_model = self.caps_net.manipulation(self.perturbations)

        # load models
        self.training_model = self.training_model.load('./models/training_model.cntk')
        self.prediction_model = self.prediction_model.load('./models/prediction_model.cntk')
        self.manipulation_model = self.manipulation_model.load('./models/manipulation_model.cntk')

        print('Models loaded.')

    def apply_perturbations_img(self, source_image):
        images = []
        for dim in range(16):
            for value in [-0.25, -0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25]:
                perturbations = np.zeros((1, 16), dtype=np.float32)
                perturbations[0, dim] = value
                manipulated_img = self.manipulation_model.eval({
                    self.manipulation_model.arguments[0]: np.reshape(source_image, (-1, 1, 28, 28)),
                    self.manipulation_model.arguments[1]: perturbations
                })
                images.append(np.reshape(manipulated_img, (1, 28, 28)))

        return images

    def apply_perturbations(self, count, reader):

        minibatch = reader.next_minibatch(count)
        input, labels = minibatch.items()
        source_images = input[1].data.asarray()
        source_labels = labels[1].data.asarray()

        for ix, i in enumerate(range(count)):
            img = source_images[i]
            images = self.apply_perturbations_img(img)
            manipulated_grid = np.reshape(image_grid(images, cols=11, rows=16), (448, 308))
            save_image('./images/manipulated/imgix_' + str(ix) + '.png', manipulated_grid * 255.)

        return None

    def capsule_network(self, count, data_dir):

        test_file = os.path.join(data_dir, "Test-28x28_cntk_text.txt")
        self.reader_test = create_reader(test_file, False, self.input_dim, self.num_output_classes)

        # load models
        self.load_models()

        # save papper perturbations as images, one image per digit with 16 * 12 perturbations.
        self.apply_perturbations(count, self.reader_test)

        # TODO: export json with:
        #  - reconstruction network weights
        #  - mini testset with all digitcaps outputs

        return None

if __name__ == '__main__':

    data_dir = os.path.join("data", "MNIST")
    Export().capsule_network(100, data_dir)
