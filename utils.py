import numpy as np
import cntk as ct
import cv2

def create_reader(path, randomize, input_dim, num_label_classes):
    '''
    Read a CTF formatted text using the CTF deserializer from a file
    '''
    ctf = ct.io.CTFDeserializer(path, ct.io.StreamDefs(
        labels=ct.io.StreamDef(field='labels', shape=num_label_classes, is_sparse=False),
        features=ct.io.StreamDef(field='features', shape=input_dim, is_sparse=False)))

    return ct.io.MinibatchSource(ctf,
        randomize = randomize, max_sweeps = ct.io.INFINITELY_REPEAT if randomize else 1)

def get_number_of_parameters(model):
    '''
    Computes total number of parameters and tensors.

    Returns:
        Tuple:
        The total number of parameters, and the total number of tensors.

    Example:
        import cntk as ct
        x = ct.input_variable(3)
        f = ct.layers.Dense(5, activation=ct.relu)
        print("Model contains {} learneable parameters in {} parameter tensors.".format(parameters, tensors))
        Training 20 parameters in 2 parameter tensors.
    '''
    parameters = model.parameters
    from functools import reduce
    from operator import add, mul
    from cntk import InferredDimension
    if any(any(dim == InferredDimension for dim in p.shape) for p in parameters):
        total_parameters = -1
    else:
        total_parameters = sum([reduce(mul, p.shape + (1,)) for p in parameters])

    return total_parameters, len(parameters)

def image_grid(images, cols=5, rows=5, img_width=28, img_height=28, channels=1):
    '''
    Args:
        images: a vector of cols * rows images of the corresponding img_width and img_height dimensions of channels channels.
    '''
    # write reconstruction back as image
    out_img = np.array([], dtype=np.float32).reshape(0, img_width * cols)

    # make a grid with all images
    for i in range(rows):
        row = np.array([], dtype=np.float32).reshape(img_height, 0)
        for j in range(cols):
            img = images[i*cols+j]
            img = np.reshape(img, (img_height, img_width))
            row = np.concatenate([row, img], axis=1)
        out_img = np.concatenate([out_img, row], axis=0)

    # Reshape
    out_img = np.reshape(out_img, (1, channels, img_height * rows, img_width * cols))
    return out_img

def save_image(filename, image, is_float=True, is_bn=True):
        # convert to integer
        if is_float:
            image = np.uint8(np.minimum(np.maximum(image, 0), 255))

        if is_bn:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        r = cv2.imwrite(filename, image)

def get_stream_by_shape(minibatch, shape):

    for item in minibatch.items():
        if item[1].data.shape[1:] == shape:
            return item[1]

    return None

