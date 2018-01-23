
def get_number_of_parameters(model):
    """
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
    """
    parameters = model.parameters
    from functools import reduce
    from operator import add, mul
    from cntk import InferredDimension
    if any(any(dim == InferredDimension for dim in p.shape) for p in parameters):
        total_parameters = -1
    else:
        total_parameters = sum([reduce(mul, p.shape + (1,)) for p in parameters])

    return total_parameters, len(parameters)
