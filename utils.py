import torch.nn as nn

def get_conv_out(layer, input_size):
    w, h = input_size
    F = layer.kernel_size
    S = layer.stride
    P = layer.padding
    w2= (w-F[0]+2*P[0])/S[0]+1
    h2 =(h-F[1]+2*P[1])/S[1]+1
    return w2,h2

def get_pool_out(layer,input_size):
    w, h = input_size
    F = layer.kernel_size
    S = layer.stride
    P = layer.padding
    w2 = (w-F)/S+1
    h2 = (h-F)/S+1
    return w2,h2

def calculate_feature_size(model, input_size):
    for layer in model:
        if type(layer) == nn.Conv2d:
            input_size = get_conv_out(layer, input_size)
        elif type(layer) == nn.MaxPool2d:
            input_size = get_pool_out(layer, input_size)
    return input_size

def get_multilabel_accuracy(pred, target):
    """ Calculate multilabel accuracy.

        Turn prediction tensor in binary. Compare with target.
        Calculate common elements. To be used for calculating running
        accuracy and total accuracy in training.
    """
    # TODO: Check if it is actually correct
    pred = pred > 0.5
    r = (pred == target.byte())
    acc = r.float().cpu().sum().data[0]
    return acc/(pred.size()[1]*pred.size()[0])
