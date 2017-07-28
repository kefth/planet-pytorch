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
        # elif type(layer) == nn.Sequential:
        #     input_size = calculate_feature_size(layer, input_size)
    return input_size

class PlanetSimpleNet(nn.Module):
    """Simple convnet."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 17),
        )

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), 256 * 6 * 6)
            x = self.classifier(x)
            return x

if __name__ == '__main__':
    net = PlanetSimpleNet()
    size = calculate_feature_size(net.features, (224, 224))
    print(size)
