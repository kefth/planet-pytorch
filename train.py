import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
import model
import data
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-model", type=str, default='PlanetSimpleNet', help="model name")
parser.add_argument("-patience", type=int, default=5, help="early stopping patience")
parser.add_argument("-batch_size", type=int, default=16, help="batch size")
parser.add_argument("-nepochs", type=int, default=20, help="max epochs")
parser.add_argument("-nocuda", action='store_true', help="no cuda used")
args = parser.parse_args()

cuda = not args.nocuda and torch.cuda.is_available() # use cuda

# Get all model names
model_names = sorted(name for name in model.__dict__
    if name.startswith("Planet")
    and callable(model.__dict__[name]))

# Rescale to 64x64 for simple net. 224 otherwise.
if args.model=='PlanetSimpleNet':
    train_transforms = transforms.Compose([transforms.Scale(64),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor()])
    val_transforms = transforms.Compose([transforms.Scale(64),
                                transforms.ToTensor()])
else:
    train_transforms = transforms.Compose([transforms.RandomCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor()])
    val_transforms = transforms.Compose([transforms.Scale(224),
                            transforms.ToTensor()])

# Create dataloaders
trainset = data.PlanetData('data/train_set_norm.csv', 'data/train-jpg',
                'data/labels.txt', train_transforms)
train_loader = DataLoader(trainset, batch_size=args.batch_size,
                        shuffle=True, num_workers=2)
valset = data.PlanetData('data/val_set_norm.csv', 'data/train-jpg',
                'data/labels.txt', val_transforms)
val_loader = DataLoader(valset, batch_size=args.batch_size,
                        shuffle=False, num_workers=2)

def train(net, loader, criterion, optimizer):
    net.train()
    running_loss = 0
    running_accuracy = 0

    start = time.time()
    for i, (X,y) in enumerate(loader):
        if cuda:
            X, y = X.cuda(), y.cuda()
        X, y = Variable(X), Variable(y)

        output = net(X)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.data[0]
        output = torch.nn.functional.sigmoid(output)>0.5
        r = (output == y.byte())
        acc = r.float().cpu().sum().data[0]
        running_accuracy += acc/17
    loss, accuracy = running_loss/len(loader), running_accuracy/len(loader)
    return loss, accuracy


if __name__ == '__main__':
    net = model.__dict__[args.model]()
    optimiser = optim.Adam(net.parameters())
    criterion = torch.nn.MultiLabelSoftMarginLoss()
    loss, acc = train(net, train_loader, criterion, optimiser)
