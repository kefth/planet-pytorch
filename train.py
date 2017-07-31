import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
import model
import data
import utils
import time
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-model", type=str, default='PlanetSimpleNet', help="model name")
parser.add_argument("-patience", type=int, default=5, help="early stopping patience")
parser.add_argument("-batch_size", type=int, default=64, help="batch size")
parser.add_argument("-nepochs", type=int, default=20, help="max epochs")
parser.add_argument("-nocuda", action='store_true', help="no cuda used")
parser.add_argument("-v", action='store_true', help="verbose")
parser.add_argument("-nworkers", type=int, default=4, help="number of workers")
parser.add_argument("-seed", type=int, default=1, help="random seed(def:1)")
args = parser.parse_args()

cuda = not args.nocuda and torch.cuda.is_available() # use cuda

# Set seeds. If using numpy this must be seeded too.
torch.manual_seed(args.seed)
if cuda:
    torch.cuda.manual_seed(args.seed)

# Setup tensorboard folders. Each run must have it's own folder. It creates
# a logs folder for each model rather than each run of a model. Demonstration purposes.
out_dir = 'logs/{}'.format(args.model)
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
    os.mkdir('{}/train'.format(out_dir))
    os.mkdir('{}/val'.format(out_dir))
logfile = open('{}/log.txt'.format(out_dir), 'w')
print(args, file=logfile)

# Tensorboard viz. tensorboard --logdir {yourlogdir}. Requires tensorflow.
# from tensorboard_logger import configure, log_value
# configure(out_dir, flush_secs=5)
import tensorboard_logger as tl
train_viz = tl.Logger('{}/train'.format(out_dir), flush_secs=5)
val_viz = tl.Logger('{}/val'.format(out_dir), flush_secs=5)

# Setup folders for saved models
if not os.path.exists('saved-models/'):
    os.mkdir('saved-models/')

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

# Create dataloaders. Use pin memory if cuda.
kwargs = {'pin_memory': True} if cuda else {}
trainset = data.PlanetData('data/train_set_norm.csv', 'data/train-jpg',
                'data/labels.txt', train_transforms)
train_loader = DataLoader(trainset, batch_size=args.batch_size,
                        shuffle=True, num_workers=args.nworkers, **kwargs)
valset = data.PlanetData('data/val_set_norm.csv', 'data/train-jpg',
                'data/labels.txt', val_transforms)
val_loader = DataLoader(valset, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.nworkers, **kwargs)

def train(net, loader, criterion, optimizer, verbose = False):
    net.train()
    running_loss = 0
    running_accuracy = 0

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
        acc = utils.get_multilabel_accuracy(output, y)
        running_accuracy += acc
        if i%400 == 0 and verbose:
            pct = float(i+1)/len(loader)
            curr_loss = running_loss/(i+1)
            curr_acc = running_accuracy/(i+1)
            print('Complete: {:.2f}, Loss: {:.2f}, Accuracy: {:.4f}'.format(pct*100,
                        curr_loss, curr_acc))
    return running_loss/len(loader), running_accuracy/len(loader)

def validate(net, loader, criterion):
    net.eval()
    running_loss = 0
    running_accuracy = 0

    for i, (X,y) in enumerate(loader):
        if cuda:
            X, y = X.cuda(), y.cuda()
        X, y = Variable(X, volatile=True), Variable(y)

        output = net(X)
        loss = criterion(output, y)
        acc = utils.get_multilabel_accuracy(output, y)
        running_loss += loss.data[0]
        running_accuracy += acc
    return running_loss/len(loader), running_accuracy/len(loader)


if __name__ == '__main__':
    net = model.__dict__[args.model]()
    optimizer = optim.Adam(net.parameters())
    criterion = torch.nn.BCELoss()
    if cuda:
        net, criterion = net.cuda(), criterion.cuda()
    # early stopping parameters
    patience = args.patience
    best_loss = 1e4

    # Print model to logfile
    print(net, file=logfile)

    for e in range(args.nepochs):
        start = time.time()
        train_loss, train_acc = train(net, train_loader,
            criterion, optimizer, args.v)
        val_loss, val_acc = validate(net, val_loader, criterion)
        end = time.time()

        # print stats
        stats ="""Epoch: {}\t train loss: {:.3f}, train acc: {:.3f}\t
                val loss: {:.3f}, val acc: {:.3f}\t time: {:.1f}s""".format(
                e, train_loss, train_acc, val_loss, val_acc, end-start
                )
        print(stats)
        print(stats, file=logfile)
        train_viz.log_value('loss', train_loss, e)#same name to apper on single graph.
        val_viz.log_value('loss', val_loss, e)

        #early stopping and save best model
        if val_loss < best_loss:
            best_loss = val_loss
            patience = args.patience
            torch.save(net.state_dict(), 'saved-models/{}.pth.tar'.format(args.model))
        else:
            patience -= 1
            if patience == 0:
                print('Run out of patience!')
                break
