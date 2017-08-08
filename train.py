import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import fbeta_score
import model
import data
import utils
import time
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='PlanetSimpleNet', help="model: PlanetSimpleNet")
parser.add_argument("--patience", type=int, default=5, help="early stopping patience")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--nepochs", type=int, default=200, help="max epochs")
parser.add_argument("--nocuda", action='store_true', help="no cuda used")
parser.add_argument("-v", action='store_true', help="verbose")
parser.add_argument("--nworkers", type=int, default=4, help="number of workers")
parser.add_argument("--seed", type=int, default=1, help="random seed")
args = parser.parse_args()

cuda = not args.nocuda and torch.cuda.is_available() # use cuda
print('Training on cuda: {}'.format(cuda))

# Set seeds. If using numpy this must be seeded too.
torch.manual_seed(args.seed)
if cuda:
    torch.cuda.manual_seed(args.seed)

# Setup folders for saved models and logs
if not os.path.exists('saved-models/'):
    os.mkdir('saved-models/')
if not os.path.exists('logs/'):
    os.mkdir('logs/')

# Setup tensorboard folders. Each run must have it's own folder. Creates
# a logs folder for each model and each run.
out_dir = 'logs/{}'.format(args.model)
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
run = 0
current_dir = '{}/run-{}'.format(out_dir, run)
while os.path.exists(current_dir):
	run += 1
	current_dir = '{}/run-{}'.format(out_dir, run)
os.mkdir(current_dir)
logfile = open('{}/log.txt'.format(current_dir), 'w')
print(args, file=logfile)

# Tensorboard viz. tensorboard --logdir {yourlogdir}. Requires tensorflow.
from tensorboard_logger import configure, log_value
configure(current_dir, flush_secs=5)

# Get all model names
model_names = sorted(name for name in model.__dict__
    if name.startswith("Planet")
    and callable(model.__dict__[name]))

# Define transforms. Pretrained models expect input of at least 224x224.
# If using pretrained models this should also be added.
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    #  std=[0.229, 0.224, 0.225])
train_transforms = transforms.Compose([transforms.RandomCrop(224),
                        transforms.RandomHorizontalFlip(),
                        utils.RandomRotation(),
                        utils.RandomTranslation(),
                        utils.RandomVerticalFlip(),
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
    targets = torch.FloatTensor(0,17) # For fscore calculation
    predictions = torch.FloatTensor(0,17)
    for i, (X,y) in enumerate(loader):
        if cuda:
            X, y = X.cuda(), y.cuda()
        X, y = Variable(X, volatile=True), Variable(y)
        output = net(X)
        loss = criterion(output, y)
        acc = utils.get_multilabel_accuracy(output, y)
        targets = torch.cat((targets, y.cpu().data), 0)
        predictions = torch.cat((predictions,output.cpu().data), 0)
        running_loss += loss.data[0]
        running_accuracy += acc
    fscore = fbeta_score(targets.numpy(), predictions.numpy() > 0.23,
                beta=2, average='samples')
    return running_loss/len(loader), running_accuracy/len(loader), fscore


if __name__ == '__main__':
    net = model.__dict__[args.model]()
    criterion = torch.nn.BCELoss()

    if cuda:
        net, criterion = net.cuda(), criterion.cuda()
    # early stopping parameters
    patience = args.patience
    best_loss = 1e4

    # Print model to logfile
    print(net, file=logfile)

    # Change optimizer for finetuning
    if args.model=='PlanetSimpleNet':
        optimizer = optim.Adam(net.parameters())
    else:
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for e in range(args.nepochs):
        start = time.time()
        train_loss, train_acc = train(net, train_loader,
            criterion, optimizer, args.v)
        val_loss, val_acc, fscore = validate(net, val_loader, criterion)
        end = time.time()

        # print stats
        stats ="""Epoch: {}\t train loss: {:.3f}, train acc: {:.3f}\t
                val loss: {:.3f}, val acc: {:.3f}\t fscore: {:.3f}\t
                time: {:.1f}s""".format( e, train_loss, train_acc, val_loss,
                val_acc, fscore, end-start)
        print(stats)
        print(stats, file=logfile)
        log_value('train_loss', train_loss, e)
        log_value('val_loss', val_loss, e)
        log_value('fscore', fscore, e)

        #early stopping and save best model
        if val_loss < best_loss:
            best_loss = val_loss
            patience = args.patience
            utils.save_model({
                'arch': args.model,
                'state_dict': net.state_dict()
            }, 'saved-models/{}-run-{}.pth.tar'.format(args.model, run))
        else:
            patience -= 1
            if patience == 0:
                print('Run out of patience!')
                break
