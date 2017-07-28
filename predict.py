import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import model
import data
import utils
import argparse
from sklearn.metrics import fbeta_score

parser = argparse.ArgumentParser()
parser.add_argument("-model", type=str, default='PlanetSimpleNet', help="model name")
parser.add_argument("-batch_size", type=int, default=64, help="batch size")
parser.add_argument("-nocuda", action='store_true', help="no cuda used")
parser.add_argument("-nworkers", type=int, default=4, help="number of workers")
parser.add_argument("-outfile", type=str, default='pred.csv', help="output file")
args = parser.parse_args()

cuda = not args.nocuda and torch.cuda.is_available() # use cuda

# Rescale to 64x64 for simple net. Validation set used for fscore.
# 224x224 for pretrained models.
if args.model=='PlanetSimpleNet':
    test_transforms = transforms.Compose([transforms.Scale(64),
                                transforms.ToTensor()])
    val_transforms = transforms.Compose([transforms.Scale(64),
                                transforms.ToTensor()])
else:
    test_transforms = transforms.Compose([transforms.RandomCrop(224),
                            transforms.ToTensor()])
    val_transforms = transforms.Compose([transforms.Scale(224),
                            transforms.ToTensor()])

# Create dataloaders
testset = data.PlanetData('data/sample_submission_v2.csv', 'data/test-jpg',
                'data/labels.txt', test_transforms)
test_loader = DataLoader(testset, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.nworkers)
valset = data.PlanetData('data/val_set_norm.csv', 'data/train-jpg',
                'data/labels.txt', val_transforms)
val_loader = DataLoader(valset, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.nworkers)


def fscore(prediction):
    """ Get the fscore of the validation set. Gives a good indication
    of score on puclic leaderboard"""
    target = torch.FloatTensor(0, 17)
    for i, (_,y) in enumerate(val_loader):
        target = torch.cat((target, y), 0)
    fscore = fbeta_score(target.numpy(), prediction.numpy() > 0.23,
                beta=2, average='samples')
    return fscore

def predict(net, loader):
    net.eval()
    predictions = torch.FloatTensor(0, 17)
    for i, (X,_) in enumerate(loader):
        if cuda:
            X = X.cuda()
        X = Variable(X, volatile=True)
        output = net(X)
        predictions = torch.cat((predictions, output.cpu().data), 0)
    return predictions

if __name__ == '__main__':
    net = model.__dict__[args.model]()
    net.load_state_dict(torch.load('saved-models/{}.pth.tar'.format(args.model)))
    if cuda:
        net = net.cuda()

    # predict on the validation set to calculate fscore
    val_pred = predict(net, val_loader)
    print("fscore on validation set: {:.4f}".format(fscore(val_pred)))

    # predict on the test data where we don't know the labels
    pred = predict(net, test_loader)
    print(pred.size())
