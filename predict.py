import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import model
import data
import utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-model", type=str, default='PlanetSimpleNet', help="model name")
parser.add_argument("-batch_size", type=int, default=64, help="batch size")
parser.add_argument("-nocuda", action='store_true', help="no cuda used")
parser.add_argument("-nworkers", type=int, default=4, help="number of workers")
parser.add_argument("-outfile", type=str, default='pred.csv', help="output file")
args = parser.parse_args()

cuda = not args.nocuda and torch.cuda.is_available() # use cuda

# Rescale to 64x64 for simple net. Validation set used for thresholds.

# Rescale to 64x64 for simple net. 224 otherwise.
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

# Optimize thersholds
def optimise_f2_thresholds(y, p, verbose=True, resolution=100):
  def mf(x):
    p2 = torch.zeros_like(p)
    for i in range(17):
      p2[:, i] = (p[:, i] > x[i]).astype(np.int)
    score = fbeta_score(y, p2, beta=2, average='samples')
    return score

  x = [0.2]*17
  for i in range(17):
    best_i2 = 0
    best_score = 0
    for i2 in range(resolution):
      i2 /= resolution
      x[i] = i2
      score = mf(x)
      if score > best_score:
        best_i2 = i2
        best_score = score
    x[i] = best_i2
    if verbose:
      print(i, best_i2, best_score)

  return x

def predict(net, loader):
    net.eval()
    predictions = torch.FloatTensor(0, 17)
    for i, (X,y) in enumerate(loader):
        if cuda:
            X, y = X.cuda(), y.cuda()
        X, y = Variable(X, volatile=True), Variable(y)
        output = net(X)
        predictions = torch.cat((predictions, output.cpu().data), 0)
    return predictions

if __name__ == '__main__':
    net = model.__dict__[args.model]()
    net.load_state_dict(torch.load('saved-models/{}.pth.tar'.format(args.model)))
    if cuda:
        net = net.cuda()
    pred = predict(net, val_loader)
    print(pred[0:30])
