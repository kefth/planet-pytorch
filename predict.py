import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import model
import data
import utils
import argparse
from sklearn.metrics import fbeta_score
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("model", type=str, help="saved model")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--scale", type=int, default=224, help="image scaling")
parser.add_argument("--nocuda", action='store_true', help="no cuda used")
parser.add_argument("--nworkers", type=int, default=4, help="number of workers")
parser.add_argument("--output_file", type=str, default="pred.csv", help="output file")
args = parser.parse_args()

cuda = not args.nocuda and torch.cuda.is_available() # use cuda
print('...predicting on cuda: {}'.format(cuda))

# Define transformations
test_transforms = transforms.Compose([transforms.RandomCrop(args.scale),
                        transforms.ToTensor()])
val_transforms = transforms.Compose([transforms.Scale(args.scale),
                        transforms.ToTensor()])

# Create dataloaders
kwargs = {'pin_memory': True} if cuda else {}
testset = data.PlanetData('data/sample_submission_v2.csv', 'data/test-jpg',
                'data/labels.txt', test_transforms)
test_loader = DataLoader(testset, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.nworkers, **kwargs)
valset = data.PlanetData('data/val_set_norm.csv', 'data/train-jpg',
                'data/labels.txt', val_transforms)
val_loader = DataLoader(valset, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.nworkers, **kwargs)


def fscore(prediction):
    """ Get the fscore of the validation set. Gives a good indication
    of score on public leaderboard"""
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
    loaded_model = torch.load(args.model)
    net = model.__dict__[loaded_model['arch']]()
    net.load_state_dict(loaded_model['state_dict'])
    print('...loaded {}'.format(loaded_model['arch']))
    if cuda:
        net = net.cuda()

    # predict on the test set
    y_test = predict(net, test_loader)

    # Ready dataframe for submission.
    labels, _, _ = data.get_labels('data/labels.txt')
    y_test = y_test.numpy()
    y_test = pd.DataFrame(y_test, columns = labels)

    # Populate the submission csv
    predictions = []
    for i in range(y_test.shape[0]):
        a = y_test.ix[[i]]
        a = a.apply(lambda x: x > 0.24, axis=1)
        a = a.transpose()
        a = a.loc[a[i] == True]
        ' '.join(list(a.index))
        predictions.append(' '.join(list(a.index)))
    df_test = pd.read_csv('data/sample_submission_v2.csv')
    df_test['tags'] = pd.Series(predictions).values
    df_test.to_csv(args.output_file, index=False)
