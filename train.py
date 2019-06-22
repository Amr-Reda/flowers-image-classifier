import argparse
from torch import nn, optim
from helper_functions import load_data, save_checkpoint, build_network, train_network, get_accuracy

parser = argparse.ArgumentParser(description='train a deep neural network on the flower data set')

parser.add_argument('data_directory', action="store", type=str, help="path of data dir")
parser.add_argument('--save_dir', action="store", type=str, dest="save_dir", help="path to save checkpoint in, default is checkpoint.pth", default='checkpoint.pth')
parser.add_argument('--arch', action="store", dest="arch", type=str, help="model of the network, default is vgg16", default="vgg16")
parser.add_argument('--learning_rate', action="store", dest="learning_rate", type=float, help="learning_rate of model, default is 0.001", default=0.001)
parser.add_argument('--hidden_units', action="store", dest="hidden_units", type=int, help="hidden_units of the network, default is 1024", default=1024)
parser.add_argument('--epochs', action="store", dest="epochs", type=int, help="number of epochs used to train the network, default is 5", default=5)
parser.add_argument('--gpu', action="store_true", default=False, help='GPU mode for training, default is off')

results = parser.parse_args()

data_directory = results.data_directory
save_dir = results.save_dir
arch = results.arch
learning_rate = results.learning_rate
hidden_units = results.hidden_units
epochs = results.epochs
gpu = results.gpu

trainloader, testloader, validloader, train_datasets, test_datasets, valid_datasets = load_data(data_directory)


model = build_network(arch, hidden_units)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

model, optimizer = train_network(model, epochs, gpu, criterion, optimizer, trainloader, validloader)

get_accuracy(testloader, gpu, model)
save_checkpoint(model, epochs, optimizer, train_datasets, save_dir, arch)