from torchvision import datasets, transforms, models
from torch import nn, optim
import torch
from collections import OrderedDict
from PIL import Image
import numpy as np
import torch.nn.functional as F

def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                         transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    
    train_datasets = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform = test_transforms)

    
    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_datasets, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size=32)
    
    return trainloader, testloader, validloader, train_datasets, test_datasets, valid_datasets

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open(image)
    
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                         [0.229, 0.224, 0.225])])
    
    return np.array(transform(im))

def validation(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0
    
    for images, labels in testloader:
        
        images, labels = images.to(device), labels.to(device)
        
        output = model.forward(images)
        test_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy

def build_network(arch, hidden_units):
    model = getattr(models, arch)(pretrained=True)
    input_size = model.classifier[0].in_features
    output_size=102

    for param in model.parameters():
        param.requires_grade=False
    
    classifier = nn.Sequential(OrderedDict([('fc1',nn.Linear(input_size, hidden_units)),
                                           ('relu1',nn.ReLU()),
                                           ('dropout1', nn.Dropout(0.05)),
                                           ('fc2', nn.Linear(hidden_units, output_size)),
                                           ('output', nn.LogSoftmax(dim=1))
                                           ]))

    model.classifier = classifier
    return model

def train_network(model, epochs, gpu, criterion, optimizer, trainloader, validloader):
    device = torch.device("cuda:0" if gpu else "cpu")
    model = model.to(device)
    steps = 0
    running_loss = 0
    print_every = 40
    for e in range(epochs):
        model.train()
        
        for ii, (images, labels) in enumerate(trainloader):
            steps += 1
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # Make sure network is in eval mode for inference
                model.eval()

                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validloader, criterion, device)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Valid Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                      "Valid Accuracy: {:.3f}".format(accuracy/len(validloader)))

                running_loss = 0

                # Make sure training is back on
                model.train()
    return model, optimizer


def get_accuracy(testloader, gpu, model):
    device = torch.device("cuda:0" if gpu else "cpu")
    correct_classified = 0
    total_classified = 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            # Get probabilities
            outputs = model(images)
            # Turn probabilities into predictions
            _, predicted_output = torch.max(outputs.data, 1)

            total_classified += labels.size(0)
            correct_classified += (predicted_output == labels).sum().item()

    return print(f"Accuracy of the model: {(100 * correct_classified / total_classified)}%")

def save_checkpoint(model, epochs, optimizer, train_datasets, path, arch):
    model.class_to_idx = train_datasets.class_to_idx

    checkpoint = {'state_dict': model.state_dict(),
                  'arch': arch,
                  'epochs': epochs,
                  'optimizer_state': optimizer.state_dict,
                  'classifier': model.classifier,
                  'class_to_idx': model.class_to_idx}

    torch.save(checkpoint, path)
    return

def load_checkpoint(filepath):
    
    checkpoint = torch.load(filepath)
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    model.classifier = checkpoint['classifier'] 
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def predict(image_path, model, topk, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = torch.device("cuda:0" if gpu else "cpu")
    image = process_image(image_path)
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image = image.unsqueeze_(0)
    
    
    with torch.no_grad():
        output = model.cpu().forward(image)
        
    probability = F.softmax(output.data,dim=1)
    probs,index = probability.topk(topk)
    
    probs_list = np.array(probs)[0]
    index_list = np.array(index)[0]
    
    class_to_idx = model.class_to_idx
    indx_to_class={}
    for x,y in class_to_idx.items():
        indx_to_class[y] = x

    classes_list = []
    for index in index_list:
        classes_list.append(indx_to_class[index])
        
    return probs_list, classes_list

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    im = Image.open(image)
    
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                         [0.229, 0.224, 0.225])])
    
    return np.array(transform(im))