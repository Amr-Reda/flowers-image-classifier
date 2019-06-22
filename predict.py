import argparse
from helper_functions import load_checkpoint, predict
import json

parser = argparse.ArgumentParser(description='Predict flower name from an image')

parser.add_argument('image_path', action="store", type=str, help="path of the image")
parser.add_argument('checkpoint', action="store", type=str, help="checkpoint file to load the network data")
parser.add_argument('--top_k', action="store", dest="top_k", type=int, help="top K most likely classes, default is 5", default=5)
parser.add_argument('--category_names ', action="store", dest="category_names", type=str, help="mapping of categories to real names file, default is cat_to_name.json", default='cat_to_name.json')
parser.add_argument('--gpu', action="store_true", default=False, help='GPU mode for training, default is off')

results = parser.parse_args()

image_path = results.image_path
checkpoint = results.checkpoint
top_k = results.top_k
category_names = results.category_names
gpu = results.gpu

model = load_checkpoint(checkpoint)

with open(category_names, 'r') as f:
    cat_to_name = json.load(f)

probs, classes = predict(image_path, model, top_k, gpu)

names = []
for i in classes:
    names += [cat_to_name[str(i)]]
    
print(probs)
print(classes)
print(names)