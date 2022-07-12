import argparse
import json
import re
import os
import pickle
import numpy as np
import pandas as pd
import seaborn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import metrics

from utils import NumpyEncoder


parser = argparse.ArgumentParser()
parser.add_argument("--representations", dest="representations")
parser.add_argument("--random_seed", dest="random_seed", type=int)
parser.add_argument("--classifier", dest="classifer", choices=["Nearest Neighbors","Linear SVM","RBF SVM","Gaussian Process","Decision Tree","Random Forest","Neural Net","AdaBoost","Naive Bayes","QDA"], default="Naive Bayes")
parser.add_argument("--model", dest="model", default='model')
parser.add_argument("--author_index", dest="author_index", default='author2idx.json')
parser.add_argument("--ft_index", dest="ft_index", default='ft2idx.json')
parser.add_argument("--predictions", dest="predictions", default="predictions")
args = parser.parse_args()


# load model
with open(f'{args.model}.pkl', 'rb') as f:
    model = pickle.load(f)

# load feature vectors
with open(args.representations, "rt") as ifd:
    representations = json.loads(ifd.read())

# load lookup tables
with open(args.author_index, 'rt') as ifd:
    author_to_index = json.loads(ifd.read())
with open(args.ft_index, 'rt') as ifd:
    feature_to_index = json.loads(ifd.read())

# format the data for prediction
features = set()
authors = set()
for item in representations:
    authors.add(item["author"])
    for k in item["representation"].keys(): # the words
        features.add(k)

data = np.zeros(shape=(len(representations), len(features)))
y_data = []
for row, item in enumerate(representations):
    y_data.append(author_to_index[item["author"]])
    for feature, value in item["representation"].items():
        data[row, feature_to_index[feature]] = value

# get prediction on val data
y_predicted = model.predict(data)
# score = metrics.accuracy_score(y_predicted, y_data)
score = metrics.f1(y_predicted, y_data)


results = {'accuracy': score, 'preds': list(y_predicted), 'actual': y_data }

with open(f'{args.predictions}.json', 'wt') as ifd:
    ifd.write(json.dumps(results, indent=4, cls=NumpyEncoder))
