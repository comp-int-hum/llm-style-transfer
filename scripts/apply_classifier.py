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
parser.add_argument("--label_index", dest="label_index", default='label_lookup.json')
parser.add_argument("--feature_index", dest="feature_index", default='feature_lookup.json')
# NOTE: can get around including these if we can look up the fields in the lookup tables
parser.add_argument("--features", dest="feats", nargs="+", help="the feature set(s) to use in the representation")
parser.add_argument("--labels", dest="labels", help="the field to use as training labels")
parser.add_argument("--predictions", dest="predictions", default="predictions")
args = parser.parse_args()


# load model
with open(f'{args.model}.pkl', 'rb') as f:
    model = pickle.load(f)

# load feature vectors
with open(args.representations, "rt") as ifd:
    representations = json.loads(ifd.read())

# load lookup tables
with open(args.labels_index, 'rt') as ifd:
    label_to_index = json.loads(ifd.read())
with open(args.feature_index, 'rt') as ifd:
    feature_to_index = json.loads(ifd.read())

# format the data for prediction
features = {}
labels = set()

for item in representations:
    labels.add(item["provenance"][args.labels])
    for k,v in item["features"].items(): # the feature sets
        if k in args.feats:
            if k not in features.keys():
                features[k] = set()
            for l in item["features"][k].keys():
                features[k].add(l)

data = np.zeros(shape=(len(representations), len(features)))
y_data = []
for row, item in enumerate(representations):
    y_data.append(labels_to_index[item["provenance"][args.labels]])
    for feature, value in item["features"].items():
        data[row, feature_to_index[feature]] = value

# get prediction on val data
y_predicted = model.predict(data)
# score = metrics.accuracy_score(y_predicted, y_data)
score = metrics.f1(y_predicted, y_data)


results = {'accuracy': score, 'preds': list(y_predicted), 'actual': y_data }

with open(f'{args.predictions}.json', 'wt') as ifd:
    ifd.write(json.dumps(results, indent=4, cls=NumpyEncoder))
