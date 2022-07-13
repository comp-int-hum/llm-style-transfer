import argparse
import json
import re
import os
import pickle
import numpy as np

from sklearn.naive_bayes import GaussianNB
# TODO: implement support for these classifiers
# from sklearn.neural_network import MLPClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn import metrics


parser = argparse.ArgumentParser()
parser.add_argument("--representations", dest="representations")
parser.add_argument("--random_seed", dest="random_seed", type=int)
parser.add_argument("--classifier", dest="classifer", choices=["Nearest Neighbors","Linear SVM","RBF SVM","Gaussian Process","Decision Tree","Random Forest","Neural Net","AdaBoost","Naive Bayes","QDA"], default="Naive Bayes")
parser.add_argument("--model", dest="model", default='model')
parser.add_argument("--features", dest="feats", nargs="+", help="the feature set(s) to use in the representation")
parser.add_argument("--labels", dest="labels", help="the field to use as training labels")
parser.add_argument("--output_dir", dest="output_dir")
args = parser.parse_args()


with open(args.representations, "rt") as ifd:
    representations = json.loads(ifd.read())

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

labels_to_index = {k : i for i, k in enumerate(labels)}
index_to_labels = {i : k for k, i in labels_to_index.items()}

# save labels lookup
with open(os.path.join(args.output_dir, f'{args.labels}_lookup.json'), 'wt') as f:
    f.write(json.dumps(labels_to_index, indent=4))

# get the labels
labels_vector = []
for item in representations:
    labels_vector.append(labels_to_index[item["provenance"][args.labels]])

output_vector = []
ft_lookups = []
for feat_type in args.feats:
    feature_to_index = {k : i for i, k in enumerate(features[feat_type])}
    index_to_feature = {i : k for k, i in feature_to_index.items()}
    
    # this shape is key, it means that the dense features will still be saved as sparse
    data = np.zeros(shape=(len(representations), len(features[feat_type])))
    for row, item in enumerate(representations):
        for feature, value in item["features"][feat_type].items():
            data[row, feature_to_index[feature]] = value
    output_vector.append(data)
    
    ft_lookups.append({feat_type:feature_to_index})

# save feature lookup
with open(os.path.join(args.output_dir,f'feature_lookup.json'), 'wt') as f:
    f.write(json.dumps(ft_lookups, indent=4))

# stack the distributions together
output_vector = np.hstack(output_vector) # num_examples, features

X_train, X_test, y_train, y_test = train_test_split(output_vector, labels_vector, 
                                    test_size=0.2, 
                                    random_state=args.random_seed)

# TODO: add support for other classifiers, group into a fn
model = GaussianNB()
model.fit(X_train, y_train)

# get validation
y_predicted = model.predict(X_test)
print(f'F1 on validation set: {metrics.f1_score(y_predicted, y_test)}')

# save model
# with open(os.path.join(args.output_dir, f'{args.model}.pkl'),'wb') as f:
#     pickle.dump(model,f)

