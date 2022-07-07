import argparse
import json
import re
import os
import pickle
import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics


parser = argparse.ArgumentParser()
parser.add_argument("--representations", dest="representations")
parser.add_argument("--random_seed", dest="random_seed", type=int)
parser.add_argument("--classifier", dest="classifer", choices=["Nearest Neighbors","Linear SVM","RBF SVM","Gaussian Process","Decision Tree","Random Forest","Neural Net","AdaBoost","Naive Bayes","QDA"], default="Naive Bayes")
parser.add_argument("--model", dest="model", default='model')
parser.add_argument("--author_index", dest="author_index", default='author2idx')
parser.add_argument("--ft_index", dest="ft_index", default='ft2idx')
args = parser.parse_args()


with open(args.representations, "rt") as ifd:
    representations = json.loads(ifd.read())

features = set()
authors = set()
for item in representations:
    authors.add(item["author"])
    for k in item["representation"].keys(): # the words
        features.add(k)
    
feature_to_index = {k : i for i, k in enumerate(features)}
index_to_feature = {i : k for k, i in feature_to_index.items()}

author_to_index = {k : i for i, k in enumerate(authors)}
index_to_author = {i : k for k, i in author_to_index.items()}
  
data = np.zeros(shape=(len(representations), len(features)))
y_data = []
for row, item in enumerate(representations):
    y_data.append(author_to_index[item["author"]])
    for feature, value in item["representation"].items():
        data[row, feature_to_index[feature]] = value

# X_train, X_test, y_train, y_test = train_test_split(data, y_data, 
#                                     test_size=0.2, 
#                                     random_state=args.random_seed)
# NOTE: actually not splitting here bc we will want to evaluate this model on 
# or do we split and then save unmodified test set to disk, then 

model = GaussianNB()
model.fit(data, y_data)

y_predicted = model.predict(data)
print(metrics.accuracy_score(y_predicted, y_data))

# save model
with open(f'{args.model}.pkl','wb') as f:
    pickle.dump(model,f)

# save author lookup
with open(f'{args.author_index}.json', 'wt') as f:
    f.write(json.dumps(author_to_index, indent=4))

# save feature lookup
with open(f'{args.ft_index}.json', 'wt') as f:
    f.write(json.dumps(feature_to_index, indent=4))
