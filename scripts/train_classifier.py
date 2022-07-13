import sklearn.naive_bayes as nb
# from sklearn.neural_network import MLPClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import json
import gzip
import argparse
import pickle
import pandas

parser = argparse.ArgumentParser()
parser.add_argument("--train", dest="train")
parser.add_argument("--dev", dest="dev")
parser.add_argument("--classifier", dest="classifier")
parser.add_argument("--feature_sets", dest="feature_sets", nargs="+")
parser.add_argument("--target_class", dest="target_class", nargs="+")
parser.add_argument("--model", dest="model")
args = parser.parse_args()

train = []
with gzip.open(args.train, "rt") as ifd:
    for item in json.loads(ifd.read()):
        datum = {"id" : item["id"], "features" : {}}
        for feature_set in args.feature_sets:
            denom = sum(item["feature_sets"][feature_set]["values"].values()) if item["feature_sets"][feature_set].get("categorical_distribution", False) else 1.0
            datum["label"] = tuple([str(item["provenance"].get(x, None)) for x in args.target_class])
            for name, val in item["feature_sets"][feature_set]["values"].items():
                datum["features"]["{}-{}".format(feature_set, name)] = val / denom
        train.append(datum)


dev = []
with gzip.open(args.dev, "rt") as ifd:
    for item in json.loads(ifd.read()):
        datum = {"id" : item["id"], "features" : {}}
        for feature_set in args.feature_sets:
            denom = sum(item["feature_sets"][feature_set]["values"].values()) if item["feature_sets"][feature_set].get("categorical_distribution", False) else 1.0
            datum["label"] = tuple([str(item["provenance"].get(x, None)) for x in args.target_class])
            for name, val in item["feature_sets"][feature_set]["values"].items():
                datum["features"]["{}-{}".format(feature_set, name)] = val / denom
        dev.append(datum)


# add more classifier-types here
if args.classifier == "naive_bayes":
    model = nb.MultinomialNB(fit_prior=False)
    X = pandas.DataFrame.from_records([d["features"] for d in train])
    X[X.isnull()] = 0.0
    Y = [str(d["label"]) for d in train]
    model.fit(X, Y)


with gzip.open(args.model, "wb") as ofd:
    ofd.write(pickle.dumps(model))
