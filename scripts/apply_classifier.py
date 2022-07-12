import sklearn.naive_bayes as nb
import sklearn.linear_model as lm
import sklearn.neural_network as nn
import json
import gzip
import argparse
import pickle
import pandas

parser = argparse.ArgumentParser()
parser.add_argument("--test", dest="test")
parser.add_argument("--model", dest="model")
parser.add_argument("--feature_sets", dest="feature_sets", nargs="+")
parser.add_argument("--target_class", dest="target_class", nargs="+")
parser.add_argument("--results", dest="results")
args = parser.parse_args()

with gzip.open(args.model, "rb") as ifd:
    model = pickle.loads(ifd.read())

test = []
with gzip.open(args.test, "rt") as ifd:
    items = json.loads(ifd.read())
    for item in items:
        datum = {"id" : item["id"], "features" : {}}
        datum["label"] = tuple([str(item["provenance"].get(x, None)) for x in args.target_class])
        for feature_set in args.feature_sets:
            denom = sum(item["feature_sets"][feature_set]["values"].values()) if item["feature_sets"][feature_set].get("categorical_distribution", False) else 1.0
            for name, val in item["feature_sets"][feature_set]["values"].items():
                feat_name = "{}-{}".format(feature_set, name)
                if feat_name in model.feature_names_in_:
                    datum["features"][feat_name] = val / denom
        test.append(datum)

for feat_name in model.feature_names_in_:
    test[0]["features"][feat_name] = test[0]["features"].get(feat_name, 0.0)


X = pandas.DataFrame.from_records([d["features"] for d in test])
X[X.isnull()] = 0.0
X = X[model.feature_names_in_]

preds = [dict([(k, v) for k, v in zip(args.target_class, eval(x))]) for x in model.classes_]

for i, probs in enumerate(model.predict_log_proba(X)):
    items[i]["predicted_provenances"] = list(sorted([(prob, pred) for pred, prob in zip(preds, probs)], key=lambda x : x[0], reverse=True))

with gzip.open(args.results, "wt") as ofd:
    ofd.write(json.dumps(items, indent=4))
    
