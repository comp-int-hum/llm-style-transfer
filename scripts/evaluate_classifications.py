# this builds a plot of a heatmap
import seaborn as sns
from sklearn.metrics import confusion_matrix

import argparse
import json
import re
import pandas as pd
import seaborn
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--summary", dest="summary")
parser.add_argument("--author_index", dest="author_index", default='author2idx.json')
parser.add_argument("--eval_method", choices=["heat map", "accuracy"], default="heat map")
# parser.add_argument("--predictions", dest="predictions", default="predictions.json")
args, rest = parser.parse_known_args()

seaborn.set_theme(style="whitegrid")

# load predictions 
# with open(args.predictions, "rt") as ifd:
#     predictions = json.loads(ifd.read())

# load lookup tables
with open(args.author_index, 'rt') as ifd:
    author_to_index = json.loads(ifd.read())

# make inverse lookup
index_to_author = {i : k for k, i in author_to_index.items()}

results = []

for i in range(int(len(rest) / 2)):
    with open(rest[i * 2 + 1], "rt") as ifd:
        configuration = json.loads(ifd.read())
    with open(rest[i * 2], "rt") as ifd:
        preds = json.loads(ifd.read())
        configuration["accuracy"] = preds["accuracy"]
        configuration["preds"] = preds["preds"]
        configuration["actual"] = preds["actual"]
    results.append(configuration)



df = pd.DataFrame(results)

# HACK making fake data so I can check the plotting
df.loc[len(df.index)] = ['20', 'True', '200', 'stopwords', '5', '2', 0.978, preds['preds'], preds['actual']]
print(df.head())
fields = [
    f for f in [
        "CLUSTER_COUNT",
        "WORDS_PER_SUBDOCUMENT",
        "NUM_FEATURES_TO_KEEP",
        "LOWERCASE",
        "FEATURE_SELECTION_METHOD"
    ] if df[f].nunique() > 1
]

if args.eval_method == "accuracy":
    fig, axs = plt.subplots(len(fields), 1, figsize=(15, 5 * len(fields)))

    for i, field in enumerate(fields):
        uvs = df[field].unique()
        if all([v.isdigit() for v in uvs]):
            lookup = {float(v) : v for v in uvs}
            order = [lookup[k] for k in sorted(lookup.keys())]
        else:
            order = uvs
        
        # HACK
        df["accuracy"] = df["accuracy"].astype(float)
        df["CLUSTER_COUNT"] = df["CLUSTER_COUNT"].astype(int)
        # df["LOWERCASE"] = df["LOWERCASE"].astype(bool)
        
        
        sp = seaborn.pointplot(x=field, y="accuracy", data=df, order=order, ax=axs[i]).set_ylabel("Accuracy")
elif args.eval_method == "heat map":
    fig, axs = plt.subplots(len(fields), 1, figsize=(15, 5 * len(fields)))
        pass
        
        # cm = df[['preds','actual']]
        # print(cm.shape)
        # print(cm.dtypes)

        # print(type(cm[0]), cm[1].shape)
        # make a confusion matrix of the predictions, then plot
        # have to incorporate field somehow
        # sp = seaborn.heatmap(cm)

        
    

# fig.tight_layout()
# fig.savefig(f'{args.eval_method}_{args.summary}.png')