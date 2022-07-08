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
parser.add_argument("--summary", dest="summary", default='summary')
parser.add_argument("--labels_index", dest="labels_index", default='author2idx.json')
parser.add_argument("--eval_method", dest="eval_method", nargs="+",choices=["heat map", "f1"], default="heat map")
args, rest = parser.parse_known_args()

seaborn.set_theme(style="whitegrid")

# load lookup tables
with open(args.labels_index, 'rt') as ifd:
    labels_to_index = json.loads(ifd.read())

# make inverse lookup
index_to_labels = {i : k for k, i in labels_to_index.items()}

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

if "heat map" in args.eval_method:
    # print(df.head())
    for index, row in df.iterrows():
        plt.figure()
        cm = row[["actual", "preds"]]
        confusion_matrix = pd.crosstab(cm["actual"], cm["preds"], rownames=['Actual'], colnames=['Predictions'])
        seaborn.heatmap(confusion_matrix, annot=True)
        plt.show()
        plt.savefig(f'{args.eval_method}_{args.summary}_{index}.png')

elif "table" in args.eval_method:
    # NOTE: we want the accuracy method to spit out a tabular format of all 
    pass
elif "f1" in args.eval_method:
    
    fields = [
        f for f in [
            "CLUSTER_COUNT",
            "WORDS_PER_SUBDOCUMENT",
            "NUM_FEATURES_TO_KEEP",
            "LOWERCASE",
            "FEATURE_SELECTION_METHOD"
        ] if df[f].nunique() > 1
    ]

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



    fig.tight_layout()
    fig.savefig(f'{args.eval_method}_{args.summary}.png')