import argparse
import json
import re
import pandas
import seaborn
import gzip
import matplotlib.pyplot as plt
from sklearn import metrics

# TODO: add heatmap plotting and LaTeX tabular summary

parser = argparse.ArgumentParser()
parser.add_argument("--summary", dest="summary")
args, rest = parser.parse_known_args()

results = []
for i in range(int(len(rest) / 6)):
    with gzip.open(rest[i * 6], "rt") as ifd:
        configuration = json.loads(ifd.read())    
    with gzip.open(rest[i * 6 + 1], "rt") as ifd:
        output = json.loads(ifd.read())
    model_fname, train_fname, dev_fname, test_fname = rest[i * 6 + 2 : i * 6 + 6]

    # evaluate a single experimental configuration here:
    guesses = [tuple(sorted([(k, v) for k, v in x["predicted_provenances"][0][1].items() if k in configuration["TARGET_CLASS"] and v != "None"])) for x in output]
    golds = [tuple(sorted([(k, v) for k, v in x["provenance"].items() if k in configuration["TARGET_CLASS"] and v != "None"])) for x in output]
    labels = list(sorted(set([str(x) for x in golds] + [str(x) for x in guesses])))
    cm = metrics.confusion_matrix([str(x) for x in golds], [str(x) for x in guesses], labels=labels)
    fscore = metrics.f1_score([str(x) for x in golds], [str(x) for x in guesses], average="macro")
    configuration["labels"] = [eval(x) for x in labels]
    configuration["f_score"] = fscore
    configuration["confusion_matrix"] = cm.tolist()
    results.append(configuration)

df = pandas.DataFrame.from_records(results)
print(df)

with gzip.open(args.summary, "wt") as ofd:
    ofd.write(json.dumps(results, indent=4))
    
# df = pandas.DataFrame(results)    
# fields = [
#     f for f in [
#         "CLUSTER_COUNT",
#         "WORDS_PER_SUBDOCUMENT",
#         "NUM_FEATURES_TO_KEEP",
#         "LOWERCASE",
#         "FEATURE_SELECTION_METHOD"
#     ] if df[f].nunique() > 1
# ]


# seaborn.set_theme(style="whitegrid")
# fig, axs = plt.subplots(len(fields), 1, figsize=(15, 5 * len(fields)))

# for i, field in enumerate(fields):
#     uvs = df[field].unique()
#     if all([v.isdigit() for v in uvs]):
#         lookup = {float(v) : v for v in uvs}
#         order = [lookup[k] for k in sorted(lookup.keys())]
#     else:
#         order = uvs
#     sp = seaborn.pointplot(x=field, y="ami", data=df, order=order, ax=axs[i]).set_ylabel("Adjusted mutual information with ground truth")

# fig.tight_layout()
# fig.savefig(args.summary)
