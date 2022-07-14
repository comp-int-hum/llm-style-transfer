import argparse
import gzip
import json
import random

parser = argparse.ArgumentParser()
parser.add_argument("--representations", dest="representations")
parser.add_argument("--train_dev_test_proportions", dest="train_dev_test_proportions", nargs=3, type=float)
parser.add_argument("--train", dest="train")
parser.add_argument("--dev", dest="dev")
parser.add_argument("--test", dest="test")
parser.add_argument("--scale", dest="scale", default=False, action="store_true")
parser.add_argument("--random_seed", dest="random_seed", type=int)
args = parser.parse_args()

if args.random_seed:
    random.seed(args.random_seed)

with gzip.open(args.representations, "rt") as ifd:
    items = json.loads(ifd.read())
    random.shuffle(items)
    if args.scale:
        test = [item for item in items if "original_author" in item["provenance"]]
        non_test = [item for item in items if "original_author" not in item["provenance"]]
        total_non_test = len(non_test)
        train_count = int(0.8 * total_non_test)
        train = non_test[:train_count]
        dev = non_test[train_count:]
    else:
        total = len(items)
        train_count = int(args.train_dev_test_proportions[0] * total)
        dev_count = int(args.train_dev_test_proportions[1] * total)
        test_count = total - (train_count + dev_count)
        train = items[:train_count]
        dev = items[train_count:train_count + dev_count]
        test = items[train_count + dev_count:]
        
    with gzip.open(args.train, "wt") as ofd:
        ofd.write(json.dumps(train, indent=4))
    with gzip.open(args.dev, "wt") as ofd:
        ofd.write(json.dumps(dev, indent=4))
    with gzip.open(args.test, "wt") as ofd:
        ofd.write(json.dumps(test, indent=4))
