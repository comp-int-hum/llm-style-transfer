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
parser.add_argument("--random_seed", dest="random_seed", type=int)
args = parser.parse_args()

if args.random_seed:
    random.seed(args.random_seed)

with gzip.open(args.representations, "rt") as ifd:
    items = json.loads(ifd.read())
    total = len(items)
    train_count = int(args.train_dev_test_proportions[0] * total)
    dev_count = int(args.train_dev_test_proportions[1] * total)
    test_count = total - (train_count + dev_count)
    random.shuffle(items)
    with gzip.open(args.train, "wt") as ofd:
        ofd.write(json.dumps(items[:train_count], indent=4))
    with gzip.open(args.dev, "wt") as ofd:
        ofd.write(json.dumps(items[train_count:train_count + dev_count], indent=4))
    with gzip.open(args.test, "wt") as ofd:
        ofd.write(json.dumps(items[train_count + dev_count:], indent=4))
