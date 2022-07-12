import json
import argparse
import gzip

parser = argparse.ArgumentParser()
parser.add_argument("--configuration", dest="configuration")
args, rest = parser.parse_known_args()

config = {}
#for i in range(int(len(rest) / 2)):
while len(rest) != 0:
    name = rest[0].lstrip("--").upper()
    rest = rest[1:]    
    vals = []
    while len(rest) > 0 and not rest[0].startswith("--"):
        vals.append(rest[0])
        rest = rest[1:]
    if len(vals) == 1:
        config[name] = vals[0]
    elif len(vals) > 1:
        config[name] = vals

with gzip.open(args.configuration, "wt") as ofd:
    ofd.write(json.dumps(config))
