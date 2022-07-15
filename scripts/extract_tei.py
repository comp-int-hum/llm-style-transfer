import argparse
import xml.etree.ElementTree as etree
import gzip
import json
import re
import os
from nltk.tokenize import word_tokenize
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--primary_source", dest="primary_source", default="data/wlc_dh")
# NOTE: this is redundany, output will always be .json of the input file
# parser.add_argument("--document", dest="document", default="exodus.json")
# TODO: want to grab this from env variables, rather than explicitly, if possible
parser.add_argument("--experiment_name", dest="experiment_name", default="wlc_dh")
args = parser.parse_args()

# NOTE: this is how to load a .json.gz
# if args.primary_source.endswith("json.gz"):
#     with gzip.open(args.primary_source, "rt") as f:
#         data = json.loads(f.read())
#     with open(f'{args.primary_source}.json', "wt") as ofd:
#         ofd.write(json.dumps(data, indent=4))
    # print(data)
result = []

# NOTE: different behaviour based on if it is a directory or an individual file.
ID_COUNTER=0
if os.path.isdir(args.primary_source):
    # NOTE: let's output it all into one .json
    # all of them are going to be at the verse level
    # if args.primary_source == "wlc_dh":
    
    for fname in os.listdir(args.primary_source):
        print(f"****** \n Processing text in {fname} \n ******")
        provenance = {}
        with open(os.path.join(args.primary_source, fname), "rt") as ifd:
            xml = etree.parse(ifd)
        root = xml.getroot()
        # HACK: grab book title
        title = xml.findall(".//{*}titleStmt/{*}title")[-1].text
        if title.endswith(".DH"):
            provenance["DH"] = True
            title = title[:-3]
        provenance["title"] = title
        lang = xml.find(".//{*}language").attrib['ident'].lower()
        provenance["language"] = lang

        for chapter in tqdm(xml.findall(".//{*}c")):
            for i, verse in enumerate(chapter):
                item = {"text" : [], "provenance": provenance, "id": f'{args.experiment_name}_{ID_COUNTER}'}
                item["provenance"]["source"] = verse.attrib.get('s')
                item["provenance"]["chapter"] = chapter.attrib.get('n')
                item["provenance"]["verse"] = verse.attrib.get('n')

                # NOTE: this catches the <vs> tags that don't have text
                if None not in [item["provenance"]["chapter"], item["provenance"]["verse"]]:
                    ID_COUNTER += 1
                    # TODO: check for ellipses
                    # HACK: for whatever reason, it flips the text? I manually reverse it again
                    # line = [x[::-1].strip() for x in verse.itertext()]
                    line = [x.strip() for x in verse.itertext()]

                    # remove the empty entries
                    while("" in line):
                        line.remove("")
                    line = " ".join(line)
                    line = line.replace('\n','')
                    item["text"] = line
                    result.append(item)
                    

                    
                    
             
#### saving the result to disk ####
# json_str = json.dumps(result, indent=4) 

# with open(f'{args.experiment_name}.json', "wt") as ofd:
#     ofd.write(json_str)

# json_bytes = json_str.encode('utf-8')           

# with gzip.open(f'{args.experiment_name}.json.gz', 'w') as ofd:       # 4. fewer bytes (i.e. gzip)
#     ofd.write(json_bytes) 



