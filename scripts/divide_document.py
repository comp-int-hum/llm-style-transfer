import gzip
import argparse
import xml.etree.ElementTree as etree
import json
import re
import logging
from nltk.tokenize import word_tokenize

parser = argparse.ArgumentParser()
parser.add_argument("--documents", dest="documents")
parser.add_argument("--subdocuments", dest="subdocuments")
parser.add_argument("--tokens_per_subdocument", dest="tokens_per_subdocument", type=int, default=0)
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)

subdocuments = []
with gzip.open(args.documents, "rt") as ifd:
    for document in json.loads(ifd.read()):
        if args.tokens_per_subdocument != 0:
            current_subdoc_tokens = []
            for word in word_tokenize(document["text"]):
                current_subdoc_tokens.append(word)
                if len(current_subdoc_tokens) == args.tokens_per_subdocument:
                    subdocument = {
                        "provenance" : document["provenance"],
                        "text" : " ".join(current_subdoc_tokens),
                        "id" : "{}_subdoc_{}".format(document["id"], len(subdocuments))
                    }
                    subdocuments.append(subdocument)
                    current_subdoc_tokens = []
            if len(current_subdoc_tokens) > 0:
                    subdocument = {
                        "provenance" : document["provenance"],
                        "text" : " ".join(current_subdoc_tokens),
                        "id" : "{}_subdoc_{}".format(document["id"], len(subdocuments))
                    }
                    subdocuments.append(subdocument)
        else:
            subdocuments.append(document)    
    
with gzip.open(args.subdocuments, "wt") as ofd:
    logging.info("Divided data into {} subdocuments".format(len(subdocuments)))
    ofd.write(json.dumps(subdocuments, indent=4))
