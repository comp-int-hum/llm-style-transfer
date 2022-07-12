import argparse
import xml.etree.ElementTree as etree
import gzip
import json
import re
import tarfile
from nltk.tokenize import word_tokenize

parser = argparse.ArgumentParser()
parser.add_argument("--primary_sources", dest="primary_sources")
parser.add_argument("--language", dest="language")
parser.add_argument("--documents", dest="documents")
args = parser.parse_args()

documents = []
if args.primary_sources.endswith("tgz"):
    tf = tarfile.open(args.primary_sources, "r:gz")
    for fname in tf.getnames():
        if fname.endswith("tei"):
            with tf.extractfile(fname) as ifd:
                document = {
                    "provenance" : {"language" : args.language},
                    "id" : fname,
                    "text" : []
                }
                xml = etree.parse(ifd)
                title = xml.find(".//{*}titleStmt/{*}title").text
                author_element = xml.find(".//{*}author")
                pers_element = author_element.find(".//{*}persName") if author_element else None
                author_name_components = list(pers_element.itertext() if pers_element else author_element.itertext() if author_element else ["Anonymous"])
                author = re.sub(r"\s+", " ", " ".join(author_name_components)).strip()
                document["provenance"]["author"] = author
                document["provenance"]["title"] = title
                current_doc = []
                for paragraph in xml.findall(".//{*}text//{*}p"):
                    paragraph_contents = " ".join(list(paragraph.itertext())).strip()
                    document["text"].append(paragraph_contents)
                document["text"] = re.sub(r"\s+", " ", " ".join(document["text"]))
                documents.append(document)
elif args.primary_sources.endswith("json.gz"):
    with gzip.open(args.primary_sources, "rt") as ifd:
        documents = json.loads(ifd.read())

else:
    raise Exception("The input document '{}' does not appear to be in a recognized format (either the extension is unknown, or you need to add handling logic for it to 'scripts/divide_documents.py')".format(args.primary_sources))


for i in range(len(documents)):
    # maybe check formatting here, too?
    documents[i]["provenance"]["language"] = args.language


with gzip.open(args.documents, "wt") as ofd:
    ofd.write(json.dumps(documents, indent=4))
