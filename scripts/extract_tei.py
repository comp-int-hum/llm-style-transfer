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


# TODO: options for provenance:
    #author 
    #title
    #source (for DH)
    #superscription (for psalms)
    #language 

# NOTE: this is how to load a .json.gz
# if args.primary_source.endswith("json.gz"):
#     with gzip.open(args.primary_source, "rt") as f:
#         data = json.loads(f.read())
#     with open(f'{args.primary_source}.json', "wt") as ofd:
#         ofd.write(json.dumps(data, indent=4))
    # print(data)
result = []
# if the data folder is wlc_dh, then the text fields we look at are 
# also each verse has to be its own entry bc we have the source
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
                    
                    # HACK: for whatever reason, it flips the text? I manually reverse it again
                    line = [x[::-1].strip() for x in verse.itertext()]
                    # remove the empty entries
                    while("" in line):
                        line.remove("")
                    line = " ".join(line)
                    line = line.replace('\n','')
                    item["text"] = line
                    result.append(item)

                    
                    
               



json_str = json.dumps(result, indent=4)         # 2. string (i.e. JSON)
json_bytes = json_str.encode('utf-8')            # 3. bytes (i.e. UTF-8)

with gzip.open(f'{args.experiment_name}.json.gz', 'w') as ofd:       # 4. fewer bytes (i.e. gzip)
    ofd.write(json_bytes) 
# save it as a compressed .json so I can load all the datasets with the gzip logic
with open(f'{args.experiment_name}.json', "wt") as ofd:
    ofd.write(json.dumps(result, indent=4))


# if args.primary_source.endswith("tei") or args.primary_source.endswith("xml"):
#     with open(args.primary_source, "rt") as ifd:
#         xml = etree.parse(ifd)
#         title = xml.find(".//{*}titleStmt/{*}title").text
#         author_element = xml.find(".//{*}author")
#         pers_element = author_element.find(".//{*}persName") if author_element else None
#         author_name_components = list(pers_element.itertext() if pers_element else author_element.itertext() if author_element else ["Anonymous"])
#         author = re.sub(r"\s+", " ", " ".join(author_name_components)).strip()
#         result["author"] = author
#         result["title"] = title
#         current_doc = []
#         for paragraph in xml.findall(".//{*}text//{*}p"):
#             paragraph_contents = " ".join(list(paragraph.itertext())).strip()
#             result["text"].append(paragraph_contents)
#         result["text"] = re.sub(r"\s+", " ", " ".join(result["text"]))
# else:
#     raise Exception("The input document '{}' does not appear to be in a recognized format (either the extension is unknown, or you need to add handling logic for it to 'scripts/divide_documents.py')".format(args.source_document))
    



def getDataRecursive(element):
    """
    adapted from: 
    https://stackoverflow.com/questions/51401826/parse-xml-in-python-without-manually-calling-attribute-tags-and-child-number
    """
    data = list()
    # get attributes of element, necessary for all elements
    for key in element.attrib.keys():
        data.append(element.tag + '.' + key + '.' + element.attrib.get(key))
    # only end-of-line elements have important text, at least in this example
    if len(element) == 0:
        if element.text is not None:
            data.append(element.tag + '.' + element.text)
    # otherwise, go deeper and add to the current tag
    else:
        for el in element:
            within = getDataRecursive(el)
            for data_point in within:
                data.append(element.tag + '.' + data_point)
    return data
