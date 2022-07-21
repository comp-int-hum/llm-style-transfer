import argparse
import xml.etree.ElementTree as etree
import gzip
import json
import re
import os
import tarfile
from nltk.tokenize import word_tokenize, sent_tokenize
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--primary_sources", dest="primary_sources", default="data/tirant_lo_blanc.txt")
parser.add_argument("--documents", dest="documents")
parser.add_argument("--language", dest="language", default="catalan")
parser.add_argument("--aggregation_method", dest="aggregation_method", choices=['line', 'verse', 'chapter', 'book'], 
                    help="What level of granularity to group the text spans", default='chapter')

args = parser.parse_args()



PREFIX = ''.join(args.primary_sources.split('.')[:-1]).split('/')[-1]

ID_COUNTER=0

def parse_xml(xml):
    result = []
    provenance = {}
    
    # root = xml.getroot()
    # HACK: grab book title
    title = xml.findall(".//{*}titleStmt/{*}title")[-1].text
    if title.endswith(".DH"):
        provenance["DH"] = True
        title = title[:-3]
    provenance["title"] = title
    # lang = xml.find(".//{*}language").attrib['ident'].lower()
    # provenance["language"] = lang
    for chapter in xml.findall(".//{*}c"):
        for i, verse in enumerate(chapter):
            item = {
                "text" : [], 
                "provenance": provenance, 
                "id": f'{PREFIX}_{ID_COUNTER}'}
            item["provenance"]["source"] = verse.attrib.get('s')
            item["provenance"]["chapter"] = chapter.attrib.get('n')
            item["provenance"]["verse"] = verse.attrib.get('n')

            # NOTE: this catches the <vs> tags that don't have text
            if item["provenance"]["verse"] != None:
            # if None not in [item["provenance"]["chapter"], item["provenance"]["verse"]]:
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
    return result

def parse_txt(text):
    # TODO: generalize, take in different delimiters
    if "tirant_lo_blanc" in PREFIX:
        if args.aggregation_method == "line":
            text = text.replace('\n', ' ')
            subdocs = sent_tokenize(text)
        elif args.aggregation_method == "chapter":
            # HACK: I first split on full stop, sub roman numerals, recombine, then split on numerals
            # text = text.replace('\n', ' ')
            lines = text.split('\n')
            reg = r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"
            nums = [line for line in lines if re.match(reg, line.rstrip()[:-1])]
            print(nums)
            # words = word_tokenize(text)
            
            # print(words[1290:1330])
            # print(words[1303])
            # print(words[1303].isalnum())
            # print(len(text.split('II.')))
            
            # print(re.search(reg, words[1334]))

            # formatted = []
            # ct = 0
            
            # print([re.search(reg, word).group(1) for word in words])
            # result = re.split(reg, text.strip())
            # print(len(result))

            
    
            # for word in words:
            #     match = re.search(reg, word)
            #     if match:
            #         print(match.group(0))
            #     if re.search(reg,subdoc):
            #         print(re.search(reg, subdoc))
            #         ct += 1
                # formatted.append(re.sub(r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$",'ROMAN_NUMERAL',subdoc))
            # print(ct)
            # subdocs = ".".join(formatted)
            # subdocs = subdocs.split("ROMAN_NUMERAL")
            # print(len(subdocs))
            # print(subdocs[0])
            # print(subdocs[1])

        elif args.aggregation_method == "book":
            subdocs = len(text.split("PART"))[1:]
        else:
            raise Exception("Please choose an aggregation method that is one of: line, chapter, and book")
    result = []
    return result

print(PREFIX)
results = []
if args.primary_sources.endswith("tgz"):
    tf = tarfile.open(args.primary_sources, "r:gz")
    for fname in tf.getnames():
        if fname.endswith(("tei", "xml")):
            with tf.extract_file(fname) as ifd:
                xml = etree.parse(fname)
            results.extend(parse_xml(xml))
elif args.primary_sources.endswith("json.gz"):               
    with gzip.open(args.primary_sources, "rt") as ifd:
        results.extend(json.loads(ifd.read()))
elif args.primary_sources.endswith(".txt"):
    with open(args.primary_sources, "rt") as ifd:
        text = ifd.read()
    results.extend(parse_txt(text))
else:
    raise Exception("The input document '{}' does not appear to be in a recognized format (either the extension is unknown, or you need to add handling logic for it to 'scripts/divide_documents.py')".format(args.primary_sources))

print(results)
# for fname in documents:
#     # print(f"****** \n Processing text in {fname} \n ******")
#     provenance = {}
#     try:
#         with tf.extractfile(fname) as ifd:
#             xml = etree.parse(ifd)
#     except:
#         with open(fname) as ifd:
#             xml = etree.parse(ifd)
#     print(xml)
#     
#                 # NOTE: here the 'verse' and 'source' fields are correct
#             # NOTE: here they're not


# for i in range(len(results)):
#     # maybe check formatting here, too?
#     results[i]["provenance"]["language"] = args.language

# # print('*****\n\n\n')

# #### saving the result to disk ####

# json_str = json.dumps(results, indent=4) 

# with open(f'{args.documents}', "wt") as ofd:
#     ofd.write(json_str)

# json_bytes = json_str.encode('utf-8')           

# with gzip.open(f'{args.experiment_name}.json.gz', 'w') as ofd:       # 4. fewer bytes (i.e. gzip)
#     ofd.write(json_bytes) 



