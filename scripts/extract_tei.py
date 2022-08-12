import argparse
from collections import Counter
import collections
from typing import Dict
import xml.etree.ElementTree as etree
import gzip
import json
import re
import os
import tarfile
from nltk.tokenize import word_tokenize, sent_tokenize
from tqdm import tqdm
import logging
from itertools import groupby

logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S %d-%m-%Y"
)
logger = logging.getLogger()

def aggregate(items: list[Dict], method: str)-> list[Dict]:
    """Takes in a list of items and aggregates them based on the specified method."""
    
    result = []
    if method == "line" or method == "verse":
        return items
    elif method == "chapter":
        for book, b_dicts in groupby(items, key=lambda i:i['provenance']['book']):
            # each b_dist is a list of dictionaries, one per verse for the whole book
            b_dicts = list(b_dicts)
            for chapter, c_dicts in groupby(b_dicts, key=lambda i:i['provenance']['chapter']):
            
                c_dicts = list(c_dicts)
                # NOTE: could be more careful with the IDs, but they'll still be unique so who cares
                # what do we do about the sources? just yeet them for now

                item = c_dicts[0].copy()
                del item['provenance']['verse']
                try:
                    del item['provenance']['s']
                except KeyError:
                    pass
                text = ' '.join([d['text'] for d in c_dicts])
                item['text'] = text
                #HACK
                item = {k : ({sk : sv for sk, sv in v.items()} if isinstance(v, dict) else v) for k, v in item.items()}

                result.append(item)
    # here we're going to groupby the book and chapter and then aggregate the verses
    elif method == "book":
        # here we're going to groupby the book and then aggregate the chapters
        for book, dicts in groupby(items, key=lambda i:i['provenance']['book']):
            # each b_dist is a list of dictionaries, one per verse for the whole book
            dicts = list(dicts)
            item = dicts[0].copy()
            del item['provenance']['verse']
            del item['provenance']['chapter']
            try:
                del item['provenance']['s']
            except KeyError:
                pass
            text = ' '.join([d['text'] for d in dicts])
            item['text'] = text
            
            #HACK 
            item = {k : ({sk : sv for sk, sv in v.items()} if isinstance(v, dict) else v) for k, v in item.items()}
            result.append(item)
    else:
        raise ValueError(f"{method} is not a valid aggregation method")

    return result

def parse_xml(xml):
    
    PREFIX = ''.join(args.primary_sources.split('.')[:-1]).split('/')[-1]
    ID_COUNTER=0
    
    result = []
    provenance = {}
    # TODO: have a list of these that is accessible to user
    # this is useful for generalization. 
    doc_specific = {"EOL": ':',
                    "verse_tag" : None,
                    "chapter_tag": None,
                    "verse_tag": None,
                    "to_ignore": None}
    

    # HACK: grab book title
    title = xml.findall(".//{*}titleStmt/{*}title")[-1].text

    # adding dataset metadata
    if title.endswith(".DH"):
        provenance["subset"] = "DH"
        title = title[:-3]
    if title in ['Isaiah', 'Psalms']:
        provenance["subset"] = title

    provenance["book"] = title
    # lang = xml.find(".//{*}language").attrib['ident'].lower()
    provenance["language"] = args.language

    for chapter in xml.findall(".//{*}c"):
        for i, child in enumerate(chapter):
            item = {
                "text" : [], 
                "id": f'{PREFIX}_{ID_COUNTER}',
                "provenance": provenance
            }
            if child.tag not in ['v', 'vs']:
                item["provenance"][child.tag] = child.text
            item["provenance"].update(chapter.attrib)
            item["provenance"]['chapter'] = item["provenance"]['n']
        
            item["provenance"].update(child.attrib)
            item["provenance"]['verse'] = item["provenance"]['n']
            
            del item["provenance"]['n']
    
            ID_COUNTER += 1
            
            # HACK: for whatever reason, it flips the text? I manually reverse it again
            # TODO: check which way around this is meant to be, could be a viewing error
            # line = [x[::-1].strip() for x in verse.itertext()]
            line = [x.strip() for x in child.itertext()]
    
            # remove the empty entries
            while("" in line):
                line.remove("")
            
            # remove the ellipses
            while("." in line):
                line.remove(".")

            line = " ".join(line)
            line = line.replace('\n','')
            item["text"] = line

            # HACK: for some reason, this is necessary
            item = {k : ({sk : sv for sk, sv in v.items()} if isinstance(v, dict) else v) for k, v in item.items()}
            result.append(item)
    # slightly different processing
    if title == 'Psalms':
        result = psalm_superscription_parse(result)
    return result


def psalm_superscription_parse(items: list[Dict]) -> list[Dict]:
    superscription_ids = []
    result = items
    return result

# TODO: not working yet, need to fix this
def parse_txt(text):

    PREFIX = ''.join(args.primary_sources.split('.')[:-1]).split('/')[-1]
    ID_COUNTER=0
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

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--primary_sources", dest="primary_sources", default="data/tirant_lo_blanc.txt")
    parser.add_argument("--documents", dest="documents")
    parser.add_argument("--language", dest="language", default="catalan")
    parser.add_argument("--aggregation_method", dest="aggregation_method", choices=['line', 'verse', 'chapter', 'book'], 
                        help="What level of granularity to group the text spans", default='chapter')

    args = parser.parse_args()

    logger.setLevel(logging.INFO)
 
    results = []
    if args.primary_sources.endswith("tgz"):
        tf = tarfile.open(args.primary_sources, "r:gz")
        for fname in tf.getnames()[0:4]:
            if fname.endswith(("tei", "xml")):
                logger.info('Parsing {}'.format(fname))
                xml = etree.parse(tf.extractfile(fname))
                results.extend(parse_xml(xml))

    # elif args.primary_sources.endswith("json.gz"):               
    #     with gzip.open(args.primary_sources, "rt") as ifd:
    #         results.extend(json.loads(ifd.read()))
    # elif args.primary_sources.endswith(".txt"):
    #     with open(args.primary_sources, "rt") as ifd:
    #         text = ifd.read()
    #     results.extend(parse_txt(text))
    else:
        raise Exception("The input document '{}' does not appear to be in a recognized format (either the extension is unknown, or you need to add handling logic for it to 'scripts/divide_documents.py')".format(args.primary_sources))


    
    # aggregating
    results = aggregate(results, args.aggregation_method)
    
    
    ##### saving the result to disk ####

    with gzip.open(f'{args.documents}', "wt") as ofd:
        ofd.write(json.dumps(results, indent=4))
        logger.info("Wrote output to '%s'", args.representations)

    # json_str = json.dumps(agg_results, indent=4) 

    # with open(f'{args.documents}_{args.aggregation_method}', "wt") as ofd:
    #     ofd.write(json_str)
    

