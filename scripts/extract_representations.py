import argparse
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
import numpy as np
import textstat
from numpy import array, average

parser = argparse.ArgumentParser()
parser.add_argument("--subdocuments", dest="subdocuments", nargs="+")
parser.add_argument("--representations", dest="representations")
parser.add_argument("--lowercase", dest="lowercase", default=False, action="store_true")
parser.add_argument("--minimum_count", dest="minimum_count", default=1)
parser.add_argument("--num_features_to_keep", dest="num_features_to_keep", type=int, default=200)
parser.add_argument("--feature_selection_method", dest="feature_selection_method", choices=["stopwords", "frequency"], default="stopwords")
args = parser.parse_args()

def frequency_based_list(fnames, lowercase, number_of_words):
    document_frequencies = {}
    author_frequencies = {}
    title_frequencies = {}
    total_document_count = 0
    unique_authors = set()
    unique_titles = set()
    for fname in fnames:
        with open(fname, "rt") as ifd:
            text = json.loads(ifd.read())
            author = text["author"]
            title = text["title"]
            unique_authors.add(author)
            unique_titles.add(title)
            total_document_count += len(text["subdocuments"])
            for subdocument in text["subdocuments"]:
                subdocument = [w.lower() for w in subdocument] if lowercase else subdocument
                for word in set(subdocument):
                    document_frequencies[word] = document_frequencies.get(word, 0) + 1
                    author_frequencies[word] = author_frequencies.get(word, set())
                    author_frequencies[word].add(author)
                    title_frequencies[word] = title_frequencies.get(word, set())
                    title_frequencies[word].add(title)
    scored_words = []
    for word, document_count in document_frequencies.items():
        scored_words.append((document_count / total_document_count, word))
    top_words = sorted(scored_words, reverse=True)[0:number_of_words]
    return [w for _, w in top_words]

word_list = stopwords.words("english") if args.feature_selection_method == "stopwords" else frequency_based_list(args.subdocuments, args.lowercase, args.num_features_to_keep)

#####
def character_freqs(tokens): #, number_of_features):
    special_chars = ['~', '@', '#', '$', '%', '^', '&', '*', '_', '+', '=', '<', '>', '/', '\\', '|']
    special_char_counts = [tokens.count(char) for char in special_chars]
    special_char_freqs = dict(zip(special_chars, special_char_counts))

    punct_marks = [ '.', '?', '!', ',', ';', ':', '-', '(', ')', '[', ']', '{', '}', '\'', '"', '`']
    punct_counts = [tokens.count(punct) for punct in punct_marks]
    punct_freqs = dict(zip(punct_marks, punct_counts))
    
    return special_char_freqs, punct_freqs 

def postag_freqs(tokens):
    postag_dict = {}
    for (word, tag) in pos_tag(tokens):

        # TODO: add punct tags?
        #if word == '-' and tag == ':':
        #    eos_hypen += 1

        # some overlap possible in these tag categories
        if tag in ['DT', 'PDT', 'WDT']:
            postag_dict['Det'] += 1
        if tag in ['POS', 'PRP$', 'WP$']:
            postag_dict['Poss'] += 1
        if tag in ['PRP', 'PRP$', 'WP', 'WP$']:
            postag_dict['Pron'] += 1
        if tag in ['RB', 'RBR', 'RBS', 'WRB']:
            postag_dict['RB*'] += 1
        if tag in ['WDT', 'WP', 'WP$', 'WRB']:
            postag_dict['Wh*'] += 1

        # no tag category overlap
        elif tag in ['CC']:
            postag_dict['CC'] += 1
        elif tag in ['CD']:
            postag_dict['CD'] += 1
        elif tag in ['EX']:
            postag_dict['EX'] += 1
        elif tag in ['FW']:
            postag_dict['FW'] += 1
        elif tag in ['IN']:
            postag_dict['IN'] += 1
        elif tag.startswith('J'):
            postag_dict['JJ*'] += 1
        elif tag in ['LS']:
            postag_dict['LS'] += 1
        elif tag in ['MD']:
            postag_dict['MD'] += 1
        elif tag.startswith('N'):
            postag_dict['NN*'] += 1
        elif tag in ['RP']:
            postag_dict['RP'] += 1
        elif tag in ['SYM']:
            postag_dict['SYM'] += 1
        elif tag in ['TO']:
            postag_dict['TO'] += 1
        elif tag in ['UH']:
            postag_dict['UH'] += 1
        elif tag.startswith('V'):
            postag_dict['VB*'] += 1
        
        if len(word) >= 8:
            postag_dict['long words'] += 1
        elif len(word) in [2, 3, 4]:
            postag_dict['short words'] += 1
        if word.isupper():
            postag_dict['all caps'] += 1
        elif word[0].isupper():
            postag_dict['uppercase'] += 1
    
    return postag_dict

#def comparison_lists():
    #return

def readability_features(subdoc):
    textstat_scores = [textstat.flesch_reading_ease(subdoc),
                         textstat.smog_index(subdoc),
                         textstat.flesch_kincaid_grade(subdoc),
                         textstat.coleman_liau_index(subdoc),
                         textstat.automated_readability_index(subdoc),
                         textstat.dale_chall_readability_score(subdoc),
                         textstat.difficult_words(subdoc),
                         textstat.linsear_write_formula(subdoc),
                         textstat.gunning_fog(subdoc)]
    return textstat_scores

#####
items = []
for fname in args.subdocuments:
    with open(fname, "rt") as ifd:
        text = json.loads(ifd.read())
        author = text["author"]
        title = text["title"]
        for subdocument in text["subdocuments"]:
            item = {
                "author" : author,
                "title" : title,
                "representation" : {},
                "features" : {}
            }
            tokenized_subdoc = word_tokenize(subdocument)
            spec_char_dict, punct_dict = character_freqs(tokenized_subdoc) 

            item["features"]["readability"] = readability_features(subdocument) 
            item["features"]["special charss"] = spec_char_dict
            item["features"]["punctuation"] = punct_dict
            item["features"]["POS"] = postag_freqs(tokenized_subdoc)

            for word in subdocument:
                word = word.lower() if args.lowercase else word
                
                item["representation"][word] = item["representation"].get(word, 0) + 1
            item["representation"] = {k : v / len(subdocument) for k, v in item["representation"].items() if v >= args.minimum_count and k in word_list}
            
            
            
            items.append(item)
            
        
with open(args.representations, "wt") as ofd:
    ofd.write(json.dumps(items, indent=4))
