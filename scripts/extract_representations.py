import argparse
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
import numpy as np
import textstat
from numpy import array, average
import spacy
# python -m spacy download en_core_web_trf

parser = argparse.ArgumentParser()
parser.add_argument("--subdocuments", dest="subdocuments", nargs="+")
parser.add_argument("--representations", dest="representations")
parser.add_argument("--lowercase", dest="lowercase", default=False, action="store_true")
parser.add_argument("--minimum_count", dest="minimum_count", default=1)
parser.add_argument("--num_features_to_keep", dest="num_features_to_keep", type=int, default=200)
parser.add_argument("--feature_selection_method", dest="feature_selection_method", choices=["stopwords", "frequency"], default="stopwords")
args = parser.parse_args()

# Load English tokenizer, tagger, parser and NER
nlp = spacy.load("en_core_web_trf")


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
                tokenized_subdoc = word_tokenize(subdocument)
                tokens = [w.lower() for w in tokenized_subdoc] if lowercase else tokenized_subdoc
                for word in set(tokens):
                    document_frequencies[word] = document_frequencies.get(word, 0) + 1
                    author_frequencies[word] = author_frequencies.get(word, set())
                    author_frequencies[word].add(author)
                    title_frequencies[word] = title_frequencies.get(word, set())
                    title_frequencies[word].add(title)
                # Use below commented code if use old divide_document.py (which tokenizes all the words)
                # subdocument = [w.lower() for w in subdocument] if lowercase else subdocument
                # for word in set(subdocument):
                    # document_frequencies[word] = document_frequencies.get(word, 0) + 1
                    # author_frequencies[word] = author_frequencies.get(word, set())
                    # author_frequencies[word].add(author)
                    # title_frequencies[word] = title_frequencies.get(word, set())
                    # title_frequencies[word].add(title)
    scored_words = []
    for word, document_count in document_frequencies.items():
        scored_words.append((document_count / total_document_count, word))
    top_words = sorted(scored_words, reverse=True)[0:number_of_words]
    return [w for _, w in top_words]

word_list = stopwords.words("english") if args.feature_selection_method == "stopwords" else frequency_based_list(args.subdocuments, args.lowercase, args.num_features_to_keep)

# Get readability scores
def readability_features(subdoc):
    textstat_tests = ['Flesch reading ease','Smog index','Flesch-Kincaid grade','Coleman-Liau index','Automated readability index','Dale-Chall readability score','Difficult words','Linsear write formula','Gunning Fog index']
    textstat_scores = [textstat.flesch_reading_ease(subdoc),
                         textstat.smog_index(subdoc),
                         textstat.flesch_kincaid_grade(subdoc),
                         textstat.coleman_liau_index(subdoc),
                         textstat.automated_readability_index(subdoc),
                         textstat.dale_chall_readability_score(subdoc),
                         textstat.difficult_words(subdoc),
                         textstat.linsear_write_formula(subdoc),
                         textstat.gunning_fog(subdoc)]
    return dict(zip(textstat_tests, textstat_scores))

# Functions for checking_lists
def count_occurence(check_word_list, word_list_all):
    num_count = 0
    for w in check_word_list:
        if w in word_list_all:
            num_count += word_list_all[w]
    return num_count

def count_occurence_phrase(phrase_list, str_spacy_doc):
    num_count = 0
    for phrase in phrase_list:
        num_count += str_spacy_doc.lower().count(phrase)
    return num_count

# Get properties of sentences
def sentence_props(spacy_doc):
    sent_lengths = ['0-10','10-20','20-30','30-40','40-50','>50']
    sent_length_list = [0, 0, 0, 0, 0, 0]  # 0-10,10-20,20-30,30-40,40-50,>50
    for sent in spacy_doc.sents:
        num_tokens_sent = len(sent)
        if num_tokens_sent >= 50:
            sent_length_list[-1] += 1
        else:
            sent_length_list[int(num_tokens_sent / 10)] += 1
    sent_props = dict(zip(sent_lengths, sent_length_list))
    sent_props['# sentences'] = len(list(spacy_doc.sents))
    return sent_props

# Get stylistic choices and (augmented) function words
def checking_lists(str_spacy_doc, word_dict):

    # Stylistic choices (taken from PAN21 winner)
    with open('./scripts/comparison_lists.json', 'r') as f:
        comparison_dict = json.load(f)

    comparison_counts = [count_occurence(comparison_dict['low numbers'][0], word_dict),
                        count_occurence(comparison_dict['low numbers'][1], word_dict),
                        count_occurence(comparison_dict['high numbers'][0], word_dict),
                        count_occurence(comparison_dict['high numbers'][1], word_dict),
                        count_occurence(comparison_dict['ordinal numbers'][0], word_dict),
                        count_occurence(comparison_dict['ordinal numbers'][1], word_dict),
                        count_occurence(comparison_dict['spelling'][0], word_dict),
                        count_occurence(comparison_dict['spelling'][1], word_dict),
                        count_occurence_phrase(comparison_dict['contractions'][0], str_spacy_doc),
                        count_occurence_phrase(comparison_dict['contractions'][1], str_spacy_doc)]
    
    comparison_names = ['low digits','low numbers','high digits','high numbers','ordinal digits','ordinal numbers','British','American','contracted','not contracted']
    comparison_feats = dict(zip(comparison_names, comparison_counts))
    
    # Function words (taken from PAN21 winner)
    with open('./scripts/function_words.json', 'r') as f:
        function_words = json.load(f) #augmented from the NLTK stopwords list

        function_word_feature = {}
        function_phrase_feature = {}
        for w in function_words['words']:
            if w in word_dict and word_dict[w] != 0:
                function_word_feature[w] = word_dict[w]
        
        function_phrase_feature = {p: str_spacy_doc.lower().count(p) for p in function_words['phrases'] if str_spacy_doc.lower().count(p) != 0}

    function_features = function_word_feature | function_phrase_feature
    return comparison_feats, function_features          

# Get special character frequencies
def character_freqs(str_spacy_doc): 
    special_chars = ['~', '@', '#', '$', '%', '^', '&', '*', '_', '+', '=', '<', '>', '/', '\\', '|', ' ']
    special_char_counts = [str_spacy_doc.count(char) for char in special_chars]
    special_char_freqs = dict(zip(special_chars, special_char_counts))
    return special_char_freqs

# Get punctuation mark frequencies
def punctuation_freqs(tokens): 
    punct_marks = [ '.', '?', '!', ',', ';', ':', '-', '--', '---', '..', '...', '(', ')', '[', ']', '{', '}', '\'', '"', '`']
    punct_counts = [tokens.count(punct) for punct in punct_marks]
    punct_freqs = dict(zip(punct_marks, punct_counts))
    return punct_freqs 

# Get POS tag frequencies and word properties 
def postag_freqs(spacy_doc):
    pos_categories = ['ADJ','ADP','ADV','AUX','CCONJ','DET','INTJ','NOUNs','NUM','PART','PRON','PUNCT','SCONJ','SYM','VERB','X']
    postag_list = [0] * 16
    word_properties = ['long words','short words','all caps','uppercase']
    wordprop_list = [0] * 4

    for token in spacy_doc:
        if token.pos_ in ['ADJ']:
            postag_list[0] += 1
        elif token.pos_ in ['ADP']:
            postag_list[1] += 1
        elif token.pos_ in ['ADV']:
            postag_list[2] += 1
        elif token.pos_ in ['AUX']:
            postag_list[3] += 1
        elif token.pos_ in ['CCONJ']:
            postag_list[4] += 1
        elif token.pos_ in ['DET']:
            postag_list[5] += 1
        elif token.pos_ in ['INTJ']:
            postag_list[6] += 1
        elif token.pos_ in ['NOUN', 'PROPN']:
            postag_list[7] += 1
        elif token.pos_ in ['NUM']:
            postag_list[8] += 1
        elif token.pos_ in ['PART']:
            postag_list[9] += 1
        elif token.pos_ in ['PRON']:
            postag_list[10] += 1
        elif token.pos_ in ['PUNCT']:
            postag_list[11] += 1
        elif token.pos_ in ['SCONJ']:
            postag_list[12] += 1
        elif token.pos_ in ['SYM']:
            postag_list[13] += 1
        elif token.pos_ in ['VERB']:
            postag_list[14] += 1
        elif token.pos_ in ['X']:
            postag_list[15] += 1
    
        # Word properties
        if len(token.text) >= 15: #arbitrary
            wordprop_list[0] += 1
        elif len(token.text) in [2, 3, 4]: #not 1 to exclude punctuation
            wordprop_list[1] += 1
        if token.text.isupper():
            wordprop_list[2] += 1
        elif token.text[0].isupper():
            wordprop_list[3] += 1

    postag_dict = dict(zip(pos_categories, postag_list))
    wordprop_dict = dict(zip(word_properties, wordprop_list))
    return postag_dict, wordprop_dict
       
###
# Extract all features
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
                "features" : {}
            }

            # Subdocument (list of strings) features
            item["features"]["readability"] = readability_features(subdocument) 

            # Run SpaCy on the subdocument
            spacy_doc = nlp(subdocument)

            # Get different versions of spacy_doc
            str_spacy_doc = str(spacy_doc)
            tokens = [token.text.lower() for token in spacy_doc]
            word_dict = {}
            for token in spacy_doc:  
                word_dict.setdefault(token.text, 0)
                word_dict[token.text] += 1  

            # Extract features
            item['features']["sentence properties"] = sentence_props(spacy_doc)
 
            comparison_list, function_list = checking_lists(str_spacy_doc, word_dict)
            item['features']["comparisons"] = comparison_list
            item['features']["function words"] = function_list

            item["features"]["special chars"] = character_freqs(str_spacy_doc) 
            item["features"]["punctuation"] = punctuation_freqs(tokens)
            
            pos_dict, word_prop_dict = postag_freqs(spacy_doc)
            item["features"]["POS"] = pos_dict
            num_words = len(tokens)
            word_prop_dict["# words"] = num_words
            num_tokens = len(word_dict)
            word_prop_dict["# tokens"] = num_tokens
            item["features"]["word properties"] = word_prop_dict

            # for word in tokenized_subdoc:
            #     word = word.lower() if args.lowercase else word
                
            #     item["representation"][word] = item["representation"].get(word, 0) + 1
            # item["representation"] = {k : v / len(subdocument) for k, v in item["representation"].items() if v >= args.minimum_count and k in word_list}
            
            items.append(item)
            
with open(args.representations, "wt") as ofd:
    ofd.write(json.dumps(items, indent=4))
