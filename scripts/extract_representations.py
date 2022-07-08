import argparse
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
import numpy as np
import textstat
from numpy import array, average

parser = argparse.ArgumentParser()
parser.add_argument("--subdocuments", dest="subdocuments")
parser.add_argument("--representations", dest="representations")
parser.add_argument("--lowercase", dest="lowercase", default=False, action="store_true")
parser.add_argument("--minimum_count", dest="minimum_count", default=1)
parser.add_argument("--num_features_to_keep", dest="num_features_to_keep", type=int, default=200)
parser.add_argument("--feature_selection_method", dest="feature_selection_method", choices=["stopwords", "frequency"], default="stopwords")
args = parser.parse_args()

def count_occurence(check_word_list, word_list_all):
    num_count = 0
    for w in check_word_list:
        if w in word_list_all:
            num_count += word_list_all[w]
    return num_count

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
            author = text["author_id"]
            title = text["subreddit"]
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

##--- Subdocument level features
# Get properties of sentences
def sentence_props(subdoc):
    sent_list = sent_tokenize(subdoc)
    sent_length_list = [0, 0, 0, 0, 0, 0]  # 0-10,10-20,20-30,30-40,40-50,>50
    for sent in sent_list:
        word_list = word_tokenize(sent)
        num_words_sent = len(word_list)
        if num_words_sent >= 50:
            sent_length_list[-1] += 1
        else:
            sent_length_list[int(num_words_sent / 10)] += 1

    props = ['# sentences', 'sentence lengths']
    sent_props = [len(sent_list), sent_length_list]
    sent_props_dict = dict(zip(props, sent_props))
    return sent_props_dict

# Get stylistic choices and (augmented) function words
def checking_lists(subdoc):
    word_dict = {}
    sent_list = sent_tokenize(subdoc)
    for sent in sent_list:
        word_list = word_tokenize(sent.lower())
        for word in word_list:
            word_dict.setdefault(word, 0)
            word_dict[word] += 1
    
    # Stylistic choices (taken from PAN21 winner)
    with open('./scripts/comparison_lists.json', 'r') as f:
        comparison_dict = json.load(f)

    comparison_counts = [count_occurence(comparison_dict['numbers'][0], word_dict),
                        count_occurence(comparison_dict['numbers'][1], word_dict),
                        count_occurence(comparison_dict['spelling'][0], word_dict),
                        count_occurence(comparison_dict['spelling'][1], word_dict),
                        count_occurence(comparison_dict['contractions'][0], word_dict),
                        count_occurence(comparison_dict['contractions'][1], word_dict)]
    
    comparison_names = ['digits','numbers','British','American','contracted','not contracted']
    comparison_feats = dict(zip(comparison_names, comparison_counts))
    
    # Function words (taken from PAN21 winner)
    with open('./scripts/function_words.json', 'r') as f:
        function_words = json.load(f) #augmented from the NLTK stopwords list

        function_word_feature = []
        for w in function_words['words']:
            if w in word_dict:
                function_word_feature.append(word_dict[w])
            else:
                function_word_feature.append(0)
            if w in word_dict:
                function_word_feature.append(word_dict[w])
            else:
                function_word_feature.append(0)
        function_phrase_feature = [subdoc.lower().count(p) for p in function_words['phrases']]

    function_features = [function_word_feature, function_phrase_feature]
    return comparison_feats, function_features          

# Get readability scores
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


##--- Tokenized level features
# Get special character and punctuation mark frequencies
def character_freqs(tokens): #, number_of_features):
    special_chars = ['~', '@', '#', '$', '%', '^', '&', '*', '_', '+', '=', '<', '>', '/', '\\', '|']
    special_char_counts = [tokens.count(char) for char in special_chars]
    special_char_freqs = dict(zip(special_chars, special_char_counts))

    punct_marks = [ '.', '?', '!', ',', ';', ':', '-', '--', '---', '...', '(', ')', '[', ']', '{', '}', '\'', '"', '`']
    punct_counts = [tokens.count(punct) for punct in punct_marks]
    punct_freqs = dict(zip(punct_marks, punct_counts))
    
    return special_char_freqs, punct_freqs 

# Get POS tag frequencies and word properties
def postag_freqs(tokens):
    pos_categories = ['Det','Poss','Pron','Adv','Wh','CC','CD','EX','FW','IN','Adj','LS','MD','Noun','RP','SYM','TO','UH','Verb']
    postag_list = [0] * 19
    word_properties = ['long words','short words','all caps','uppercase']
    wordprop_list = [0] * 4
    for (word, tag) in pos_tag(tokens):
        # TODO: add punct tags?
        #if word == '-' and tag == ':':
        #    eos_hypen += 1

        # some overlap possible in these tag categories
        if tag in ['DT', 'PDT', 'WDT']:
            postag_list[0] += 1
        if tag in ['POS', 'PRP$', 'WP$']:
            postag_list[1] += 1
        if tag in ['PRP', 'PRP$', 'WP', 'WP$']:
            postag_list[2] += 1
        if tag in ['RB', 'RBR', 'RBS', 'WRB']:
            postag_list[3] += 1
        if tag in ['WDT', 'WP', 'WP$', 'WRB']:
            postag_list[4] += 1

        # no tag category overlap
        elif tag in ['CC']:
            postag_list[5] += 1
        elif tag in ['CD']:
            postag_list[6] += 1
        elif tag in ['EX']:
            postag_list[7] += 1
        elif tag in ['FW']:
            postag_list[8] += 1
        elif tag in ['IN']:
            postag_list[9] += 1
        elif tag.startswith('J'):
            postag_list[10] += 1
        elif tag in ['LS']:
            postag_list[11] += 1
        elif tag in ['MD']:
            postag_list[12] += 1
        elif tag.startswith('N'):
            postag_list[13] += 1
        elif tag in ['RP']:
            postag_list[14] += 1
        elif tag in ['SYM']:
            postag_list[15] += 1
        elif tag in ['TO']:
            postag_list[16] += 1
        elif tag in ['UH']:
            postag_list[17] += 1
        elif tag.startswith('V'):
            postag_list[18] += 1
    
        # Word properties
        if len(word) >= 15: #arbitrary
            wordprop_list[0] += 1
        elif len(word) in [2, 3, 4]:
            wordprop_list[1] += 1
        if word.isupper():
            wordprop_list[2] += 1
        elif word[0].isupper():
            wordprop_list[3] += 1
    
    postag_dict = dict(zip(pos_categories, postag_list))
    wordprop_dict = dict(zip(word_properties, wordprop_list))
    return postag_dict, wordprop_dict
        
###
items = []
# rather than each doc being specific to an author, it's a list of all of them
with open(args.subdocuments, "rt") as ifd:
    examples = json.loads(ifd.read())
    for text in examples:
        author = text["author_id"]
        title = text["subreddit"]
        for idx, subdocument in enumerate(text["texts"]):
            item = {
                "author" : author,
                "title" : title[idx],
                "representation" : {},
                "features" : {}
            }
          
            # Subdocument (list of strings) features
            item["features"]["readability"] = readability_features(subdocument) 
            item['features']["sentence properties"] = sentence_props(subdocument)
            comparison_list, function_list = checking_lists(subdocument)
            item['features']["comparisons"] = comparison_list
            item['features']["function words"] = function_list

            # Tokenized doc (list of tokens) features
            tokenized_subdoc = word_tokenize(subdocument)

            spec_char_dict, punct_dict = character_freqs(tokenized_subdoc) 
            item["features"]["special chars"] = spec_char_dict
            item["features"]["punctuation"] = punct_dict
            pos_dict, word_prop_dict = postag_freqs(tokenized_subdoc)
            item["features"]["POS"] = pos_dict
            item["features"]["word properties"] = word_prop_dict

            for word in tokenized_subdoc:
                word = word.lower() if args.lowercase else word
                item["representation"][word] = item["representation"].get(word, 0) + 1
            item["representation"] = {k : v / len(subdocument) for k, v in item["representation"].items() if v >= args.minimum_count and k in word_list}
            items.append(item)
            
with open(args.representations, "wt") as ofd:
    ofd.write(json.dumps(items, indent=4))
