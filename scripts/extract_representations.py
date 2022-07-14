import gzip
import argparse
import json
import numpy as np
import textstat
from collections import Counter
from numpy import array, average
import spacy
from tqdm import tqdm
import transformers
import torch
import time
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
# python -m spacy download en_core_web_trf

# TODO: add BERT support, depending on language flag, and fluency score

parser = argparse.ArgumentParser()
parser.add_argument("--subdocuments", dest="subdocuments")
parser.add_argument("--representations", dest="representations")
parser.add_argument("--language", dest="language", )
args = parser.parse_args()

# Load SpaCy English tokenizer, tagger, parser and NER
nlp = spacy.load("en_core_web_trf")

## Currently not being used; uses nltk
def frequency_based_list(fnames, lowercase, number_of_words):
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk import pos_tag

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

# word_list = stopwords.words("english") if args.feature_selection_method == "stopwords" else frequency_based_list(args.subdocuments, args.lowercase, args.num_features_to_keep)
### --- NEURAL FEATURES --- ###
def fluency_score(subdoc,model,tokenizer,):

    inputs = tokenizer(subdoc, return_tensors="pt")
    
    with torch.no_grad():
        logits = model(**inputs).logits
    
    # convert logits to probabilities with softmax
    m = torch.nn.Softmax(dim=1)
    probs = m(logits)
    # predicted_class_id = logits.argmax().item()
    # predicted_class_label = model.config.id2label[predicted_class_id]
    # return logits, predicted_class_id, predicted_class_label
    # return predicted_class_id # 0: fluent, 1: disfluent
    # just the fluency probability
    return probs[0]

def get_BERT_model(lang):
    # most of these models are publically available
    private=False
    # specifying the path to grab the model from HF Hub
    if lang == "english":
        model_name = "bert-base-uncased"
    elif lang == "hebrew":
        model_name = "onlplab/alephbert-base"
    elif lang == "latin":
        model_name = "hmcgovern/latin_bert"
        private=True
        access_token = "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"
    elif lang == "greek":
        model_name = "pranaydeeps/Ancient-Greek-BERT"
    elif lang == "arabic":
        model_name = "asafaya/bert-base-arabic"
    elif lang == "catalan":
        # this model is uncased
        model_name = "ClassCat/roberta-base-catalan"
    else:
        raise Exception("No suitable BERT model found for this language")
    model = AutoModel.from_pretrained(model_name, use_auth_token=private)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=private)
    return model, tokenizer

def get_sent_emb(subdoc, model, tokenizer):
    """Returns a BERT-style document embedding"""
    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    # should I split the subdoc into sentences?
    # this is language specific, well, the hebrew will already be Hebrew
    encoded_input = tokenizer(subdoc, padding=True, truncation=True, return_tensors='pt')
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling. In this case, max pooling.
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # print("Sentence embeddings:")
    # print(sentence_embeddings.shape)
    return sentence_embeddings.numpy()


### --- STYLOMETRIC FEATURES --- ###
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

# Get properties of sentences
def sentence_props(spacy_doc):
    sent_lengths = ['0-10','10-20','20-30','30-40','40-50','>50']
    sent_length_list = [0, 0, 0, 0, 0, 0]  # 0-10,10-20,20-30,30-40,40-50,>50 tokens (not words)
    for sent in spacy_doc.sents:
        num_tokens_sent = len(sent)
        if num_tokens_sent >= 50:
            sent_length_list[-1] += 1
        else:
            sent_length_list[int(num_tokens_sent / 10)] += 1
    sent_props = dict(zip(sent_lengths, sent_length_list))
    # sent_props['# sentences'] = len(list(spacy_doc.sents))
    return sent_props, len(list(spacy_doc.sents))

# Functions needed for checking_lists
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

# Get (augmented) stylistic choices and function words
def checking_lists(str_spacy_doc, word_dict):

    # Stylistic choices (augmented from PAN21 winner)
    with open('./scripts/comparison_lists.json', 'r') as f:
        comparison_dict = json.load(f) #augmented from the original (orig = NLTK stopwords + Zlatkova 2018)

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
    
    # Function words (augmented from PAN21 winner)
    with open('./scripts/function_words.json', 'r') as f:
        function_words = json.load(f) #augmented from the original (orig = NLTK stopwords + Zlatkova 2018)

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

# Get n most frequent tokens (NB: tokens, not words, so includes punctuation)
def most_freq_tokens(counter, min_count, num_top_tokens):
    most_freq = counter.most_common(num_top_tokens)
    most_freq_dict = {k : v for k, v in most_freq if v >= min_count}
    return most_freq_dict

####

# NOTE: we need to be more careful with the language flag here, it only makes sense to call some of these features
# if the language is English, maybe it doesn't matter, it's
# loading models that I only want to do once rather than call in a fn
# fluency_tokenizer = AutoTokenizer.from_pretrained("cointegrated/roberta-large-cola-krishna2020")
# fluency_model = AutoModelForSequenceClassification.from_pretrained("cointegrated/roberta-large-cola-krishna2020")

# bert_model, bert_tokenizer = get_BERT_model(args.language)

# Extract all features
items = []

with gzip.open(args.subdocuments, "rt") as ifd:
    # examples = json.loads(ifd.read())

    for subdocument in tqdm(json.loads(ifd.read())):
        uid = subdocument["id"]
        text = subdocument["text"]
        # language = subdocument["provenance"]["language"]
        item = {
            "id" : uid,
            "text" : text,
            "feature_sets" : {
            },
            "provenance" : subdocument["provenance"]
        }
    
        #### Neural Features ####
        # item["feature_sets"]["fluency"] = {"values": fluency_score(text,fluency_model,fluency_tokenizer)}
        # TODO: make sure this outputs to an okay format, currently a torch tensor
        # item["feature_sets"]["sentence_embedding"] = get_sent_emb(subdocument, bert_model, bert_tokenizer)
        
        #### Stylometric Features ####
        # Subdocument (list of strings) features
        item["feature_sets"]["readability"] = {"values": readability_features(text)}

        # Run SpaCy on the subdocument
        spacy_doc = nlp(text)

        # Get different versions of spacy_doc
        str_spacy_doc = str(spacy_doc)
        tokens = [token.text.lower() for token in spacy_doc]
        counter = Counter(tokens) 
        word_dict = {}
        for token in spacy_doc:  
            word_dict.setdefault(token.text, 0)
            word_dict[token.text] += 1  

        # Extract features
        sent_props, num_sents = sentence_props(spacy_doc)
        item['feature_sets']["sentence properties"] = {"values" : sent_props}
        item['feature_sets']["# sentences"]= {"values" : num_sents}

        comparison_list, function_list = checking_lists(str_spacy_doc, word_dict)
        item['feature_sets']["comparisons"] = {"values" : comparison_list}
        item['feature_sets']["function words"] = {"values" : function_list}

        item["feature_sets"]["special chars"] = {"values" : character_freqs(str_spacy_doc)} 
        
        item["feature_sets"]["punctuation"] = {"values" : punctuation_freqs(tokens)} 
        
        pos_dict, word_prop_dict = postag_freqs(spacy_doc)
        item["feature_sets"]["POS"] = {"values" : pos_dict}
        
        num_tokens = len(tokens)
        word_prop_dict["# tokens in doc"] = num_tokens #tokens not words b/c includes punct
        num_unique_tokens = len(word_dict)
        word_prop_dict["# unique tokens"] = num_unique_tokens #also includes punct
        item["feature_sets"]["word properties"] = {"values" : word_prop_dict}

        # NOTE: this never seems to return tokens, the text examples are too short
        # but keep it in for other use cases
        item["feature_sets"]["most freq tokens"] = most_freq_tokens(counter, 3, 30) #min_count, #num_top_tokens

        ## Include the below if using the function frequency_based_list (needs to be updated)
        # for word in subdocument:
        #     word = word.lower() if args.lowercase else word
            
        #     item["representation"][word] = item["representation"].get(word, 0) + 1
        # item["representation"] = {k : v / len(subdocument) for k, v in item["representation"].items() if v >= args.minimum_count and k in word_list}

        items.append(item)
            
with gzip.open(args.representations, "wt") as ofd:
    ofd.write(json.dumps(items, indent=4))

##### Tom's code #####
# representations = []
# with gzip.open(args.subdocuments, "rt") as ifd:
#     for subdocument in json.loads(ifd.read()):
#         uid = subdocument["id"]
#         text = subdocument["text"]
#         language = subdocument["provenance"]["language"]
#         representation = {
#             "id" : uid,
#             "text" : text,
#             "feature_sets" : {
#             },
#             "provenance" : subdocument["provenance"]
#         }

#         #
#         # here is where you should extract features (the first example is simple stopword-lookup)
#         #
#         if language in stopwords.fileids():
#             stopword_list = stopwords.words(language)
#             counts = {}

#             for word in word_tokenize(text.lower()):
#                 if word in stopword_list:
#                     counts[word] = counts.get(word, 0) + 1
#             representation["feature_sets"]["stopwords_from_nltk"] = {
#                 "categorical_distribution" : True,
#                 "values" : counts
#             }

#         # after all features have been extracted:
#         representations.append(representation)
        
# with gzip.open(args.representations, "wt") as ofd:
#     ofd.write(json.dumps(representations, indent=4))
