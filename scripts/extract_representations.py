import gzip
import argparse
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

parser = argparse.ArgumentParser()
parser.add_argument("--subdocuments", dest="subdocuments")
parser.add_argument("--representations", dest="representations")
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

representations = []
with gzip.open(args.subdocuments, "rt") as ifd:
    for subdocument in json.loads(ifd.read()):
        uid = subdocument["id"]
        text = subdocument["text"]
        language = subdocument["provenance"]["language"]
        representation = {
            "id" : uid,
            "text" : text,
            "feature_sets" : {
            },
            "provenance" : subdocument["provenance"]
        }

        #
        # here is where you should extract features (the first example is simple stopword-lookup)
        #
        if language in stopwords.fileids():
            stopword_list = stopwords.words(language)
            counts = {}

            for word in word_tokenize(text.lower()):
                if word in stopword_list:
                    counts[word] = counts.get(word, 0) + 1
            representation["feature_sets"]["stopwords_from_nltk"] = {
                "categorical_distribution" : True,
                "values" : counts
            }

        # after all features have been extracted:
        representations.append(representation)
        
with gzip.open(args.representations, "wt") as ofd:
    ofd.write(json.dumps(representations, indent=4))
