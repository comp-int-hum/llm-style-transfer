import argparse
import json
from glob import glob
import os.path
import tensorflow as tf
import gzip

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", dest="source")
    parser.add_argument("--target", dest="target")
    parser.add_argument("--transferred", dest="transferred")
    parser.add_argument("--output", dest="output")
    args = parser.parse_args()

    source_authors = set()
    target_authors = set()
    source_documents = []

    for user in tf.data.experimental.load(args.source):
        texts = [x for x in user["body"]]
        uid = user["user_id"].numpy().decode("utf-8")
        source_authors.add(uid)
        subreddits = [x.numpy().decode("utf-8") for x in user["subreddit"]]
        utcs = [int(x) for x in user["created_utc"]]
        for text, subreddit, utc in zip(texts, subreddits, utcs):
            source_documents.append(
                {
                    "text" : text.numpy().decode("utf-8").strip(),
                    "provenance" : {
                        "author" : uid,
                        "subreddit" : subreddit
                    },
                    "id" : "reddit_style_transfer_{}".format(len(source_documents))
                }
            )

            
    target_documents = []
    for user in tf.data.experimental.load(args.target):
        texts = [x for x in user["body"]]
        uid = user["user_id"].numpy().decode("utf-8")
        target_authors.add(uid)
        subreddits = [x.numpy().decode("utf-8") for x in user["subreddit"]]
        created = [int(x) for x in user["created_utc"]]
        for text, subreddit, utc in zip(texts, subreddits, utcs):
            target_documents.append(
                {
                    "text" : text.numpy().decode("utf-8").strip(),
                    "provenance" : {
                        "author" : uid,
                        "subreddit" : subreddit,
                    },
                    "id" : "reddit_style_transfer_{}".format(len(source_documents) + len(target_documents))
                }
            )
            

    restyled_documents = {}
    for fname in glob(os.path.join(args.transferred, "*")):
        style = "_".join(os.path.basename(fname).split("_")[0:-1])
        restyled_documents[style] = []
        for user in tf.data.experimental.load(fname):            
            texts = [x for x in user["body"]]
            for text in texts: #, subreddit, utc in zip(texts, utcs):
                doc_num = len(restyled_documents[style])
                #author = source_documents[doc_num]["provenance"]["author"]
                restyled_documents[style].append(
                    {
                        "text" : text.numpy().decode("utf-8").strip(),
                        "provenance" : {
                            "original_document" : source_documents[doc_num]["id"],
                            "original_author" : source_documents[doc_num]["provenance"]["author"],
                            "style" : style
                        },
                        "id" : "reddit_style_transfer_{}".format(len(source_documents) + len(target_documents) + sum([len(x) for x in restyled_documents.values()]))
                    }
                )

    
    with gzip.open(args.output, "wt") as ofd:
        docs = source_documents + target_documents + sum(restyled_documents.values(), [])
        ofd.write(json.dumps(docs, indent=4))
