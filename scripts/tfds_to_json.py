import tensorflow as tf
# import author_id_datasets
import argparse
import json
import numpy as np
from nltk.tokenize import word_tokenize

from utils import NumpyEncoder

parser = argparse.ArgumentParser()
parser.add_argument("--tfds_path", dest="tfds_path", default="/exp/apatel/style_transfer/style_transfer_datasets/reddit_test_query/random/source_author_posts/")
parser.add_argument("--representations", dest="representations")

args = parser.parse_args()


dataset = tf.data.experimental.load(args.tfds_path, 
            element_spec={'subreddit': tf.TensorSpec(shape=(None,), dtype=tf.string, name=None), 
                          'body': tf.TensorSpec(shape=(None,), dtype=tf.string, name=None), 
                          'user_id': tf.TensorSpec(shape=(), dtype=tf.string, name=None), 
                          'created_utc': tf.TensorSpec(shape=(None,), dtype=tf.string, name=None)})


items = []
for elem in dataset:
    item = {"author_id": elem['user_id'].numpy(),
            "subreddit": elem['subreddit'].numpy(),
            "texts": [word_tokenize(x.decode()) for x in elem['body'].numpy()],
          }
    items.append(item)


with open(args.representations, "wt") as ofd:
    ofd.write(json.dumps(items, indent=4, cls=NumpyEncoder))

