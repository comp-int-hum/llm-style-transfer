# import tensorflow_datasets as tfds
import tensorflow as tf
import author_id_datasets
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--tfds_path", dest="tfds_path")
parser.add_argument("--representations", dest="representations")

args = parser.parse_args()

# Load the `train_query` split
# dataset = tf.data.experimental.load(tfds_path, split="train_query", shuffle=true)
# dataset = tfds.load("reddit_user_id", split="train_query", shuffle_files=True)
dataset = tf.data.experimental.load(args.tfds_path, 
            element_spec={'subreddit': tf.TensorSpec(shape=(None,), dtype=tf.string, name=None), 
                          'body': tf.TensorSpec(shape=(None,), dtype=tf.string, name=None), 
                          'user_id': tf.TensorSpec(shape=(), dtype=tf.string, name=None), 
                          'created_utc': tf.TensorSpec(shape=(None,), dtype=tf.string, name=None)})


# tf.TensorSpec(shape=(16,), dtype=tf.dtypes.string)
# print(dataset)


# The returned object works as a Python iterator. Below, we take the
# first element and print the first comment by the first user.
items = []
for elem in dataset.take(2):
    item = {"text":elem['body'],
          "author_id": elem['user_id'],
            "topic": elem['subreddit']}
    print(item)

# for thing in dataset.take(5):
#     print(thing)
  
#   print(user)

# You can see the features associated with each user by printing
# the `element_spec` property
# print(dataset.element_spec)

# okay so it's text, author_id, topic

# with open(args.representations, "wt") as ofd:
#     ofd.write(json.dumps(items, indent=4))

