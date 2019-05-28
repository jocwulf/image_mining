#!/usr/bin/env python
#
# Copyright 2017 The Open Images Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Classifier inference utility.

This code takes a resnet_v1_101 checkpoint, runs the classifier on the image and
prints predictions in human-readable form.

-------------------------------
Example command:
-------------------------------

# 0. Create directory for model/data
WORK_PATH="/tmp/oidv2"
mkdir -p "${WORK_PATH}"
cd "${WORK_PATH}"

# 1. Download the model, inference code, and sample image
wget https://storage.googleapis.com/openimages/2017_07/classes-trainable.txt
wget https://storage.googleapis.com/openimages/2017_07/class-descriptions.csv
wget https://storage.googleapis.com/openimages/2017_07/oidv2-resnet_v1_101.ckpt.tar.gz
wget https://raw.githubusercontent.com/openimages/dataset/master/tools/classify_oidv2.py
tar -xzf oidv2-resnet_v1_101.ckpt.tar.gz

wget -O cat.jpg https://farm6.staticflickr.com/5470/9372235876_d7d69f1790_b.jpg

# 2. Run inference
python classify_oidv2.py \
--checkpoint_path='D:\Dominique_Code\oidv2-resnet_v1_101.ckpt' \
--labelmap='classes-trainable.txt' \
--dict='class-descriptions.csv' \
--image="cat.jpg" \
--top_k=10 \
--score_threshold=0.3

# Sample output:
Image: "cat.jpg"

3272: /m/068hy - Pet (score = 0.96)
1076: /m/01yrx - Cat (score = 0.95)
0708: /m/01l7qd - Whiskers (score = 0.90)
4755: /m/0jbk - Animal (score = 0.90)
2847: /m/04rky - Mammal (score = 0.89)
2036: /m/0307l - Felidae (score = 0.79)
3574: /m/07k6w8 - Small to medium-sized cats (score = 0.77)
4799: /m/0k0pj - Nose (score = 0.70)
1495: /m/02cqfm - Close-up (score = 0.55)
0036: /m/012c9l - Domestic short-haired cat (score = 0.40)

-------------------------------
Note on image preprocessing:
-------------------------------

This is the code used to perform preprocessing:
--------
from preprocessing import preprocessing_factory

def PreprocessImage(image, network='resnet_v1_101', image_size=299):
  # If resolution is larger than 224 we need to adjust some internal resizing
  # parameters for vgg preprocessing.
  if any(network.startswith(x) for x in ['resnet', 'vgg']):
    preprocessing_kwargs = {
        'resize_side_min': int(256 * image_size / 224),
        'resize_side_max': int(512 * image_size / 224)
    }
  else:
    preprocessing_kwargs = {}
  preprocessing_fn = preprocessing_factory.get_preprocessing(
      name=network, is_training=False)

  height = image_size
  width = image_size
  image = preprocessing_fn(image, height, width, **preprocessing_kwargs)
  image.set_shape([height, width, 3])
  return image
--------

Note that there appears to be a small difference between the public version
of slim image processing library and the internal version (which the meta
graph is based on). Results that are very close, but not exactly identical to
that of the metagraph.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from pymongo import MongoClient
import time
import argparse
import six.moves.urllib as urllib


########set group IDs here
flickr_group_ids=[""]

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('labelmap', 'classes-trainable.txt',
                    'Labels, one per line.')

flags.DEFINE_string('dict', 'class-descriptions.csv',
                    'Descriptive string for each label.')

flags.DEFINE_string('checkpoint_path', 'oidv2-resnet_v1_101.ckpt',
                    'Path to checkpoint file.')

flags.DEFINE_string('image', '',
                    'Comma separated paths to image files on which to perform '
                    'inference.')

flags.DEFINE_integer('top_k', 5000, 'Maximum number of results to show.')

flags.DEFINE_float('score_threshold',0.005 , 'Score threshold.') #None


def LoadLabelMap(labelmap_path, dict_path):
  """Load index->mid and mid->display name maps.

  Args:
    labelmap_path: path to the file with the list of mids, describing
        predictions.
    dict_path: path to the dict.csv that translates from mids to display names.
  Returns:
    labelmap: an index to mid list
    label_dict: mid to display name dictionary
  """
  labelmap = [line.rstrip() for line in tf.gfile.GFile(labelmap_path)]

  label_dict = {}
  for line in tf.gfile.GFile(dict_path):
    words = [word.strip(' "\n') for word in line.split(',', 1)]
    label_dict[words[0]] = words[1]

  return labelmap, label_dict

# Load labelmap and dictionary from disk.
labelmap, label_dict = LoadLabelMap(FLAGS.labelmap, FLAGS.dict)

g = tf.Graph()
with g.as_default():
  with tf.Session() as sess:
    saver = tf.train.import_meta_graph(FLAGS.checkpoint_path + '.meta')
    saver.restore(sess, FLAGS.checkpoint_path)

    input_values = g.get_tensor_by_name('input_values:0')
    predictions = g.get_tensor_by_name('multi_predictions:0')
    
    ####OFFLINE###########
    client = MongoClient('localhost:27017')
    db = client.flickr

    #for image_filename in FLAGS.image.split(','):
    for group_id in flickr_group_ids:
      print("starting oid processing group_id"+str(group_id))        
      photoid_dicts = list()
      for entry in db.flickr.find({"oid":{"$exists":0},'saved_image_extension':{'$exists':1}},{"id":1,"saved_image_extension":1}):

  
          photoid_dicts.append(entry)
  
      for photoid_dict in photoid_dicts:
          try:
              #start_time = time.time()
              
              image_filename = "/mnt/volume-fra1-01/flickr/picextraction/"+photoid_dict['id']+"."+photoid_dict['saved_image_extension']
                      
              compressed_image = tf.gfile.FastGFile(image_filename, 'rb').read()
              predictions_eval = sess.run(
                  predictions, feed_dict={
                      input_values: [compressed_image]
                  })
              top_k = predictions_eval.argsort()[::-1]  # indices sorted by score
              if FLAGS.top_k > 0:
                top_k = top_k[:FLAGS.top_k]
              if FLAGS.score_threshold is not None:
                top_k = [i for i in top_k
                         if predictions_eval[i] >= FLAGS.score_threshold]
              #print('Image: "%s"\n' % image_filename)
              resultdict = dict()
              for idx in top_k:
                mid = labelmap[idx]
                display_name = label_dict[mid]
                score = predictions_eval[idx]
                resultdict[display_name] = score.item()

              #print(str(resultdict))
              db.flickr.update_one({'_id': photoid_dict['_id']}, {"$set":{"oid":resultdict}}, upsert=False)
      
              #print('Iteration: %.3f sec'%(time.time()-start_time))
          except Exception as e:
              print("ERROR: problems with oid mining, skipping image :"+str(photoid_dict["id"]))
              continue        

