import argparse
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

import time

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops
if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

#from utils import label_map_util
from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

from pymongo import MongoClient

#####see here for the input tutorial: https://docs.python.org/2/howto/argparse.html
parser = argparse.ArgumentParser()
parser.add_argument("grouplist", help="provide list of flickr group_ids")
args = parser.parse_args()
flickr_group_ids = args.grouplist.split(",") #we need the urls separated with comma as input


##change logging level to avoid excessive logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# What model to download.

#######################################################################################
#http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
 
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
 
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
 
NUM_CLASSES = 90
#####################################################################################

#######################################################################################
#http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz
# MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28'
# MODEL_FILE = MODEL_NAME + '.tar.gz'
# DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
# 
# # Path to frozen detection graph. This is the actual model that is used for the object detection.
# PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
# 
# # List of the strings that is used to add correct label for each box.
# PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
# 
# NUM_CLASSES = 90
#####################################################################################

#######################################################################################
#http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz
# MODEL_NAME = 'faster_rcnn_resnet101_coco_2018_01_28'
# MODEL_FILE = MODEL_NAME + '.tar.gz'
# DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
# 
# # Path to frozen detection graph. This is the actual model that is used for the object detection.
# PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
# 
# # List of the strings that is used to add correct label for each box.
# PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
# 
# NUM_CLASSES = 90
#####################################################################################

#######################################################################################
#http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_lowproposals_oid_2018_01_28.tar.gz
# MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_lowproposals_oid_2018_01_28'
# MODEL_FILE = MODEL_NAME + '.tar.gz'
# DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
#  
# # Path to frozen detection graph. This is the actual model that is used for the object detection.
# PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
#  
# # List of the strings that is used to add correct label for each box.
# PATH_TO_LABELS = os.path.join('data', 'oid_bbox_trainable_label_map.pbtxt')
#  
# NUM_CLASSES = 546
#####################################################################################

#######################################################################################
#http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28.tar.gz
# MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28'
# MODEL_FILE = MODEL_NAME + '.tar.gz'
# DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
# 
# # Path to frozen detection graph. This is the actual model that is used for the object detection.
# PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
# 
# # List of the strings that is used to add correct label for each box.
# PATH_TO_LABELS = os.path.join('data', 'oid_bbox_trainable_label_map.pbtxt')
# 
# NUM_CLASSES = 546
#####################################################################################

#######################################################################################
#http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_ava_v2.1_2018_04_30.tar.gz
# MODEL_NAME = 'faster_rcnn_resnet101_ava_v2.1_2018_04_30'
# MODEL_FILE = MODEL_NAME + '.tar.gz'
# DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
# 
# # Path to frozen detection graph. This is the actual model that is used for the object detection.
# PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
# 
# # List of the strings that is used to add correct label for each box.
# PATH_TO_LABELS = os.path.join('data', 'ava_label_map_v2.1.pbtxt')
# 
# NUM_CLASSES = 80
#####################################################################################


    
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
    
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)
 

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


###the function
def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict


###these we grapped from image function
with detection_graph.as_default():
  with tf.Session() as sess:
      
    # Get handles to input and output tensors
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in [
        'num_detections', 'detection_boxes', 'detection_scores',
        'detection_classes', 'detection_masks'
    ]:
      tensor_name = key + ':0'
      if tensor_name in all_tensor_names:
        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
            tensor_name)
    if 'detection_masks' in tensor_dict:
      # The following processing is only for single image
      detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
      detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
      # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
      real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
      detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
      detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
      detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
          detection_masks, detection_boxes, image.shape[0], image.shape[1])
      detection_masks_reframed = tf.cast(
          tf.greater(detection_masks_reframed, 0.5), tf.uint8)
      # Follow the convention by adding back the batch dimension
      tensor_dict['detection_masks'] = tf.expand_dims(
          detection_masks_reframed, 0)
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
    
        
    #open database connection
    client = MongoClient('localhost:27017')
    db = client.flickr
     
    for group_id in flickr_group_ids:
        print("starting coco processing group_id"+str(group_id))        
        photoid_dicts = list()
        for entry in db.flickr.find({"group_id":group_id,"coco":{"$exists":0},'saved_image_extension':{'$exists':1}},{"id":1,"saved_image_extension":1}):
            photoid_dicts.append(entry)
    
        for photoid_dict in photoid_dicts:
            try:
                #this is from here: https://github.com/tensorflow/models/issues/3270
                #start_time = time.time()
        
                imagename = "/mnt/volume-fra1-01/flickr/picextraction/"+photoid_dict['id']+"."+photoid_dict['saved_image_extension']
                
                image = Image.open(imagename)
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                image_np = load_image_into_numpy_array(image)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                # Run inference
                output_dict = sess.run(tensor_dict,
                                       feed_dict={image_tensor: image_np_expanded})
                
                  # all outputs are float32 numpy arrays, so convert types as appropriate

                theclasses = output_dict[
                    'detection_classes'][0].astype(np.uint8)
                thescores = output_dict['detection_scores'][0]

                numscores=np.where(thescores==0)[0][0]#find the first position where likelihood is zero
                scoresdict=dict()
                if numscores >0:
                    for i in range(0,(numscores)):
                      labelname = category_index[theclasses[i]]['name']
                      if labelname not in scoresdict: #we are just intested in the max values for now
                          scoresdict[labelname]=thescores[i].item()
                db.flickr.update_one({'_id': photoid_dict['_id']}, {"$set":{"coco":scoresdict}}, upsert=False)
        
                #print('Iteration: %.3f sec'%(time.time()-start_time))
            except Exception as e:
                print("ERROR: problems with mining, skipping image :"+str(photoid_dict["id"]))
                continue
