"""Fractal dataset preprocessing and specifications."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile
import numpy as np
from six.moves import cPickle
from six.moves import urllib
import tensorflow as tf


LOCAL_DIR = os.path.join("data/")
DATA_DIR = "fractal-images/"
TRAIN_BATCHES = ["data_batch_%d" % (i + 1) for i in range(5)]
TEST_BATCHES = ["test_batch"]

IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512


def read(split):
  """Create an instance of the dataset object."""
  """An iterator that reads and returns images and labels from cifar."""
  batches = {
    tf.estimator.ModeKeys.TRAIN: TRAIN_BATCHES,
    tf.estimator.ModeKeys.EVAL: TEST_BATCHES
  }[split]

  all_images = []

  for batch in batches:
    with open(os.path.join(LOCAL_DIR, DATA_DIR, batch), "rb") as fo:
      dict = cPickle.load(fo)
      images = np.array(dict["data"])

      num = images.shape[0]
      images = np.reshape(images, [num, 3, IMAGE_HEIGHT, IMAGE_WIDTH])
      images = np.transpose(images, [0, 2, 3, 1])
      print("Loaded %d examples." % num)

      all_images.append(images)

  all_images = np.concatenate(all_images)

  return tf.contrib.data.Dataset.from_tensor_slices((all_images))

def parse(image, label):
  """Parse input record to features and labels."""
  image = tf.to_float(image) / 255.0
  image = tf.reshape(image, [IMAGE_HEIGHT, IMAGE_WIDTH, 3])
  return {"image": image}, {"label": label}
