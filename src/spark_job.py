#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from pyspark.sql import SparkSession

import pandas as pd
import pickle

spark = SparkSession.builder     .master("local[*]")     .appName("covid19")     .getOrCreate()

print(sys.version, "\n", tf.__version__)


# In[12]:


# Helper Functions
def distributed_dataset(dataset, batch_size):
    return windowed_dataset(dataset, 20, batch_size, 1000)

def windowed_dataset(dataset, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(dataset)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1:]))
    dataset = dataset.repeat().batch(batch_size).prefetch(1)
    return dataset


# # Setup
# ## Data

# In[13]:


prices = spark.read.csv("ProcessedData/testing_set.csv", 
                        header=True, 
                        inferSchema=True).drop("_c0").toPandas()
prices, time = prices.iloc[:, 1:], prices.iloc[:, :1]

mono_worker_dataset = windowed_dataset(prices, 
                           window_size=20, 
                           batch_size=32, 
                           shuffle_buffer=1000)


# ## LSTM

# In[5]:


lstm_model = keras.models.load_model("model.h5")


# # Parallelization
# Note: Run strategy only once

# In[6]:


import os

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
os.environ["TF_CONFIG"] = json.dumps({
    "cluster": {
        "worker": ["localhost:12345", "localhost:23456"]
    },
    "task": {"type": "worker", "index": 0}
})


# In[14]:


num_workers = 3
per_worker_batch_size = 20
batch_size = per_worker_batch_size * num_workers

multi_worker_dataset = distributed_dataset(prices, batch_size)

options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
dataset_no_auto_shard = multi_worker_dataset.with_options(options)


# In[15]:


with strategy.scope():
    multi_worker_model = lstm_model
    
multi_worker_model.fit(multi_worker_dataset, epochs=500, steps_per_epoch=70)

