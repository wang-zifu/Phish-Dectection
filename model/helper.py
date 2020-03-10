# coding=utf-8

"""
Model helper functionality.
"""

import logging
import os
import tensorflow as tf

logger = logging.getLogger(__name__)

def summary(summary_writer, metrics, global_step, tag=None):
    """Write summary message to event file."""
    summary_values = []
    for key, value in metrics.items():
        summary_values.append(tf.Summary.Value(tag="/".join([tag, key]) if tag else key,
                                               simple_value=value))
    summary_writer.add_summary(tf.Summary(value=summary_values), global_step)
    summary_writer.flush()

def makedirs(dirname):
    """Make directory safely.

    Make directory when it not exists.
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def make_iterator(features,labels,batch_size):
    # create dataset : (x,labels)
    dataset = (features,labels)
    print('features.shape:',features.shape)
    dataset = tf.data.Dataset.from_tensor_slices(dataset)
    # shuffle
    dataset = dataset.shuffle(buffer_size=100)
    # batch size
    dataset = dataset.batch(batch_size).repeat()
    # create a iterator of the create shape and type
    # iter = tf.data.Iterator.from_structure(dataset.output_types,
    #                                         dataset.output_shapes)
    iter = dataset.make_one_shot_iterator()

    return iter.get_next()



