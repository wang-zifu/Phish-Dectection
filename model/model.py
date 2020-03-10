# coding=utf-8

"""
A base model class provide a frame of deep learning model.
"""

import os
import abc
import pickle
import logging
import tempfile

import numpy as np
import tensorflow as tf
from tqdm import trange

from .data import getdata  # 加载的数据为X_train,y_train,X_test,y_test，且为np.array格式
from .helper import makedirs, make_iterator, summary

logger = logging.getLogger(__name__)

class BaseModel(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 model_path,
                 num_targets,
                 vocab_size=103,
                 embed_size=128,
                 hidden_size=64,
                 hidden_layer=128,
                 attention_size=64,
                 ):

        self.num_targets = num_targets
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.hidden_size = hidden_layer
        self.attention_size = attention_size

        self._model_path = model_path
        self._meta_path = os.path.join(model_path, "meta.pickle")
        self._ckpt_path = os.path.join(model_path, "checkpoint")

        makedirs(self._model_path)  
        if not os.path.exists(self._meta_path):
            with open(self._meta_path, "wb") as f:
                pickle.dump(self._hparams, f)  

    @abc.abstractmethod  
    def call(self, *args, **kwargs):
        pass

    @property
    def ckpt(self):
        return os.path.join(self._ckpt_path, 'model.ckpt')

    def train(self,
              data_path='/dataprocess/crawl/trData/',
              batch_size=256,
              numb_epoch=0,
              keep_prob=0.8,
              learning_rate=0.005,
              max_grad_norm=5.0,
              threshold=0.5,  
              verbose_period=100,
              url=True,
              domain=False,
              ):
        """Train interface."""
        # model_path = self._model_path
        tempdir = tempfile.mkdtemp()  
        graph = tf.Graph()  
        # Build training components
        with graph.as_default():
            train_x, valid_x, train_y, valid_y = getdata(data_path, url=url, domain=domain)
            valid_data = [valid_x, valid_y]

            x_train, y_train = make_iterator(train_x, train_y, batch_size)
            y_train = tf.cast(y_train, dtype=tf.float32)

            logits = self.call(x_train, keep_prob=keep_prob)  

            probabilities = tf.nn.sigmoid(logits)
            predictions = tf.greater(probabilities, threshold)  

            with tf.name_scope('train'):
                cross_entropy = tf.losses.sigmoid_cross_entropy(y_train,
                                                                logits,
                                                                reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)  # if logits=y_tru,将会出现梯度无变化的错误
                global_step = tf.train.get_or_create_global_step()

                # Clip gradients
                optimizer = tf.train.AdamOptimizer(learning_rate)
                variables = tf.trainable_variables()
                gradients = tf.gradients(cross_entropy, variables)  
                clipped_grads, grad_norm = tf.clip_by_global_norm(gradients,
                                                                  max_grad_norm)  

                # Guarantee to update batch normalization's moving variables.
                train_op = optimizer.apply_gradients(zip(clipped_grads, variables),
                                                     global_step=global_step)

            with tf.name_scope('metrics'):
                """
                评估指标参数
                """
                auc, auc_update = tf.metrics.auc(labels=y_train, predictions=probabilities)
                accuracy, accuracy_update = tf.metrics.accuracy(labels=y_train, predictions=predictions)
                precision, precision_update = tf.metrics.precision(labels=y_train, predictions=predictions)
                recall, recall_update = tf.metrics.recall(labels=y_train, predictions=predictions)
                fpr, fpr_update = tf.contrib.metrics.streaming_false_positive_rate(labels=y_train,
                                                                                   predictions=predictions)

                loss, loss_update = tf.metrics.mean(cross_entropy)
                metrics_update = tf.group([auc_update, accuracy_update,
                                           precision_update, recall_update,
                                           fpr_update, loss_update])
                metrics = tf.group([auc, accuracy,
                                    precision, recall,
                                    fpr, loss])

                saver = tf.train.Saver(max_to_keep=3)  # model to save
                local_variables_initializer = tf.local_variables_initializer()

        with tf.Session(graph=graph) as sess:
            best_epoch, best_loss = 0, np.inf
            # Initialize or restore variable
            if os.path.exists(self._ckpt_path):
                
                saver.restore(sess, self.ckpt)
                summary_writer = tf.summary.FileWriter(os.path.join(self._model_path, 'event'))
            else:
                sess.run(tf.global_variables_initializer())
                saver.save(sess, self.ckpt)
                summary_writer = tf.summary.FileWriter(os.path.join(self._model_path, 'event'))

            for epoch in range(numb_epoch):
                print('The epoch of training:', epoch)
                sess.run(local_variables_initializer)  
                steps_per_epoch = train_x.shape[0] // batch_size + 1  
                process_bar = trange(steps_per_epoch, desc='epoch {}'.format(epoch + 1))  
                for _ in process_bar:
                    ce, gn, gs, mu, tp = sess.run([cross_entropy,
                                                   grad_norm,
                                                   global_step,
                                                   metrics_update,
                                                   train_op])

                    metrics = {'loss': loss.eval(),
                                'accuracy': accuracy.eval(),
                                'auc': auc.eval(),
                                'precision': precision.eval(),
                                'recall': recall.eval(),
                                'fpr': fpr.eval(), }

                    process_bar.set_postfix(grad_norm=gn, **metrics)

                    # Summary statistics
                    if gs % verbose_period == 0:
                        summary(summary_writer, metrics, global_step=gs, tag='train')


            if valid_data:
                saver.save(sess, os.path.join(tempdir, 'model.ckpt'))
                eval_metrics = self.evaluate(
                                            valid_data,
                                            batch_size,
                                            threshold=threshold,
                                            keep_prob=keep_prob,
                                            _ckpt_path=tempfile
                                        )

                summary(summary_writer, eval_metrics, global_step=global_step.eval(), tag='valid')

                for key, value in eval_metrics.items():
                    logger.info('Evaluate metric {0}:{1}'.format(key, value))

                    if eval_metrics['loss'] < best_loss:
                        best_loss = eval_metrics['loss']
                        logger.info('Best model in epoch {0} with loss {1}.'.format(epoch + 1, best_loss))
                        logger.info('Overwrite saved model.')
                        saver.save(sess, self.ckpt)
                    elif tolerance is not None and epoch + 1 > tolerance + best_epoch:
                        logger.info('Advance terminate on epoch {0}.'.format(epoch + 1))
                        break
            else:
                logger.info('Overwrite saved model.')
                saver.save(sess, self.ckpt)

    def evaluate(self,
                 valid_data,
                 batch_size,
                 threshold=0.5,
                 keep_prob=0.8,
                 _ckpt_path=None):
        """Evaluate interface"""
        graph = tf.Graph()  
        # Build training components
        with graph.as_default():
            valid_x, valid_y = valid_data

            x_valid, y_valid = make_iterator(valid_x, valid_y, batch_size)

            logits = self.call(x_valid, keep_prob=keep_prob)  

            probabilities = tf.nn.sigmoid(logits)
            predictions = tf.greater(probabilities, threshold)  

            with tf.name_scope('train'):
                cross_entropy = tf.losses.sigmoid_cross_entropy(y_valid,
                                                                logits,
                                                                reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)

            with tf.name_scope('metrics'):
                auc, auc_update = tf.metrics.auc(labels=y_valid, predictions=probabilities)  
                accuracy, accuracy_update = tf.metrics.accuracy(labels=y_valid, predictions=predictions)
                precision, precision_update = tf.metrics.precision(labels=y_valid, predictions=predictions)
                recall, recall_update = tf.metrics.recall(labels=y_valid, predictions=predictions)
                fpr, fpr_update = tf.contrib.metrics.streaming_false_positive_rate(labels=y_valid,
                                                                                   predictions=predictions)

                loss, loss_update = tf.metrics.mean(cross_entropy)
                metrics_update = tf.group([auc_update, accuracy_update,
                                           precision_update, recall_update,
                                           fpr_update, loss_update])

                saver = tf.train.Saver(max_to_keep=3)  # model to save

        with tf.Session(graph=graph) as sess:
            model_checkpoint_path = os.path.join(_ckpt_path or self._ckpt_path, 'model.ckpt')
            saver.restore(sess, model_checkpoint_path)
            sess.run(tf.local_variables_initializer())

            steps_per_epoch = valid_x.shape[0] // batch_size + 1
            process_bar = trange(steps_per_epoch, desc='epoch {}'.format(1))
            for _ in process_bar:
                try:
                    sess.run(metrics_update)
                    process_bar.set_postfix(accuracy=accuracy.eval(),
                                            auc=auc.eval(),
                                            precision=precision.eval(),
                                            recal=recall.eval(),
                                            loss=loss.eval(),
                                            fpr=fpr.eval(),
                                            refresh=True)
                except tf.errors.OutOfRangeError:
                    process_bar.close()
                    break

            metrics = {"accuracy": accuracy.eval(), "auc": auc.eval(), "precision": precision.eval(),
                       "recall": recall.eval(), "fpr": fpr.eval(), "loss": loss.eval()}
        return metrics


    @classmethod
    def load(cls, model_path):
        with open(os.path.join(model_path, 'meta.pickle'), 'rb') as f:
            return cls(model_path, **pickle.load(f))
