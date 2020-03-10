import tensorflow as tf
from .model import BaseModel
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn

from tensorflow.contrib.rnn import LSTMCell
from .Attention import SentenceMode, Attention  


class LstmModel(BaseModel):
    def __init__(self,
                 model_path,
                 num_targets,
                 vocab_size=103,
                 embed_size=128,
                 hidden_size=64,
                 hidden_layer=128,
                 attention_size=64,
                 sentence_mode=SentenceMode.ATTENTION, ):
        super(LstmModel, self).__init__(model_path,
                                        num_targets,
                                        vocab_size,
                                        embed_size,
                                        hidden_size,
                                        hidden_layer,
                                        attention_size,
                                        )
        self.sentence_mode = sentence_mode

    def call(self, data, keep_prob=0.8):
        # data : [batch_size,data_length]
        max_sentence_length = data.shape[1]
        print('data.shape[1]:', data.shape[1])

        with tf.variable_scope('embedding_layer'), tf.device("/cpu:0"):
            embedding = tf.get_variable('embedding',
                                        shape=[self.vocab_size, self.embed_size],
                                        initializer=tf.initializers.random_uniform(-1.0, 1.0))

            tf.summary.histogram('embeddings_var', embedding)
            # w2v : [batch_size,max_sentence_length,embed_size]
            data = tf.cast(data, dtype=tf.int32)
            w2v = tf.nn.embedding_lookup(embedding, data)

        with tf.variable_scope('bilstm_layer'):
            # final_outputs is tuple
            final_outputs, final_state = bi_rnn(LSTMCell(self.hidden_size),
                                                LSTMCell(self.hidden_size),
                                                inputs=w2v,
                                                dtype=tf.float32)

            tf.summary.histogram('RNN_outputs', final_outputs)

            if self.sentence_mode == SentenceMode.ATTENTION:
                attention_ = Attention(final_outputs, self.attention_size, time_major=False, return_alphas=True)
                outputs, alphas = attention_.attentionModel()
                # outputs ：[batc_size,vocab_size]
                tf.summary.histogram('alphas', alphas)

            elif self.sentence_mode == SentenceMode.FINAL_STATE:
                final_state_fw, final_state_bw = final_state
                # outputs = tf.concat([final_state_fw, final_state_bw], axis=-1)
                outputs = tf.concat(final_state, 2)
                
            else:
                raise ValueError("sentence mode `{0}` dose not "
                                 "supported on gru model.".format(self.sentence_mode))

        with tf.variable_scope('fully_connected_layer'):
            # rnn_output = [batch_size,sentence_length]
            rnn_output = tf.nn.dropout(outputs, keep_prob=keep_prob)
            # h : [batch_size,sentence_length]
            print('rnn_output.shape:', rnn_output.shape)
            h = tf.layers.Dense(rnn_output.shape.as_list()[-1], activation=tf.nn.relu)(rnn_output)
            # h = tf.layers.Dense(64,activation=tf.nn.relu)(rnn_output)
            print('h.shape:', h.shape)
            # logits:[batch_size,num_targets]
            logits = tf.layers.Dense(self.num_targets)(h)
            print('logits.shape:', logits.shape)

        return logits


if __name__ == '__name__':
    modelpath = '/root/ylj/DPhishDetect/savemodel/nn/'
