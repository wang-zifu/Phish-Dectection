# coding=utf-8
# 调用BiLSTM-Attention方式

from .LSTM import LstmModel

model_path='/savemodel/nn/' # 训练模型保存路径
lstm = LstmModel(model_path=model_path,num_targets=1,sentence_mode=1) # 不添加Attention
lstm_att = LstmModel(model_path=model_path,num_targets=1).train() # 默认调用Attention




