# bert_regression

工作中遇到一个用文本做回归的问题，原版bert本来有run_reg.py，但不知为什么后来就又移除了，这个版本的y是从0到1，我的需求是从0到几十万，所以需要改改。

在bert基础上，有几点修改：

1、extract_features.py中的read_examples函数读文件，改成自己的输入文件的格式，即“数值y\t文本x”的形式。

2、modeling.py文件的227行改成，用倒数第二层。

3、run_reg_v2.3.1.py在run_reg.py的基础上，做了如下修改：

   （1）去掉softmax：让y的范围从0到1变成从0到无穷大。
    
   （2）再加两个relu和全连接：增加模型的拟合能力。
    
   （3）用Xavier给w做初始化：更快、更好的效果。
    
run_reg.py来自：https://github.com/google-research/bert/pull/503/commits/f005e159ffb40591b7e16d257ab4abc4e137182a

运行和原版一样：
'''
export PY_FILE_DIR=/usr/xujianzhi/formal_jobs/5_claim_cost_prediction/3_code/predict_compensation_amount_v1.1/albert_model/code_albert
export PYTHON_PATH=/usr/xujianzhi/virtual_env/env_python3_experiment_1/bin/python

export VERSION=v8.1.5_2
export BERT_BASE_DIR=/usr/xujianzhi/pretrained_model/bert/chinese_L-12_H-768_A-12
export FINE_TUNED_MODEL_10_epoch=$DATA_DIR_SOURCE/4_fine_tuned_model

$PYTHON_PATH $PY_FILE_DIR/run_reg_v2.3.1.py \
  --task_name=regression \
  --do_train=true \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$DATA_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=256 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=10.0 \
  --output_dir=$FINE_TUNED_MODEL_10_epoch > $PY_FILE_DIR/logs/$VERSION/10_epoch.log
'''
