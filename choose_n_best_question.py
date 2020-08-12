# coding=utf-8
"""BERT finetuning runner."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import csv
import time
import collections
import tensorflow as tf
from . import modeling
from .run_classifier_manual_base import *
from . import optimization_finetuning as optimization
from . import tokenization


tf.flags.DEFINE_string("predictor_save_dir", None, "predictor_save_dir.")
tf.flags.DEFINE_string("fine_tuned_model_dir", None, "fine_tuned_model_dir.")
tf.flags.DEFINE_string("output_albert_dir", None, "output_albert_dir.")



path_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#######     train
FLAGS.task_name = 'sentence_pair'
FLAGS.do_train = True
FLAGS.do_eval = False
FLAGS.do_predict = False
FLAGS.data_dir = os.path.join(path_dir, 'data_input')
FLAGS.output_dir = os.path.join(path_dir, 'data_albert_output')
FLAGS.vocab_file = os.path.join(path_dir, 'albert_pretrained_model', 'albert_tiny_489k', 'vocab.txt')
FLAGS.bert_config_file = os.path.join(path_dir, 'albert_pretrained_model', 'albert_tiny_489k', 'albert_config_tiny.json')
FLAGS.fine_tuned_model_dir = os.path.join(path_dir, 'fine_tuned_model')
FLAGS.predictor_save_dir = os.path.join(path_dir, 'predictor_save_dir')

###
tf.logging.set_verbosity(tf.logging.INFO)
processors = {
    "sentence_pair": SentencePairClassificationProcessor,
    # "lcqmc_pair":LCQMCPairClassificationProcessor
}


tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                              FLAGS.init_checkpoint)


# if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
  # raise ValueError(
      # "At least one of `do_train`, `do_eval` or `do_predict' must be True.")


bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))


# tf.gfile.MakeDirs(FLAGS.output_dir)
task_name = FLAGS.task_name.lower()
if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))


processor = processors[task_name]()
label_list = processor.get_labels()
tokenizer = tokenization.FullTokenizer(
    vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)


tpu_cluster_resolver = None
if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)


is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
# Cloud TPU: Invalid TPU configuration, ensure ClusterResolver is passed to tpu.
print("###tpu_cluster_resolver:",tpu_cluster_resolver)
# run_config = tf.contrib.tpu.RunConfig(
    # cluster=tpu_cluster_resolver,
    # master=FLAGS.master,
    # model_dir=FLAGS.output_dir,
    # save_checkpoints_steps=FLAGS.save_checkpoints_steps,
    # tpu_config=tf.contrib.tpu.TPUConfig(
        # iterations_per_loop=FLAGS.iterations_per_loop,
        # num_shards=FLAGS.num_tpu_cores,
        # per_host_input_for_training=is_per_host))


# train_examples = None
# num_train_steps = None
# num_warmup_steps = None
# if FLAGS.do_train:
  # train_examples =processor.get_train_examples(FLAGS.data_dir) # TODO
  # print("###length of total train_examples:",len(train_examples))
  # num_train_steps = int(len(train_examples)/ FLAGS.train_batch_size * FLAGS.num_train_epochs)
  # num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)


# model_fn = model_fn_builder(
    # bert_config=bert_config,
    # num_labels=len(label_list),
    # init_checkpoint=FLAGS.init_checkpoint,
    # learning_rate=FLAGS.learning_rate,
    # num_train_steps=num_train_steps,
    # num_warmup_steps=num_warmup_steps,
    # use_tpu=FLAGS.use_tpu,
    # use_one_hot_embeddings=FLAGS.use_tpu)


# # If TPU is not available, this will fall back to normal Estimator on CPU
# # or GPU.
# estimator = tf.contrib.tpu.TPUEstimator(
    # use_tpu=FLAGS.use_tpu,
    # model_fn=model_fn,
    # config=run_config,
    # train_batch_size=FLAGS.train_batch_size,
    # eval_batch_size=FLAGS.eval_batch_size,
    # predict_batch_size=FLAGS.predict_batch_size)

# ################  save & reload model  #################

# # estimator.export_saved_model(model_dir, serving_input_receiver_fn)

# # from tensorflow.contrib import predictor
# # predict_fn = predictor.from_saved_model(model_dir)
# # result = predict_fn(...)

# ########################################################

# ####
# train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
# train_file_exists=os.path.exists(train_file)
# print("###train_file_exists:", train_file_exists," ;train_file:",train_file)
# if not train_file_exists: # if tf_record file not exist, convert from raw text file. # TODO
    # file_based_convert_examples_to_features(train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
# tf.logging.info("***** Running training *****")
# tf.logging.info("  Num examples = %d", len(train_examples))
# tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
# tf.logging.info("  Num steps = %d", num_train_steps)
# train_input_fn = file_based_input_fn_builder(
    # input_file=train_file,
    # seq_length=FLAGS.max_seq_length,
    # is_training=True,
    # drop_remainder=True)
# # estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

# #####################################
# #######     server      #############
# def serving_input_fn():
  # label_ids = tf.placeholder(tf.int32, [None], name='label_ids')
  # input_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='input_ids')
  # input_mask = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='input_mask')
  # segment_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='segment_ids')
  # input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
      # 'label_ids': label_ids,
      # 'input_ids': input_ids,
      # 'input_mask': input_mask,
      # 'segment_ids': segment_ids,
  # })()
  # return input_fn

# estimator._export_to_tpu = False
# estimator.export_savedmodel(export_dir_base=FLAGS.predictor_save_dir, serving_input_receiver_fn=serving_input_fn)

###     reload
predict_fn = tf.contrib.predictor.from_saved_model(os.path.join(FLAGS.predictor_save_dir, '1573012268'))

def func_predict_by_server(text_a, text_b):
    predict_example = InputExample('id', text_a, text_b, '1')
    feature = convert_single_example(100, predict_example, label_list,
                                        FLAGS.max_seq_length, tokenizer)
    prediction = predict_fn({
        "input_ids":[feature.input_ids],
        "input_mask":[feature.input_mask],
        "segment_ids":[feature.segment_ids],
        "label_ids":[feature.label_id],
    })
    # {'probabilities': array([[0.81027454, 0.18972546]], dtype=float32)}
    probabilities = prediction["probabilities"]
    # label = label_list[probabilities.argmax()]
    return probabilities

##

number = 150
print('#######################################')
print('#######################################')
print('#######################################')
# for __ in range(3):
def func_choose_n_best_question(q_in):
    time_start = time.time()    
    text_a, text_b = q_in, '所有的投保、续保都是见费出单吗？'
    for _ in range(number):
        predict_example = InputExample('id', text_a, text_b, '1')
        feature = convert_single_example(100, predict_example, label_list,
                                              FLAGS.max_seq_length, tokenizer)
    print('convert_single_example use {} seconds during {} samples.'.format(time.time()  - time_start, number))
    print('#######################################')
    prediction = predict_fn({
        "input_ids":[feature.input_ids] * number,
        "input_mask":[feature.input_mask] * number,
        "segment_ids":[feature.segment_ids] * number,
        "label_ids":[feature.label_id] * number,
    })
    # {'probabilities': array([[0.81027454, 0.18972546]], dtype=float32)}
    probabilities = prediction["probabilities"]
    print(probabilities[0])
    print('albert_server use {} seconds during {} samples.'.format(time.time()  - time_start, number))
    print('#######################################')
    print('#######################################')
    print('#######################################')
    return 'abc'

# if __name__ == "__main__":
  # # flags.mark_flag_as_required("data_dir")
  # # flags.mark_flag_as_required("task_name")
  # # flags.mark_flag_as_required("vocab_file")
  # # flags.mark_flag_as_required("bert_config_file")
  # # flags.mark_flag_as_required("output_dir")
  # tf.app.run()