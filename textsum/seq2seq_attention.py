# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Trains a seq2seq model.

WORK IN PROGRESS.

Implement "Abstractive Text Summarization using Sequence-to-sequence RNNS and
Beyond."

"""
import os
import sys
import time
import Queue
from os.path import join as pjoin
import subprocess

import tensorflow as tf
import numpy as np
import batch_reader
import data
import seq2seq_attention_decode
import seq2seq_attention_model

from textsum_cfg import TRAIN_DATA_PATH
from textsum_cfg import TRAIN_EVAL_DATA_PATH
from textsum_cfg import EVAL_DATA_PATH
from textsum_cfg import DECODE_DATA_PATH
from textsum_cfg import VOCAB_PATH

from textsum_cfg import TRAIN_SRC_PATH
from textsum_cfg import TRAIN_TGT_PATH

from data_utils import NgramData
from data_utils import corrupt_batch
from data_utils import add_blank_token

import logging

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('article_key', 'article',
                           'tf.Example feature key for article.')
tf.app.flags.DEFINE_string('abstract_key', 'headline',
                           'tf.Example feature key for abstract.')
tf.app.flags.DEFINE_string('log_root', '', 'Directory for model root.')
tf.app.flags.DEFINE_string('decode_dir', '', 'Directory for decode summaries.')
tf.app.flags.DEFINE_string('mode', 'train', 'train/eval/decode mode')
tf.app.flags.DEFINE_integer('max_run_steps', 10000000,
                            'Maximum number of run steps.')
tf.app.flags.DEFINE_integer('max_article_sentences', 2,
                            'Max number of first sentences to use from the '
                            'article')
tf.app.flags.DEFINE_integer('max_abstract_sentences', 100,
                            'Max number of first sentences to use from the '
                            'abstract')
tf.app.flags.DEFINE_integer('epochs', 1,
                            'number of epochs to train for.')
tf.app.flags.DEFINE_integer('beam_size', 4,
                            'beam size for beam search decoding.')
tf.app.flags.DEFINE_bool('truncate_input', False,
                         'Truncate inputs that are too long. If False, '
                         'examples that are too long are discarded.')
tf.app.flags.DEFINE_integer('num_gpus', 0, 'Number of gpus used.')
tf.app.flags.DEFINE_integer('random_seed', 111, 'A seed value for randomness.')

tf.app.flags.DEFINE_string('noise_scheme', 'none', 'noising scheme (none, drop, swap)')
tf.app.flags.DEFINE_string('swap_scheme', 'unigram', 'swap scheme, one of {unigram, uniform, ad, kn, mkn}')
tf.app.flags.DEFINE_float('delta', 0.0, 'unscaled noising probability')

tf.app.flags.DEFINE_integer('max_anneals', 8, 'maximum number of times to anneal learning rate')

tf.app.flags.DEFINE_string('resume_model_path', None, 'path to load model and resume with')
tf.app.flags.DEFINE_integer('resume_epoch', None, 'epoch to resume with')
tf.app.flags.DEFINE_float('resume_lr', None, 'learning rate to resume with')
tf.app.flags.DEFINE_integer('resume_step', None, 'step to resume with')

AVG_LOSS_DECAY = 0.99

def _RunningAvgLoss(running_avg_loss, loss, summary_writer, step, epoch=None,
        decay=AVG_LOSS_DECAY, fetch_time=None, run_time=None):
  """Calculate the running average of losses."""
  if running_avg_loss == 0:
    running_avg_loss = loss
  else:
    running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
  running_avg_loss = float(min(running_avg_loss, 12))
  loss_summ = tf.Summary()
  loss_summ.value.add(tag='running_avg_loss', simple_value=running_avg_loss)
  if summary_writer is not None:
    summary_writer.add_summary(loss_summ, step)
  msg = "step %d | running loss: %f" % (step, running_avg_loss)
  if epoch is not None:
    msg = ("epoch %d | " % epoch) + msg
  if fetch_time is not None:
    assert run_time is not None
    msg = msg + " | fetch time: %f | run time: %f" % (fetch_time, run_time)
  logging.info(msg)
  return running_avg_loss


def _Train(sess, model, data_batcher, saver, summary_writer, epoch, src_ngd=None, tgt_ngd=None):
  """Runs model training."""
  running_avg_loss = 0
  step = 0
  while step < FLAGS.max_run_steps:
    tic = time.time()
    try:
      (article_batch, abstract_batch, targets, article_lens, abstract_lens,
       loss_weights, _, _) = data_batcher.NextBatch()
    except Queue.Empty:
      break

    if FLAGS.delta > 0 and FLAGS.noise_scheme != "none":
      # NOTE just passing in article_batch twice since no output y in encoder
      article_batch, _ = corrupt_batch(article_batch, article_batch, article_lens, FLAGS, src_ngd)
      abstract_batch, targets = corrupt_batch(abstract_batch, targets, abstract_lens, FLAGS, tgt_ngd)

    toc = time.time()
    fetch_time = toc - tic
    tic = time.time()
    (_, summaries, loss, train_step) = model.run_train_step(
        sess, article_batch, abstract_batch, targets, article_lens,
        abstract_lens, loss_weights)
    toc = time.time()
    run_time = toc - tic

    summary_writer.add_summary(summaries, train_step)
    running_avg_loss = _RunningAvgLoss(
        running_avg_loss, loss, summary_writer, train_step, epoch=epoch, fetch_time=fetch_time, run_time=run_time)
    step += 1
    if step % 100 == 0:
      summary_writer.flush()
  return running_avg_loss


def _Eval(sess, model, data_batcher, vocab=None):
  """Runs model eval."""
  summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir)
  running_avg_loss = 0
  step = 0
  total_weights = 0.0
  total_loss = 0.0
  while True:
    tic = time.time()
    try:
      (article_batch, abstract_batch, targets, article_lens, abstract_lens,
       loss_weights, _, _) = data_batcher.NextBatch()
    except Queue.Empty:
      break
    toc = time.time()
    fetch_time = toc - tic
    tic = time.time()
    # NOTE eval_step is just current train_step
    (summaries, loss, eval_step) = model.run_eval_step(
        sess, article_batch, abstract_batch, targets, article_lens,
        abstract_lens, loss_weights)
    toc = time.time()
    run_time = toc - tic
    sum_weights = np.sum(loss_weights)
    total_weights += sum_weights
    total_loss += sum_weights * loss

    # Just prints first of the batch
    logging.debug(
        'article:  %s',
        ' '.join(data.Ids2Words(article_batch[0][:].tolist(), vocab)))
    logging.debug(
        'abstract: %s',
        ' '.join(data.Ids2Words(abstract_batch[0][:].tolist(), vocab)))

    # Only write one loss summary at the end
    #summary_writer.add_summary(summaries, eval_step)
    running_avg_loss = _RunningAvgLoss(
        running_avg_loss, loss, None, eval_step,
        fetch_time=fetch_time, run_time=run_time)
    #if step % 100 == 0:
      #summary_writer.flush()

  weighted_avg_loss = total_loss / float(total_weights)
  loss_summ = tf.Summary()
  loss_summ.value.add(tag="loss", simple_value=weighted_avg_loss)
  summary_writer.add_summary(loss_summ, eval_step)
  summary_writer.flush()
  return weighted_avg_loss


def main(unused_argv):
  vocab = data.Vocab(VOCAB_PATH, 1000000)
  logging.info("%d words in vocab" % vocab.NumIds())
  add_blank_token(vocab)
  # Check for presence of required special tokens.
  assert vocab.WordToId(data.PAD_TOKEN) > 0
  assert vocab.WordToId(data.UNKNOWN_TOKEN) >= 0
  assert vocab.WordToId(data.SENTENCE_START) > 0
  assert vocab.WordToId(data.SENTENCE_END) > 0

  FLAGS.train_dir = pjoin(FLAGS.log_root, "train")
  FLAGS.eval_dir = pjoin(FLAGS.log_root, "eval")
  # We'll specify decode directory in flags
  #FLAGS.decode_dir = pjoin(FLAGS.log_root, "decode")

  if not os.path.exists(FLAGS.log_root):
    os.mkdir(FLAGS.log_root)
  file_handler = logging.FileHandler(pjoin(FLAGS.log_root, "log.txt"))
  logging.getLogger().addHandler(file_handler)
  logging.info(FLAGS.__flags)

  batch_size = 64
  if FLAGS.mode == 'decode':
    batch_size = FLAGS.beam_size

  hps = seq2seq_attention_model.HParams(
      mode=FLAGS.mode,  # train, eval, decode
      lr=1.0,  # learning rate
      batch_size=batch_size,
      enc_layers=2,
      enc_timesteps=50,
      dec_timesteps=25,
      min_input_len=2,  # discard articles/summaries < than this
      num_hidden=256,  # for rnn cell
      emb_dim=128,  # If 0, don't use embedding
      max_grad_norm=2,
      num_softmax_samples=0)  # If 0, no sampled softmax.

  num_epochs = FLAGS.epochs
  if hps.mode != "train":
    num_epochs = 1
  get_batch_reader = lambda path: batch_reader.Batcher(
      path, vocab, hps, FLAGS.article_key,
      FLAGS.abstract_key, FLAGS.max_article_sentences,
      FLAGS.max_abstract_sentences, bucketing=(FLAGS.mode == "train"),
      truncate_input=FLAGS.truncate_input, num_epochs=1)
  ckpt_path = pjoin(FLAGS.log_root, "model.ckpt")

  tf.set_random_seed(FLAGS.random_seed)
  np.random.seed(FLAGS.random_seed)

  if hps.mode == 'train':
    tic = time.time()
    model = seq2seq_attention_model.Seq2SeqAttentionModel(
        hps, vocab, num_gpus=FLAGS.num_gpus)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    model.build_graph()
    saver = tf.train.Saver(max_to_keep=FLAGS.epochs)

    # Stuff for noising
    src_ngram_data = NgramData(TRAIN_SRC_PATH, vocab)
    tgt_ngram_data = NgramData(TRAIN_TGT_PATH, vocab)

    total_params = np.sum([np.prod(v.get_shape()) for v in tf.trainable_variables()])
    logging.info("%d total parameters" % total_params)

    start_epoch = 1
    if FLAGS.resume_model_path is not None:
      logging.info("Restoring model from %s, epoch %d, lr %f" % (FLAGS.resume_model_path,
          FLAGS.resume_epoch, FLAGS.resume_lr))
      saver.restore(sess, FLAGS.resume_model_path)
      start_epoch = FLAGS.resume_epoch
      model.set_lr(sess, FLAGS.resume_lr)
      model.set_global_step(sess, FLAGS.resume_step)
    else:
      init = tf.initialize_all_variables()
      sess.run(init)

    toc = time.time()
    logging.info("Took %fs to set up model and session" % (toc-tic))
    # Train dir is different from log_root to avoid summary directory
    # conflict with Supervisor.
    #summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph=sess.graph)
    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir)
    prev_eval_cost = _Eval(sess, model, get_batch_reader(TRAIN_EVAL_DATA_PATH), vocab=vocab)
    logging.info("Starting eval cost: %f" % prev_eval_cost)
    eval_costs = list()
    lr_value = hps.lr
    anneals = 0
    for epoch in xrange(start_epoch, num_epochs+1):
      tic = time.time()
      _Train(sess, model, get_batch_reader(TRAIN_DATA_PATH), saver,
              summary_writer, epoch=epoch, src_ngd=src_ngram_data, tgt_ngd=tgt_ngram_data)
      toc = time.time()
      logging.info("Epochs %d took %fs" % (epoch, toc-tic))
      save_path = saver.save(sess, pjoin(FLAGS.log_root, "model_epoch%d.ckpt" % epoch))
      logging.info("Saved model to %s" % save_path)
      # NOTE Lose last mod batch_size examples here
      eval_cost = _Eval(sess, model, get_batch_reader(TRAIN_EVAL_DATA_PATH), vocab=vocab)
      logging.info("Epoch %d eval cost: %f" % (epoch, eval_cost))
      if eval_cost > prev_eval_cost:
        # Load model from best epoch (previous epoch assuming no patience)
        saver.restore(sess, ckpt_path)
        # Anneal
        lr_value = lr_value / 2.0
        anneals = anneals + 1
        if anneals > FLAGS.max_anneals:
          break
        logging.info("Annealing learning rate to: %f" % lr_value)
        model.set_lr(sess, lr_value)
      prev_eval_cost = eval_cost
      eval_costs.append(eval_cost)
      # Symlink model.ckpt to the latest saved model since better
      subprocess.check_call("ln -sf %s %s" % (os.path.abspath(save_path), os.path.abspath(ckpt_path)), shell=True)
  elif hps.mode == 'eval':
    model = seq2seq_attention_model.Seq2SeqAttentionModel(
        hps, vocab, num_gpus=FLAGS.num_gpus)
    model.build_graph()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      logging.info('Loading checkpoint %s', ckpt_path)
      saver = tf.train.Saver()
      saver.restore(sess, ckpt_path)
      eval_cost = _Eval(sess, model, get_batch_reader(EVAL_DATA_PATH), vocab=vocab)
      logging.info("Eval cost: %f" % eval_cost)
  elif hps.mode == 'decode':
    decode_mdl_hps = hps
    # Only need to restore the 1st step and reuse it since
    # we keep and feed in state for each step's output.
    decode_mdl_hps = hps._replace(dec_timesteps=1)
    model = seq2seq_attention_model.Seq2SeqAttentionModel(
        decode_mdl_hps, vocab, num_gpus=FLAGS.num_gpus)
    model.build_graph()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      logging.info('Loading checkpoint %s', ckpt_path)
      saver = tf.train.Saver()
      saver.restore(sess, ckpt_path)
      decoder = seq2seq_attention_decode.BSDecoder(model, get_batch_reader(DECODE_DATA_PATH), hps, vocab)
      decoder.DecodeLoop()

if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  tf.app.run()
