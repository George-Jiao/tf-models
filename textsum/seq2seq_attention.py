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
import sys
import time
import Queue

import tensorflow as tf
import batch_reader
import data
import seq2seq_attention_decode
import seq2seq_attention_model

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_path',
                           '', 'Path expression to tf.Example.')
tf.app.flags.DEFINE_string('vocab_path',
                           '', 'Path expression to text vocabulary file.')
tf.app.flags.DEFINE_string('article_key', 'article',
                           'tf.Example feature key for article.')
tf.app.flags.DEFINE_string('abstract_key', 'headline',
                           'tf.Example feature key for abstract.')
tf.app.flags.DEFINE_string('log_root', '', 'Directory for model root.')
tf.app.flags.DEFINE_string('train_dir', '', 'Directory for train.')
tf.app.flags.DEFINE_string('eval_dir', '', 'Directory for eval.')
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
tf.app.flags.DEFINE_integer('eval_interval_secs', 60, 'How often to run eval.')
tf.app.flags.DEFINE_integer('checkpoint_secs', 60, 'How often to checkpoint.')
tf.app.flags.DEFINE_bool('use_bucketing', False,
                         'Whether bucket articles of similar length.')
tf.app.flags.DEFINE_bool('truncate_input', False,
                         'Truncate inputs that are too long. If False, '
                         'examples that are too long are discarded.')
tf.app.flags.DEFINE_integer('num_gpus', 0, 'Number of gpus used.')
tf.app.flags.DEFINE_integer('random_seed', 111, 'A seed value for randomness.')

AVG_LOSS_DECAY = 0.99

def _RunningAvgLoss(loss, running_avg_loss, summary_writer, step, epoch=None, decay=AVG_LOSS_DECAY, fetch_time=None, run_time=None):
  """Calculate the running average of losses."""
  if running_avg_loss == 0:
    running_avg_loss = loss
  else:
    running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
  running_avg_loss = min(running_avg_loss, 12)
  loss_sum = tf.Summary()
  loss_sum.value.add(tag='running_avg_loss', simple_value=running_avg_loss)
  summary_writer.add_summary(loss_sum, step)
  msg = "step %d | running_avg_loss: %f" % (step, running_avg_loss)
  if epoch is not None:
    msg = ("epoch %d | " % epoch) + msg
  if fetch_time is not None:
    assert run_time is not None
    msg = msg + " | fetch_time: %f | run_time: %f" % (fetch_time, run_time)
  sys.stdout.write(msg + "\n")
  return running_avg_loss


def _Train(sess, model, data_batcher, sv, saver, summary_writer, epoch):
  """Runs model training."""
  running_avg_loss = 0
  step = 0
  while not sv.should_stop() and step < FLAGS.max_run_steps:
    tic = time.time()
    try:
      (article_batch, abstract_batch, targets, article_lens, abstract_lens,
       loss_weights, _, _) = data_batcher.NextBatch()
    except Queue.Empty:
      break
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


def _Eval(model, data_batcher, vocab=None):
  """Runs model eval."""
  model.build_graph()
  saver = tf.train.Saver()
  summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir)
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  running_avg_loss = 0
  step = 0
  while True:
    time.sleep(FLAGS.eval_interval_secs)
    try:
      ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
    except tf.errors.OutOfRangeError as e:
      tf.logging.error('Cannot restore checkpoint: %s', e)
      continue

    if not (ckpt_state and ckpt_state.model_checkpoint_path):
      tf.logging.info('No model to eval yet at %s', FLAGS.train_dir)
      continue

    tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
    saver.restore(sess, ckpt_state.model_checkpoint_path)

    try:
      (article_batch, abstract_batch, targets, article_lens, abstract_lens,
       loss_weights, _, _) = data_batcher.NextBatch()
    except Queue.Empty:
      break
    (summaries, loss, eval_step) = model.run_eval_step(
        sess, article_batch, abstract_batch, targets, article_lens,
        abstract_lens, loss_weights)
    tf.logging.info(
        'article:  %s',
        ' '.join(data.Ids2Words(article_batch[0][:].tolist(), vocab)))
    tf.logging.info(
        'abstract: %s',
        ' '.join(data.Ids2Words(abstract_batch[0][:].tolist(), vocab)))

    summary_writer.add_summary(summaries, eval_step)
    running_avg_loss = _RunningAvgLoss(
        running_avg_loss, loss, summary_writer, eval_step)
    if step % 100 == 0:
      summary_writer.flush()


def main(unused_argv):
  vocab = data.Vocab(FLAGS.vocab_path, 1000000)
  # Check for presence of required special tokens.
  assert vocab.WordToId(data.PAD_TOKEN) > 0
  assert vocab.WordToId(data.UNKNOWN_TOKEN) >= 0
  assert vocab.WordToId(data.SENTENCE_START) > 0
  assert vocab.WordToId(data.SENTENCE_END) > 0

  batch_size = 64
  if FLAGS.mode == 'decode':
    batch_size = FLAGS.beam_size

  hps = seq2seq_attention_model.HParams(
      mode=FLAGS.mode,  # train, eval, decode
      min_lr=0.001,  # min learning rate.
      lr=0.001,  # learning rate
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
  get_batcher = lambda: batch_reader.Batcher(
      FLAGS.data_path, vocab, hps, FLAGS.article_key,
      FLAGS.abstract_key, FLAGS.max_article_sentences,
      FLAGS.max_abstract_sentences, bucketing=FLAGS.use_bucketing,
      truncate_input=FLAGS.truncate_input, num_epochs=1)
  tf.set_random_seed(FLAGS.random_seed)

  if hps.mode == 'train':
    tic = time.time()
    model = seq2seq_attention_model.Seq2SeqAttentionModel(
        hps, vocab, num_gpus=FLAGS.num_gpus)
    model.build_graph()
    saver = tf.train.Saver()
    sv = tf.train.Supervisor(logdir=FLAGS.log_root,
                             is_chief=True,
                             saver=saver,
                             summary_op=None,
                             save_summaries_secs=60,
                             save_model_secs=FLAGS.checkpoint_secs,
                             global_step=model.global_step)
    sess = sv.prepare_or_wait_for_session(config=tf.ConfigProto(
        allow_soft_placement=True))
    toc = time.time()
    print("Took %fs to set up model and session" % (toc-tic))
    # Train dir is different from log_root to avoid summary directory
    # conflict with Supervisor.
    #summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph=sess.graph)
    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir)
    for epoch in xrange(1, num_epochs+1):
      tic = time.time()
      _Train(sess, model, get_batcher(), sv, saver, summary_writer, epoch=epoch)
      toc = time.time()
      print("Epochs %d took %fs" % (epoch, toc-tic))
    sv.Stop()
  elif hps.mode == 'eval':
    model = seq2seq_attention_model.Seq2SeqAttentionModel(
        hps, vocab, num_gpus=FLAGS.num_gpus)
    _Eval(model, get_batcher(), vocab=vocab)
  elif hps.mode == 'decode':
    decode_mdl_hps = hps
    # Only need to restore the 1st step and reuse it since
    # we keep and feed in state for each step's output.
    decode_mdl_hps = hps._replace(dec_timesteps=1)
    model = seq2seq_attention_model.Seq2SeqAttentionModel(
        decode_mdl_hps, vocab, num_gpus=FLAGS.num_gpus)
    decoder = seq2seq_attention_decode.BSDecoder(model, get_batcher(), hps, vocab)
    decoder.DecodeLoop()


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
