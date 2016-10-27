CUDA_VISIBLE_DEVICES=1 ../bazel-bin/textsum/seq2seq_attention \
  --mode=decode \
  --article_key=article \
  --abstract_key=abstract \
  --vocab_path=data/giga.vocab \
  --log_root=log_root_sandbox \
  --decode_dir=log_root_sandbox/decode \
  --beam_size=8 \
  --num_gpus 1
