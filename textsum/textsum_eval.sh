CUDA_VISIBLE_DEVICES=1 ../bazel-bin/textsum/seq2seq_attention \
  --mode=eval \
  --article_key=article \
  --abstract_key=abstract \
  --vocab_path=data/giga.vocab \
  --log_root=log_root_sandbox \
  --num_gpus 1
