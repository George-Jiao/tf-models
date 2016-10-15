CUDA_VISIBLE_DEVICES=1 ../bazel-bin/textsum/seq2seq_attention \
  --mode=eval \
  --article_key=article \
  --abstract_key=abstract \
  --data_path=data/giga.valid \
  --vocab_path=data/giga.vocab \
  --log_root=log_root \
  --eval_dir=log_root/eval \
  --num_gpus 1
