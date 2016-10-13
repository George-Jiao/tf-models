CUDA_VISIBLE_DEVICES=0 ../bazel-bin/textsum/seq2seq_attention \
  --mode=train \
  --article_key=article \
  --abstract_key=abstract \
  --data_path=data/giga.train \
  --vocab_path=data/giga.vocab \
  --log_root=log_root \
  --train_dir=log_root/train
