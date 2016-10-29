# Data fetching/queueing
FIRST_GET_WAIT_SEC = 200
GET_WAIT_SEC = 1
BUCKET_CACHE_BATCH = 10
# Maximum number
QUEUE_NUM_BATCH = 1000

TRAIN_DATA_PATH = "data/giga.train"
#TRAIN_DATA_PATH = "data/giga.4321.train"
# Just use this since all of validation too slow
# to run at end of every epoch
TRAIN_EVAL_DATA_PATH ="data/giga.4000.valid"
# Run full eval and decoding on full thing though
EVAL_DATA_PATH = "data/giga.valid"
DECODE_DATA_PATH = "data/giga.valid"
# Using process_gigaword_data.py from other nlm-noising
VOCAB_PATH = "data/giga.vocab"

TRAIN_SRC_PATH = "/bak/alex_summary/train.src"
TRAIN_TGT_PATH = "/bak/alex_summary/train.tgt"
