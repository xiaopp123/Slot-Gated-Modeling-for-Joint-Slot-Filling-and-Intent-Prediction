import argparse
import os
import logging
import sys
import tensorflow as tf
import numpy as np

#allow_abbrev 参数是否需要简写
parser = argparse.ArgumentParser(allow_abbrev=False)

parser.add_argument("--num_units", type=int, default=64, help="Network size", dest="layer_size")
parser.add_argument("--model_type", type=str, default="full", help="""full(default) | intent_only \
                                                                       full: full attention model \
                                                                       intent_only: intent attention model""")

#Training Environment
parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
parser.add_argument("--max_epochs", type=int, default=20, help="Max epochs to train")
parser.add_argument("--no_early_stop", action="store_false", dest="early_stop", help="Disable early stop, which is based on sentence level accuracy")
parser.add_argument("--patience", type=int, default=5, help="Patience to wait before stop")

#model and vocab
parser.add_argument("--dataset", type=str, default=None, help="""Type 'atis' or 'snips' to use dataset provided by us or enter what ever you named your own dataset. Note, if you don't want to use this part, enter --dataset=''. It can not be None""")
parser.add_argument("--model_path", type=str, default="./model", help="path to sava model")
parser.add_argument("--vocab_path", type=str, default="./vocab", help="path to vocabulary files")

#data
parser.add_argument("--train_data_path", type=str, default='train', help="Path to training data files.")
parser.add_argument("--test_data_path", type=str, default='test', help="Path to testing data files.")
parser.add_argument("--valid_data_path", type=str, default='valid', help="Path to validation data files.")
parser.add_argument("--input_file", type=str, default='seq.in', help="Input file name.")
parser.add_argument("--slot_file", type=str, default='seq.out', help="Slot file name.")
parser.add_argument("--intent_file", type=str, default='label', help="Intent file name.")

arg = parser.parse_args()

#print arguments
for k, v in sorted(vars(arg).items()):
    print(k, " = ", v)

print()
