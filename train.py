import argparse
import os
import logging
import sys
import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
import numpy as np
from utils import create_vocabulary
from utils import load_vocabulary

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

#add final state to intent 恒为True
if arg.model_type == "full":
    add_final_state_to_intent = True
    remove_slot_attn = False
elif arg.model_type == "intent_only":
    add_final_state_to_intent = True
    remove_slot_attn = True
else:
    print("unknow model type!")
    exit(1)

#full path ./data/ + dataset + train/test/valid
if arg.dataset == None:
    print("name of dataset can not be None")
elif arg.dataset == "snip":
    print("use snip dataset")
elif arg.dataset == "atis":
    print("use atis dataset")
else:
    print("use own dataset: ", arg.dataset)

full_train_path = os.path.join("./data", arg.dataset, arg.train_data_path)
full_test_path = os.path.join('./data',arg.dataset,arg.test_data_path)
full_valid_path = os.path.join('./data',arg.dataset,arg.valid_data_path)

create_vocabulary(os.path.join(full_train_path, arg.input_file), os.path.join(arg.vocab_path, "in_vocab"))
create_vocabulary(os.path.join(full_train_path, arg.slot_file), os.path.join(arg.vocab_path, "slot_vocab"))
create_vocabulary(os.path.join(full_train_path, arg.intent_file), os.path.join(arg.vocab_path, "intent_vocab"))

# {word 2 id, words list}
in_vocab = load_vocabulary(os.path.join(arg.vocab_path, "in_vocab"))
slot_vocab = load_vocabulary(os.path.join(arg.vocab_path, "slot_vocab"))
intent_vocab = load_vocabulary(os.path.join(arg.vocab_path, "intent_vocab"))

def create_model(input_data, input_size, sequence_length, slot_size, intent_size, layer_size = 128, is_training = True):
    """
    input_data: 输入数据[batch, len]
    input_size: 输入数据中单词的个数
    sequence_length: 数据的长度[batch]
    slot_size: slot中单词的个数
    intent_size: intent中单词的个数
    layer_size: 隐藏层维度
    """
    cell_fw = tf.nn.rnn_cell.LSTMCell(layer_size)
    cell_bw = tf.nn.rnn_cell.LSTMCell(layer_size)

    if is_training == True:
        # rnn drop out https://www.tensorflow.org/api_docs/python/tf/nn/rnn_cell/DropoutWrapper
        cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, input_keep_prob=0.5, output_keep_prob=0.5)
        cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, input_keep_prob=0.5, output_keep_prob=0.5)

    embedding = tf.get_variable("embedding", [input_size, layer_size])
    inputs = tf.nn.embedding_lookup(embedding, input_data)
    """
    默认time_major=False，bidirectional_dynamic_rnn返回是（outputs, output_states)元组
    outputs是一个(output_fw, output_bw)元组，output_fw = [batch_size, time, cell_fw.output_size]
    output_states是一个(output_state_fw, output_state_bw)元组,包含前向后向的最终状态，output_state_fw包含两个元素，一个是\
            c状态，另一个是h状态 shape都是[batch, layer_size]
    """
    state_outputs, final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, \
            sequence_length=sequence_length, dtype=tf.float32)

    #将前向后向的c,h状态拼接
    final_state = tf.concat([final_state[0][0], final_state[0][1], final_state[1][0], final_state[1][1]], 1)
    #前向输出和后向输出拼接
    state_outputs = tf.concat([state_outputs[0], state_outputs[1]], 2)
    state_shape = state_outputs.get_shape()

    with tf.variable_scope("attention"):
        slot_inputs = state_outputs
        if remove_slot_attn == False:
            with tf.variable_scope("slot_attn"):
                attn_size = state_shape[2].value
                origin_shape = tf.shape(state_outputs)
                hidden = tf.expand_dims(state_outputs, 1)
                # hidden_con = [batch, time, 1, size]
                hidden_conv = tf.expand_dims(state_outputs, 2)
                k = tf.get_variable("AttenW", [1, 1, attn_size, attn_size])
                # hidden_features = [batch, time, 1, size]
                hidden_features = tf.nn.conv2d(hidden_conv, k, [1, 1, 1, 1], "SAME")
                hidden_features = tf.reshape(hidden_features, origin_shape)
                #hidden_features = [batch, 1, time, size]
                hidden_features = tf.expand_dims(hidden_features, 1)
                v = tf.get_variable("AttnV", [attn_size])

                slot_inputs_shape = tf.shape(slot_inputs)
                slot_inputs = tf.reshape(slot_inputs, [-1, attn_size])
                # y = [batch*time, attn_size]
                y = core_rnn_cell._linear(slot_inputs, attn_size, True)
                # y = [batch, time, attn_size]
                y = tf.reshape(y, slot_inputs_shape)
                # y = [batch, time, 1, attn_size]
                y = tf.expand_dims(y, 2)
                # s = [batch ,time, time], 每个输出都和其他的时刻的输出有一个权重
                s = tf.reduce_sum(v * tf.tanh(hidden_features + y), [3])
                a = tf.nn.softmax(s)
                # a = [ batch, time, time, 1]
                a = tf.expand_dims(a, -1)
                # slot_d = [batch, time, attn_size], a是权重，hidden是前几个时刻的输出，slot_d是加权和
                slot_d = tf.reduce_sum(a * hidden, [2])
        else:
            attn_size = state_shape[2].value
            slot_inputs = tf.reshape(slot_inputs, [-1, attn_size])

        
        intent_input = final_state
        with tf.variable_scope("intent_attn"):
            attn_size = state_shape[2].value
            #同上，输出变成[batch, time, 1, size]
            hidden = tf.expand_dims(state_outputs, 2)
            k = tf.get_variable("AttenW", [1, 1, attn_size, attn_size])
            # hidden_features = [batch, time, 1, size]
            hidden_features = tf.nn.conv2d(hidden, k, [1, 1, 1, 1], "SAME")
            v = tf.get_variable("AttnV", [attn_size])

            # y = [batch, attn_size]
            y = core_rnn_cell._linear(intent_input, attn_size, True)
            y = tf.reshape(y, [-1, 1, 1, attn_size])
            # s = [batch, time]
            s = tf.reduce_sum(v * tf.tanh(hidden_features + y), [2, 3])
            a = tf.nn.softmax(s)
            a = tf.expand_dims(a, -1)
            a = tf.expand_dims(a, -1)
            # a = [batch, time, 1, 1], hidden = [batch, time, 1, size]
            # d = [batch, size]
            d = tf.reduce_sum(a * hidden, [1, 2])

            if add_final_state_to_intent == True:
                intent_output = tf.concat([d, intent_input], 1)
            else:
                intent_output = d

        with tf.variable_scope("slot_gated"):
            intent_gate = core_rnn_cell._linear(intent_output, attn_size, True)
            #intent_gate = [batch, 1, attn_size]
            intent_gate = tf.reshape(intent_gate, [-1, 1, intent_gate.get_shape()[1].value])
            v1 = tf.get_variable("gataV", [attn_size])
            if remove_slot_attn == False:
                slot_gate = v1 * tf.tanh(slot_d + intent_gate)
            else:
                slot_gate = v1 * tf.tanh(state_outputs + intent_gate)
            # slot_gate = [batch, attn_size]
            slot_gate = tf.reshape(slot_gate, [-1, attn_size])
            # slot_input = [batch*time, attn_size], slot_gate = [batch * time, attn_size]
            slot_output = tf.concat([slot_gate, slot_inputs], 1)

    with tf.variable_scope("intent_proj"):
        intent = core_rnn_cell._linear(intent_output, intent_size, True)
    with tf.variable_scope("slot_proj"):
        slot = core_rnn_cell._linear(slot_output, slot_size, True)

    # slot = [batch * time, slot_size], intent = [batch, intent_size]
    outputs = [slot, intent]

    return outputs


# create training model
#[batch, len]
input_data = tf.placeholder(tf.int32, [None, None], name="inputs")
sequence_length = tf.placeholder(tf.int32, [None], name="sequence_length")
global_step = tf.Variable(0, trainable=False, name="global_step")
slots = tf.placeholder(tf.int32, [None, None], name="slots")
slot_weights = tf.placeholder(tf.float32, [None, None], name="slot_weights")
intent = tf.placeholder(tf.int32, [None], name="intent")

with tf.variable_scope("model"):
    training_outputs = create_model(input_data, len(in_vocab["vocab"]), sequence_length,\
            len(slot_vocab["vocab"]), len(intent_vocab["vocab"]), layer_size=arg.layer_size)

slots_shape = tf.shape(slots)
slots_reshape = tf.reshape(slots, [-1])

slot_outputs = training_outputs[0]
with tf.variable_scope("slot_loss"):
    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=slots_reshape, logits=slot_outputs)
    crossent = tf.reshape(crossent, slots_shape)
    slot_loss = tf.reduce_sum(crossent * slot_weights, 1)
    total_size = tf.reduce_sum(slot_weights, 1)
    total_size += 1e-12
    slot_loss = slot_loss / total_size

intent_output = training_outputs[1]
with tf.variable_scope("intent_loss"):
    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=intent, logits=intent_output)
    intent_loss = tf.reduce_sum(crossent) / tf.cast(arg.batch_size, tf.float32)

params = tf.trainable_variables()
opt = tf.train.AdamOptimizer()

intent_params = []
slot_params = []

for p in params:
    if not "slot_" in p.name:
        intent_params.append(p)
    #这里有点小疑问
    if "slot_" in p.name or "bidirectional_rnn" in p.name or "embedding" in p.name:
        slot_params.append(p)

gradients_slot = tf.gradients(slot_loss, slot_params)
gradients_intent = tf.gradients(intent_loss, intent_params)

clipped_gradients_slot, norm_slot = tf.clip_by_global_norm(gradients_slot, 5.0)
clipped_gradients_intent, norm_intent = tf.clip_by_global_norm(gradients_intent, 5.0)
