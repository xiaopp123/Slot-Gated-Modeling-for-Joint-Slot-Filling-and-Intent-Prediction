import numpy as np

def create_vocabulary(input_path, output_path, no_pad=False):
    if not isinstance(input_path, str):
        raise TypeError("input_path should be string")
    
    if not isinstance(output_path, str):
        raise TypeError("output path should be string")

    vocab = {}
    with open(input_path, "r") as fr, open(output_path, "w") as fw:
        for line in fr:
            line = line.rstrip("\r\n")
            words = line.split()

            for w in words:
                if w == "_UNK":
                    continue
                if str.isdigit(w) == True:
                    w = '0'
                if w in vocab:
                    vocab[w] += 1
                else:
                    vocab[w] = 1
        if no_pad == False:
            vocab = ["_PAD", "_UNK"] + sorted(vocab, key=vocab.get, reverse=True)
        else:
            vocab = ["_UNK"] + sorted(vocab, key=vocab.get, reverse=True)

        for v in vocab:
            fw.write(v + "\n")

def load_vocabulary(path):
    """
        返回字典类型
        第一个key值是 word to id
        第二个key值是 word list
    """
    if not isinstance(path, str):
        raise TypeError("path should be a string")

    vocab = []
    rev = []
    with open(path) as fr:
        for line in fr:
            line = line.rstrip("\r\n")
            rev.append(line)

        vocab = dict([(x, y) for (y, x) in enumerate(rev)])

    return {"vocab" : vocab, "rev" : rev}

def sentence_to_ids(data, vocab):
    if not isinstance(vocab, dict):
        raise TypeError("vocab should be a dict that contains vocab and rev")
    vocab = vocab["vocab"]
    if isinstance(data, str):
        words = data.split()
    elif isinstance(data, list):
        words = dta
    else:
        raise TypeError("data should be a string or a list contains words")

    ids = []
    for w in words:
        if str.isdigit(w) == True:
            w = '0'
        ids.append(vocab.get(w, vocab["_UNK"]))

    return ids

def pad_sentence(s, max_length, vocab):

    return s + [vocab["vocab"]["_PAD"]] * (max_length - len(s))

class DataProcessor(object):
    def __init__(self, in_path, slot_path, intent_path, in_vocab, slot_vocab, intent_vocab):
        self.__fd_in = open(in_path, "r")
        self.__fd_slot = open(slot_path, "r")
        self.__fd_intent = open(intent_path, "r")
        self.__in_vocab = in_vocab
        self.__slot_vocab = slot_vocab
        self.__intent_vocab = intent_vocab
        self.end = 0

    def close(self):
        self.__fd_in.close()
        self.__fd_slot.close()
        self.__fd_intent.close()

    def get_batch(self, batch_size):
        in_data =[]
        slot_data = []
        slot_weight = []
        length = []
        intents = []

        batch_in = []
        batch_slot = []
        max_len = 0

        in_seq = []
        slot_seq = []
        intent_seq = []

        for i in range(batch_size):
            inp = self.__fd_in.readline()
            if inp == "":
                self.end = 1
                break
            slot = self.__fd_slot.readline()
            intent = self.__fd_intent.readline()
            inp = inp.rstrip()
            slot = slot.rstrip()
            intent = intent.rstrip()

            in_seq.append(inp)
            slot_seq.append(slot)
            intent_seq.append(intent)

            in_ids = sentence_to_ids(inp, self.__in_vocab)
            slot_ids = sentence_to_ids(slot, self.__slot_vocab)
            intent_ids = sentence_to_ids(intent, self.__intent_vocab)

            batch_in.append(np.array(in_ids))
            batch_slot.append(np.array(slot_ids))
            length.append(len(in_ids))
            intents.append(intent_ids[0])

            if len(in_ids) != len(slot_ids):
                print(inp, slot)
                print(in_ids, slot_ids)
                exit(0)
            if len(in_ids) > max_len:
                max_len = len(in_ids)

        length = np.array(length)
        intents = np.array(intents)

        for i, s in zip(batch_in, batch_slot):
            in_data.append(pad_sentence(list(i), max_len, self.__in_vocab))
            slot_data.append(pad_sentence(list(s), max_len, self.__slot_vocab))

        in_data = np.array(in_data)
        slot_data = np.array(slot_data)

        for s in slot_data:
            weight = np.not_equal(s, np.zeros(s.shape))
            weight = weight.astype(np.float32)
            slot_weight.append(weight)
        slot_weight = np.array(slot_weight)

        return in_data, slot_data, slot_weight, length, intents, in_seq, slot_seq, intent_seq

def compute_f1_score(correct_slots, pred_slots):
    pass

if __name__ == "__main__":
    pass
