import torch
from pyvi import ViTokenizer
import json

class Vocab():
    def __init__(self, path: str, max_len: int = 60, src_lang: str = 'vietnamese', tar_lang: str = 'english'):
        self.path = path
        self.max_len = max_len
        self.src_lang = src_lang
        self.tar_lang = tar_lang

        self.pad_piece = '<PAD>'
        self.sos_piece = '<SOS>'
        self.eos_piece = '<EOS>'
        self.unk_piece = '<UNK>'

        self.pad_id = 0
        self.sos_id = 1
        self.eos_id = 2
        self.unk_id = 3


        self.src_vocab = set()
        self.tar_vocab = set()

        self.src_w2i = {
            self.pad_piece: self.pad_id,
            self.sos_piece: self.sos_id,
            self.eos_piece: self.eos_id,
            self.unk_piece: self.unk_id
        }
        self.tar_w2i = dict(self.src_w2i)

        self.src_i2w = {}
        self.tar_i2w = {}

        self.make_vocab()

        self.src_vocab_size = len(self.src_i2w)
        self.tar_vocab_size = len(self.tar_i2w)

    def make_vocab(self):
        data = json.load(open(self.path, 'r', encoding='utf-8'))

        for item in data:
            src_sentence = item[self.src_lang]
            tar_sentence = item[self.tar_lang]

            src_words = src_sentence.lower().split()
            tar_words = tar_sentence.lower().split()
        
            self.src_vocab.update(src_words)
            self.tar_vocab.update(tar_words)


        idx = len(self.src_w2i)
        for word in self.src_vocab:
            self.src_w2i[word] = idx
            idx += 1

        self.src_i2w = {idx : word for word, idx in self.src_w2i.items()}

        idx = len(self.tar_w2i)
        for word in self.tar_vocab:
            self.tar_w2i[word] = idx
            idx += 1

        self.tar_i2w = {idx : word for word, idx in self.tar_w2i.items()}

    def encode_sentence(self, sentence: str, lang: str):
        words = sentence.lower().split()

        if lang == self.src_lang:
            w2i = self.src_w2i
        elif lang == self.tar_lang:
            w2i = self.tar_w2i
        else:
            raise ValueError(f"Language must be {self.src_lang} or {self.tar_lang}")

        input_ids = [w2i.get(word, self.unk_id) for word in words] # Encode

        input_ids = input_ids[: self.max_len - 2] # Truncation
        input_ids = [self.sos_id] + input_ids + [self.eos_id] # SOS + ids + EOS
        input_ids = input_ids + [self.pad_id] * (self.max_len - len(input_ids)) # Padding
        
        return torch.tensor(input_ids, dtype=torch.long)

    def decode_sentence(self, sentence_ids, lang):
        if lang == self.src_lang:
            i2w = self.src_i2w
        elif lang == self.tar_lang:
            i2w = self.tar_i2w
        else:
            raise ValueError(f"Language must be {self.src_lang} or {self.tar_lang}")

        words = []
        for _id in sentence_ids:
            idx = _id.item() if hasattr(_id, "item") else int(_id)

            if idx in {self.pad_id, self.sos_id}:
                continue

            if idx == self.eos_id:
                break

            words.append(i2w.get(idx, self.unk_piece))

        return " ".join(words)
        




if __name__ == '__main__':
    path = r'src/data/small-PhoMT\small-train.json'
    vocab = Vocab(path, 100)
    sample = {
        "english": "﻿Hurricane Dorian , one of the most powerful storms ever recorded in the Atlantic Ocean , made landfall as a Category 5 storm on Great Abaco Island in the northern Bahamas on Sunday morning , September 1 , 2019 .",
        "vietnamese": "Vào chủ nhật ngày 1-9-2019 , cơn bão Dorian , một trong những cơn bão mạnh nhất được ghi nhận ở Đại Tây Dương , với sức gió 362 km/h đổ bộ vào đảo Great Abaco , miền bắc Bahamas ."
    }

    src_lang ='english'
    tar_lang = 'vietnamese'
    # ids = vocab.encode_sentence(sample[src_lang], src_lang)
    # print(ids)
    # print(vocab.decode_sentence(ids, src_lang))

    print(vocab.src_vocab_size, vocab.tar_vocab_size)

    # ids = vocab.encode_sentence(sample[src_lang], src_lang)
    # print(ids.max(), vocab.src_vocab_size)


