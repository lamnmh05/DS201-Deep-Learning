import torch
from pyvi import ViTokenizer
import json

class Vocab():
    def __init__(self, path, sentence_key: str, label_key:str, max_len: int = 100):
        self.path = path
        self.sentence_key = sentence_key
        self.label_key = label_key
        self.max_len = max_len

        self.vocab = [] 
        self.labels = set()

        self.pad_id = 0
        self.unk_id = 1

        self.w2i = {}
        self.i2w = {}

        self.i2l = {}
        self.l2i = {}

        self.make_vocab()

    def make_vocab(self):
        data = json.load(open(self.path, 'r', encoding='utf-8'))

        for item in data:

            try:
                # Text classification
                if isinstance(item[self.sentence_key], str):
                    tokenized_sentence = ViTokenizer.tokenize(item[self.sentence_key])
                    self.vocab.extend(tokenized_sentence.split())
                
                if isinstance(item[self.label_key], str):
                    self.labels.add(item[self.label_key])


                # Sequential labeling
                if isinstance(item[self.sentence_key], list):
                    for word in item[self.sentence_key]:
                        self.vocab.append(word)
                
                if isinstance(item[self.label_key], list):
                    for i in item[self.label_key]:
                        self.labels.add(i)

            except:
                raise Exception('Wrong input type')

        self.vocab = list(set(self.vocab))

        self.w2i = {word : idx for idx, word in enumerate(self.vocab, 2)}
        self.i2w = {idx : word for word, idx in self.w2i.items()}

        self.l2i = {label : idx for idx, label in enumerate(self.labels)}
        self.i2l = {idx: label for label, idx in self.l2i.items()}


    def encode_sentence(self, input: str):
        try:
            if isinstance(input, str): # Text classification
                tokenized_sentence = ViTokenizer.tokenize(input)
                tokens = tokenized_sentence.split()

            if isinstance(input, list): # Sequential labeling
                tokens = input
        except:
            raise Exception('Wrong input type')


        input_ids = []
        for token in tokens:
            try:
                input_ids.append(self.w2i[token])
            except:
                input_ids.append(self.unk_id)


        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len] # Truncation
        else:
            input_ids.extend([self.pad_id] * (self.max_len - len(input_ids))) # Padding

        return torch.tensor(input_ids, dtype=torch.long)
    
        
    def encode_label(self, labels: str):
        if isinstance(labels, str): # Text classification
            return torch.tensor([self.l2i[labels]], dtype=torch.long)
        
        elif isinstance(labels, list): # Sequential labeling
            final = [self.l2i[label] for label in labels]
            
            if len(labels) > self.max_len:
                final = final[:self.max_len]
                
            else:
                final.extend([-100] * (self.max_len - len(labels)))


            return torch.tensor(final, dtype=torch.long)
        
        else:
            raise Exception('Wrong input type')


    def decode_label(self, label_vec: torch.Tensor):
        labels = []
        for label_id in label_vec:
            idx = label_id.item()
            if idx == -100: # ignore index for cross entropy loss
                continue
            labels.append(self.i2l[idx])
        return labels
    
    @property
    def vocab_size(self):
        return len(self.vocab) + 2 # unk, pad id

    @property
    def num_labels(self):
        return len(self.labels)

if __name__ == '__main__':
    # path = r'PhoNER\word\test_word.json'
    # data = []
    # with open(path, 'r', encoding='utf-8') as file:
    #     for line in file:
    #         item = json.loads(line)
    #         data.append(item)

    # with open("PhoNER/test.json", "w", encoding="utf-8") as f:
    #     json.dump(data, f, ensure_ascii=False, indent=4)

    path = r'UIT-VSFC\UIT-VSFC-train.json'
    sentence_key='sentence' 
    label_key='sentiment'

    vocab = Vocab(path, sentence_key, label_key)

    sample = {
        'sentence' : 'giảng buồn ngủ',
        'label': 'negative'
    }

    print(vocab.encode_sentence(sample['sentence']))
    print(vocab.vocab_size)
    print(vocab.num_labels)
    print(vocab.l2i)
    print(vocab.encode_label(sample['label']))
    print(vocab.decode_label(vocab.encode_label(sample['label'])))

    sample = {
        "words": [
            "Từ",
            "24",
            "-",
            "7",
            "đến",
            "31",
            "-",
            "7",
            ",",
            "bệnh",
            "nhân",
            "được",
            "mẹ",
            "là",
            "bà",
            "H.T.P",
            "(",
            "47",
            "tuổi",
            ")",
            "đón",
            "về",
            "nhà",
            "ở",
            "phường",
            "Phước",
            "Hoà",
            "(",
            "bằng",
            "xe",
            "máy",
            ")",
            ",",
            "không",
            "đi",
            "đâu",
            "chỉ",
            "ra",
            "Tạp",
            "hoá",
            "Phượng",
            ",",
            "chợ",
            "Vườn",
            "Lài",
            ",",
            "phường",
            "An",
            "Sơn",
            "cùng",
            "mẹ",
            "bán",
            "tạp",
            "hoá",
            "ở",
            "đây",
            "."
        ],
        "tags": [
            "O",
            "B-DATE",
            "I-DATE",
            "I-DATE",
            "O",
            "B-DATE",
            "I-DATE",
            "I-DATE",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "B-NAME",
            "O",
            "B-AGE",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "B-LOCATION",
            "I-LOCATION",
            "I-LOCATION",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "B-LOCATION",
            "I-LOCATION",
            "I-LOCATION",
            "O",
            "B-LOCATION",
            "I-LOCATION",
            "I-LOCATION",
            "O",
            "B-LOCATION",
            "I-LOCATION",
            "I-LOCATION",
            "O",
            "O",
            "B-JOB",
            "I-JOB",
            "I-JOB",
            "O",
            "O",
            "O"
        ]
    }
    path = 'PhoNER\dev.json'
    sentence_key='words' 
    label_key='tags'

    vocab = Vocab(path, sentence_key, label_key)
    print(vocab.vocab_size)
    print(vocab.num_labels)
    print(vocab.l2i)


    print(vocab.encode_sentence(sample['words']))

    encoded_label = vocab.encode_label(sample['tags'])
    print(encoded_label)
    print(vocab.decode_label(encoded_label))