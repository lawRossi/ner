from typing import Sequence
import jieba
from torchtext.vocab import Vocab
from torchtext import data
from torchtext.datasets import SequenceTaggingDataset


def is_all_chinese(word):
    for _char in word:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True


def tokenize(utterance):
    tokens = jieba.lcut(utterance)
    flat_tokens = []
    for token in tokens:
        if not is_all_chinese(token):
            flat_tokens.append(token)
        else:
            flat_tokens.extend(token)
    return flat_tokens


class NERDataset(data.Dataset):
    @staticmethod
    def sort_key(example):
        for attr in dir(example):
            if not callable(getattr(example, attr)) and \
                    not attr.startswith("__"):
                return len(getattr(example, attr))
        return 0

    def __init__(self, path, fields, encoding="utf-8", separator="\t", **kwargs):
        examples = []
        columns = []

        with open(path, encoding=encoding) as input_file:
            for line in input_file:
                line = line.rstrip()
                if line == "":
                    if columns:
                        examples.append(data.Example.fromlist(columns, fields))
                    columns = []
                else:
                    for i, column in enumerate(line.split(separator)):
                        if len(columns) < i + 1:
                            columns.append([])
                        columns[i].append(column)

            if columns:
                examples.append(data.Example.fromlist(columns, fields))
        super().__init__(examples, fields, **kwargs)


def load_datasets(train_file, dev_file):
    Token = data.Field(lower=True, batch_first=True)
    Tag = data.Field(batch_first=True)
    fileds = [("Token", Token), ("Tag", Tag)]

    train_dataset = NERDataset(train_file, fileds)
    dev_dataset = NERDataset(dev_file, fileds)

    Token.build_vocab(train_dataset)
    Tag.build_vocab(train_dataset)
    return train_dataset, dev_dataset
