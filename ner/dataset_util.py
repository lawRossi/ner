from torchtext import data


def tokenize(text):
    return list(text)


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
    Token = data.Field(batch_first=True, tokenize=tokenize)
    Tag = data.Field(batch_first=True)

    fileds = [("Token", Token), ("Tag", Tag)]
    train_dataset = NERDataset(train_file, fileds)
    dev_dataset = NERDataset(dev_file, fileds)

    Token.build_vocab(train_dataset)
    Tag.build_vocab(train_dataset)
    return train_dataset, dev_dataset
