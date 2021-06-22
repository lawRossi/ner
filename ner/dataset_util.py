from torchtext import data
from torchtext.data.iterator import batch


def tokenize(text):
    return list(text)

def tokenize_bichar(text):
    bichars = [text[i:i+2] for i in range(len(text)-1)]
    bichars.append(text[-1] + "<eos>")
    return bichars


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
        use_bichar = kwargs.get("use_bichar", False)
        with open(path, encoding=encoding) as input_file:
            for line in input_file:
                line = line.rstrip()
                if line == "":
                    if columns:
                        if use_bichar:
                            columns = self._add_bichar(columns)
                        examples.append(data.Example.fromlist(columns, fields))
                    columns = []
                else:
                    for i, column in enumerate(line.split(separator)):
                        if len(columns) < i + 1:
                            columns.append([])
                        columns[i].append(column)

            if columns:
                if use_bichar:
                    columns = self._add_bichar(columns)
                examples.append(data.Example.fromlist(columns, fields))
        if "use_bichar" in kwargs:
            del kwargs["use_bichar"]
        super().__init__(examples, fields, **kwargs)
    
    def _add_bichar(self, columns):
        chars, tags = columns
        bichars = ["".join(chars[i:i+2]) for i in range(len(chars)-1)]
        bichars.append(chars[-1] + "<eos>")
        columns = [chars, bichars, tags]
        return columns


def load_datasets(train_file, dev_file, use_bichar=False, min_tf=2, min_bichar_tf=2):
    Char = data.Field(batch_first=True, tokenize=tokenize)
    Tag = data.Field(batch_first=True)
    if use_bichar:
        BiChar = data.Field(batch_first=True, tokenize=tokenize_bichar)
        fields = [("Char", Char), ("BiChar", BiChar), ("Tag", Tag)]
    else:
        fields = [("Char", Char), ("Tag", Tag)]
    train_dataset = NERDataset(train_file, fields, use_bichar=use_bichar)
    dev_dataset = NERDataset(dev_file, fields, use_bichar=use_bichar)

    Char.build_vocab(train_dataset, min_freq=min_tf)
    if use_bichar:
        BiChar.build_vocab(train_dataset, min_freq=min_bichar_tf)
    Tag.build_vocab(train_dataset)
    return train_dataset, dev_dataset
