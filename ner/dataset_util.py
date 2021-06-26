from torchtext import data
from torchtext.data import field
import ahocorasick


def identity(x):
    return x


def tokenize(text):
    return list(text)

def tokenize_bichar(text):
    bichars = [text[i:i+2] for i in range(len(text)-1)]
    bichars.append(text[-1] + "<eos>")
    return bichars


def build_automaton(dict_file):
    automaton = ahocorasick.Automaton()
    with open(dict_file, encoding="utf-8") as fi:
        for line in fi:
            word, entity_type = line.strip().split("\t")
            automaton.add_word(word, (word, entity_type))
    automaton.make_automaton()
    return automaton


class AcTokenizer:
    def __init__(self, dict_file):
        self.automaton = build_automaton(dict_file)

    def __call__(self, text):
        tags = [[] for _ in range(len(text))]
        for pos, match in self.automaton.iter(text):
            word, entity_type = match
            i = pos - len(word) + 1
            if len(word) == 1:
                tags[i].append(f"S_{entity_type}")
            else:
                tags[i].append(f"START_{entity_type}")
            for j in range(len(word)-2):
                tags[i+j+1].append(f"INTER_{entity_type}")
            if len(word) > 1:
                tags[i+len(word)-1].append(f"END_{entity_type}")
        return tags


class NERDataset(data.Dataset):
    @staticmethod
    def sort_key(example):
        for attr in dir(example):
            if not callable(getattr(example, attr)) and \
                    not attr.startswith("__"):
                return len(getattr(example, attr))
        return 0

    def __init__(self, path, fields, separator="\t", use_bichar=False, 
            use_lexicon=False, **kwargs):
        examples = []
        columns = []
        with open(path, encoding="utf-8") as input_file:
            for line in input_file:
                line = line.rstrip()
                if line == "":
                    if columns:
                        if use_bichar:
                            self._add_bichar(columns)
                        if use_lexicon:
                            self._add_lexicon_features(columns, fields)
                        examples.append(data.Example.fromlist(columns, fields))
                    columns = []
                else:
                    for i, column in enumerate(line.split(separator)):
                        if len(columns) < i + 1:
                            columns.append([])
                        columns[i].append(column)
            if columns:
                if use_bichar:
                    self._add_bichar(columns)
                if use_lexicon:
                    self._add_lexicon_features(columns, fields)
                examples.append(data.Example.fromlist(columns, fields))
        super().__init__(examples, fields, **kwargs)

    def _add_bichar(self, columns):
        chars = columns[0]
        bichars = ["".join(chars[i:i+2]) for i in range(len(chars)-1)]
        bichars.append(chars[-1] + "<eos>")
        columns.append(bichars)

    def _add_lexicon_features(self, columns, fileds):
        chars = columns[0]
        text = "".join(chars)
        Lexicon = fileds[-1][1]
        lexicon_features = Lexicon.preprocess(text)
        columns.append(lexicon_features)


def load_datasets(train_file, dev_file, use_bichar=False, dict_file=None, 
        min_tf=2, min_bichar_tf=2):
    Char = data.Field(batch_first=True, tokenize=tokenize)
    Tag = data.Field(batch_first=True, is_target=True, unk_token=None)
    fields = [("Char", Char), ("Tag", Tag)]
    if use_bichar:
        BiChar = data.Field(batch_first=True, tokenize=tokenize_bichar)
        fields.append(("BiChar", BiChar))
    if dict_file is not None:
        use_lexicon = True
        nesting_field = data.Field(batch_first=True, tokenize=identity, unk_token=None, fix_length=3)
        Lexicon = data.NestedField(nesting_field, tokenize=AcTokenizer(dict_file))
        fields.append(("Lexicon", Lexicon))
    else:
        use_lexicon = False
    train_dataset = NERDataset(train_file, fields, use_bichar=use_bichar, use_lexicon=use_lexicon)
    dev_dataset = NERDataset(dev_file, fields, use_bichar=use_bichar, use_lexicon=use_lexicon)

    Char.build_vocab(train_dataset, min_freq=min_tf)
    if use_bichar:
        BiChar.build_vocab(train_dataset, min_freq=min_bichar_tf)
    if dict_file is not None:
        Lexicon.build_vocab(train_dataset)
    Tag.build_vocab(train_dataset)

    return train_dataset, dev_dataset
