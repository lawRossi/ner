from torchtext import data
from transformers.models.auto.tokenization_auto import AutoTokenizer
# import ahocorasick
import os.path


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


class TransformersTokenizer:
    def __init__(self, model_name_or_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def __call__(self, text):
        tokens = tokenize(text)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        token_ids.insert(0, self.tokenizer.cls_token_id)
        token_ids.append(self.tokenizer.sep_token_id)
        token_type_ids = [0] * len(token_ids)
        attention_mask = [1] * len(token_ids)
        return {"input_ids": token_ids, "token_type_ids": token_type_ids, "attention_mask": attention_mask}


class BertNERDataset(data.Dataset):
    @staticmethod
    def sort_key(example):
        for attr in dir(example):
            if not callable(getattr(example, attr)) and \
                    not attr.startswith("__"):
                return len(getattr(example, attr))
        return 0

    def __init__(self, path, fields, tokenizer, separator="\t", use_lexicon=False, **kwargs):
        examples = []
        columns = []
        with open(path, encoding="utf-8") as input_file:
            for line in input_file:
                line = line.rstrip()
                if line == "":
                    if columns:
                        if use_lexicon:
                            self._add_lexicon_features(columns, fields)
                        columns = self._convert_tokens(tokenizer, columns)
                        examples.append(data.Example.fromlist(columns, fields))
                    columns = []
                else:
                    for i, column in enumerate(line.split(separator)):
                        if len(columns) < i + 1:
                            columns.append([])
                        columns[i].append(column)
            if columns:
                if use_lexicon:
                    self._add_lexicon_features(columns, fields)
                columns = self._convert_tokens(tokenizer, columns)
                examples.append(data.Example.fromlist(columns, fields))
        super().__init__(examples, fields, **kwargs)
    
    def _convert_tokens(self, tokenizer, columns):
        text = "".join(columns[0])
        result = tokenizer(text)
        tags = columns[1]
        return [result["input_ids"], result["token_type_ids"], result["attention_mask"], tags]


def load_datasets_for_bert(data_dir, tokenizer, dict_file=None):
    Token = data.Field(batch_first=True, use_vocab=False, unk_token=None, pad_token=0)
    TokenType = data.Field(batch_first=True, use_vocab=False, unk_token=None, pad_token=0)
    Mask = data.Field(batch_first=True, use_vocab=False, unk_token=None, pad_token=0)
    Tag = data.Field(batch_first=True, is_target=True, unk_token=None)
    fields = [("Token", Token), ("TokenType", TokenType), ("Mask", Mask), ("Tag", Tag)]
    train_file = os.path.join(data_dir, "train.txt")
    dev_file = os.path.join(data_dir, "dev.txt")
    train_dataset = BertNERDataset(train_file, fields, tokenizer)
    dev_dataset = BertNERDataset(dev_file, fields, tokenizer)
    # Token.build_vocab(train_dataset)
    # TokenType.build_vocab(train_dataset)
    # Mask.build_vocab(train_dataset)
    Tag.build_vocab(train_dataset)
    return train_dataset, dev_dataset


# def tokenize_and_align_labels(tokenizer, tokens, labels, label_all_tokens=True):
#     tokenized_inputs = tokenizer(
#         tokens,
#         truncation=True,
#         # We use this argument because the texts in our dataset are lists of words
#         #  (with a label for each word).
#         is_split_into_words=True,
#     )
#     new_labels = []
#     word_ids = tokenized_inputs.word_ids(batch_index=0)
#     previous_word_idx = None
#     label_ids = []
#     for word_idx in word_ids:
#         # Special tokens have a word id that is None. We set the label to "<pad>" 
#         # so they are automatically ignored in the loss function.
#         if word_idx is None:
#             new_labels.append("<pad>")
#         # We set the label for the first token of each word.
#         elif word_idx != previous_word_idx:
#             label_ids.append(labels[word_idx]])
#         # For the other tokens in a word, we set the label to either the current label or <pad>,
#         # depending on the label_all_tokens flag.
#         else:
#             label_ids.append(labels[word_idx] if label_all_tokens else "<pad>")
#         previous_word_idx = word_idx

#     tokenized_inputs["labels"] = labels
#     return tokenized_inputs


