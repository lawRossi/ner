from transformers import BertTokenizer
import logging
import os
import importlib


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class WhitespaceTokenizer(BertTokenizer):
    # def __init__(self, vocab_file, do_lower_case=True, max_len=None, do_basic_tokenize=True,
    #              never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
    #     super().__init__(vocab_file, do_lower_case, max_len, do_basic_tokenize, never_split)
    #     self.do_lower_case = do_lower_case
    def __init__(self, vocab_file, do_lower_case=True, do_basic_tokenize=True, never_split=None,
                 unk_token="[UNK]", sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]",
                 mask_token="[MASK]", tokenize_chinese_chars=True, **kwargs):
        super().__init__(vocab_file, do_lower_case, do_basic_tokenize, never_split, unk_token,sep_token,
        pad_token, cls_token, mask_token, tokenize_chinese_chars, **kwargs)
        self.do_lower_case = do_lower_case

    def _tokenize(self, text):
        if self.do_lower_case:
            text = text.lower()
        tokens = text.split(" ")
        tokens = map(self.check_in_vocab, tokens)
        return list(tokens)

    def check_in_vocab(self, token):
        if token in self.vocab:
            return token
        else:
            return "[UNK]"


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


class DataProcessor:
    """Base class for data converters for sequence classification data sets."""

    def __init__(self, labels=None):
        self.labels = labels

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class ConllProcessor(DataProcessor):
    """Processor for the NER model."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.txt")))
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "dev.txt")), "dev")

    def get_labels(self):
        """See base class."""
        return list(sorted(self.labels))

    def _read_file(self, path):
        samples = []
        with open(path, encoding="utf-8") as fi:
            text = []
            tags = []
            for line in fi:
                line = line.strip()
                if line.startswith("-DOCSTART- "):
                    if text:
                        samples.append((" ".join(text), tags))
                        text = []
                        tags = []
                elif line:
                    splits = line.split("\t")
                    text.append(splits[0])
                    tags.append(splits[-1])
                else:
                    if text:
                        samples.append((" ".join(text), tags))
                        text = []
                        tags = []
            if text:  # the last document
                samples.append((" ".join(text), tags))
        return samples

    def _create_examples(self, samples, set_type):
        """Creates examples for the training and dev sets."""
        if self.labels is None:
            self.labels = set()
        examples = []
        for (i, (text, tags)) in enumerate(samples):
            guid = "%s-%s" % (set_type, i)
            text_a = text
            labels = tags
            if set_type == "train":
                self.labels = self.labels.union(tags)
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=labels))
        return examples


class FeatureConverter():
    def __init__(self, tokenizer, max_seq_len=120):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def convert_examples_to_features(self, examples, label_list):
        raise NotImplementedError


class BertFeatureConverter(FeatureConverter): 
    def convert_examples_to_features(self, examples, label_list):
        """Loads a data file into a list of `InputBatch`s."""
        logger.info("labels: %s" % " ".join(label_list))
        label_map = {label: i + 1 for i, label in enumerate(label_list)}  # 0 reserved for padding

        features = []
        for (ex_index, example) in enumerate(examples):
            tokens_a = self.tokenizer.tokenize(example.text_a)
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > self.max_seq_len - 2:
                logger.warning("sequence to0 long")
                tokens_a = tokens_a[:(self.max_seq_len - 2)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids: 0   0   0   0  0     0 0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambigiously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            label_ids = None
            if example.label:
                label_ids = [0] + [label_map[item] for item in example.label] + [0]
                if len(input_ids) != len(label_ids):
                    logger.warning("tokenization error")
                    continue

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (self.max_seq_len - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding
            if label_ids:
                label_ids += padding

            assert len(input_ids) == self.max_seq_len
            assert len(input_mask) == self.max_seq_len
            assert len(segment_ids) == self.max_seq_len
            if label_ids:
                assert len(label_ids) == self.max_seq_len

            # if ex_index < 3:
            #     logger.info("*** Example ***")
            #     logger.info("guid: %s" % (example.guid))
            #     logger.info("tokens: %s" % " ".join(
            #             [str(x) for x in tokens]))
            #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            #     logger.info(
            #             "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            #     if label_ids:
            #         logger.info("label: %s " % " ".join([str(x) for x in label_ids]))

            features.append(
                    InputFeatures(input_ids=input_ids,
                                input_mask=input_mask,
                                segment_ids=segment_ids,
                                label_ids=label_ids))
        return features


def find_class(import_path):
    module = import_path[:import_path.rfind(".")]
    class_name = import_path[import_path.rfind(".")+1:]
    ip_module = importlib.import_module(".", module)
    class_ = getattr(ip_module, class_name)
    return class_
