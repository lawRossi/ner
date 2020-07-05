import torch
import torch.nn as nn
from .modeling_albert_zh import AlbertModel, AlbertPreTrainedModel, BertConfig
from torchcrf import CRF
import os
from ..dataset_util import WhitespaceTokenizer, ConllProcessor, BertFeatureConverter, find_class
from .bert_crf_ner import BertCrfNerTagger, prepare_parser


class AlbertCrfForTokenClassification(AlbertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        emissions = self.classifier(sequence_output)

        if labels is not None:
            log_likelyhood, tags = self.crf(emissions, labels),  self.crf.decode(emissions)
            tags = torch.tensor(tags, dtype=torch.long)
            return (-log_likelyhood, tags)
        else:
            tags = self.crf.decode(emissions)
            tags = torch.tensor(tags, dtype=torch.long)
            return (tags, )


class AlbertCrfNerTagger(BertCrfNerTagger):
    def initialize_model(self, args):
        num_labels = len(self.data_processor.get_labels()) + 1  # account for padding
        config = BertConfig.from_pretrained(args.bert_model,
                                            num_labels=num_labels)
        model = AlbertCrfForTokenClassification.from_pretrained(args.bert_model, config=config, from_tf=False)
        self.model = model

    @classmethod
    def load_model(cls, model_dir):
        output_args_file = os.path.join(model_dir, "training_args.bin")
        args = torch.load(output_args_file)
        label_file = os.path.join(model_dir, "labels.txt")
        with open(label_file, encoding="utf-8") as fi:
            labels = set(fi.read().strip().split("||"))
        if args.data_processor is not None:
            data_processor = find_class(args.data_processor)(labels)
        else:
            data_processor = ConllProcessor(labels)
        if args.tokenizer is not None:
            tokenizer = find_class(args.tokenizer)()
        else:
            tokenizer = WhitespaceTokenizer.from_pretrained(model_dir, do_lower_case=args.do_lower_case)
        if args.converter is not None:
            converter = find_class(args.converter)(tokenizer, args.max_seq_length)
        else:
            converter = BertFeatureConverter(tokenizer, args.max_seq_length)
        num_labels = len(labels) + 1  # accounts for padding
        config = BertConfig.from_pretrained(model_dir, num_labels=num_labels)
        model = AlbertCrfForTokenClassification.from_pretrained(model_dir, config=config, from_tf=False)
        return cls(data_processor, converter, model)


def main():
    parser = prepare_parser()
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_index
    if args.do_train:
        data_processor = ConllProcessor()
        tokenizer = WhitespaceTokenizer.from_pretrained(args.bert_model)
        converter = BertFeatureConverter(tokenizer)
        tagger = AlbertCrfNerTagger(data_processor, converter)
        tagger.train(args)
    elif args.do_eval:
        for dir_ in os.listdir(args.output_dir):
            if dir_.startswith("checkpoint"):
                checkpoint_dir = os.path.join(args.output_dir, dir_)
                clf = AlbertCrfNerTagger.load_model(checkpoint_dir)
                clf.test(args)


if __name__ == "__main__":
    main()
