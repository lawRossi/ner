import torch
from .bert_ner import BertNerTagger, prepare_parser
from .modeling_albert import AlbertForTokenClassification, BertConfig
from ..dataset_util import WhitespaceTokenizer, BertFeatureConverter, ConllProcessor, find_class
import os


class AlbertNerTagger(BertNerTagger):
    def initialize_model(self, args):
        num_labels = len(self.data_processor.get_labels()) + 1  # account for padding
        config = BertConfig.from_pretrained(args.bert_model,
                                            num_labels=num_labels, share_type="all")
        model = AlbertForTokenClassification.from_pretrained(args.bert_model, config=config, from_tf=False)
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
        config = BertConfig.from_pretrained(model_dir, num_labels=num_labels, share_type="all")
        model = AlbertForTokenClassification.from_pretrained(model_dir, config=config, from_tf=False)
        return cls(data_processor, converter, model)


def main():
    parser = prepare_parser()
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_index
    if args.do_train:
        data_processor = ConllProcessor()
        tokenizer = WhitespaceTokenizer.from_pretrained(args.bert_model)
        converter = BertFeatureConverter(tokenizer)
        tagger = AlbertNerTagger(data_processor, converter)
        tagger.train(args)
    elif args.do_eval:
        for dir_ in os.listdir(args.output_dir):
            if dir_.startswith("checkpoint"):
                checkpoint_dir = os.path.join(args.output_dir, dir_)
                clf = AlbertNerTagger.load_model(checkpoint_dir)
                clf.test(args)


if __name__ == "__main__":
    # main()
    model = AlbertNerTagger.load_model("output/checkpoint_1572420239/")
    print(model.recognize_nes("梅西进球了吗？"))
