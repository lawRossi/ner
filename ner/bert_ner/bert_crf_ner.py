from transformers.configuration_bert import BertConfig
from transformers.modeling_bert import BertPreTrainedModel, BertModel
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torchcrf import CRF
from .bert_ner import BertNerTagger, prepare_parser
import os
from ..dataset_util import ConllProcessor, BertFeatureConverter, WhitespaceTokenizer, find_class


class BertCrfForTokenClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.num_labels)``
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForTokenClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, scores = outputs[:2]
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
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


class BertCrfNerTagger(BertNerTagger):
    def initialize_model(self, args):
        num_labels = len(self.data_processor.get_labels()) + 1  # account for padding
        config = BertConfig.from_pretrained(args.bert_model,
                                            num_labels=num_labels)
        model = BertCrfForTokenClassification.from_pretrained(args.bert_model, config=config, from_tf=False)
        self.model = model

    def collect_labels(self, out, seq_lens, label_list, labels=None):
        label_map = {i+1: label for i, label in enumerate(label_list)}
        label_map[0] = "O"
        preds = []
        if labels is not None:
            true_labels = []
        for i in range(seq_lens.shape[0]):
            preds_ = out[i][:seq_lens[i]].tolist()[1:-1]
            preds.append([label_map[pred] for pred in preds_])
            if labels is not None:
                labels_ = labels[i][:seq_lens[i]].tolist()[1:-1]
                true_labels.append([label_map[label] for label in  labels_])
        if labels is not None:
            return preds, true_labels
        return preds
    
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
        model = BertCrfForTokenClassification.from_pretrained(model_dir, config=config, from_tf=False)
        return cls(data_processor, converter, model)


def main():
    parser = prepare_parser()
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_index
    if args.do_train:
        data_processor = ConllProcessor()
        tokenizer = WhitespaceTokenizer.from_pretrained(args.bert_model)
        converter = BertFeatureConverter(tokenizer)
        tagger = BertCrfNerTagger(data_processor, converter)
        tagger.train(args)
    elif args.do_eval:
        for dir_ in os.listdir(args.output_dir):
            if dir_.startswith("checkpoint"):
                checkpoint_dir = os.path.join(args.output_dir, dir_)
                clf = BertCrfNerTagger.load_model(checkpoint_dir)
                clf.test(args)


if __name__ == "__main__":
    main()
