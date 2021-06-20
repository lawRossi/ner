from ..base import NerTagger
import torch
import torch.nn as nn
from torchcrf import CRF
import tqdm
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from torch.optim import Adam
import os.path
from ..dataset_util import ConllProcessor, CharTokenizer, InputExample, TextConverter
import argparse
import json
from ..ner_evaluate import evaluate


class BilstmCrfNerTagger(NerTagger):
    def __init__(self, data_processor, converter):
        super().__init__(data_processor, converter)

    def train(self, args):
        device = torch.device(args.device)
        examples = self.data_processor.get_train_examples(args.data_dir)
        label_list = self.data_processor.get_labels()
        self.converter.build_vocabulary(examples, label_list)
        features = self.converter.convert_examples_to_features(examples, label_list)
        model = BilstmCrfModel(len(self.converter.vocab)+1, args.emb_dims, args.hidden_dims, len(label_list), args.dropout)
        model.to(device)

        all_input_ids = torch.tensor([feature.input_ids for feature in features])
        all_labels = torch.tensor([feature.label_ids for feature in features])
        dataset = TensorDataset(all_input_ids, all_labels)
        sampler = RandomSampler(dataset)
        data_loader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size)
        optimizer = Adam(model.parameters(), lr=args.learning_rate)
        total_loss = 0
        for _ in range(args.num_train_epochs):
            tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
            for i, values in enumerate(tk0):
                values = [value.to(device) for value in values]
                sentences, tags = values
                loss = model(sentences, tags)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                if (i + 1) % args.log_interval == 0:
                    tk0.set_postfix(loss=total_loss/args.log_interval)
                    total_loss = 0
        model_dir = args.output_dir
        model_path = os.path.join(model_dir, "bilstm_crf.pt")
        torch.save(model, model_path)
        if args is not None:
            output_args_file = os.path.join(model_dir, "training_args.bin")
            torch.save(args, output_args_file)
        label_file = os.path.join(model_dir, "labels.txt")
        with open(label_file, "w", encoding="utf-8") as fo:
            fo.write("||".join(self.data_processor.get_labels()))
        with open(os.path.join(model_dir, "vocab.json"), "w", encoding="utf-8") as fo:
            json.dump(self.converter.vocab, fo)
        with open(os.path.join(model_dir, "label_vocab.json"), "w", encoding="utf-8") as fo:
            json.dump(self.converter.label_vocab, fo)
    
    def test(self, args):
        device = torch.device(args.device)
        examples = self.data_processor.get_dev_examples(args.data_dir)
        label_list = self.data_processor.get_labels()
        self.converter.build_vocabulary(examples, label_list)
        features = self.converter.convert_examples_to_features(examples, label_list)
        self.model.to(device)

        all_input_ids = torch.tensor([feature.input_ids for feature in features])
        all_labels = torch.tensor([feature.label_ids for feature in features])
        dataset = TensorDataset(all_input_ids, all_labels)
        sampler = SequentialSampler(dataset)
        data_loader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size)
        tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
        all_pred_labels = []
        all_true_labels = []
        for i, values in enumerate(tk0):
            values = [value.to(device) for value in values]
            sentences, tags = values
            preds = self.model(sentences)
            all_pred_labels.extend([[label_list[idx] for idx in pred] for pred in preds])
            all_true_labels.extend([[label_list[idx] for idx in tag] for tag in tags])
        
        evaluate(all_true_labels, all_pred_labels)

    def predict_batch(self, texts):
        examples = [InputExample(guid=None, text_a=text) for text in texts]
        label_list = self.data_processor.get_labels()
        features = self.converter.convert_examples_to_features(examples, label_list)
        all_input_ids = torch.tensor([feature.input_ids for feature in features])
        tags_idxes = self.model(all_input_ids)
        tags = [[label_list[idx] for idx in idxes] for idxes in tags_idxes]
        return tags

    @classmethod
    def load_model(cls, model_dir):
        output_args_file = os.path.join(model_dir, "training_args.bin")
        args = torch.load(output_args_file)
        label_file = os.path.join(model_dir, "labels.txt")
        with open(label_file, encoding="utf-8") as fi:
            labels = set(fi.read().strip().split("||"))
        with open(os.path.join(model_dir, "vocab.json"), encoding="utf-8") as fi:
            vocab = json.load(fi)
        with open(os.path.join(model_dir, "label_vocab.json"), encoding="utf-8") as fi:
            label_vocab = json.load(fi)
        data_processor = ConllProcessor(labels)
        tokenizer = CharTokenizer()
        converter = TextConverter(tokenizer, args.max_seq_length, vocab, label_vocab)
        model_path = os.path.join(model_dir, "bilstm_crf.pt")
        model = torch.load(model_path)
        tagger= cls(data_processor, converter)
        tagger.model = model
        return tagger


class BilstmCrfModel(nn.Module):
    def __init__(self, vocab_size, emb_dims, hidden_dims, num_labels, dropout=0.3):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_labels = num_labels
        self.hidden_dims = hidden_dims
        self.embedding = nn.Embedding(vocab_size, emb_dims, padding_idx=0)
        self.bilstm = nn.LSTM(emb_dims, hidden_dims//2, num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tags = nn.Linear(hidden_dims, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequences, labels=None):
        """
        Args:
            sequences (tensor): tensor with shape (batch_size, seq_len)
        """
        batch_size = sequences.shape[0]
        embedded = self.dropout(self.embedding(sequences))
        hidden = self._init_hidden(batch_size)
        lstm_output, hidden = self.bilstm(embedded, hidden)
        emissions = self.hidden2tags(self.dropout(lstm_output))
        mask = self._compute_mask(sequences)
        if labels is not None:
            log_likelihood = self.crf(emissions, labels, mask=mask)
            return -log_likelihood
        else:
            tags = self.crf.decode(emissions, mask=mask)
            return tags

    def _compute_mask(self, sequences):
        return sequences != 0

    def _init_hidden(self, batch_size):
        return (torch.zeros(2, batch_size, self.hidden_dims // 2),
                torch.zeros(2, batch_size, self.hidden_dims // 2))


def prepare_parser():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")

    parser.add_argument("--output_dir",
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")

    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")

    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")

    parser.add_argument("--emb_dims",
                        default=100,
                        type=int,
                        help="embedding dimentions")

    parser.add_argument("--hidden_dims",
                        default=100,
                        type=int,
                        help="hidden dimentions")

    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")

    parser.add_argument("--num_train_epochs",
                        default=3,
                        type=int,
                        help="Total number of training epochs to perform.")
    
    parser.add_argument("--dropout",
                        default=0.3,
                        type=float,
                        help="dropout rate")

    parser.add_argument("--log_interval",
                        type=int,
                        default=5)

    parser.add_argument("--cuda_index",
                        type=str,
                        default="-1",
                        help="Specify GPU to use")

    parser.add_argument("--device",
                        type=str,
                        default="cpu",
                        help="Specify which device to use")
    return parser


def main():
    parser = prepare_parser()
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_index
    if args.do_train:
        data_processor = ConllProcessor()
        tokenizer = CharTokenizer()
        converter = TextConverter(tokenizer, args.max_seq_length)
        tagger = BilstmCrfNerTagger(data_processor, converter)
        tagger.train(args)

    elif args.do_eval:
        tagger = BilstmCrfNerTagger.load_model(args.output_dir)
        tagger.test(args)


if __name__ == "__main__":
    print("??")
    main()
