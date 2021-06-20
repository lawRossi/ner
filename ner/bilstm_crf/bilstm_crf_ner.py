from ..base import NerTagger
import torch
import torch.nn as nn
from torchcrf import CRF
import tqdm
from torch.optim import Adam
import os.path
import json


class BilstmCrfNerTagger(NerTagger):
    def __init__(self, data_processor, converter):
        super().__init__(data_processor, converter)

    def predict_batch(self, texts):
        self.model.eval()
        examples = [InputExample(guid=None, text_a=text) for text in texts]
        label_list = self.data_processor.get_labels()
        features = self.converter.convert_examples_to_features(examples, label_list)
        all_input_ids = torch.tensor([feature.input_ids for feature in features])
        tags_idxes = self.model(all_input_ids)
        tags = [[label_list[idx] for idx in idxes] for idxes in tags_idxes]
        return tags

    @classmethod
    def load_model(cls, model_dir, device="cpu"):
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
        model = torch.load(model_path, map_location=device)
        tagger= cls(data_processor, converter)
        tagger.model = model
        tagger.model.eval()
        return tagger


class BilstmCrfModel(nn.Module):
    def __init__(self, vocab_size, emb_dims, hidden_dims, num_labels, padding_idx, dropout=0.3):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_labels = num_labels
        self.hidden_dims = hidden_dims
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(vocab_size, emb_dims, padding_idx=padding_idx)
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
        hidden = self._init_hidden(batch_size, sequences.device)
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
        return sequences != self.padding_idx

    def _init_hidden(self, batch_size, device):
        return (torch.zeros(2, batch_size, self.hidden_dims // 2, device=device),
                torch.zeros(2, batch_size, self.hidden_dims // 2, device=device))


if __name__ == "__main__":
    # main()
    tagger = BilstmCrfNerTagger.load_model("output")
    print(tagger.predict("你喜欢梅西吗"))