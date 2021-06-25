import pickle
from ..base import NerTagger
import torch
import torch.nn as nn
from torchcrf import CRF
import os.path
import json


class BilstmCrfNerTagger(NerTagger):
    def __init__(self, model, Char, label_vocab, BiChar=None, Lexicon=None, device="cpu") -> None:
        super().__init__()
        self.model = model
        self.Char = Char
        self.BiChar = BiChar
        self.Lexicon = Lexicon
        self.label_vocab = label_vocab
        self.device = device

    def predict_batch(self, texts):
        chars = [self.Char.preprocess(text) for text in texts]
        char_tensors = self.Char.process(chars, device=self.device)
        if self.BiChar is not None:
            bichars = [self.BiChar.preprocess(text) for text in texts]
            bichar_tensors = self.BiChar.process(bichars, device=self.device)
        else:
            bichar_tensors = None
        if self.Lexicon is not None:
            lex_features = [self.Lexicon.preprocess(text) for text in texts]
            lex_tensors = self.Lexicon.process(lex_features, device=self.device)
        else:
            lex_tensors = None
        tags_list = self.model(char_tensors, bichars=bichar_tensors, lex_features=lex_tensors)
        tags = [[self.label_vocab[str(tag)] for tag in tags] for tags in tags_list]
        return tags

    @classmethod
    def load_model(cls, model_dir, device="cpu"):
        model_path = os.path.join(model_dir, "bilstm_crf.pt")
        model = torch.load(model_path, map_location=device)
        model.eval()
        with open(os.path.join(model_dir, "Char.pkl"), "rb") as fi:
            Char = pickle.load(fi)
        bichar_file = os.path.join(model_dir, "BiChar.pkl")
        if os.path.exists(bichar_file):
            with open(bichar_file, "rb") as fi:
                BiChar = pickle.load(fi)
        else:
            BiChar = None
        lex_file = os.path.join(model_dir, "Lexicon.pkl")
        if os.path.exists(lex_file):
            with open(lex_file, "rb") as fi:
                Lexicon = pickle.load(fi)
        else:
            Lexicon = None
        with open(os.path.join(model_dir, "label_vocab.json"), encoding="utf-8") as fi:
            label_vocab = json.load(fi)
        tagger= cls(model, Char, label_vocab, BiChar, Lexicon, device)
        return tagger


class BilstmCrfModel(nn.Module):
    def __init__(self, vocab_size, emb_dims, hidden_dims, num_labels, padding_idx, dropout=0.3, 
            bichar_vocab_size=0, lex_vocab_size=0, lex_emb_dims=20):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_labels = num_labels
        self.hidden_dims = hidden_dims
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(vocab_size, emb_dims, padding_idx=padding_idx)
        input_dims = emb_dims
        if bichar_vocab_size != 0:
            self.bichar_embedding = nn.Embedding(bichar_vocab_size, emb_dims, padding_idx)
            input_dims += emb_dims
        if lex_vocab_size != 0:
            self.lex_embedding = nn.Embedding(lex_vocab_size, lex_emb_dims, padding_idx=0)
            input_dims += lex_emb_dims
        self.bilstm = nn.LSTM(input_dims, hidden_dims//2, num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tags = nn.Linear(hidden_dims, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, chars, labels=None, bichars=None, lex_features=None):
        """
        Args:
            chars (tensor): tensor with shape (batch_size, seq_len)
        """
        batch_size = chars.shape[0]
        embedded_chars = self.embedding(chars)
        features = [embedded_chars]
        if bichars is not None:
            embedded_bichars = self.bichar_embedding(bichars)
            features.append(embedded_bichars)
        if lex_features is not None:
            embedded_lex = self.lex_embedding(lex_features).mean(dim=2)
            features.append(embedded_lex)
        if len(features) > 1:
            embedded_features = torch.cat(features, dim=-1)
        else:
            embedded_features = features[0]
        hidden = self._init_hidden(batch_size, chars.device)
        lstm_output, hidden = self.bilstm(embedded_features, hidden)
        emissions = self.hidden2tags(self.dropout(lstm_output))
        mask = self._compute_mask(chars)
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
    tagger = BilstmCrfNerTagger.load_model("output")
    print(tagger.recognize_nes("巴萨下场比赛什么时候"))
