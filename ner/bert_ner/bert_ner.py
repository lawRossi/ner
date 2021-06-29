from ner.base import NerTagger
from transformers import AutoModel, AutoConfig, AutoTokenizer
import torch
import torch.nn as nn
from torchcrf import CRF
import os.path
import pickle
from ..dataset_util import TransformersTokenizer
import json



class BertForNER(nn.Module):
    def __init__(self, model_name_or_path, num_labels, use_crf=False, dropout=0.3) -> None:
        super().__init__()
        self.bert_model = AutoModel.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        config = AutoConfig.from_pretrained(model_name_or_path)
        hidden_dims = config.hidden_size
        self.hidden2tags = nn.Linear(hidden_dims, num_labels)
        self.use_crf = use_crf
        if use_crf:
            self.crf = CRF(num_labels, batch_first=True)
        else:
            self.loss = nn.CrossEntropyLoss(ignore_index=0)  # 0 is used for padding
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
        output = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        hidden = output.last_hidden_state
        logits = self.hidden2tags(self.dropout(hidden))
        mask = self._compute_mask(input_ids)
        if self.use_crf:
            if labels is not None:
                log_likelihood = self.crf(logits, labels, mask=mask)
                return -log_likelihood
            else:
                tags = self.crf.decode(logits, mask=mask)
                return tags
        else:
            if labels is not None:
                logits = logits.view(-1, logits.shape[-1])
                labels = labels.view(-1)
                loss = self.loss(logits, labels)
                return loss
            else:
                tags_list = logits.argmax(dim=-1).cpu().detach().numpy()
                tags = []
                seq_lens = mask.cpu().numpy().sum(axis=1)
                for i, tags_ in enumerate(tags_list):
                    tags.append(tags_[:seq_lens[i]])
                return tags

    def _compute_mask(self, input_ids):
        mask = input_ids != 0
        return mask
 

class BertNERTagger(NerTagger):
    def __init__(self, model, tokenizer, label_vocab, Lexicon=None, device="cpu") -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.label_vocab = label_vocab
        self.Lexicon = Lexicon
        self.device = device

    @classmethod
    def load_model(cls, model_dir, device="cpu"):
        model_path = os.path.join(model_dir, "bert_ner.pt")
        model = torch.load(model_path, map_location=device)
        tokenizer = TransformersTokenizer(model_dir)
        model.eval()
        # lex_file = os.path.join(model_dir, "Lexicon.pkl")
        # if os.path.exists(lex_file):
        #     with open(lex_file, "rb") as fi:
        #         Lexicon = pickle.load(fi)
        #     args = torch.load(os.path.join(model_dir, "args.pkl"))
        #     Lexicon.tokenize = AcTokenizer(args.dict_file)
        # else:
        Lexicon = None
        with open(os.path.join(model_dir, "label_vocab.json"), encoding="utf-8") as fi:
            label_vocab = json.load(fi)
        tagger= cls(model, tokenizer, label_vocab, Lexicon, device)
        return tagger

    def predict_batch(self, texts):
        input_ids = []
        token_type_ids = []
        attention_masks = []

        for text in texts:
            result = self.tokenizer(text)
            input_ids.append(result["input_ids"])
            token_type_ids.append(result["token_type_ids"])
            attention_masks.append(result["attention_mask"])
        input_ids = self.pad_to_max_len(input_ids)
        token_type_ids = self.pad_to_max_len(token_type_ids)
        attention_masks = self.pad_to_max_len(attention_masks)
        tags_list = self.model(input_ids, token_type_ids, attention_masks)
        for i in range(len(tags_list)):
            tags_list[i] = [self.label_vocab[str(tag)] for tag in tags_list[i][1:-1]]  # drop tag for [CLS] and [SEP]
        return tags_list

    def pad_to_max_len(self, input_ids):
        max_len = max(map(len, input_ids))
        for i in range(len(input_ids)):
            input_ids[i] = input_ids[i] + ([0] * (max_len - len(input_ids[i])))
        return torch.tensor(input_ids, dtype=torch.long, device=self.device)
