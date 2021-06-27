from transformers import AutoModel, AutoConfig
import torch.nn as nn
from torchcrf import CRF


class BertForNER(nn.Module):
    def __init__(self, model_name_or_path, num_labels, use_crf=False, dropout=0.3) -> None:
        super().__init__()
        self.bert_model = AutoModel.from_pretrained(model_name_or_path)
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
        hidden = output.last_hidden_state[:, 1:-1, :]  # drop [CLS] and [SEP]
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
        mask = mask[:, 1:-1]  # drop mssk for [CLS] and [SEP]
        return mask
