import argparse
from ner.bert_ner.bert_ner import BertForNER
from ..dataset_util import BertNERDataset, load_datasets_for_bert, TransformersTokenizer
import os
from torchtext import data
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import tqdm
import torch
from ..ner_evaluate import evaluate
import json
import pickle
import math


def prepare_parser():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model_name_or_path",
                        default=None,
                        type=str,
                        required=True,
                        help="specify which model to use.")

    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir.")

    parser.add_argument("--output_dir",
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")

    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")

    parser.add_argument("--use_crf",
                        action='store_true',
                        help="Whether to use crf layer.")
    
    parser.add_argument("--dict_file",
                        type=str,
                        default=None,
                        help="the path of dict")

    parser.add_argument("--batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")

    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")

    parser.add_argument("--num_train_epochs",
                        default=1,
                        type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")

    parser.add_argument("--dropout",
                        default=0.3,
                        type=float,
                        help="dropout rate")

    parser.add_argument("--device",
                        type=str,
                        default="cpu",
                        help="Specify which device to use")
    return parser


def train(model, train_data, dev_data, optimizer, scheduler, args):
    data_iter = data.BucketIterator(train_data, args.batch_size, shuffle=True, device=args.device)
    for _ in tqdm.trange(args.num_train_epochs):
        model.train()
        total_loss = 0
        tbar = tqdm.tqdm(data_iter)
        for i, batch in enumerate(tbar):
            input_ids = batch.Token
            token_type_ids = batch.TokenType
            attention_mask = batch.Mask
            tags = batch.Tag
            # lex_features = None if args.dict_file is None else batch.Lexicon
            loss = model(input_ids, token_type_ids, attention_mask, tags)
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            total_loss += loss.item()
            if (i + 1) % 5 == 0:
                tbar.set_postfix(loss=total_loss/5)
                total_loss = 0
        if args.do_eval:
            test(model, dev_data, args)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    model_path = os.path.join(args.output_dir, "bert_ner.pt")
    torch.save(model, model_path)

    if args.dict_file is not None:
        with open(os.path.join(args.output_dir, "Lexicon.pkl"), "wb") as fo:
            Lexicon = train_data.fields.get("Lexicon")
            Lexicon.tokenize = None
            pickle.dump(Lexicon, fo)

    torch.save(args, os.path.join(args.output_dir, "args.pkl"))

    tag_vocab = train_data.fields.get("Tag").vocab
    tag_map = {i: tag for i, tag in enumerate(tag_vocab.itos)}
    tag_map[tag_vocab.stoi["<pad>"]] = "O"
    with open(os.path.join(args.output_dir, "label_vocab.json"), "w", encoding="utf-8") as fo:
        json.dump(tag_map, fo)


def test(model, eval_data, args):
    tag_vocab = eval_data.fields.get("Tag").vocab
    tag_map = {i: tag for i, tag in enumerate(tag_vocab.itos)}
    tag_map[tag_vocab.stoi["<pad>"]] = "O"
    data_iter = data.BucketIterator(eval_data, args.batch_size, device=args.device, sort=False, train=False)
    tbar = tqdm.tqdm(data_iter)
    model = model.eval()
    all_pred_labels = []
    all_true_labels = []
    for batch in tbar:
        input_ids = batch.Token
        token_type_ids = batch.TokenType
        attention_mask = batch.Mask
        # lex_features = None if args.dict_file is None else batch.Lexicon
        tags_list = batch.Tag.cpu().numpy()
        preds_list = model(input_ids, token_type_ids, attention_mask)
        # if args.use_crf:
        all_true_labels.extend([[tag_map[tag] for tag in tags if tag != tag_vocab.stoi["<pad>"]] for tags in tags_list])
        # else:
        #     all_true_labels.extend([[tag_map[tag] for tag in tags] for tags in tags_list])
        all_pred_labels.extend([[tag_map[pred] for pred in preds] for preds in preds_list])
    evaluate(all_true_labels, all_pred_labels)


def setup_optimizer(model, args, dataset):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    num_train_steps = math.ceil(len(dataset) * args.num_train_epochs / args.batch_size)
    num_warmup_steps = int(args.warmup_proportion * num_train_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_train_steps)
    return optimizer, scheduler


def main():
    parser = prepare_parser()
    args = parser.parse_args()
    data_dir = args.data_dir
    tokenizer = TransformersTokenizer(args.model_name_or_path)
    train_data, dev_data = load_datasets_for_bert(data_dir, tokenizer, args.dict_file)
    if args.do_train:
        if args.dict_file is not None:
            Lexicon = train_data.fields.get("Lexicon")
            lex_vocab_size = len(Lexicon.vocab)
        else:
            lex_vocab_size = 0
        Tag = train_data.fields.get("Tag")
        num_tags = len(Tag.vocab)
        model = BertForNER(args.model_name_or_path, num_tags)
        model.to(args.device)
        optimizer, scheduler = setup_optimizer(model, args, train_data)
        train(model, train_data, dev_data, optimizer, scheduler, args)

if __name__ == "__main__":
    main()
