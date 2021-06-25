import argparse
from ner.bilstm_crf.bilstm_crf_ner import BilstmCrfModel
from ..dataset_util import load_datasets
import os.path
from torchtext import data
from torch.optim import Adam
import tqdm
import torch
from ..ner_evaluate import evaluate
import json
import pickle


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

    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")

    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    
    parser.add_argument("--use_bichar",
                        action='store_true',
                        help="Whether to use bichar features.")
    
    parser.add_argument("--dict_file",
                        type=str,
                        default=None,
                        help="the path of dict")

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


def train(model, train_data, optimizer, args):
    data_iter = data.BucketIterator(train_data, args.batch_size, shuffle=True, device=args.device)
    for _ in tqdm.trange(args.num_train_epochs):
        model.train()
        total_loss = 0
        tbar = tqdm.tqdm(data_iter)
        for i, batch in enumerate(tbar):
            chars = batch.Char
            tags = batch.Tag
            bichars = None if not args.use_bichar else batch.BiChar
            lex_features = None if args.dict_file is None else batch.Lexicon
            loss = model(chars, tags, bichars, lex_features)
            loss.backward()
            optimizer.step()
            model.zero_grad()
            total_loss += loss.item()
            if (i + 1) % 50 == 0:
                tbar.set_postfix(loss=total_loss/50)
                total_loss = 0
    model_path = os.path.join(args.output_dir, "bilstm_crf.pt")
    torch.save(model, model_path)
    with open(os.path.join(args.output_dir, "Char.pkl"), "wb") as fo:
        pickle.dump(train_data.fields.get("Char"), fo)
    if args.use_bichar:
        with open(os.path.join(args.output_dir, "BiChar.pkl"), "wb") as fo:
            pickle.dump(train_data.fields.get("BiChar"), fo)
    if args.dict_file is not None:
        with open(os.path.join(args.output_dir, "Lexicon.pkl"), "wb") as fo:
            pickle.dump(train_data.fields.get("Lexicon"), fo)
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
        chars = batch.Char
        bichars = None if not args.use_bichar else batch.BiChar
        lex_features = None if args.dict_file is None else batch.Lexicon
        tags_list = batch.Tag.cpu().numpy()
        preds_list = model(chars, bichars=bichars, lex_features=lex_features)
        all_true_labels.extend([[tag_map[tag] for tag in tags if tag != tag_vocab.stoi["<pad>"]] for tags in tags_list])
        all_pred_labels.extend([[tag_map[pred] for pred in preds] for preds in preds_list])
    evaluate(all_true_labels, all_pred_labels)


def main():
    parser = prepare_parser()
    args = parser.parse_args()
    data_dir = args.data_dir
    train_file = os.path.join(data_dir, "train.txt")
    dev_file = os.path.join(data_dir, "dev.txt")
    train_data, dev_data = load_datasets(train_file, dev_file, use_bichar=args.use_bichar, 
        dict_file=args.dict_file)
    if args.do_train:
        Char = train_data.fields.get("Char")
        if args.use_bichar:
            BiChar = train_data.fields.get("BiChar")
            bichar_vocab_size = len(BiChar.vocab)
        else:
            bichar_vocab_size = 0
        if args.dict_file is not None:
            Lexicon = train_data.fields.get("Lexicon")
            lex_vocab_size = len(Lexicon.vocab)
        else:
            lex_vocab_size = 0
        Tag = train_data.fields.get("Tag")
        vocab_size = len(Char.vocab)
        num_tags = len(Tag.vocab)
        padding_idx = Char.vocab.stoi["<pad>"]
        model = BilstmCrfModel(vocab_size, args.emb_dims, args.hidden_dims, num_tags, padding_idx, 
            args.dropout, bichar_vocab_size, lex_vocab_size)
        optimizer = Adam(model.parameters(), lr=args.learning_rate)
        model.to(args.device)
        train(model, train_data, optimizer, args)
    if args.do_eval:
        model_path = os.path.join(args.output_dir, "bilstm_crf.pt")
        model = torch.load(model_path, map_location=args.device)
        test(model, dev_data, args)


if __name__ == "__main__":
    main()
