import argparse
from ner.bilstm_crf.bilstm_crf_ner import BilstmCrfModel
from ..dataset_util import load_datasets
import os.path
from torchtext import data
from torch.optim import Adam
import tqdm
import torch
from ..ner_evaluate import evaluate


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


def train(model, data_iter, optimizer, args):
    for _ in tqdm.trange(args.num_train_epochs):
        model.train()
        total_loss = 0
        tbar = tqdm.tqdm(data_iter)
        for i, batch in enumerate(tbar):
            tokens = batch.Token
            tags = batch.Tag
            loss = model(tokens, tags)
            loss.backward()
            optimizer.step()
            model.zero_grad()
            total_loss += loss.item()
            if (i + 1) % 50 == 0:
                tbar.set_postfix(loss=total_loss/50)
                total_loss = 0
    model_path = os.path.join(args.output_dir, "bilstm_crf.pt")
    torch.save(model, model_path)


def test(model, eval_data, args):
    tag_vocab = eval_data.fields.get("Tag").vocab
    tag_map = {i: tag for i, tag in enumerate(tag_vocab.itos)}
    tag_map[tag_vocab.stoi["<unk>"]] = "O"
    tag_map[tag_vocab.stoi["<pad>"]] = "O"
    data_iter = data.BucketIterator(eval_data, args.batch_size, device=args.device, sort=False, train=False)
    tbar = tqdm.tqdm(data_iter)
    model = model.eval()
    all_pred_labels = []
    all_true_labels = []
    for batch in tbar:
        tokens = batch.Token
        tags_list = batch.Tag.cpu().numpy()
        preds_list = model(tokens)
        all_true_labels.extend([[tag_map[tag] for tag in tags if tag != tag_vocab.stoi["<pad>"]] for tags in tags_list])
        all_pred_labels.extend([[tag_map[pred] for pred in preds] for preds in preds_list])
    evaluate(all_true_labels, all_pred_labels)


def main():
    parser = prepare_parser()
    args = parser.parse_args()
    data_dir = args.data_dir
    train_file = os.path.join(data_dir, "train.txt")
    dev_file = os.path.join(data_dir, "dev.txt")
    train_data, dev_data = load_datasets(train_file, dev_file)
    if args.do_train:
        Token = train_data.fields.get("Token")
        Tag = train_data.fields.get("Tag")
        vocab_size = len(Token.vocab)
        num_tags = len(Tag.vocab)
        padding_idx = Token.vocab.stoi["<pad>"]
        model = BilstmCrfModel(vocab_size, args.emb_dims, args.hidden_dims, num_tags, padding_idx)
        optimizer = Adam(model.parameters(), lr=args.learning_rate)
        train_iter = data.BucketIterator(train_data, args.batch_size, shuffle=True, device=args.device)
        model.to(args.device)
        train(model, train_iter, optimizer, args)
    if args.do_eval:
        model_path = os.path.join(args.output_dir, "bilstm_crf.pt")
        model = torch.load(model_path, map_location=args.device)
        test(model, dev_data, args)


if __name__ == "__main__":
    main()
