from ..base import NerTagger
import torch
import logging
import random
import numpy as np
import os
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers.modeling_bert import BertForTokenClassification, BertConfig
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from ..ner_evaluate import evaluate
import argparse
from tensorboardX import SummaryWriter
from ..dataset_util import ConllProcessor, BertFeatureConverter, WhitespaceTokenizer
from ..dataset_util import find_class, InputExample


class BertNerTagger(NerTagger):
    def __init__(self, data_processor, converter, model=None):
        super().__init__(data_processor, converter)
        self.model = model
        self.device = None

    def train(self, args):
        train_examples = self.data_processor.get_train_examples(args.data_dir)
        self.train_(train_examples, args)

    def train_(self, train_examples, args):
        self.prepare_device(args)
        self.check_output_dir(args)
        if args.local_rank in [-1, 0]:
            tb_writer = SummaryWriter()
        args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
        train_dataloader = self.prepare_data(train_examples, args)
        num_train_optimization_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        args.num_train_optimization_steps = num_train_optimization_steps
        model = self.prepare_model(args)
        optimizer, scheduler = self.prepare_optimizer(model, args)

        logger = logging.getLogger(__name__)
        logger.info("**** Running training ***")
        logger.info(" batch size = %d", args.train_batch_size)

        model.train()

        global_step = 0
        for _ in trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss = model(input_ids, attention_mask=input_mask, token_type_ids=segment_ids, labels=label_ids)[0]
                if args.n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    global_step += 1
                    if args.local_rank in [-1, 0]:
                        tb_writer.add_scalar("loss", loss.item(), global_step)
            if args.save_checkpoints and _ + 1 >= args.save_checkpoints_after:
                self.save_checkpoint(args.output_dir, args)

    def prepare_device(self, args):
        if args.local_rank == -1 or args.no_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
            n_gpu = torch.cuda.device_count()
        else:
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            n_gpu = 1
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.distributed.init_process_group(backend='nccl')
        self.device = device

        logger = logging.getLogger(__name__)
    
        logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            device, n_gpu, bool(args.local_rank != -1), args.fp16))

        if args.gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                                args.gradient_accumulation_steps))

        args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()
        args.n_gpu = n_gpu
    
    def check_output_dir(self, args):
        if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
            raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
    
    def prepare_data(self, examples, args):
        label_list = self.data_processor.get_labels()
        features = self.converter.convert_examples_to_features(examples, label_list)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = None
        if features[0].input_mask is not None:
            all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = None
        if features[0].segment_ids is not None:
            all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        if args.local_rank == -1:
            if args.do_train:
                sampler = RandomSampler(data)
            else:
                sampler = SequentialSampler(data)
        else:
            sampler = DistributedSampler(data)
        
        if args.do_train:
            dataloader = DataLoader(data, sampler=sampler, batch_size=args.train_batch_size)
        else:
            dataloader = DataLoader(data, sampler=sampler, batch_size=args.eval_batch_size)
        logger = logging.getLogger(__name__)
        logger.info(" Num examples = %d" % len(features))

        return dataloader

    def prepare_model(self, args):
        if args.do_train and not args.continue_training:
            self.initialize_model(args)
        
        if args.local_rank == 0:
            torch.distributed.barrier()
        
        model = self.model 
        model.to(self.device)
        if args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[args.local_rank],
                output_device=args.local_rank, 
                find_unused_parameters=True
            )
        elif args.n_gpu > 1:
            model = torch.nn.DataParallel(self.model)
        return model

    def prepare_optimizer(self, model, args):
        # Prepare optimizer
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, correct_bias=False)
        warmup_steps = int(args.warmup_proportion * args.num_train_optimization_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=args.num_train_optimization_steps)
        return optimizer, scheduler

    def initialize_model(self, args):
        num_labels = len(self.data_processor.get_labels()) + 1  # account for padding
        config = BertConfig.from_pretrained(args.bert_model,
                                            num_labels=num_labels)
        model = BertForTokenClassification.from_pretrained(args.bert_model, config=config, from_tf=False)
        self.model = model

    def test(self, args):
        self.prepare_device(args)
        if args.local_rank in [-1, 0]:
            tb_writer = SummaryWriter()

        eval_examples = self.data_processor.get_dev_examples(args.data_dir)
        eval_dataloader = self.prepare_data(eval_examples, args)
        label_list = self.data_processor.get_labels()
        logger = logging.getLogger(__name__)
        logger.info("*** running evaluation ***")
        logger.info("*** batch size = %d", args.eval_batch_size)

        model = self.prepare_model(args)

        model.eval()
        
        eval_loss = 0
        nb_eval_steps, nb_eval_examples = 0, 0
        all_preds = []
        all_true_labels = []

        for step, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating", disable=args.local_rank not in [-1, 0])):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.no_grad():
                tmp_eval_loss, logits = model(input_ids, attention_mask=input_mask, token_type_ids=segment_ids, labels=label_ids)
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            seq_lens = input_mask.cpu().numpy().sum(axis=1)
            preds, true_labels = self.collect_labels(logits, seq_lens, label_list, label_ids)

            if args.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()
            
            eval_loss += tmp_eval_loss.item()
            all_preds.extend(preds)
            all_true_labels.extend(true_labels)

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

            if args.local_rank in [-1, 0]:
                tb_writer.add_scalar("eval_loss", tmp_eval_loss.mean(), nb_eval_steps)

        logger.info("evaluation result")
        evaluate(all_true_labels, all_preds)
    
    def collect_labels(self, out, seq_lens, label_list, labels=None):
        label_map = {i+1: label for i, label in enumerate(label_list)}
        label_map[0] = "O"
        outputs = np.argmax(out, axis=2)
        preds = []
        if labels is not None:
            true_labels = []
        for i in range(seq_lens.shape[0]):
            preds_ = outputs[i][:seq_lens[i]].tolist()[1:-1]
            preds.append([label_map[pred] for pred in preds_])
            if labels is not None:
                labels_ = labels[i][:seq_lens[i]].tolist()[1:-1]
                true_labels.append([label_map[label] for label in  labels_])
        if labels is not None:
            return preds, true_labels
        return preds

    def save_model(self, model_dir, args=None):
        model_to_save = self.model
        model_to_save.save_pretrained(model_dir)
        self.converter.tokenizer.save_pretrained(model_dir)

        if args is not None:
            output_args_file = os.path.join(model_dir, "training_args.bin")
            torch.save(args, output_args_file)

        label_file = os.path.join(model_dir, "labels.txt")
        with open(label_file, "w", encoding="utf-8") as fo:
            fo.write("||".join(self.data_processor.get_labels()))
    
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
        model = BertForTokenClassification.from_pretrained(model_dir, config=config, from_tf=False)
        return cls(data_processor, converter, model)

    def predict_batch(self, texts):
        texts = [" ".join(text) for text in texts]
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
            n_gpu = torch.cuda.device_count()
            if n_gpu > 1:
                self.model = torch.nn.DataParallel(self.model)
        self.model.eval()

        examples = []
        for text in texts:
            examples.append(InputExample(None, text))
        label_list = self.data_processor.get_labels()
        features = self.converter.convert_examples_to_features(examples, label_list)
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        logits = self.model(input_ids, attention_mask=input_mask, token_type_ids=segment_ids)[0]
        logits = logits.detach().cpu().numpy()
        seq_lens = input_mask.cpu().numpy().sum(axis=1)
        label_list = self.data_processor.get_labels()
        preds = self.collect_labels(logits, seq_lens, label_list)
        return preds


def prepare_parser():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
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
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--num_train_optimization_steps",
                        type=int)
    parser.add_argument("--adam_epsilon", 
                        default=1e-8, 
                        type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", 
                        default=1.0, 
                        type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--n_gpu",
                        default=0,
                        type=int,
                        help="number of gpu availuable")
    parser.add_argument('--continue_training',
                        action='store_true',
                        help="Whether to continue training using a checkpoint")
    parser.add_argument('--save_checkpoints',
                        action='store_true',
                        help="Whether to save checkpoints while training")
    parser.add_argument('--save_checkpoints_after',
                        type=int,
                        default=1)
    parser.add_argument("--cuda_index",
                        type=str,
                        default="0",
                        help="Specify GPU to use")
    parser.add_argument("--data_processor",
                        type=str,
                        help="Specify the data processor")
    parser.add_argument("--tokenizer",
                        type=str,
                        help="Specify the tokenizer")
    parser.add_argument("--converter",
                        type=str,
                        help="Specify the feature converter")
    return parser


def main():
    parser = prepare_parser()
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_index
    if args.do_train:
        data_processor = ConllProcessor()
        tokenizer = WhitespaceTokenizer.from_pretrained(args.bert_model)
        converter = BertFeatureConverter(tokenizer)
        tagger = BertNerTagger(data_processor, converter)
        tagger.train(args)
    elif args.do_eval:
        for dir_ in os.listdir(args.output_dir):
            if dir_.startswith("checkpoint"):
                checkpoint_dir = os.path.join(args.output_dir, dir_)
                clf = BertNerTagger.load_model(checkpoint_dir)
                clf.test(args)

if __name__ == "__main__":
    main()
