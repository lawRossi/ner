python -m ner.bilstm_crf.train --do_train --do_eval --dict_file data/dict.csv --data_dir data --batch_size 64 --learning_rate 5e-5 --num_train_epochs 1 --output_dir output