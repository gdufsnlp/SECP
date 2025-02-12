#!/usr/bin/bash

# SECP-BERT
python train.py --config_path config/bert_laptop_config.yml --do_eval_first --do_warmup
python train.py --config_path config/bert_restaurant_config.yml --do_eval_first --do_warmup

# SECP-RoBERTa
python train.py --config_path config/roberta_laptop_config.yml --do_eval_first --do_warmup
python train.py --config_path config/roberta_restaurant_config.yml --do_eval_first --do_warmup