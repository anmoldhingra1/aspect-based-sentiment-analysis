# Sentiment classifier module

This is a single-label classification task. Given a text and the recognised aspects in it, we have to predict whether the text has a negative, positive or neutral sentiment with respect to the aspect.

## Setup

### Pre-requisites

**This module only runs on python3**

Install the necessary packages

```pip install pytorch numpy nltk sklearn```


### Prepare dataset

The training and testing datasets should be named "train.csv" and "test.csv" respectively and have the following format

|id|aspect|text|sentiment|
|:---:|:---:|:---:|:---:|
|0|product|The premium is too high|0|
|1|process|You have not updated my details online|0|
|2|people|You have not updated my details online|0|
|3|system|You have not updated my details online|0|
|4|product|no claim bonus huh?, nice|1|
|5|process|ok, i will check my email and make the payment|2|

Here we use the integer equivalents of the sentiments where

['negative','positive','neutral'] is mapped to [0,1,2]

## Run the module

Begin training the model by running the script 

```
python run_classifier.py \
--data_dir <DATA_PATH> \
--vocab_file <VOCAB_PATH> \
--bert_config_file <CONFIG_PATH> \
--init_checkpoint <CKPT_PATH> \
--eval_test \
--do_lower_case \
--max_seq_length 320 \
--train_batch_size 24 \
--learning_rate 2e-5 \
--num_train_epochs 6 \
--output_dir <OUTPUT_DIR> \
```

The run_classifier.py scripts accepts the following arguments:

* `--data_dir`: The input data dir. Should contain the .csv files (or other data files) for the task.
* `--vocab_file`:The vocabulary file that the BERT model was trained on. (path to the vocab.txt file)
* `--bert_config_file`: The config json file corresponding to the pre-trained BERT model. This specifies the model architecture. (path to the bert_config.json file)
* `--init_checkpoint`: Initial checkpoint (usually from a pre-trained BERT model). (path to the pytorch_model.bin file)
* `--output_dir`: The output directory where the model checkpoints will be written.[Note: <OUTPUT_DIR> should not exist and if it does it should be empty]

Optional arguments

* `--eval_test`: Whether to run eval on the test set.
* `--do_lower_case`: Whether to lower case the input text. True for uncased models, False for cased models.
* `--max_seq_length`: The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded.
* `--train_batch_size`: Total batch size for training.
* `--eval_batch_size`: Total batch size for eval.
* `--learning_rate`: The initial learning rate for Adam.
* `--num_train_epochs`: Total number of training epochs to perform.
* `--warmup_proportion`: Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% of training.
* `--no_cuda`: Whether not to use CUDA when available
* `--accumulate_gradients`: Number of steps to accumulate gradient on (divide the batch_size and accumulate)
* `--local_rank`: local_rank for distributed training on gpus
* `--seed`: random seed for initialization
* `--gradient_accumulation_steps`: Number of updates steps to accumualte before performing a backward/update pass.
* `--save_model`: Path to save the trained model to
* `--load_model`: Path to load the trained model from

The results will be saved in <OUTPUT_DIR>. 




