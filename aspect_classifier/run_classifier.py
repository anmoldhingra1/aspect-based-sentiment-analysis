from fast_bert.data_cls import BertDataBunch

from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy_multilabel
 
import torch

import argparse

import pandas as pd

import os

def main():

  parser = argparse.ArgumentParser()

  parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="Path specifying the location of the dataset"
                      )
  
  parser.add_argument("--label_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="Path specifying the location of the labels.csv file")

  parser.add_argument("--output_dir",
                        default=None,
                        required=True,
                        type=str,
                        help="Path specifying the location to save the results")

  parser.add_argument("--text_col",
                        default=None,
                        required=True,
                        type=str,
                        help="The column name of the text")

  parser.add_argument("--batch_size",
                        default=16,
                        required=False,
                        type=int,
                        help="Batch size per GPU")

  parser.add_argument("--max_seq_len",
                        default=320,
                        required=False,
                        type=int,
                        help="Maximum length of the token sequence to input to BERT")

  parser.add_argument("--multi_gpu",
                        default=False,
                        required=False,
                        type=bool,
                        help="Whether to use multi-gpu for training")

  parser.add_argument("--epochs",
                      default=6,
                      type=int,
                      required=False,
                      help="Number of epochs to train")

  parser.add_argument("--lr",
                        default=6e-5,
                        type=float,
                        required=False,
                        help="Initial learning rate for training")

  parser.add_argument("--save_model",
                        required=False,
                        default=None,
                        help="Whether to save the model or not")

  parser.add_argument("--eval",
                      required=False,
                      type=bool,
                      default=True,
                      help="Whether to run evaluation after each epoch")


  args = parser.parse_args()
                      

  DATA_PATH = args.data_dir

  LABEL_PATH = args.label_dir

  OUTPUT_PATH = args.output_dir

  EPOCHS = args.epochs

  LR = args.lr

  EVAL = args.eval

  TEXT_COL = args.text_col

  BATCH_SIZE = args.batch_size

  MAX_SEQ_LEN = args.max_seq_len

  MULTI_GPU = args.multi_gpu

  labels =  pd.read_csv(os.path.join(DATA_PATH,'labels.csv'),header=None).values

  LABEL_LIST = [val[0] for val in labels]


  databunch = BertDataBunch(DATA_PATH, LABEL_PATH,
                            tokenizer='bert-base-uncased',
                            train_file='m_aspect_train.csv',
                            val_file='m_aspect_test.csv',
                            label_file='labels.csv',
                            text_col=TEXT_COL,
                            label_col=LABEL_LIST,
                            batch_size_per_gpu=BATCH_SIZE,
                            max_seq_length=MAX_SEQ_LEN,
                            multi_gpu=MULTI_GPU,
                            multi_label=True,
                            model_type='bert',
                            no_cache=True)

  # display(databunch.get_dl_from_texts)

  device_cuda = torch.device("cuda")
  metrics = [{'name': 'accuracy', 'function': accuracy_multilabel}]

  learner = BertLearner.from_pretrained_model(
              databunch,
              pretrained_path='bert-base-uncased',
              metrics=metrics,
              device=device_cuda,
              logger=None,
              output_dir=OUTPUT_PATH,
              finetuned_wgts_path=None,
              warmup_steps=500,
              multi_gpu=MULTI_GPU,
              is_fp16=False,
              multi_label=True,
              logging_steps=50)

  global_step, loss = learner.fit(epochs=EPOCHS,
        lr=LR,
        validate=EVAL, 	# Evaluate the model after each epoch
        schedule_type="warmup_cosine",
        optimizer_type="lamb")

  print("global_Step:",global_step,"loss:", loss)

  if args.save_model:
    learner.save_model()


if __name__ == "__main__":
    main()