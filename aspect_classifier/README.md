# Aspect classifier module

The learning of aspects is a multi-label classification task. Initially this module was trained on five higher level aspects namely:
* *Product* : Representing queries related to the insurance product itself.
* *Process* : Representing queries related to the processes in insurance.
* *People* : Representing queries related to the people involved in insurance.
* *System* : Representing queries related to the system architecture required for maintainence.
* *Other* : Representing queries which did not belong to any of the above aspects


# Setup

## Pre-requisites

**This module only runs on Python3**

Install the necessary packages by running the following command 

```pip install pytorch-lamb tensorboardX fastprogress sklearn spacy transformers>=2.3.0 pandas python-box tokenizers tensorflow```

As of now distributed training is not supported. 

## Prepare the dataset

The training data should be named "train.csv" and the testing data as "test.csv".

All the possible aspects should be present in a file named "labels.csv" in the format

```
product
process
people
system
other
```

Both should have the following format (in case of the data has 5 aspects as mentioned above)

|id|TEXT_COL|product|process|people|system|other|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|0|I did not get the policy|1|1|0|0|0|
|1|The premium is too high|1|0|0|0|0|
|2|You have not updated my policy details online|1|0|1|1|0|

The presence of an aspect is indicated by a 1 at its respective column for each text.

# Running the module

After the performing all the steps for setup, run the script

```
python run_classifier.py \
--data_dir <DATA PATH> \
--label_dir <LABEL PATH> \
--output_dir <OUTPUT_PATH> \
--text_col <TEXT_COL> \
```

The following arguments are accepted in run_classifier.py:

* `--data_dir`: Path specifying the location of the dataset
* `--label_dir`: Path specifying the location of the labels.csv file
* `--output_dir`: Path specifying the location to save the results [Note: The directory should already be created and be empty]
* `--text_col`: The column name of the text

Optional arguments

* `--batch_size`: Batch size per GPU (default=16)
* `--max_seq_len`: Maximum length of the token sequence to input to BERT (default=320)
* `--multi_gpu`: Whether to use multi-gpu for training (default=False) [Note: This requires the apex package for python3]
* `--epochs`: Number of epochs to train (default=6)
* `--lr`: Initial learning rate for training (default=6e-5)
* `--save_model`: Whether to save the model or not
* `--eval`: Whether to run evaluation after each epoch (default=True)

The results will be saved in the <OUTPUT_PATH>




