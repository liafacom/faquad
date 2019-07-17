## FaQuAD: Reading Comprehension Dataset in the Domain of Brazilian Higher Education

The FaQuAD is a Portuguese reading comprehension dataset which follows the format of the Stanford Question Answering Dataset (SQuAD). As far as we know, FaQuAD is a pioneer Portuguese reading comprehension dataset with the SQuAD's challenging format.

Besides the FaQuAD dataset this repository provides the required source code to reproduce the experiments presented by the paper. Mostly of the code is a workaround for reusing the default AllenNLP set for training the BiDAF network with SQuAD. Further code improvements are required, but not a goal for awhile.

### Attention

Some AllenNLP's source code modifications are required in order to reproduce the experiments, hereupon, make sure to have a clean newly created python 3.6+ virtual environment. We recommend using miniconda3.

## Dependencies

In order to reproduce the experiments you'll only need allennlp 0.8.3 and pandas 0.24.2.

The complete list of dependencies can be found in requirements.txt.

## Changing the spaCy word splitter

Since we are taking advantage of the already implemented SQuAD dataset reader, it's necessary to change the spaCy word splitter language directly in AllenNLP's source code from "en_core_web_sm" to "pt_core_news_sm". The SpacyWordSplitter class is located at allennlp/data/tokenizers/word_splitter.py 

## Pre-trained embeddings

Portuguese GloVe and ELMo weights are required to execute mostly of the experiments. GloVe weights must be placed at a "/glove" directory as "glove_s{}.zip" (replace {} with the dimensions amount) and can be downloaded at http://nilc.icmc.usp.br/embeddings. ELMo weights and options must be placed at a "/elmo" directory as "elmo_pt_weights.hdf5" and "elmo_pt_options.json" and can be download at https://allennlp.org/elmo. 

## Running

After changing the spaCy word splitter language and putting the pre-trained models into their respective directories, you may want to start training a model. In this case, just activate your virtual environment and run the top-level file in "/aurelio" directory. 

The top-level file invokes a function from the experiment file that preprocess the dataset and commands AllenNLP to train the model. This function takes the following args:

* config_file_path: an AllenNLP configuration file for BiDAF
* train_dataset_path: the absolute path of the training dataset
* dev_dataset_path: the absolute path of the dev dataset
* serialization_dir: the directory which the model will be serialized
* reduce_train_dataset: if the training's alternative answers should be considered
* reduce_dev_dataset: if the validation's alternative answers should be considered
* expand_train_qas: if each training's alternative answers should have its own question
* elmo: if elmo should be used
* dev_dataset_portion: the amount of data that should be removed from the training fold (specially useful for learning curves)
* embedding_dim: the word embeddings' dimensions (0 for not using word embeddings).  

Obs: train and dev datasets will be merged so it can be split into folds. They are taken separately because AllenNLP's BiDAF dataset reader supports only holdout.

## Dataset

The full dataset is located at "data/dataset.json". "train.json" and "dev.json" are the splitted into training and validation version of the full dataset and may be used for holdout. 

For further instructions about AllenNLP access https://allennlp.org.