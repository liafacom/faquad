## FaQuAD: Reading Comprehension Dataset in the Domain of Brazilian Higher Education

The FaQuAD is a Portuguese reading comprehension dataset which follows the format of the Stanford Question Answering Dataset (SQuAD). As far as we know, FaQuAD is a pioneer Portuguese reading comprehension dataset with the SQuAD's challenging format.

Besides the FaQuAD dataset this repository provides the required source code to reproduce the experiments presented by the paper. Mostly of the code is a workaround for reusing the default AllenNLP set for training the BiDAF network with SQuAD. Further code improvements are required, but not a goal for awhile.

### Attention

Some AllenNLP's source code modifications are required in order to reproduce the experiments, hereupon, make sure to have a clean newly created python 3.6+ virtual environment.


## Dependencies

All dependencies may be installed with pip. Make sure to create a python 3.6+ environment.

| Dependency | Version |
|------------|---------|
| allennlp   | 0.8.3   |
| pandas     | 0.24.2  |
| sklearn    | 0.20.3  |
| seaborn    | 0.9.0   |

The complete list of dependencies can be found in requirements.txt.

## Changing the spaCy word splitter

Since we are taking advantage of the already implemented SQuAD dataset reader, it's necessary to change the spaCy word splitter language directly in AllenNLP's source code from "en_core_web_sm" to "pt_core_news_sm". The SpacyWordSplitter class is located at allennlp/data/tokenizers/word_splitter.py 

## Pre-trained embeddings

Portuguese GloVe and ELMo weights are required to execute all experiments. GloVe weights must be placed at a "/glove" directory and can be downloaded at http://nilc.icmc.usp.br/embeddings. ELMo weights must be placed at a "/elmo" directory and can be download at https://allennlp.org/elmo.
 
## Code Overview

#### experiment.py

Preprocess the dataset, splits into K folds and commands AllenNLP to train the model with a given dataset.

#### dataset_utils.py
A set of useful, mostly not optimized, workaround functions to allow AllenNLP's default dataset reader for SQuAD to be reused with FaQuAD as the experiments were made. These works fine with a small dataset like FaQuAD, still, optimization may be required to work with large datasets like SQuAD.

