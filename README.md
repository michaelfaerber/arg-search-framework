# Argument Search Framework

## Requirements
python3.7, tensorflow, numpy, pandas, scipy, nltk, seqeval, transformers, scikit-learn, gensim, spacy, bcubed, matplotlib, tensorflow-addons, hdbscan, umap-learn

## Usage

### Pre-calculating embeddings
The calculation of embeddings with BERT can take quite a while. Therefore, embeddings can be calculated in advance and not newly for every training of the model. For this, adjust and execute a python script with the following content.

```python
import sys
project_path = # insert absolute path to project here
sys.path.append(project_path)
sys.path.append(project_path + '/scripts')
sys.path.append(project_path + '/classes')
import sentence_dataset as sd
import argument_mining as AM
from argument_mining import Clustering, Segmentation

path_data = project_path + '/data/' + # insert dataset name (plus slash) here
path_embed = # insert path here

#1 Embeddings for segmentation
#arg_model = Segmentation(path_data, path_embed, file_size = 100, dir_size = 1000, mode = 'debatepediaSEG')
#arg_model.set_generators(batch_size=300, shuffle=False, stratify = False)
#arg_model.compute_and_save_embeddings(type = 'bert', separate_embedding=None, word_embedding=False)

#2 Embeddings for clustering
#arg_model = Clustering(path_data, path_embed, file_size = 100, dir_size = 1000, mode = 'debatepediaDS')
#arg_model.set_generators(batch_size=300, shuffle=False, stratify = False)
#arg_model.compute_and_save_embeddings(type = 'bert', separate_embedding=True, word_embedding=True)

```
_For information about the usage of the parameters, refer to the documentation in the code in `argument_mining.py` and `sentence_dataset`._

One row in the resulting csv files constitutes one embeddings vector. For the clustering tasks (AS and DS), one embedding is calculated over one discussion pair for DS (or argument pair for AS). For the segmentation task, the embeddings are calculated sentence-wise and one file corresponds to one text document (resulting in different file sizes).

### Next steps
From the command line navigate to the location of `main.py` and execute it with respective parameter setting (all ten parameters must be set). See the example usage below.

```bash
python main.py execution_mode dataset task embed_type machine batch_size epochs eval_partition model_type layer [layers]
```

1. `execution_mode `: Either `train`, `resume`, `evaluate`, `cluster` or `plot_history`.
2. `dataset`: Name of the dataset which will be used in the paths to locate the data.
3. `task`: Either `AS` (argument clustering), `DS` (discussion clustering), `SEG` (segmentation into arguments)
4. `embed_type`: If using pre-calculated embeddings, this string will be used to differ the different embedding variants per dataset (will be used in the path).
5. `machine`: Either `default` or `custom`. Determines where to find the pre-calculated embeddings. If using `custom`, set the respective location in `main.py`. The default location of pre-calculated embeddings is `dirname(PARENT_DIR) + '/data/'`.
6. `batch_size`: The batch size for training the model
7. `epochs`: The number of epochs to train the model
8. `eval_partition`: Either `val`, `train`, `test`. The partition to use for evaluating the training.
9. `model_type`: Either `FNN`, `LINEAR` or `BILSTM`
10. `layer`: Number of neurons
11. `layers`: Optional: Number of neurons for each additional layer if using `FNN` or `BILSTM`



```bash
python main.py train debatepedia DS BERT default 64 10 val FNN 300 200 100
```

## Folder structure
`main.py` main entry point to all functionality apart from pre-calculating the embeddings
`sentence_dataset.py` Calculating BERT embeddings using the transformers library of huggingface
### data
Contains the data. Holds one subdirectory for each dataset which each contain three files: `train.json`, `val.json` and `test.json`.
### model
The trained models will be saved here.
### classes
The main functionality is located in `argument_mining.py`. `data_generator.py` is used to process the data in batches.
### scripts
`create_features.py` processes the JSON files and `preprocess_general.py` contains helper functions. They are used in other files throughout the project.

## Datasets
The JSON files must be split into `train.json`, `val.json` and `test.json` and must show the following structure:
```
{'ID': topic ID
 'topic': discussion title, 
 'subtopics': [{'ID': subtopic ID
                'title': sub heading,
                'arguments':[{'claim': claim,
                              'premise': premise,
                            'stance': pro/con}]
                }]
}
```


