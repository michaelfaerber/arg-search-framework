import os
from os.path import abspath, dirname
import sys
from pathlib import Path
PARENT_DIR = dirname(abspath(__file__))
sys.path.append(PARENT_DIR)
sys.path.append(str(Path(PARENT_DIR + '/classes')))

import argument_mining as am
from argument_mining import Clustering, Segmentation

def am_run(machine = 'default', task = 'AS', execution_mode = 'train', dataset = 'debatepedia', embed_type = 'BERT_difference',  batch_size = 100, epochs = 100, eval_partition = 'val', model_type= 'FNN', layers =[200]):
    """Run training, evaluation, clustering etc.
    """
    model_name = task + '_' + embed_type + '_' + str(batch_size) + '_' + model_type + '-'.join([str(x) for x in layers]) + '_' + str(dataset)
    print(model_name)
    
    assert (execution_mode in ['train', 'resume', 'evaluate', 'cluster', 'plot_history'])
    #assert (dataset in ['debatepedia', 'essay', 'debateorg'])
    assert (task in ['AS', 'DS', 'SEG'])
    assert (eval_partition in ['val', 'test', 'train', None])
    assert (model_type in ['FNN', 'LINEAR', 'BILSTM'])
    #assert (embed_type in ['BERT_difference', 'BERT_pair', 'BERT', 'BERT_difference_word_average', 'BERT_word_average','Fasttext_difference', 'Fasttext'])

    #SET PATH TO JSON FILES
    path_data = PARENT_DIR + '/data/' + dataset + '/'
    #SET PATH TO PRECALCULATED EMBEDDINGS
    if machine in ['default']:
        path_embed = dirname(PARENT_DIR) + '/data/'
    elif machine == 'custom':
        path_embed = '' #TODO Change this to set path to embeddings individually
    path_embed = path_embed + task + '/' + dataset + '/' + embed_type + '/'

    shuffle = True
    stratify = False
    mode = 'debatepedia' + task 

    if task in ['AS', 'DS']:
        file_size = 100
        dir_size = 1000
        arg_model = Clustering(path_data, path_embed, file_size = file_size, dir_size = dir_size, mode = mode)
        if execution_mode in ['train', 'resume']:
            stratify = True 
        elif execution_mode in  ['cluster', 'evaluate']:
            stratify = False
    elif task == 'SEG':
        file_size = None
        dir_size = None
        arg_model = Segmentation(path_data = path_data, path_embed = path_embed, file_size = file_size, dir_size = dir_size, mode=mode)
        if execution_mode in ['train', 'resume','evaluate']:
            stratify = False
    
    arg_model.set_generators(batch_size=batch_size, shuffle=shuffle, stratify = stratify)

    if execution_mode == 'train':
        arg_model.train_model(model_name, epochs = epochs, model_type = model_type)
    elif execution_mode == 'resume':
        arg_model.resume_training(model_name, epochs = epochs)
    elif execution_mode == 'evaluate':
        arg_model.evaluate_model(model_name=model_name, partition = eval_partition, model_type = model_type)
    elif execution_mode == 'cluster':
        arg_model.compute_cluster(model_name, partition = eval_partition)
    elif execution_mode == 'plot_history':
        arg_model.plot_history(model_name)

execution_mode = sys.argv[1]
dataset =sys.argv[2]
task = sys.argv[3]
embed_type =sys.argv[4]
machine = sys.argv[5]
batch_size = int(sys.argv[6])
epochs = int(sys.argv[7])
eval_partition = sys.argv[8]
model_type = sys.argv[9]
layers = []
for l in sys.argv[10:]:
    layers.append(int(l))

am_run(execution_mode=execution_mode, dataset = dataset, task = task, embed_type=embed_type, machine= machine, batch_size=batch_size, epochs= epochs, eval_partition= eval_partition, model_type = model_type, layers = layers)