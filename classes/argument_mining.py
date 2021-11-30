import csv
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
#import seaborn as sns
import hdbscan
from gensim.parsing.preprocessing import strip_non_alphanum
import sklearn.cluster as cluster
import matplotlib.patches as mpatches
from matplotlib import cm
import numpy as np
import pandas as pd
import numpy.ma as ma
import math
import random
import os
import bcubed
from os import path
from os.path import abspath, dirname
import scipy.cluster.hierarchy as sch
import tensorflow as tf
import tensorflow_addons as tfa
import sklearn
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import homogeneity_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import completeness_score
from sklearn.linear_model import LinearRegression
from tensorflow.keras.layers import Activation, Dense, Masking, Bidirectional, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import umap
import sys
PARENT_DIR = dirname(dirname(abspath(__file__)))
sys.path.append(PARENT_DIR)
sys.path.append(str(Path(PARENT_DIR + '/scripts')))
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from data_generator import DataGenerator, DataGeneratorSequence
import preprocess_general as pp
import create_features as feat
import sentence_dataset as sd
from itertools import chain
import scipy.stats
from sklearn.metrics import make_scorer
#from sklearn.cross_validation import cross_val_score
#from sklearn.grid_search import RandomizedSearchCV
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

class ArgumentMining():
    def __init__(self, path_data, path_embed, mode, file_size, dir_size, dim = 768):

        """
        Parameters
        ----------
        path_data: str
            path to dir where the train.json, val.json and test.json files are
        path_embed: str
            path to dir where csv files with the embeddings are or should be
        mode: str
            either 'debatepedia' or 'debateorg'
        file_size: int
            number of lines (examples) within a file.
            Should be small enough so embeddings of examples within a file can be calculated without running into OOM issues.

        """

        self.path_data = path_data
        self.path_embed = path_embed
        self.mode = mode
        self.file_size = file_size
        self.dir_size = dir_size
        self.doc_embed_dim = dim

        self.train_dict = pp.load_dict_from_json(path_data, 'train')
        self.val_dict = pp.load_dict_from_json(path_data, 'val')
        self.test_dict = pp.load_dict_from_json(path_data, 'test')

        self.train_docs = feat.create_documents(self.mode, self.train_dict)
        self.val_docs = feat.create_documents(self.mode, self.val_dict)
        self.test_docs = feat.create_documents(self.mode, self.test_dict)

        #Merge into one documents dictionary
        train_docs_len = len(self.train_docs['documents'])
        self.documents = self.train_docs.copy()
        self.documents['documents'] = pp.merge_dict([self.train_docs['documents'], self.val_docs['documents'], self.test_docs['documents']])
        assert(len(self.documents['documents']) == train_docs_len + len(self.val_docs['documents']) + len(self.test_docs['documents']))
        self.train_docs = feat.create_documents(self.mode, self.train_dict)
        assert(train_docs_len== len(self.train_docs['documents']))

        self.dictionary = self.train_dict + self.val_dict + self.test_dict

    def set_generators(self, batch_size = 32, shuffle=True, stratify = False):

        """ Set generators for training, validation and testing

        Parameters
        ----------
        batch_size: int (default 32)
        shuffle: bool (default True)
            whether to shuffle the training data for each epoch
        stratify: bool
            whether to stratify the training and validation dataset
        """

        #TODO redundant!
        train_docs = feat.create_documents(self.mode, self.train_dict)
        val_docs = feat.create_documents(self.mode, self.val_dict)
        test_docs = feat.create_documents(self.mode, self.test_dict)

        (train_examples, train_labels) = feat.create_examples(train_docs, offset = 0)
        (val_examples, val_labels) = feat.create_examples(val_docs, offset = len(train_labels))
        (test_examples, test_labels) = feat.create_examples(test_docs, offset = len(train_labels) + len(val_labels))
        train_examples_len = len(train_examples)
        self.labels = pp.merge_dict([train_labels, val_labels, test_labels])
        assert(len(self.labels) == train_examples_len + len(val_labels) + len(test_labels))

        #Merge into one documents dictionary 
        train_docs_len = len(train_docs['documents'])
        self.documents = train_docs
        self.documents['documents'] = pp.merge_dict([train_docs['documents'], val_docs['documents'], test_docs['documents']])
        assert(len(self.documents['documents']) == train_docs_len + len(val_docs['documents']) + len(test_docs['documents']))

        #Partition of IDs
        self.partition = dict()
        self.partition['train'] = list(train_examples.keys())
        self.partition['val'] = list(val_examples.keys())
        self.partition['test']  = list(test_examples.keys())
        #print('Original number of train examples: {}'.format(len(self.partition['train'])))
        #print('Original number of val examples: {}'.format(len(self.partition['val'])))
        #print('Original number of test examples: {}'.format(len(self.partition['test'])))

        if stratify:
            self.partition['train'] = self.__stratify_IDs(self.partition['train'])
            self.partition['val'] = self.__stratify_IDs(self.partition['val'])
        print('Number of train examples after stratification: {}'.format(len(self.partition['train'])))
        print('Number of val examples after stratification: {}'.format(len(self.partition['val'])))



        #Merge into single pairs dictionaries
        self.examples = pp.merge_dict([train_examples, val_examples, test_examples])
        assert(len(self.examples) == train_examples_len + len(val_examples) + len(test_examples))
        assert(sorted(list(self.examples.keys())) == list(range(0, len(self.examples))))

        # Generators
        pass


    def __stratify_IDs(self, IDs):

        """ Returns a stratified list of IDs (by reducing and not augmenting)

        Calculates a list of IDs where the number of samples of each class are equal.
        Randomly removes samples from lists which are above the minimum number of samples for a class.

        Parameters
        ----------
        IDs: list of int
            list of IDs to change

        Returns
        -------
        list of int
            a list of IDs based on the input where the number of samples for each class is the same

        """
        random.seed(24)

        labels_distinct = set(self.labels.values())

        #SORT the IDs by their labels
        min_size = len(self.labels)
        label_dict = dict()
        for l in labels_distinct:
            label_dict[l] = [ID for ID in IDs if self.labels[ID] == l]
            if len(label_dict[l]) < min_size:
                min_size = len(label_dict[l])

        for l in labels_distinct:
            random.shuffle(label_dict[l]) #shuffle the list of IDs for each label
            label_dict[l] = label_dict[l][:min_size] #and the cut off after desired size

        IDs_new = [ID for sublist in list(label_dict.values()) for ID in sublist]
        random.shuffle(IDs_new)
        return IDs_new

    def compute_and_save_embeddings(self, separate_embedding = True, word_embedding = False):
        """Compute embeddings for train exaples and save the on disk"""
        pass

    def get_document(self, example):

        """ Get the text associated with the document ID(s)

        Parameters:
        -----------
        example: int or (int, int)
            document ID(s)

        Returns
        -------
        str
            text associated with the document ID(s)
        """

        pass

    def resume_training(self, model_name, name = 'r1', epochs = 100):

        """ Resume the training from a checkpoint

        Parameters
        ----------
        model_name: str
            name of the model to continue training from (must be in the /model/ directory)

        Returns
        -------
        training history

        """

        mc = ModelCheckpoint(PARENT_DIR + '/model/' + model_name, monitor='val_loss', mode='min', save_best_only=True, verbose=1)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
        model = self.get_model(model_name)
        history = model.fit(x=self.train_generator, validation_data=self.val_generator,
                            epochs=epochs, callbacks=[mc, es])

        # convert the history.history dict to a pandas DataFrame:
        hist_df = pd.DataFrame(history.history)
        # save to json
        hist_json_file = 'history_'+ name + '.json'
        with open(PARENT_DIR + '/model/' + model_name + '/' + hist_json_file, mode='w') as f:
            hist_df.to_json(f)

        print('Evaluation on validation dataset')
        self.evaluate_model(model = model, model_name = model_name, partition = 'val', model_type = model_type)

        #Update training history

        history_old = pd.read_json(PARENT_DIR + '/model/' + model_name + '/'+ 'history.json')
        history_updated = pd.concat([history_old, hist_df], ignore_index=True)

        hist_json_file = 'history.json'
        with open(PARENT_DIR + '/model/' + model_name + '/' + hist_json_file, mode='w') as f:
            history_updated.to_json(f)

        self.plot_history(model_name)
        return history

    def get_model(self, model_name):

        """ Load model from /model directory

        Parameters
        ----------
        model_name: str
            the name of the model (hast to be the name of a dir in /model)

        Returns
        -------
        keras model
        """

        model = tf.keras.models.load_model(PARENT_DIR + '/model/' + model_name)
        return model

    def train_model(self, model_name, model_type):

        """ Train and save a model
        Parameters
        ----------
        model_name: str
            name under which the model should be saved

        Returns
        -------
        history object
            history of the training process
        """

        pass

    def plot_history(self, model_name):
        """Plot the training history"""
        history = pd.read_json(PARENT_DIR + '/model/' + model_name + '/'+ 'history.json')
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(PARENT_DIR + '/model/' + model_name + '/'+ 'loss.png', bbox_inches='tight')
        plt.show()

    def evaluate_model(self, model_name):
        pass

    def get_topic_from_topicID(self, topicID):
        #GET TOPIC
        topic = ''
        for d in self.dictionary:
            if d['ID'] == topicID:
                topic = d['topic']
        return topic

class Clustering(ArgumentMining):

    def set_generators(self, batch_size = 32, shuffle=True, stratify = False):
        super().set_generators(batch_size, shuffle, stratify)
        # Generators
        self.train_generator = DataGenerator(self.partition['train'], self.labels, file_size = self.file_size, dir_size = self.dir_size, path=self.path_embed, batch_size=batch_size, shuffle=shuffle)
        self.val_generator = DataGenerator(self.partition['val'], self.labels, file_size = self.file_size, dir_size = self.dir_size, path=self.path_embed, batch_size=batch_size, shuffle=shuffle)
        self.test_generator = DataGenerator(self.partition['test'], self.labels, file_size = self.file_size, dir_size = self.dir_size, path=self.path_embed, batch_size=batch_size, shuffle=False)

    def compute_and_save_embeddings(self, type = 'bert', separate_embedding = True, word_embedding = False):

        """ Compute the embeddings in batches and save them on disk

        Parameters
        ----------
        separate_embeddings: bool (default True)
            whether to calculate the embeddings for each element in the pair separately
            False: Concatenate the two documents and calculate the embeddings over this paired input directly
            True: Calculate the embeddings of each element of the pair separately and the take the average

        Returns
        -------
        history object
            history of the training process
        """

        IDs = sorted(list(self.examples.keys()))
        assert(IDs == list(range(0, len(IDs)))) #must be continously!

        nr_files = math.ceil(len(IDs) / self.file_size) #divide into several files
        required = list(range(0, nr_files))
        missing = [e for e in required]
        #print('missing: {}'.format(len(missing)))

        for fileID in tqdm(missing):
            dir_nr = math.floor((fileID + 0.0) / self.dir_size)
            batch = []
            IDs_batch = IDs[fileID*self.file_size : (fileID+1) * self.file_size]
            for ID in IDs_batch:
                example = self.examples[ID]
                batch.append(self.get_document(example))

            if separate_embedding:
                #CREATE DIRECTORY
                # try:
                #     os.makedirs(self.path_embed + str(dir_nr) + '_separate'+'/')
                # except FileExistsError:
                #     pass # directory already exists

                batch1= [b[0] for b in batch]
                batch2 = [b[1] for b in batch]
                embeddings1 = sd.get_document_embeddings(batch1, type=type, word_embedding = word_embedding)
                embeddings2 = sd.get_document_embeddings(batch2, type=type, word_embedding = word_embedding)
                # with open(self.path_embed + str(dir_nr) + '_separate' +'/' +str(fileID) + '_1' + '.csv', 'w', newline='') as file:
                #     np.savetxt(file, X = embeddings1, delimiter='\t')
                # with open(self.path_embed + str(dir_nr) + '_separate' +'/' +str(fileID) + '_2' + '.csv', 'w', newline='') as file:
                #     np.savetxt(file, X = embeddings2, delimiter='\t')
                embeddings = abs(embeddings1 - embeddings2)
            else: #Calculate embeddings over pair
                embeddings = sd.get_document_embeddings(batch, type=type, word_embedding = word_embedding)

            #CREATE DIRECTORY
            try:
                os.makedirs(self.path_embed + str(dir_nr) +'/')
            except FileExistsError:
                pass # directory already exists

            with open(self.path_embed + str(dir_nr) +'/' +str(fileID) + '.csv', 'w', newline='') as file:
                np.savetxt(file, X = embeddings, delimiter='\t')

    def get_document(self, example):

        """ Get the text associated with the two document IDs

        Parameters:
        -----------
        example: (int, int)
            two document ID

        Returns
        -------
        str
            documentIDs  and text associated with the document IDs
        """

        d1 = self.documents['documents'][example[0]]['document'] #document1
        d2 = self.documents['documents'][example[1]]['document'] #document2
        return (d1, d2)

    def train_model(self, model_name, epochs=100, layers=[200], model_type = 'FNN'):

        """ Train and save a model for learning similarity of documents

        Returns
        -------
        history object
            history of the training process
        """
        assert(model_type in ['FNN', 'LINEAR', 'SVM'])
        mc = ModelCheckpoint(PARENT_DIR + '/model/' + model_name, monitor='val_loss', mode='min', save_best_only=True, verbose=1)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
        if model_type == 'FNN':
            model = Sequential()
            for nr_neurons in layers:
                model.add(Dense(nr_neurons, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))

            model.compile(optimizer='Adam', loss='mse', metrics=[tf.keras.metrics.MeanSquaredError()])

        elif model_type == 'LINEAR':
            model = Sequential()
            model.add(Dense(1, use_bias=True))
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.99, epsilon=1e-05, amsgrad=False, name='Adam')

            model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])

        history = model.fit(x=self.train_generator, validation_data=self.val_generator,
                            epochs=epochs, callbacks=[mc, es])

        # convert the history.history dict to a pandas DataFrame:
        hist_df = pd.DataFrame(history.history)
        # save to json
        hist_json_file = 'history_t.json'
        with open(PARENT_DIR + '/model/' + model_name + '/'+ hist_json_file, mode='w') as f:
            hist_df.to_json(f)
        #Once again
        hist_json_file = 'history.json'
        with open(PARENT_DIR + '/model/' + model_name + '/'+ hist_json_file, mode='w') as f:
            hist_df.to_json(f)
        self.plot_history(model_name)

        return history

    def evaluate_model(self, model_name, model_type = None, partition = 'val'):
        """ Calculated classification report on test data and returns the correctly and misclassified examples
        Returns
        -------
        dict
            {'classification_report': sklearn classification report, 
             'misclassified_examples': tuples of keys to self.documents['documents'],
             'correctly_classified_examples': tuples of keys to self.documents['documents']}
        """
        assert (partition in ['val', 'test', 'train'])
        if partition == 'val':
            generator = self.val_generator
        elif partition == 'test':
            generator = self.test_generator
        elif partition == 'train':
            generator = self.train_generator

        model = self.get_model(model_name)

        #CONFUSION MATRIX
        y_pred = model.predict(generator) #get predictions
        y_pred = np.reshape(y_pred, y_pred.size)
        y_bin = np.copy(y_pred) #holds 0 and 1 instead of float in [0,1]
        
        thres_0 = y_bin < 0.5 #thresholds for label similar or not
        thres_1 = y_bin >= 0.5
        y_bin[thres_0] = 0 #cast prediction outcome to binary values
        y_bin[thres_1] = 1
        y_bin = list(y_bin)
        y_bin = [int(i) for i in y_bin]
        y_true = [self.labels[ID] for ID in self.partition[partition]]
        cm = confusion_matrix(y_true, y_bin) #calculate confusion matrix
        print(cm)
        cr = classification_report(y_true, y_bin, digits = 3)


        documents = sorted(list(self.documents['documents'].items())) #test documents
        document_IDs = [d[0] for d in documents]
        misclassified = []
        correctly_classified = []

        for i, ID in enumerate(self.partition[partition]):
            docID_a = self.examples[ID][0]
            docID_b = self.examples[ID][1]
            matrix_index_a = document_IDs.index(docID_a)
            matrix_index_b = document_IDs.index(docID_b)
            if y_true[i] - y_bin[i] != 0:
                misclassified.append((document_IDs[matrix_index_a], document_IDs[matrix_index_b]))
            else:
                correctly_classified.append((document_IDs[matrix_index_a], document_IDs[matrix_index_b]))
        #returns tuples of indices to self.documents['documents']

        print(cr)
        return {'classification_report': cr, 'misclassified_examples': misclassified,'correctly_classified_examples': correctly_classified}

    def get_distance_matrix(self, model_name, partition = 'test'):
           
        """ Calculate the distance matrix of the documents based on model predictions

        Parameters
        ----------
        model_name: str
            the name of the model (hast to be the name of a dir in /model)
        patition: str in ['val', 'train', 'test', 'all']
            which partition to use

        Returns
        -------
        tuple (np.array, list of int, list of int)
            (distance_matrix, document_IDs, true_groups_IDs)
            The indexes of the distance matrix correspond to the indexes of the two lists
        """

        assert (partition in ['val', 'test', 'train', 'all'])
        #GET THE IDs of the DOCUMENTs
        documents = sorted(list(self.documents['documents'].items())) #test documents
        document_IDs = [d[0] for d in documents] # IDs to self.documents['documents']
        key = self.documents['parentkey'] #should be subtopicID for AS and topicID for DS
        IDs_of_true_groups = [d[1][key] for d in documents]

        model = self.get_model(model_name)
        #GET PREDICTION
        if partition == 'test':
            y_pred = model.predict(self.test_generator)
            partition = self.partition['test']
        elif partition == 'train':
            y_pred = model.predict(self.train_generator)
            partition = self.partition['train']
        elif partition == 'val':
            y_pred = model.predict(self.val_generator)
            partition = self.partition['val']
        elif partition == 'all':
            y_pred_train = model.predict(self.train_generator)
            y_pred_val = model.predict(self.val_generator)
            y_pred_test = model.predict(self.test_generator)
            y_pred = y_pred_train.flatten().tolist() + y_pred_val.flatten().tolist() + y_pred_test.flatten().tolist() 
            y_pred = np.array(y_pred)
            partition = self.partition['train'] + self.partition['val'] + self.partition['test']


        
        y_pred = np.reshape(y_pred, y_pred.size)
        y_pred = np.around(y_pred, decimals = 2) #round to two decimals

        nr_documents = len(document_IDs) 
        distance_matrix = np.full(shape=(nr_documents, nr_documents), fill_value = 10.0)
        for i, ID in tqdm(enumerate(partition)): 
            docID_a = self.examples[ID][0]
            docID_b = self.examples[ID][1]
            matrix_index_a = document_IDs.index(docID_a)
            matrix_index_b = document_IDs.index(docID_b) 
            score = y_pred[i]
            distance_matrix[matrix_index_a, matrix_index_b] = 1.0 - score #since we want distances and not similarites for the clustering
        #distance_matrix = np.around(distance_matrix, decimals = 2)
        distance_matrix_T = distance_matrix.transpose()
        M = (distance_matrix + distance_matrix_T)/2 #since distance matrix has to be symmetrical!
        #Calculation of how symmetric the results are:
        
        mask2D = distance_matrix <= 1 #values to keep
        mask1D = np.add.accumulate(mask2D, 0)[-1] == 0 #get last row of accumulated array, if entry is zero mask it
        
        differences = np.absolute(M - distance_matrix)
        avg_dev = np.sum(differences) / np.count_nonzero(mask2D)
        print('Average deviation (symmetry): {}'.format(avg_dev)) 
        M = ma.masked_array(M, mask=np.logical_not(mask2D))
        M = np.around(M, decimals = 2)
        document_IDs =  ma.masked_array(document_IDs, mask=mask1D)
        IDs_of_true_groups = ma.masked_array(IDs_of_true_groups, mask=mask1D)
        
        return (M, document_IDs, IDs_of_true_groups)

    def compute_cluster(self, model_name, partition = 'test', clustering_type = 'agg'):
  
        """ Computes a clustering of documents based on a model which predicts pairwise similarities.

        Parameters
        ----------
        model_name: str
            the name of the model which predicts pairwise similarities of text documents
            (hast to be the name of a dir in /model)
        patition: str in ['val', 'train', 'test', 'all']
            which partition to use

        Returns
        -------
        tuple
            (clustering, ARI-score)
        """

        #IDs point are keys in self.documents['documents']
        distance_matrix, document_IDs, y_true = self.get_distance_matrix(model_name, partition) #as masked arrays
        d = distance_matrix[~y_true.mask].transpose()[~y_true.mask]
        d[d.mask] = 0.0
        d_bin = np.rint(d)
        eval_dict = {'ARI': {}, 'Homogeneity': {}, 'Completeness': {}, 'F1_bcubed':{}, 'n_clusters_true':{}, 'n_clusters_pred':{} }
        for i in tqdm(list(range(170, 200, 2))):
            if clustering_type == 'agg':
                clustering = AgglomerativeClustering(n_clusters = i, linkage='average', 
                                                    affinity='precomputed', compute_full_tree=True,
                                                    distance_threshold=None)
            elif clustering_type == 'hdbscan':
                clustering = hdbscan.HDBSCAN(min_samples=2, min_cluster_size=2, metric='precomputed')
            clustering.fit(d_bin)
            y_pred = clustering.labels_
            assert(len(y_pred) == y_true.count())
            eval_results = self.evaluate_cluster_on_ground_truth(y_pred, list(y_true[~y_true.mask]))
            for key in eval_results.keys():
                    eval_dict[key][i] = eval_results[key]
        df = pd.DataFrame(eval_dict) 
        # save to json
        file = model_name + '_' + partition 
        with open(PARENT_DIR + '/model/clustering_results/' + file + '.json', mode='w') as f:
            df.to_json(f, date_format='epoch')
        self.plot_clustering_results(file, 'n_clusters')
        return None

        #Transform back into masked array with matching IDs to documents
        y_pred_masked = np.zeros(shape=document_IDs.shape)
        count = 0
        for i, mask_value in enumerate(list(document_IDs.mask)): #masked array
            if mask_value == False:
                y_pred_masked[i] = y_pred[count]
                count +=1
        y_pred_masked = np.array(y_pred_masked)
        y_pred_masked = ma.masked_array(y_pred_masked, mask=document_IDs.mask)
        assert((y_pred_masked[~y_pred_masked.mask] == y_pred).all())
        return {'clustering': clustering, 'doc_IDs': document_IDs, 'y_pred': y_pred, 'y_true': y_true}

    def plot_clustering_results(self, file, label):
        df = pd.read_json(PARENT_DIR + '/model/clustering_results/' + file + '.json')
        #plt.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.plot(df['ARI'])
        plt.plot(df['Homogeneity'])
        plt.plot(df['Completeness'])
        plt.plot(df['F1_bcubed'])
        #plt.plot(df['n_clusters_pred']/df['n_clusters_true'])
        plt.xlabel(label)
        plt.legend(['ARI', 'Homogeneity', 'Completeness', 'F1_bcubed'], loc='center left')
        plt.savefig(PARENT_DIR + '/model/clustering_results/' + file + '.png', bbox_inches='tight')
        plt.show()

    def evaluate_cluster_on_ground_truth(self, y_pred, y_true, name = ''):
        """ Computes evaluation metric
        Parameters
        ----------
        y_pred: list
        y_true: list
        """
        #apply ARI (adjusted Rand index) oder ANMI adjusted mutual information
        #cm = confusion_matrix(y_true, y_pred)
        label_mapping = dict()
        for i, l in enumerate(y_true):
            try : label_mapping[l].append(y_pred[i])
            except:
                label_mapping[l] = []
                label_mapping[l].append(y_pred[i])
        #TODO use this?
        label_mapping = self.get_most_frequent_prediction(label_mapping)

        ho = homogeneity_score(y_true, y_pred)
        co = completeness_score(y_true, y_pred)
        score = ari(y_true, y_pred)

        #Calculate Bcubed F1 score
        ldict = {ID: {label} for (ID, label) in enumerate(y_true)}
        cdict = {ID: {label} for (ID, label) in enumerate(y_pred)}
        precision = bcubed.precision(cdict, ldict)
        recall = bcubed.recall(cdict, ldict)
        f1score = bcubed.fscore(precision, recall)
        n_true = len(set(y_true)) #Number of ground truth groups
        n_pred = len(set(y_pred)) #Number of found clusters
        result = {'ARI': round(score, 3), 'Homogeneity': round(ho, 3), 'Completeness': round(co, 3), 'F1_bcubed': round(f1score,3), 'n_clusters_true': n_true,'n_clusters_pred': n_pred, 'name': name, 'n_documents': len(y_true)}
        print(result)
        return result


    def get_most_frequent_prediction(self, true_to_pred_labels):
        label_mapping = dict()
        for key, predicted_labels in true_to_pred_labels.items():
            label_mapping[key] = most_frequent(predicted_labels)

        return label_mapping


    #  if self.documents['mode'] == 'AS':
    #         #Sort arguments by topics into dict
    #         docs_by_topic = dict()
    #         for id, d in all_documents:
    #             try : docs_by_topic[d['topicID']].append((id, d))
    #             except:
    #                 docs_by_topic[d['topicID']] = []
    #                 docs_by_topic[d['topicID']].append((id, d))

    #         #Compute topic modeling for every topic separately
    #         eval_results_total = dict()
    #         for example, documents in tqdm(docs_by_topic.items()):
    #             X, mapping, vectorizer = self.compute_embeddings(documents, topicID = example, vectorizer_type=vectorizer_type, word_embedding = word_embedding,stemming = stemming, stopwords= stopwords)
    #             targets = [self.documents['documents'][dID]['subtopicID'] for dID in mapping] #targets are subtopics
    #             n_groups = math.floor(0.15 * len(documents)+ 1.71) #TODO this is only for debatepedia
    #             embedding = self.dim_reduction(n_groups, X, vectorizer, dim_reduction = dim_reduction, n_neigh = 3, n_comp = 4)
    #             topic = self.get_topic_from_topicID(example)


    #             print(topic)
    #             labels = self.clustering(embedding, clustering_type= clustering_type)
    #             eval_results = self.evaluate_cluster_on_ground_truth(labels, targets)
    #             eval_results_total[example] = eval_results

    #         name = str(dim_reduction) + '_' + str(vectorizer_type)+ '_' + str(word_embedding) + '_' +  str(stemming)
    #         #df = pd.DataFrame(eval_results_total)
    #         df = pd.DataFrame.from_dict(eval_results_total, orient='index')
    #         print(df.describe())
    #         # save to json
    #         file = 'AS_' + name
    #         with open(PARENT_DIR + '/model/clustering_results/' + file + '.json', mode='w') as f:
    #             df.to_json(f)
    #         #self.visualize_clustering(embedding, targets, labels, eval_results = eval_results, name = str(dim_reduction) + '_' + str(vectorizer_type)+ '_' + str(word_embedding) + '_' +  str(stemming), topic= topic, topicID=topicID)
    #         return df


    def topic_modeling(self, partition = 'train', dim_reduction = 'umap', vectorizer_type = 'count', word_embedding = 'onehot', stemming = False, stopwords = True, clustering_type = 'hdbscan', across_topics = False):
        assert(dim_reduction in ['lsa', 'lda', 'umap', None])
        assert (vectorizer_type in ['count', 'tfidf', None])
        assert(word_embedding in ['onehot', 'bertCLS', 'bertAverage', 'fasttext'])

        all_documents = self.documents['documents'].items()


        if self.documents['mode'] == 'AS' and across_topics == True:
            #Sort arguments by topics into dict
            docs_by_topic = dict()
            for index, d in enumerate(all_documents):
                try : docs_by_topic[d[1]['topicID']].append((d[0], index))
                except:
                    docs_by_topic[d[1]['topicID']] = []
                    docs_by_topic[d[1]['topicID']].append((d[0], index))
                

            #Compute topic modeling for every topic separately
            eval_results_total = dict()
            documents = all_documents
            print('Length all documents')
            print(str(len(all_documents)))

            X, mapping, vectorizer = self.compute_embeddings(documents, topicID = 'AS', vectorizer_type=vectorizer_type, word_embedding = word_embedding,stemming = stemming, stopwords= stopwords)
            embedding = self.dim_reduction(300, X, vectorizer, dim_reduction = dim_reduction, n_neigh = 3, n_comp = 4)
            print(embedding.shape)
            
            for tID, topics in tqdm(docs_by_topic.items()):
                topic = self.get_topic_from_topicID(tID)
                print(topic)
                mapping_topic  = [d[0] for d in topics]
                print('mapping_topic length')
                print(str(len(mapping_topic)))
                indexes = [d[1] for d in topics]
                embedding_topic = [embedding[i] for i in indexes] #?
                embedding_topic = np.array(embedding_topic)
                print('embedding')
                print(embedding_topic.shape)

                n_groups = math.floor(0.15 * len(topics)+ 1.71) #TODO this is only for debatepedia
                labels = self.clustering(embedding_topic, clustering_type= clustering_type, n_groups=n_groups)
                print('labels:')
                print(str(len(labels)))
                targets = [self.documents['documents'][dID]['subtopicID'] for dID in mapping_topic] #targets are subtopics
                print('targets:')
                print(str(len(targets)))

                eval_results = self.evaluate_cluster_on_ground_truth(labels, targets)
                eval_results_total[tID] = eval_results

            name = str(dim_reduction) + '_' + str(vectorizer_type)+ '_' + str(word_embedding) + '_' +  str(stemming)
            #df = pd.DataFrame(eval_results_total)
            df_eval_results = pd.DataFrame.from_dict(eval_results_total, orient='index')
            print(df_eval_results.describe())
            # save to json
            file = 'AS_' + name
            with open(PARENT_DIR + '/model/clustering_results/' + file + '.json', mode='w') as f:
                df_eval_results.to_json(f)
            #self.visualize_clustering(embedding, targets, labels, eval_results = eval_results, name = str(dim_reduction) + '_' + str(vectorizer_type)+ '_' + str(word_embedding) + '_' +  str(stemming), topic= topic, topicID=topicID)
            return df_eval_results

        elif self.documents['mode'] == 'AS' and across_topics == False:
            #Sort arguments by topics into dict
            docs_by_topic = dict()
            for id, d in all_documents:
                try : docs_by_topic[d['topicID']].append((id, d))
                except:
                    docs_by_topic[d['topicID']] = []
                    docs_by_topic[d['topicID']].append((id, d))

            #Compute topic modeling for every topic separately
            eval_results_total = dict()
            topic_results_total = dict()
            labels_comparison = dict()
            for example, documents in tqdm(docs_by_topic.items()): 
                for v in [1]:
                    if v == 0:
                        clustering_type = 'kmeans'
                        word_embedding = 'bertAverage'
                        vectorizer_type = None
                        stopwords = False
                        stemming = False
                        dim_reduction = None
                    else:
                        clustering_type = 'hdbscan'
                        word_embedding = 'onehot'
                        vectorizer_type = 'tfidf'
                        stopwords = True
                        stemming = True
                        dim_reduction = None
                    X, mapping, vectorizer = self.compute_embeddings(documents, topicID = example, vectorizer_type=vectorizer_type, word_embedding = word_embedding,stemming = stemming, stopwords= stopwords, max_df = 1.0)
                    targets = [self.documents['documents'][dID]['subtopicID'] for dID in mapping] #targets are subtopics
                    n_groups = math.floor(0.15 * len(documents)+ 1.71) #TODO this is only for debatepedia
                

                    embedding = self.dim_reduction(n_groups, X, vectorizer, dim_reduction = dim_reduction, n_neigh = 3, n_comp = 4)
                    topic = self.get_topic_from_topicID(example)

                    print(topic)
                    labels = self.clustering(embedding, clustering_type= clustering_type, n_groups = n_groups)

                    #SHOW WHAT THE CLUSTERS ARE LIKE!!!
                    topic_results = dict()
                    for index, clusterID in enumerate(labels):
                        #topic_name = self.get_topic_from_topicID(topicID)
                        clusterID = str(clusterID)
                        try: names = topic_results[clusterID]['subtopics']
                        except:
                            topic_results[clusterID] = dict()
                            topic_results[clusterID]['topic'] = self.get_topic_from_topicID(example)
                            topic_results[clusterID]['subtopics'] = []
                            topic_results[clusterID]['labels'] = []
                        #terms = vectorizer.get_feature_names()
                        topic_results[clusterID]['labels'].append((targets[index], mapping[index]))
                        topic_results[clusterID]['subtopics'].append((targets[index], self.get_subtopic_from_subtopicID(targets[index])))
                        topic_results[clusterID]['subtopics'] = list(set(topic_results[clusterID]['subtopics']) )
                    topic_results_total[example] = topic_results

                    labels_sure = [l for l in labels if l != -1]
                    targets_sure = [targets[i] for i, l in enumerate(labels) if l != -1]
                    eval_results = self.evaluate_cluster_on_ground_truth(labels, targets, name = topic)
                    eval_results_total[example] = eval_results
                    if v == 0:
                        labels0 = labels
                    else: labels1 = labels
                labels_comparison[example] = self.evaluate_cluster_on_ground_truth(labels1, labels1, name = topic)

            name = str(clustering_type) + "_" + str(dim_reduction) + '_' + str(vectorizer_type)+ '_' + str(word_embedding) + '_' +  str(stemming)
            #df = pd.DataFrame(eval_results_total)
            df_eval_results = pd.DataFrame.from_dict(eval_results_total, orient='index')
            print(df_eval_results.describe())
            # save to json
            file = 'AS_' + 'eval-results_' + name
            with open(PARENT_DIR + '/model/clustering_results/' + file + '.json', mode='w') as f:
                df_eval_results.to_json(f)

            df_comparison =  pd.DataFrame.from_dict(labels_comparison, orient='index')
            print(df_comparison.describe())
            # save to json
            file = 'AS_' + 'comp-results_' + name
            with open(PARENT_DIR + '/model/clustering_results/' + file + '.json', mode='w') as f:
                df_comparison.to_json(f)

            pp.write_dict_to_json(PARENT_DIR + '/model/clustering_results/unsupervised/', file + '_CLUSTERS', topic_results_total)
            return df_eval_results

        else: #in case of DS over all documents
            #TEST different parameters!
            if partition == 'val':
                #n_list = list(np.arange(0.2, 0.9, 0.02))
                n_list = list(range(25, 100, 1))
                documents = self.val_docs['documents']
                n_neigh = 24
            elif partition =='test':
                #n_list = list(np.arange(0.2, 0.9, 0.02))
                n_list = list(range(40, 100, 1))
                documents = self.test_docs['documents']
                n_neigh = 53
            elif partition == 'train':
                #n_list = list(np.arange(0.2, 0.9, 0.02))
                #n_list = list(range(150, 300, 5))
                #n_list = list(range(2, 70, 1))
                #n_list = [27] #for debateorg
                n_list = [295]
                documents = self.train_docs['documents']
                n_neigh = 27 # 7 for debateorg
            elif partition == 'all':
                documents = self.documents['documents']
                n_neigh = 27
                #n_list = list(range(200, 600, 5))
                n_list = [295]

            eval_dict = {'ARI': {}, 'Homogeneity': {}, 'Completeness': {}, 'F1_bcubed':{}, 'n_clusters_true':{}, 'n_clusters_pred':{}, 'name':{} }
            for param in tqdm(n_list):
                X, mapping, vectorizer = self.compute_embeddings(documents.items(), vectorizer_type = vectorizer_type, word_embedding = word_embedding, stemming = stemming, stopwords= stopwords)
                targets = [documents[dID]['topicID'] for dID in mapping] #targets are topics
                print('targets:')
                print(len(list(set(targets))))

                name = str(dim_reduction)+ '_' + str(vectorizer_type) + '_' + str(word_embedding) + '_' + str(stemming)
                embedding = self.dim_reduction(param, X, vectorizer, dim_reduction = dim_reduction, n_neigh = n_neigh)
                labels = self.clustering(embedding, clustering_type = clustering_type, n_groups=param)

                # topic_results = dict()
                # for index, topicID in enumerate(targets):

                #     topic_name = self.get_topic_from_topicID(topicID)
                #     try: name = topic_results[topicID]['topic']
                #     except:
                #         topic_results[topicID] = dict()
                #         topic_results[topicID]['topic'] = topic_name
                #         topic_results[topicID]['labels'] = []

                #     terms = vectorizer.get_feature_names()
                #     topic_results[topicID]['labels'].append(labels[index])
                #     #topic_results[topicID]['labels'].append(terms[labels[index]])
                # print(topic_results)

                #SHOW WHAT THE CLUSTERS ARE LIKE!!!
                topic_results = dict()
                for index, clusterID in enumerate(labels):
                    #topic_name = self.get_topic_from_topicID(topicID)
                    clusterID = str(clusterID)
                    try: names = topic_results[clusterID]['topics']
                    except:
                        topic_results[clusterID] = dict()
                        topic_results[clusterID]['topics'] = []
                        topic_results[clusterID]['labels'] = []
                    #terms = vectorizer.get_feature_names()
                    topic_results[clusterID]['labels'].append((targets[index], mapping[index]))
                    topic_results[clusterID]['topics'].append((targets[index], self.get_topic_from_topicID(targets[index])))
                    topic_results[clusterID]['topics'] = list(set(topic_results[clusterID]['topics']) )
                print(topic_results)

                #Exclude noise
                labels_sure = [l for l in labels if l != -1]
                targets_sure = [targets[i] for i, l in enumerate(labels) if l != -1]

                print('Proportion of as noise classified examples')
                print(str(
                    (len(targets) - len(targets_sure))/len(targets)
                    ))

                eval_results = self.evaluate_cluster_on_ground_truth(labels_sure, targets_sure)
                #eval_results = self.evaluate_cluster_on_ground_truth(labels, targets)
                for key in eval_results.keys():
                    eval_dict[key][param] = eval_results[key]
            df_eval_results = pd.DataFrame(eval_dict)
            # save to json
            file = str(self.documents['mode']) + '_' + partition + '_' + str(vectorizer_type) + '_' + str(dim_reduction) + '_' + str(word_embedding)
            with open(PARENT_DIR + '/model/clustering_results/unsupervised/' + file + '.json', mode='w') as f:
                df_eval_results.to_json(f)
            #PLOT PARAMETER DEVELOPMENT
            self.plot_clustering_results(file, 'n_dim LSA')
            pp.write_dict_to_json(PARENT_DIR + '/model/clustering_results/unsupervised/', file + '_CLUSTERS', topic_results)
            return (eval_dict, topic_results)


    def get_topic_from_topicID(self, topicID):
        #GET TOPIC
        topic = ''
        for d in self.dictionary:
            if d['ID'] == topicID:
                topic = d['topic']
        return topic
    def get_subtopic_from_subtopicID(self, subtopicID):
        #GET TOPIC
        subtopic = ''
        for d in self.dictionary:
            for s in d['subtopics']:
                if s['ID'] == subtopicID:
                    subtopic = s['title']
        return subtopic

    def compute_embeddings(self, documents, topicID = 'DS', vectorizer_type = 'count', word_embedding = 'onehot', stemming = False, stopwords = True, max_df = 0.2, max_features= None):
        """Compute tf-idf vectors of all documents

        Parameters
        ----------
        documents: list of tuple (int, str)
            A list of document tuples (docID, document)
        embed_type: str in ['count', 'tfidf', None]
            type of embedding for document embedding
        word_embedding: str in ['onehot', 'fasttext', 'bert']

        stemming: bool, default False
            whether to perform stemming
        stopwords: bool
            wheter to remove stopwords
        max_df: int or float in [0,1], default 0.5
            maximum document frequency for tfidf and count vectorizers
        """
        assert ((word_embedding == 'onehot' and vectorizer_type != None) or (word_embedding != 'onehot'))
        #tokenize and preprocess the documents
        documents_tokenized = [(id, pp.tokenize_and_clean_text(d['document'], stemming = stemming, stopwords = stopwords)) for id, d in documents]
        mapping = [d[0] for d in documents_tokenized]
        documents_cleaned = [' '.join(d[1]) for d in documents_tokenized]

        #VECTORIZER
        if vectorizer_type == None:
            vectorizer = None
        else:
            if vectorizer_type == 'tfidf':
                vectorizer = TfidfVectorizer(max_df = max_df, smooth_idf=True, max_features= max_features)
            elif vectorizer_type == 'count':
                vectorizer = CountVectorizer(max_df=max_df, max_features = max_features)
            X = vectorizer.fit_transform(documents_cleaned)
            print(X.shape)

        #WORD EMBEDDINGS
        if word_embedding == 'onehot':
            pass
        elif word_embedding == 'fasttext':
            for i, d in enumerate(documents_tokenized):
                  row = X[i] #the to the document corresponding tfidf or count vectorizer weights
            #     weights = row[np.flatnonzero(row)] #only the values which are relevant (not zero)
            #     size = weights.shape[0]
            #     sd.get_document_embeddings(d[0], type='fasttext')

        elif word_embedding in ['bertCLS', 'bertAverage']:
            path_e = self.path_embed +  str(topicID) + '_' + word_embedding +  '.csv'
            if False:
            #if path.exists(path_e):
                with open(path_e, 'r') as read_obj:
                    lines  = read_obj.readlines()
                examples = []
                for l in lines:
                    examples.append(np.fromstring(l, sep='\t'))
                X = np.array(examples)
            else:
                if word_embedding == 'bertCLS':
                    X = sd.get_document_embeddings(documents_cleaned, type='bert', word_embedding = False)
                elif word_embedding == 'bertAverage':
                    X = sd.get_document_embeddings(documents_cleaned, type='bert', word_embedding = True)
                #save as npy
                #with open(path_e, 'w') as file:
                #    np.savetxt(file, X, delimiter='\t')

            print('bert' + str(X.shape))
        return (X, mapping, vectorizer)

    def dim_reduction(self, n_groups, X, vectorizer = None, dim_reduction = 'umap', n_comp = 4, n_neigh =8):
        """ Perform dimensionality reduction on document embeddings

        Parameters
        ----------
        n_groups: int
            number of expected clusters (heuristic)
        X: np.array
            embedding vector of all documents
        vectorizer: default None
        dim_reduction: str in ['lsa', 'lda', 'umap', None]
            Which dimensionality reduction to perform

        """
        assert(dim_reduction in ['lsa', 'lda', 'umap', None])

        if n_neigh == 'dynamic':
            n_neighbors = max(round(X.shape[0] / 38), 2) #heuristic
            n_neigh = n_neighbors
        if n_comp == 'dynamic':
            n_comp = n_groups

        #LSA TOPIC MODELING
        if dim_reduction == 'lsa':
            model = TruncatedSVD(n_components=n_groups, algorithm='randomized', n_iter=100, random_state=122)

        #LDA TOPIC MODELING
        elif dim_reduction == 'lda':
            model = LatentDirichletAllocation(n_components=n_groups,
                            learning_method='online', random_state=0, verbose=0)
        if dim_reduction in ['lsa', 'lda']:
            model.fit(X)
            terms = vectorizer.get_feature_names()
            #PRINT TERMS REPRESENTING FOUND TOPICS
            # for i, comp in enumerate(model.components_):
            #     terms_comp = zip(terms, comp)
            #     sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:7]
            #     print("Topic "+str(i)+": ")
            #     for t in sorted_terms:
            #         print(t[0])
            #         print(" ")

            embedding = model.fit_transform(X)
            embedding = umap.UMAP(n_components=n_comp, metric='cosine', n_neighbors=n_neigh, min_dist=0.001, random_state=12).fit_transform(embedding)

        #DIRECTLY PERFORM DIMENSIONALITY REDUCTION ON EMBEDDINGS
        elif dim_reduction == 'umap':
            embedding = umap.UMAP(n_components=n_comp, metric='cosine', n_neighbors=n_neigh, min_dist=0.001, random_state=12).fit_transform(X)
        elif dim_reduction == None:
            if vectorizer == None:
                embedding = np.array(X)
            else:
                embedding = X.toarray()
            

        return embedding

    def clustering(self, embedding, clustering_type = 'hdbscan', n_max_values = 1, n_groups = 170):
        """Perform Clustering

        Parameters
        ----------
        embedding: np.array
            document embeddings
        clustering_type: str in ['hdbscan', None]

        Returns
        -------

            labels od found clusters
        """

        #HDBSCAN CLUSTERING
        if clustering_type == 'hdbscan':
            labels = hdbscan.HDBSCAN(
                min_samples=2,
                min_cluster_size=2,
            ).fit_predict(embedding)
        elif clustering_type == None:
            labels = np.argmax(embedding, axis =1)
            # n = 0 - n_max_values
            # labels = []
            # for row in embedding:
            #     indices = np.argpartition(row.flatten(), n)[n:]
            #     indices = indices.tolist()
            #     indices = [str(i) for i in indices]
            #     label = int(''.join(indices))
            #     labels.append(label)
        elif clustering_type =='agg':
            clustering = AgglomerativeClustering(n_clusters = n_groups, linkage='complete',
                                               affinity='cosine', compute_full_tree=True,
                                               distance_threshold=None)
            clustering.fit(embedding)
            labels = clustering.labels_
        elif clustering_type == 'kmeans':
            clustering = sklearn.cluster.KMeans(n_clusters=n_groups)
            clustering.fit(embedding)
            labels = clustering.labels_
        return labels

    def visualize_clustering(self, embedding,  targets, labels, eval_results = '', name='', topic = '', topicID = None):
        """ Visualize the clustering results

        Parameters
        ----------
        embedding: np.array
            the document embeddings
        targets: list of int
            the ground truth labels in the order of the documents in embeddings
        labels: list of int
            the predicted labels in the order of the documents in embeddings
        eval_results: dict
            evaluation results of the clustering
        name: str, optional, default ''
            filename addition (e.g. parameters)
        topic: str, optional, default ''
            the name of the topic the clustering is performed (only for AS)
        topic_ID: int
            the ID of the said topic, again only for AS

        """

        #DIMENSIONALITY REDUCTION TO 2D
        visualization = umap.UMAP(n_components=2, n_neighbors=3, min_dist=0.001, random_state=12).fit_transform(embedding) #Perform dimensionality reduction to 2D to plot the results!
        #visualization = embedding
        print(visualization.shape)
        #VISUALIZE RESULTS
        fig = plt.figure(figsize=(10, 10))
        #ax = plt.axes(projection ="3d")
        #ax.scatter(visualization[:, 0], visualization[:, 1], visualization[:, 2],
        plt.scatter(visualization[:, 0], visualization[:, 1],
            c = targets,
            cmap='Spectral',
            s = 15, # size
            edgecolor='none')
        n_groups_true = len(set(targets)) #Number of ground truth groups
        n_groups_pred = len(set(labels)) #Number of found clusters
        plt.suptitle(t= topic + ', N_true: ' + str(n_groups_true) +  ', N_pred: ' + str(n_groups_pred))
        plt.title(str(eval_results)) #Print the evaluation results
        #pops = []
        #for t in list(set(targets)):
        #   pops.append(mpatches.Patch(color=t, label=t))
        #plt.legend(handles=pops)
        #fig = umap.plot.points(visualization, labels=targets)

        #SAVE VISUALIZATION
        try:
            os.makedirs(PARENT_DIR + '/model/clustering_results/' )
        except FileExistsError:
            pass # directory already exists
        topic = strip_non_alphanum(topic)
        topic = '-'.join(topic.split(' '))
        plt.savefig(PARENT_DIR + '/model/clustering_results/' +  str(topicID) + '_' + topic + '_' + name +  '.png', bbox_inches='tight')

    def get_all_topics(self, partition = 'train'):
        if partition == 'train':
            dictionary = self.train_dict
        elif partition == 'val':
            dictionary = self.val_dict
        elif partition == 'test':
            dictionary = self.test_dict
        elif partition == 'all':
            dictionary = self.dictionary

        topics = [d['topic'] for d in dictionary]
        print(topics)
        return topics



class Segmentation(ArgumentMining):
    def __init__(self, path_data, path_embed, mode, file_size, dir_size):
        super().__init__(path_data, path_embed, mode, file_size, dir_size)
        self.idx2label = {0: 'B', 1: 'I', 2: 'O'}

    def set_generators(self, batch_size = 32, shuffle=True, stratify = False):
        super().set_generators(batch_size, shuffle, stratify)
        # Generators
        label2idx = {value: key for key, value in self.idx2label.items()}

        for ID, label in self.labels.items():
            for i, bio in enumerate(label):
                self.labels[ID][i] = label2idx[bio]
        self.train_generator = DataGeneratorSequence(self.partition['train'], self.labels, file_size = self.file_size, dir_size = self.dir_size, path=self.path_embed, batch_size=batch_size, shuffle=shuffle)
        self.val_generator = DataGeneratorSequence(self.partition['val'], self.labels, file_size = self.file_size, dir_size = self.dir_size, path=self.path_embed, batch_size=batch_size, shuffle=shuffle)
        self.test_generator = DataGeneratorSequence(self.partition['test'], self.labels, file_size = self.file_size, dir_size = self.dir_size, path=self.path_embed, batch_size=batch_size, shuffle=False)


    def compute_and_save_embeddings(self, type='bert', separate_embedding=None, word_embedding=False):

        """ Compute the embeddings in batches and save them on disk
        
        Parameters
        ----------
        word_embedding: bool
            True: Calculate averaged word embeddings
            False: Take sentence embedding (output of BERT associated with the [CLS] token)
        """

        IDs = sorted(list(self.examples.keys()))
        assert(IDs == list(range(0,len(IDs)))) #must be continously!

        for ID in tqdm(IDs): #one file for every example
            example = self.examples[ID]
            sentences = self.get_document(example)
            #(nr_sent, embedding_dim)
            embedding = sd.get_document_embeddings(sentences, type=type, word_embedding=word_embedding) #2-dim np.array for one example

            #save as npy
            with open(self.path_embed + str(ID) + '.csv', 'w') as file:
                np.savetxt(file, embedding, delimiter='\t')

    def get_document(self, example, incl_topic_info = True):
          
        """ Get the text associated with the document ID

        Parameters:
        -----------
        example: int
            one document ID

        Returns
        -------
        str
            text associated with the document ID
        """
        if incl_topic_info:
            d = [self.get_topic_from_topicID(self.documents['documents'][example]['topicID']) + '. ' + sentence for (label, sentence) in self.documents['documents'][example]['document']]
        else: 	
            d = [sentence for (label, sentence) in self.documents['documents'][example]['document']]
        return d

    def get_all_sentences(self, partition = 'train'):
        assert(partition in ['train', 'val', 'test'])    
        X = []
        y = []

        for ID in self.partition[partition]:
            embeddings = self.train_generator._load_example(ID = ID)
            embedding = np.vstack(embeddings) #transform to 2D numpy
            X.append(embedding)
            y.append(self.labels[ID])

        X = [d.tolist() for d in X]
        X = [np.array(sentence) for sublist in X for sentence in sublist]
        X = np.array(X)
        y_one_hot = self.train_generator._one_hot(y)
        y_one_hot = [label for sublist in y_one_hot for label in sublist] #flatten
        y_one_hot = np.array(y_one_hot)
        y_categ = [label for sublist in y for label in sublist] #flatten
        return (X, y_one_hot, y_categ)

    def train_model(self, model_name, epochs=100, model_type = 'BILSTM', layers = [200]):
        assert(model_type in ['BILSTM', 'FNN', 'CRF'])
        mc = ModelCheckpoint(PARENT_DIR + '/model/' + model_name, monitor='val_loss', mode='min', save_best_only=True, verbose=1)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=600)

        if model_type == 'BILSTM':
            #Mask timesteps were all values in the input tensor equal 0
            model = Sequential()
            model.add(Masking(mask_value=0.,input_shape=(None, self.doc_embed_dim)))
            model.add(Bidirectional(LSTM(units=200, return_sequences=True)))
            model.add(Dense(3, activation='softmax'))
            model.compile(loss=tfa.losses.SigmoidFocalCrossEntropy(), optimizer='adagrad', 
                metrics=['accuracy'])
            history = model.fit(x=self.train_generator, validation_data=self.val_generator, epochs=epochs, callbacks=[mc, es])
        elif model_type == 'FNN':
            model = Sequential()
            for nr_neurons in layers:
                model.add(Dense(nr_neurons, activation='relu'))
            model.add(Dense(3, activation='softmax'))
            model.compile(loss=tfa.losses.SigmoidFocalCrossEntropy(), optimizer='adagrad', 
                metrics=['accuracy'])
            X, y, _ = self.get_all_sentences(partition = 'train')
            val_x, val_y, _ = self.get_all_sentences(partition= 'val')
            history = model.fit(x=X, y = y, validation_data=(val_x, val_y), shuffle = True, epochs=epochs, callbacks=[mc, es])

        elif model_type == 'CRF':
            crf = sklearn_crfsuite.CRF(
                algorithm='lbfgs',
                c1=0.1,
                c2=0.1,
                max_iterations=100,
                all_possible_transitions=True
            )
            X, y, _ = self.get_all_sentences(partition = 'train')
            crf.fit(X, y)
            
            #Evaluation on test dataset
            print('Evaluation on test dataset')
            test_x, test_y, _ = self.get_all_sentences(partition= 'test')
            y_pred = crf.predict(test_x)
            print(metrics.flat_classification_report(
                test_y, y_pred, digits=3))

            #Evaluation on validation dataset
            print('Evaluation on validation dataset')
            val_x, val_y, _ = self.get_all_sentences(partition= 'val')
            y_pred = crf.predict(val_x)
            print(metrics.flat_classification_report(
                test_y, y_pred, digits=3))

        if not model_type == 'CRF':
            # convert the history.history dict to a pandas DataFrame:     
            hist_df = pd.DataFrame(history.history) 
            # save to json
            hist_json_file = 'history_t.json' 
            with open(PARENT_DIR + '/model/' + model_name + '/'+ hist_json_file, mode='w') as f:
                hist_df.to_json(f)
            #Once again
            hist_json_file = 'history.json' 
            with open(PARENT_DIR + '/model/' + model_name + '/'+ hist_json_file, mode='w') as f:
                hist_df.to_json(f)

            print('Evaluation on validation dataset')
            self.evaluate_model(model = model, model_name = model_name, partition = 'val', model_type = model_type)
            print('Evaluation on test dataset')
            self.evaluate_model(model = model, model_name = model_name, partition = 'test', model_type = model_type)
            model.save(PARENT_DIR + '/model/' + model_name)
            return history

    def evaluate_model(self, model = None, model_name = '', partition = 'val', model_type = 'BILSTM'):
            
        """ Produce evaluation results for a given model and a given dataset.

        Compute loss, accuracy, f1 score and the confusion matrix.
        
        Parameters
        ----------
        model_name : string
            The name of the saved model which to use for the prediction.
        ds : tf.dataset
            Dataset containing (ids, labels, sentences)
        
        Returns
        -------
            tuple
            (result containing loss and accuracy, F1 score, confusion matrix)
        """

        if not model:
            print('Model loaded')
            model = self.get_model(model_name)
        else:
            print('Model directly taken')
        generator = None
        assert (partition in ['val', 'test', 'train'])
        if partition == 'val':
            generator = self.val_generator
        elif partition == 'test':
            generator = self.test_generator
        elif partition == 'train':
            generator = self.train_generator

        
        y_true = [self.labels[ID] for ID in self.partition[partition]]
        y_true_old = y_true.copy()

        if model_type == 'BILSTM':
            print('Evaluate BILSTM')
            y_true = generator._one_hot(y_true)
            y_true = generator._padding(y_true)
            mask = self.extract_mask(y_true)

            y_true = self.get_unmasked_categorical_labels(y_true, mask)
            y_pred = model.predict(generator) 
            y_pred = self.get_unmasked_categorical_labels(y_pred, mask)

            # misclassified = []
            # correctly_classified = []
            # count = 0
            # for d, labels in enumerate(y_true_old[:157]):
            #     for s, true_label in enumerate(labels):
            #         pred_label = y_pred[count]
            #         dID = self.partition[partition][d]
            #         sID = self.documents['documents'][dID]['subtopicID']  
            #         sentence = self.documents['documents'][dID]['document'][s]
            #         output = {'true_label' : int(true_label), 'pred:label' : int(pred_label), 'sentence': sentence}
            #         if true_label - pred_label != 0:
            #             misclassified.append(output)
            #         else:
            #             correctly_classified.append(output)
            #         count+=1

            path = PARENT_DIR + '/model/' + model_name + '/'
            # try: pp.load_dict_from_json(path, 'qualitative_' + partition)
            # except: pp.write_dict_to_json(path, 'qualitative_' + partition , {'wrong' : misclassified, 'correct': correctly_classified})
            # else: pp.write_dict_to_json(path, 'qualitative_2' + partition , {'wrong' : misclassified, 'correct': correctly_classified})



        elif model_type == 'FNN':
            print('Evaluate FNN')
            X, _, y_true = self.get_all_sentences(partition= partition)
            #y_true = [label for label in sublist for sublist in y_true] #categorical??
            y_pred = model.predict(X) 
            y_pred = np.argmax(y_pred, axis=1)
            #y_pred = np.array(y_true.copy()) # for majority class baseline
            #y_pred.fill(1) # for majority class baseline

        cm = confusion_matrix(y_true, y_pred)
        cr = self.get_F1_score(y_true, y_pred)
        print(cr)
        self.print_confusion_matrix(cm)
        self.plot_history(model_name)
        #return (result, f1, cm)
        return (cr, cm)

    def plot_history(self, model_name):
        """Plot the training history"""
        history = pd.read_json(PARENT_DIR + '/model/' + model_name + '/'+ 'history.json', orient = 'columns')
        plt.plot(history['accuracy'])
        plt.plot(history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(PARENT_DIR + '/model/' + model_name + '/'+ 'acc.png', bbox_inches='tight')
        # summarize history for loss
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(PARENT_DIR + '/model/' + model_name + '/'+ 'loss.png', bbox_inches='tight')
        plt.show()
    def print_confusion_matrix(self, cm):

        """ Print confusion matrix in formatted output

        Parameters
        ----------
        cm: nparray
            cm produced with...

        Returns
        -------
            list
            Original confusion matrix as 2d list and with char instead of numerical labels
        """

        print("Confusion matrix")
        print(f'{"":5} {"Predicted"}')
        print(f'{"":5} {self.idx2label[0]:5}  {self.idx2label[1]:5} {self.idx2label[2]:5}')
        for i in range(0,len(cm)):
            print(f'{self.idx2label[i]} {cm[i][0]:5}  {cm[i][1]:5} {cm[i][2]:5}')
        return cm

    def get_unmasked_categorical_labels(self, y, mask):

        """ Transform array of one-hot labels into array of numerical labels.

        The array is flattened (all sentences in one dimension) and the
        masked values are removed.

        Parameters
        ----------
        y : nparray
            3d numpy array of shape (#documents, #sentences, 3)
        mask : nparray
            1d array of True and False

        Returns
        -------
            nparray
            A 1d numpy array of numerical labels
        """

        nr_sentences = y.shape[0]*y.shape[1]
        y = np.reshape(y, (nr_sentences, y.shape[2]))#Flatten arrays -> all sentences in one dimension
        y = y[mask] #Apply flattened mask
        y = np.argmax(y, axis=1) #One-hot labels back to numerical

        return y

    def extract_mask(self, X):

        """ Extract the mask from X

        Parameters
        ----------
        X: nparray
            3d numpy array of shape (samples, timesteps, features)

        Returns
        -------
            nparray
            False when all values in the correponding vector in X are zero.
        """

        X = np.reshape(X, (X.shape[0]*X.shape[1], X.shape[2]))
        masking = Masking(mask_value=0.,input_shape=(X.shape[0], [1]))
        masked_output = masking(X) #Apply masking
        mask = masked_output._keras_mask.numpy() #Extract the mask from masked X_val
        return mask

    def get_F1_score(self, y_true, y_pred):

        """ Compute F1 score.

        Parameters
        ----------
        y_true: nparray
            1d numpy array of the true labels
        y_pred: nparray
            1d numpy array of the predicted labels

        Returns
        -------
            float
            F1 score
        """

        #y_true_tag = [str(item) for item in y_true]
        #y_pred_tag = [str(item) for item in y_pred]
        #f1 = f1_score(y_true_tag, y_pred_tag)

        cr = classification_report(y_true, y_pred, digits = 3)
        return cr


def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        curr_frequency = List.count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            num = i

    return num



