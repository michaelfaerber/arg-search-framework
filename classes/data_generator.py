import numpy as np
import random 
import math
import csv
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import sys
from os.path import abspath, dirname
PARENT_DIR = dirname(dirname(abspath(__file__)))
sys.path.append(PARENT_DIR)
sys.path.append(PARENT_DIR + '/scripts')
import sentence_dataset as sd


class DataGenerator(Sequence):
    """Generates data for Keras"""
    def __init__(self, list_IDs, labels, file_size, dir_size, path, batch_size=32,
                 shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.labels = labels #the dict with {'ID': label} key-value pairs
        self.list_IDs = list_IDs #a list of IDs (training or validation)
        self.shuffle = shuffle
        self.file_size = file_size
        self.dir_size = dir_size
        self.path = path

        self._on_epoch_end() #call it once on the beginning

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(math.ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        batch = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        assert(len(batch) != 0)

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in batch]

        # Generate data
        return self._data_generation(list_IDs_temp)

    def _on_epoch_end(self):
        """Updates indexes after each epoch"""

        self.indexes = list(range(len(self.list_IDs)))
        if self.shuffle == True:
            random.shuffle(self.indexes)
                
    def _data_generation(self, list_IDs_temp):
        """ Generate data"""
        
        assert(len(list_IDs_temp) != 0)
        
        X_batch = [] #list of numpy arrays (= embeddings of the examples)
        y = [] #list of integers (= labels of the examples)
        file_nr = math.floor((list_IDs_temp[0] + 0.0) / self.file_size)
        dir_nr = math.floor((file_nr + 0.0) / self.dir_size)

        with open(self.path + str(dir_nr) + '/' + str(file_nr) + '.csv', 'r') as read_obj:
            lines = read_obj.readlines()

        for ID in list_IDs_temp:
            #OPEN THE FILE if needed
            cur_file_nr = math.floor((ID + 0.0) / self.file_size)
            if cur_file_nr != file_nr: #For continuous examples (like in test data), open file not on every example
                file_nr = cur_file_nr
                dir_nr = math.floor((file_nr + 0.0) / self.dir_size)
                with open(self.path + str(dir_nr) + '/' + str(file_nr) + '.csv', 'r') as read_obj:
                    lines = read_obj.readlines()

            #READ embeddings
            line_nr = ID % self.file_size
            example = lines[line_nr]
            embedding = np.fromstring(example, sep='\t') #transform to numpy
            X_batch.append(embedding)
            y.append(self.labels[ID])

        return self._get_X_y_from_batch(X_batch, y) #transform into numpy arrays

    def _get_X_y_from_batch(self, X_batch, y):
        X = np.vstack(X_batch)
        y = np.array(y)
        return (X, y)

class DataGeneratorSequence(DataGenerator):
    def __init__(self, list_IDs, labels, file_size, dir_size, path, batch_size=32,
                 shuffle=True):
        super().__init__(list_IDs, labels, file_size, dir_size, path, batch_size=32,
                 shuffle=True)

    def _one_hot(self, label_list):

        """One-hot encode the input list of list of categorical values

        Parameters
        ----------
        label_list: list of list of int
            list of list of categorical values

        Returns
        -------
        list of list of np.array
            The same list but with one-hot numpy arrays instead of the categorical values.   
        
        """
        one_hot = []
        labels = set([item for sublist in label_list for item in sublist])
        
        for sublist in label_list:
            one_hot.append(np.eye(len(labels))[np.array(sublist)])
        return one_hot
            

    def _padding(self, batch):

        """Pad the sequence of 2D numpy arrays

        Parameters
        ----------
        batch: list of 2D np.array
            list of (various size, fixed size) np.arrays
        
        Returns
        -------
        np.array of shape (nr examples, pad_val, embedding size)
        
        """
        padded_batch = []

        nr_ex = len(batch)
        #pad_val = max(batch, key = lambda x: x.shape[0]).shape[0]
        pad_val = 50
        try: dim = batch[0].shape[1]
        except: dim = 1

        for embedding in batch:
            padded_embedding = np.zeros((pad_val, dim)) #pad all embeddings to same 2D shape
            embedding = embedding[:pad_val]
            embedding = np.reshape(embedding, (len(embedding), dim))
            a = len(embedding)
            padded_embedding[:a] = embedding
            padded_batch.append(padded_embedding) #list of 2D np.arrays

        output = np.vstack(padded_batch)
        output = np.reshape(output, (nr_ex, pad_val, dim))

        return output
        
    def _data_generation(self, list_IDs_temp):
        assert(len(list_IDs_temp) != 0)
        X_batch = []
        y = []

        for ID in list_IDs_temp:
            embeddings = self._load_example(ID)
            embedding = np.vstack(embeddings) #transform to 2D numpy
            X_batch.append(embedding)
            y.append(self.labels[ID])
        
        return self._get_X_y_from_batch(X_batch, y)

    def _get_X_y_from_batch(self, X_batch, y):

        """
        Parameters
        ----------
        X_batch: list of 2D np.array
            list of np.arrays of shape (various size, embedding-dim)
        y: list of list of int
            list of list of categorical values

        Returns
        -------
        tuple of np.array
            Each item has shape (nr examples, pad_val, embedding size)
            For y the embedding size is the number of different labels (since they are one-hot encoded)

        """
        #Test sample_weights
        sample_weights = []
        for example in y:
            example_copy = example.copy()
            for i, label in enumerate(example):
                if label == 0:
                    example_copy[i] = 2
                elif label == 2:
                    example_copy[i] = 2
            sample_weights.append(example_copy)
        sample_weights = self._padding(sample_weights)

        X = self._padding(X_batch)
        y = self._padding(self._one_hot(y))
        
        
        # return (X, y, sample_weights)
        return (X, y)

    def _load_example(self, ID):
        
        """ Load one file per ID 
        
        Parameters
        ----------
        ID: int

        Returns
        -------
        list of np.array
            list of embeddings
        """
        
        with open(self.path + str(ID) + '.csv', 'r') as read_obj:
            lines  = read_obj.readlines()
        examples = []
        for l in lines:
            examples.append(np.fromstring(l, sep='\t'))
        return examples

        