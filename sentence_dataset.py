import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import transformers
from transformers import (TFDistilBertModel, DistilBertTokenizer,)
import os
from os.path import abspath, dirname
import sys
from pathlib import Path
PARENT_DIR = dirname(abspath(__file__))
sys.path.append(PARENT_DIR)


def get_document_embeddings_from_bert(documents,
                                      model_name="distilbert-base-uncased", word_embedding = False):
  """Extract sentence embeddings from BERT model

  Parameters
  ----------
  documents : list of str or list of tuple (str, str) 
    A list containing the sentences.
  model_name : str, optional
    The name of a huggingface model. The default is "distilbert-base-uncased"
  word_embedding: bool
    True: Calculate averaged word embeddings
    False: Take sentence embedding (output of BERT associated with the [CLS] token)

  Returns
  -------
  nparray
    A numpy array of sentence embeddings in the shape
    (number of sentences, embedding vector size)
  """

  # Load pretrained model and tokenizer
  model = TFDistilBertModel.from_pretrained(model_name)
  tokenizer = DistilBertTokenizer.from_pretrained(model_name)

  # Apply tokenizer to a batch of sentences
  # -> Performs padding, tokenizes the input and adds the special tokens
  input_str_batch = documents
  input_ids_dict = tokenizer.batch_encode_plus(input_str_batch,
                                                 add_special_tokens=True,
                                                 padding='max_length',
                                                 truncation = True,
                                                 max_length = 512)
  # Get tokenized inputs and attention mask from the tokenizer's output
  # Will be the input the for bert model
  padded_input = np.array(input_ids_dict['input_ids'])
  # mask indicating what was padded
  attention_mask = np.array(input_ids_dict['attention_mask'])

  last_hidden_states = model(padded_input, attention_mask=attention_mask)[0]

  if word_embedding:
    embedding = np.sum(last_hidden_states[:, 1:-1, :], axis=1)
    embedding = embedding / last_hidden_states.shape[0]
  else:
    # Sentence embeddings are the outputs of bert corresponding to first (CLS) token
    # (number of examples, max number of tokens in the sequence, number of hidden units in the DistilBERT model)
    embedding = last_hidden_states[:, 0, :]

  return embedding

def get_document_embeddings(documents, type="bert", word_embedding= False):
  """Get the sentence embeddings

  Parameters
  ----------
  documents : list of str
    A list containing the sentences.
  type : str, optional
    the default is 'bert', other options (like average) not implemented yet

  Returns
  -------
  nparray
    A numpy array of sentence embeddings in the shape
    (number of sentences, embedding vector size)
  """

  if type == "bert":
    return get_document_embeddings_from_bert(documents, word_embedding = word_embedding)
  else:
    raise Exception("'{}' not accepted as type.".format(type))







