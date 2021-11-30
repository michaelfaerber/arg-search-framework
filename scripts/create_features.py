import csv
import json
import sys
import random
from os.path import abspath, dirname

import gensim
import gensim.downloader as api
import nltk
import numpy as np
import math
from tqdm import tqdm
import pandas as pd
import sklearn
from gensim.models import Phrases
from gensim.models.fasttext import FastText
from gensim.test.utils import datapath, get_tmpfile
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
PARENT_DIR = dirname(dirname(abspath(__file__)))
sys.path.append(PARENT_DIR)
sys.path.append(PARENT_DIR + '/scripts')
import preprocess_general as pp


nltk.download('punkt')

def rm_frame_in_debatepedia_title(title):
         
    """ Remove frame from title (words before the colon)

    Parameters
    ----------
    title: str
        subtopic in debatepedia
    
    Returns
    -------
    str
        the title with frame and colon removed if there was one
    """

    #Remove frame from title (words before the colon)
    title = title.split(':')[-1].strip()
    return title

def pedia_create_documents_DS(pedia_discussions, weight_title):
         
    """ Creates a dictionary of documents from the discussions: one for each subtopic

    Parameters
    ----------
    pedia_discussions: list of dict
        the debatepedia dictionary list containing the discussions:
        {'ID': topic ID
         'topic': discussion title, 
         'subtopics': [{'ID': subtopic ID
                        'title': sub heading,
                        'arguments':[{'claim': claim,
                                      'premise': premise,
                                      'stance': pro/con}]
                        }]
        }
    weight_title: int
        the weight to be given to the title and thus how often it occurs in the document
    
    Returns
    -------
    dict
        of shape
           {'mode' : 'DS' (discussion similarity) or 'AS' (argument similarity)
            'parentkey': 'topicID' (for DS) or 'subtopicID' (for AS)
            'documents': {
                'documentID': { 'topicID' : , 
                                'subtopicID' : , 
                                'document' : discussion (for DS) or argument (for AS)}
                },
                'documentID': { 'topicID' : , 
                                'subtopicID' : , 
                                'document' : discussion (for DS) or argument (for AS)}
                }, ...
          
        }
        Each key in documents is the document ID (which is either the ID of the discussion (subtopicID) or of the argument(argument ID))
        and each value is a dict containing the topicID, subtopicID and the document.
        DS: Each document consists of concatenated title and arguments within a subtopic. The title occurs weight_title times.
        AS: Each document consists of one argument.
    """
    
    documents = dict()
    documents['mode'] = 'DS'
    documents['parentkey'] ='topicID'
    documents['documents'] = dict()
    for d in pedia_discussions:
        for s in d['subtopics']:
            document = (rm_frame_in_debatepedia_title(s['title']) + ' ') * weight_title
            for a in s['arguments']:
                document = document + a['claim']+ ' ' + a['premise'] + ' '
            documents['documents'][s['ID']]= {'topicID' : d['ID'], 'subtopicID' : s['ID'], 'document' : document}
    return documents

def pedia_create_documents_AS(discussions, weight_title=None):
          
    """ Creates a dictionary of documents from the discussions: one for each argument

    Parameters
    ----------
    discussions: list of dict
        the debatepedia dictionary list containing the discussions:
        {'ID': topic ID
         'topic': discussion title, 
         'subtopics': [{'ID': subtopic ID
                        title': sub heading,
                        'arguments':[{'claim': claim,
                                      'premise': premise,
                                      'stance': pro/con}]
                        }]
        }
    weight_title: None
        not used here
    
    Returns
    -------
    dict
        of shape
           {'mode' : 'DS' (discussion similarity) or 'AS' (argument similarity)
            'parentkey': 'topicID' (for DS) or 'subtopicID' (for AS)
            'documents': {
                'documentID': { 'topicID' : , 
                                'subtopicID' : , 
                                'document' : discussion (for DS) or argument (for AS)}
                },
                'documentID': { 'topicID' : , 
                                'subtopicID' : , 
                                'document' : discussion (for DS) or argument (for AS)}
                }, ...
          
        }
        Each key in documents is the document ID (which is either the ID of the discussion (subtopicID) or of the argument(argument ID))
        and each value is a dict containing the topicID, subtopicID and the document.
        DS: Each document consists of concatenated title and arguments within a subtopic. The title occurs weight_title times.
        AS: Each document consists of one argument.
    """
    
    documents = dict()
    documents['mode'] = 'AS'
    documents['parentkey'] ='subtopicID'
    documents['documents'] = dict()
    for d in discussions:
        for s in d['subtopics']:
            for a in s['arguments']:
                document = a['premise']
                documents['documents'][a['ID']]= {'topicID' : d['ID'], 'subtopicID' : s['ID'], 'document' : document}
    return documents

def org_create_documents(org_discussions, weight_title):
           
    """ Creates list of documents from the discussions: one document for each discussion

    Parameters
    ----------
    org_discussions: list of dict
        the debateorg dictionary list containing the discussions:
        {'title': discussion title,
         'category': category,
         'url': url,
         'posts':[{'post': post,
                   'stance': stance}]
        }
    weight_title: int
        the weight to be given to the title and thus how often it occurs in the document
    
    Returns
    -------
    list of str
        the documents, each documents consists of concatenated title and posts within a discussion. The title occurs weight_title times.
    """

    documents = dict()
    ID = 0
    for d in org_discussions:
        document = (d['title'] + ' ') * weight_title
        for a in d['posts']:
            document = document + a['post'] + ' '
        documents[int(ID)]= document
        ID +=1
    return documents

def create_documents_SEG(discussions, weight_title=None):
          
    """ Creates a dictionary of documents from the discussions: one for each argument

    Parameters
    ----------
    discussions: list of dict
        the debatepedia dictionary list containing the discussions:
        {'ID': topic ID
         'topic': discussion title, 
         'subtopics': [{'ID': subtopic ID
                        title': sub heading,
                        'arguments':[{
                            'ID': argument ID
                            'claim': claim,
                            'premise': premise,
                            'stance': pro/con
                         }],
                        'posts':[{
                            'ID': post ID
                            'post' : [(argID, sentence)]
                        }] 
                      }]
        }
    weight_title: None
        not used here
    
    Returns
    -------
    dict
        of shape
           {'mode' : 'DS' (discussion similarity) or 'AS' (argument similarity)
            'parentkey': 'topicID' (for DS) or 'subtopicID' (for AS)
            'documents': {
                'documentID': { 'topicID' : , 
                                'subtopicID' : ,
                                'document' : list of tuples (argumentID, sentence)
                              }
                },
                'documentID': { 'topicID' : , 
                                'subtopicID' : , 
                                'document' : list of tuples (argumentID, sentence)
                              }
                }, ...
          
        }
        Each key in documents is the document ID (which is either the ID of the discussion (subtopicID) or of the argument (argument ID))
        and each value is a dict containing the topicID, subtopicID and the document.
        DS: Each document consists of concatenated title and arguments within a subtopic. The title occurs weight_title times.
        AS: Each document consists of one argument.
        SEG:Each document consists of a list of (argumentID, sentence) tuples
    """
    
    documents = dict()
    documents['mode'] = 'SEG'
    documents['parentkey'] ='topicID'
    documents['documents'] = dict()
    for d in discussions:
        for s in d['subtopics']:
            #list of (aID, sentence) tuples
            for p in s['posts']:
                document = p['post']
                documents['documents'][p['ID']]= {'topicID' : d['ID'], 'subtopicID' : s['ID'], 'document' : document}
    return documents

def get_tfidf_weighted_bow_vectors(documents, path, filename, wv = None, documents_tokenized=None):

    """ Produce tf-idf weighted one-hot vectors

    Parameters
    ----------
    documents: list of str
        the input texts
    
    Returns
    -------
    (sparse matrix of type numpy.float64, list, callable)
        of shape (#documents, #words_in_vocab), list of vocab of size #words_in_vocab
    """

    #vectorizer = TfidfVectorizer(analyzer=lambda x:[w for w in x]) #identity function since documents come in tokenized
    #vectorizer = TfidfVectorizer(tokenizer=lambda x:x, lowercase=False, min_df=5)

    if not documents_tokenized:
        vectorizer = TfidfVectorizer(strip_accents='unicode', tokenizer=pp.tokenize_and_clean_text, min_df=12, sublinear_tf = True)
        #
        X = vectorizer.fit_transform(documents)
        vocab = vectorizer.get_feature_names()
        #TODO How can I test wheter build_tokenizer() does what I want???
        documents_tokenized = tokenize_documents(documents, vectorizer.build_tokenizer(), path, filename)
    else: #Doesn't work!!!!
        vectorizer = TfidfVectorizer(tokenizer=lambda x:x, lowercase=False)
        X = vectorizer.fit_transform(documents_tokenized)
        vocab = vectorizer.get_feature_names()
    #not tested!! probably not working because X is sparse matrix!!
    nr_errors =0
    if wv:
        for d in range(0, len(documents_tokenized)): #calculate tfidf scores for every token in each document
            weight_sum =0
            bow = np.zeros((X.shape[0], 100)) #bow matrix filled with zeros
            for token in documents_tokenized[d]: 
                try:
                    tfidf = X[d, vocab.index(token)]
                    vec = wv[token] #get word vector
                    bow[d] = bow[d] + (tfidf * vec)
                    weight_sum += tfidf
                except:
                    nr_errors = nr_errors + 1
                    print('Error for word: {} during calculation of tfidf weighted vectors.'.format(token)) 
                    pass
            bow[d] = bow[d] / weight_sum #averaging -> is this necessary?
        print('Number of errors was: {}'.format(nr_errors))
        return (bow, vocab)
    else: return (X, vocab)

def train_and_save_model(documents_tokenized):
          
    """ Learn the word vectors with a FastText model.

    Parameters
    ----------
    documents_tokenized: list of list of str
        the input tokenized documents
    
    Returns
    -------
    FastText model
        the model is additionally written to disk. Word vectors will be of dimension 100.
    """
    
    model = FastText(size=100, window=4, min_count=1)
    model.build_vocab(sentences=documents_tokenized)
    model.train(sentences=documents_tokenized, total_examples=len(documents_tokenized), epochs=5)
    filename = get_tmpfile("fasttext.model")
    model.save(filename)
    return model

def load_model(filename):
    model = FastText.load(filename)
    return model

def create_documents(mode, discussions, weight_title=1, train_size = 1.0):
          
    """ Tokenizes the documents from dict to a list of documents

    Loads the dict from disk, creates a list of documents (list of str).
    A documents consists of the title and the post concatenated.
    If weight_title is set then the title is repeated weight_title times.
    The result is written to disk.

    Parameters
    ----------
    mode: str
        either 'debateorg' or 'debatepediaDS' oder 'debatepediaAS'
    path: str
        path to the directory where the json file is
    filename: str
        name of the file without extension
    weight_title: int (default 1)
        the weight to be given to the title (useful for tfidf calculations)
    train_size: float (default 1.0)
        between 0 and 1. If 1.0 there will be no test set
    
    Returns
    -------
    dict
        documents 
    """
    
    if mode.startswith('debatepedia'):
        if mode == 'debatepediaDS':
            documents = pedia_create_documents_DS(discussions, weight_title)
        if mode == 'debatepediaAS':
            documents = pedia_create_documents_AS(discussions)
        if mode == 'debatepediaSEG':
            documents = create_documents_SEG(discussions)
    elif mode == 'debateorg':
        documents = org_create_documents(discussions, weight_title)
    else: print('Error: Wrong input variable. Has to be either "debatepedia(DS/AS)" or "debateorg" but was "{}"'.format(mode))

    return documents

def tokenize_documents(documents, tokenizer, path, filename):
    documents_tokenized = [tokenizer(d) for d in documents]
    pp.write_list_of_lists_to_csv(documents_tokenized, path, filename + '_tokenized')
    return documents_tokenized

def create_example_pairs(documents, offset=0):
        
    """ Create dictionatries mapping the examples to the documentIDs and labels

    For discussion similarity.

    Parameters
    ----------
    documents: list of (int, str) tuples
        (topicID, text) tuples
    offset: where to start with indexing
        
    Returns
    -------
    tuple of dict: (dict, dict)
        pairs dict of shape
        {
            'exampleID_1': (docID, docID),
            'exampleID_2': (docID, docID),  
            'exampleID_3': (docID, docID),
            ...
        }
        and labels dict of shape
        {
            'exampleID_1': label,
            'exampleID_2': label,  
            'exampleID_3': label,
            ...
        }
    """
    parent_key = documents['parentkey']
    if documents['mode'] == 'AS': AS = True
    else: AS = False

    pairs = dict()
    labels = dict()

    label = 0
    ID = offset
    for a in tqdm(sorted(list(documents['documents'].keys()))):
        for b in sorted(list(documents['documents'].keys())):
            condition = (not AS) or (AS and (documents['documents'][a]['topicID'] == documents['documents'][b]['topicID'])) #create only arguments pairs within a topic
            if (a != b and condition): #don't include the same document
                if documents['documents'][a][parent_key] == documents['documents'][b][parent_key]:    #create label based whether they belong to the same group (i.e. topic or subtopic)
                    label = 1
                else: label = 0
                pairs[ID] = (a,b)
                labels[ID] = label

                ID+=1
    return (pairs, labels)

def create_examples_SEG(documents, offset = 0):
    """
    Parameters
    ----------

    Returns
    -------
    tuple of dict
        (examples, labels)
    """

    examples = dict()
    labels = dict()

    label = 0
    ID = offset
    for d in sorted(list(documents['documents'].keys())):
        examples[ID] = d
        label = [label for (label, sentence) in documents['documents'][d]['document']]
        labels[ID] = convert_into_BIO(label)
        ID+=1
    return (examples, labels)

def convert_into_BIO(labels):
    
    """Builds a dataset of sentence sequences with labels and ids 

    Parameters
    ----------
    labels : list of int (argID) (or None for not labeled)
        

    Returns
    -------
    list of string
        BIO labels
        
    """
  
    #if set(labels) == {None}: #no labels at all
    #   return ['O']
        #return None
    
    bio = []
    last = -1
    for l in labels:
        if l == None: #no argID
            bio.append('O')
            last = l
        elif l != last: #new argID
            bio.append('B')
            last = l
        elif l == last: #same argID as before
            bio.append('I')
    return bio #TODO convert labels into numbers! :-() Or move to class and use class attributes




def create_examples(documents, offset = 0):
    if documents['mode']  in ['AS', 'DS']:
        return create_example_pairs(documents, offset)
    elif documents['mode'] == 'SEG':
        return create_examples_SEG(documents, offset)
    else: return None









#org_documents = create_documents_from_json('debateorg', path + "debateorg/", 'discussions_debateorg')
#documents_pedia_org = pedia_documents + org_documents
#documents_pedia_org_tokenized = pp.load_csv_into_list_of_lists(path, "discussions_org_pedia_tokenized_mindf12")

#model = train_and_save_model(documents_pedia_org_tokenized)
#fasttext_model300 = api.load('fasttext-wiki-news-subwords-300')
filepath_pretrained = "C:/Users/X/data/Embeddings/cc.en.300.bin"
#wv = gensim.models.fasttext.load_facebook_vectors(filepath_pretrained)
#fb_model = gensim.models.fasttext.load_facebook_model(filepath_pretrained)

#bow, vocab = get_tfidf_weighted_bow_vectors(pedia_documents_tokenized)
#bow_FT, vocab_FT = get_tfidf_weighted_bow_vectors(documents_pedia_org, wv= model.wv)
#bow_FT300, vocab_FT300 = get_tfidf_weighted_bow_vectors(documents_pedia_org, wv= fasttext_model300.wv)
#bow_FT_pretrained, vocab_FT_pretrained = get_tfidf_weighted_bow_vectors(documents_pedia_org, fb_model.wv)
#TODO you can add tokenized documents as a parameter here, too
#bow_OH, vocab_OH = get_tfidf_weighted_bow_vectors(documents_pedia_org, path, 'discussions_org_pedia')






#pd.DataFrame(bow.todense(), columns=tfidf.vocabulary_)
#TODO common phrases??
#pedia_documents = [tokenize_and_clean_text(d) for d in pedia_documents]
#bigram = Phrases(pedia_documents, min_count=20)
#print(list(bigram[pedia_documents[5:10]]))



# #TODO
# def prepare_training_data_debatepedia(discussions, path_manual):
#     #SOME CLEANING
#     #txt file with manually corrected titles
#     manual = []
#     with open(path_manual) as f:
#         for line in f:
#             manual.append(line.strip())

#     for d in discussions:
#         if len(d['subtopics']) > 0 and d['topic'] != 'Democrats vs. Republicans':
#             for q in d['subtopics']:
#                 #Replace 'debate' and 'issue' with specific topic
#                 pat1 = r'(in|on|into) (this|the) (issue|debate)'
#                 if re.search(pat1, q['title']):
#                     q['title'] = re.sub(pat1, 'regarding ' + d['topic'].lower(), q['title'])  
#                 #Replace titles with titles from a manually created list
#                 if q['title'] in manual:
#                     q['title'] = manual[manual.index(q['title'])+1] 
#                 #Remove too short titles
#                 if len(q['title'].split(' ')) <= 3:
#                     d['subtopics'].remove(q)
#                 #Remove frame from title (words before the colon)
#                 q['title'] = q['title'].split(':')[-1].strip()
#     #PREPARE TRAINING DATA
#     #1 shuffle
#     random.shuffle(discussions)
#     #2 train-test split
#     test_size = int(0.2 * len(discussions))
#     test_discussions = discussions[:test_size]
#     train_discussions = discussions[test_size:]
#     #3 create example triples: subtopic1, subtopic2, label
#     test_discussions = flatten_to_subtopics(test_discussions)
#     #flatten_to_subtopics(train_discussions)
#     return test_discussions 

# def flatten_to_subtopics(discussions):
#     subtopics = []
#     for d in discussions:
#         for s in d['subtopics']:
#             subtopics.append({'topic': d['topic'], 'title': s['title'], 'arguments': create_features_over_arguments(s['arguments'])})
#     return subtopics


# def create_examples_from_list_of_subtopic_dict(subtopics):
#     #TODO create stratified training set!
#     examples = []
#     for s1 in subtopics:
#         for s2 in subtopics:
#             if s1 != s2:
#                 #TODO add features for discussion here
#                 if s1[0] == s2[0]:
#                     examples.append((1, s1['title'], s2['title']))
#                 else:
#                     examples.append((0, s1['title'], s2['title']))
#     return examples


# test_now = prepare_training_data_debatepedia(pedia_discussions, path_html + "manual.txt")
