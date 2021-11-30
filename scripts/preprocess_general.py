import csv
import json
import random
import re
import os
from collections import Counter

import nltk
import pandas as pd
import spacy
from gensim.parsing.preprocessing import (strip_multiple_whitespaces,
                                          strip_non_alphanum, strip_numeric,
                                          strip_short)
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('punkt')
#from spellchecker import SpellChecker

def read_txt_into_list_of_lines(path, filename, enc=None, ext=''):
             
    """ Reads a textfile and returns a list of rows

    Parameters
    ----------
    path: str
        path to the directory where the textfile is
    filename: str
        name of the file (if without file extension the provide ext)
    enc: str (default None)
        encoding of the file
    ext: str (default empty string)
        the file extension
    
    Returns
    -------
    list of str
        list of rows in the textfile
    """
    
    with open(path + filename + ext, encoding= enc) as f:
        lines = []
        for line in f:
            lines.append(line.strip())
        return lines

def write_tuples_to_csv(path, filename, tuples, d='excel-tab', enc=None):
              
    """ Write tuples (can be triples or likewise also) to a csv, one tuple one line

    Parameters
    ----------
    path: str
        path to the directory where the csv should be written
    filename: str
        name of the file without file extension
    enc: str (default None)
        encoding of the file
    d: str (default 'excel-tab')
        delimiter
    
    Returns
    -------
    None
    """
    
    with open(path+ filename + '.csv', 'w', newline="", encoding=enc) as f:
        writer = csv.writer(f, dialect=d)
        writer.writerows(tuples)
    return None

def read_dir_into_documents(path, enc=None):
    documents = []
    for file in os.listdir(path):
        lines = read_txt_into_list_of_lines(path, file, enc)
        documents.append(lines)
    return documents

def read_dir_into_documents_with_names(path, enc=None):
    documents = []
    for file in os.listdir(path):
        lines = read_txt_into_list_of_lines(path, file, enc)
        documents.append({'filename' : str(file), 'lines': lines})
    return documents

def load_csv_into_list_of_lists(path, filename, d='excel-tab', enc='utf8'):
         
    """ Reads a csv file and returns a list of one list for each row

    Parameters
    ----------
    path: str
        path to the directory where the json file is
    filename: str
        name of the file without file extension
    
    Returns
    -------
    list of list of str
        content of the csv
    """
    
    with open(path + filename + '.csv', 'r', newline='', encoding=enc) as read_obj:
        csv_reader = csv.reader(read_obj, dialect=d)
        list_of_rows = list(csv_reader)
    return list_of_rows

def write_list_of_lists_to_csv(list_of_lists, path, filename, d='excel-tab', enc='utf8'):
           
    """ Writes a list of list to a csv file

    Parameters
    ----------
    list_of_lists: list of lists

    path: str
        path to the directory where it should be written
    filename: str
        name of the file without file extension
    
    Returns
    -------
    list of list of str
        content of the csv
    """
    
    with open(path + filename + ".csv", "w", newline='', encoding=enc) as f:
        writer = csv.writer(f, dialect=d)
        writer.writerows(list_of_lists)
    return None

def merge_dict(dict_list):
    merged = dict_list[0]
    for d in dict_list[1:]:
        merged.update(d)
    return merged

def write_dict_to_json(path, filename, d):
        
    """ Writes the dictionary to a JSON file to the specified location

    Parameters
    ----------
    path: str
        path to the directory
    filename: str
        filename to be without extension (e.g. discussions_debatepedia)

    Returns
    -------
    None
        Nothing to be returned
    """

    with open(path + filename + '.json', 'w') as outfile:
        json.dump(d, outfile, indent=2)
    return None

def load_dict_from_json(path, filename):
        
    """ Loads the data from JSON file into a python dictionary

    Parameters
    ----------
    path: str
        path to the directory with the file
    filename: str
        name of the file without extension
    
    Returns
    -------
    dict
        the dict represented in the file
    """

    with open(path + filename + '.json') as json_file: 
        data = json.load(json_file)
        return data

def tokenize_and_clean_text(text, stemming = False, stopwords = True, min = 4, max = 20):
         
    """ Cleans and tokenizes the text 

    Removes special characters and then applies nltk word tokenizer. 
    Results in lowercase tokens without stopwords and minimum size 3.

    Parameters
    ----------
    text: str
        the input text to be cleaned and tokenized
    
    Returns
    -------
    list of str
        the result are lowercase tokens without stopwords and with min size 3
    """

    text = remove_org_com_url(text)
    text = strip_numeric(strip_non_alphanum(text)) #substitute special chars with ' '
    text = rm_underscore(text)
    text = strip_multiple_whitespaces(text)
    text = word_tokenize(text) #tokenize
    text = [token.lower() for token in text] #to lower case
    if stopwords:
        text = [token for token in text if token not in STOPWORDS] #remove custom stopwords
    text = [token for token in text if len(token) >= min] #only keep tokens with size >3
    text = [token for token in text if len(token) <= max] #only keep tokens with size <=20
    if stemming:
        stemmer = PorterStemmer()
        text = [stemmer.stem(token) for token in text]
    return text

def remove_org_com_url(text):
          
    """ Removes urls ending with .org or .com from strings
    Parameters
    ----------
    text: str
        the input text to be cleaned
    
    Returns
    -------
    str
        the cleaned output, urls were replaced by empty string
    """

    #whitespace or start of string followed by non-whitespace and .com or .org
    pat = r'(\s|^)[^\s]+\.(com|org)[^\s]*'
    text = re.sub(pat, '', text)
    return text

def rm_underscore(text):
          
    """ Removes underscores

    Parameters
    ----------
    text: str
        the input text to be cleaned
    
    Returns
    -------
    str
        the cleaned output, underscores were replaces by ' '
    """

    #whitespace or start of string followed by non-whitespace and .com or .org
    pat = r'\_'
    text = re.sub(pat, ' ', text)
    return text

# def get_vocab(texts):
    #     flat_list = [item for sublist in texts_tokenized for item in sublist]
    #     freq = Counter(flat_list)
    #     vocab = freq.most_common()
    #     #word, frequency, word length
    #     vocab = sorted([(word[0], word[1], len(word[0])) for word in vocab], key=lambda x: x[1], reverse=True)
    #     vocab = [(triplet[0], triplet[1]) for triplet in vocab if triplet[2] > 2] #Remove single and two-char tokens
#     return vocab

STOPWORDS = set(["a", "able" , "about" , "above" , "according" , "accordingly" , "across" , "actually" , "after" , "afterwards" , "again" , "against" , "ain't" , "all" , "allow" , "allows" , "almost" , "alone" , "along" , "already" , "also" , "although" , "always" , "am" , "among" , "amongst" , "an" , "and" , "another" , "any" , "anybody" , "anyhow" , "anyone" , "anything" , "anyway" , "anyways" , "anywhere" , "apart" , "appear" , "appreciate" , "appropriate" , "are" , "aren't" , "around" , "as" , "aside" , "ask" , "asking" , "associated" , "at" , "available" , "away" , "awfully" , "be" , "became" , "because" , "become" , "becomes" , "becoming" , "been" , "before" , "beforehand" , "behind" , "being" , "believe" , "below" , "beside" , "besides" , "best" , "better" , "between" , "beyond" , "both" , "brief" , "but" , "by" , "c'mon" , "c's" , "came" , "can" , "can't" , "cannot" , "cant" , "cause" , "causes" , "certain" , "certainly" , "changes" , "clearly" , "co" , "com" , "come" , "comes" , "concerning" , "consequently" , "consider" , "considering" , "contain" , "containing" , "contains" , "corresponding" , "could" , "couldn't" , "course" , "currently" , "definitely" , "described" , "despite" , "did" , "didn't" , "different" , "do" , "does" , "doesn't" , "doing" , "don't" , "done" , "down" , "downwards" , "during" , "each" , "edu" , "eg" , "eight" , "either" , "else" , "elsewhere" , "enough" , "entirely" , "especially" , "et" , "etc" , "even" , "ever" , "every" , "everybody" , "everyone" , "everything" , "everywhere" , "ex" , "exactly" , "example" , "except" , "far" , "few" , "fifth" , "first" , "five" , "followed" , "following" , "follows" , "for" , "former" , "formerly" , "forth" , "four" , "from" , "further" , "furthermore" , "get" , "gets" , "getting" , "given" , "gives" , "go" , "goes" , "going" , "gone" , "got" , "gotten" , "greetings" , "had" , "hadn't" , "happens" , "hardly" , "has" , "hasn't" , "have" , "haven't" , "having" , "he" , "he's" , "hello" , "help" , "hence" , "her" , "here" , "here's" , "hereafter" , "hereby" , "herein" , "hereupon" , "hers" , "herself" , "hi" , "him" , "himself" , "his" , "hither" , "hopefully" , "how" , "howbeit" , "however" , "i'd" , "i'll" , "i'm" , "i've" , "ie" , "if" , "ignored" , "immediate" , "in" , "inasmuch" , "inc" , "indeed" , "indicate" , "indicated" , "indicates" , "inner" , "insofar" , "instead" , "into" , "inward" , "is" , "isn't" , "it" , "it'd" , "it'll" , "it's" , "its" , "itself" , "just" , "keep" , "keeps" , "kept" , "know" , "known" , "knows" , "last" , "lately" , "later" , "latter" , "latterly" , "least" , "less" , "lest" , "let" , "let's" , "like" , "liked" , "likely" , "little" , "look" , "looking" , "looks" , "ltd" , "mainly" , "many" , "may" , "maybe" , "me" , "mean" , "meanwhile" , "merely" , "might" , "more" , "moreover" , "most" , "mostly" , "much" , "must" , "my" , "myself" , "name" , "namely" , "nd" , "near" , "nearly" , "necessary" , "need" , "needs" , "neither" , "never" , "nevertheless" , "new" , "next" , "nine" , "no" , "nobody" , "non" , "none" , "noone" , "nor" , "normally" , "not" , "nothing" , "novel" , "now" , "nowhere" , "obviously" , "of" , "off" , "often" , "oh" , "ok" , "okay" , "old" , "on" , "once" , "one" , "ones" , "only" , "onto" , "or" , "other" , "others" , "otherwise" , "ought" , "our" , "ours" , "ourselves" , "out" , "outside" , "over" , "overall" , "own" , "particular" , "particularly" , "per" , "perhaps" , "placed" , "please" , "plus" , "possible" , "presumably" , "probably" , "provides" , "que" , "quite" , "qv" , "rather" , "rd" , "re" , "really" , "reasonably" , "regarding" , "regardless" , "regards" , "relatively" , "respectively" , "right" , "s", "said" , "same" , "saw" , "say" , "saying" , "says" , "second" , "secondly" , "see" , "seeing" , "seem" , "seemed" , "seeming" , "seems" , "seen" , "self" , "selves" , "sensible" , "sent" , "serious" , "seriously" , "seven" , "several" , "shall" , "she" , "should" , "shouldn't" , "since" , "six" , "so" , "some" , "somebody" , "somehow" , "someone" , "something" , "sometime" , "sometimes" , "somewhat" , "somewhere" , "soon" , "sorry" , "specified" , "specify" , "specifying" , "still" , "sub" , "such" , "sup" , "sure" , "t's" , "take" , "taken" , "tell" , "tends" , "th" , "than" , "thank" , "thanks" , "thanx" , "that" , "that's" , "thats" , "the" , "their" , "theirs" , "them" , "themselves" , "then" , "thence" , "there" , "there's" , "thereafter" , "thereby" , "therefore" , "therein" , "theres" , "thereupon" , "these" , "they" , "they'd" , "they'll" , "they're" , "they've" , "think" , "third" , "this" , "thorough" , "thoroughly" , "those" , "though" , "three" , "through" , "throughout" , "thru" , "thus" , "to" , "together" , "too" , "took" , "toward" , "towards" , "tried" , "tries" , "truly" , "try" , "trying" , "twice" , "two" , "un" , "under" , "unfortunately" , "unless" , "unlikely" , "until" , "unto" , "up" , "upon" , "us" , "use" , "used" , "useful" , "uses" , "using" , "usually" , "value" , "various" , "very" , "via" , "viz" , "vs" , "want" , "wants" , "was" , "wasn't" , "way" , "we" , "we'd" , "we'll" , "we're" , "we've" , "welcome" , "well" , "went" , "were" , "weren't" , "what" , "what's" , "whatever" , "when" , "whence" , "whenever" , "where" , "where's" , "whereafter" , "whereas" , "whereby" , "wherein" , "whereupon" , "wherever" , "whether" , "which" , "while" , "whither" , "who" , "who's" , "whoever" , "whole" , "whom" , "whose" , "why" , "will" , "willing" , "wish" , "with" , "within" , "without" , "won't" , "wonder" , "would" , "wouldn't" , "yes" , "yet" , "you" , "you'd" , "you'll" , "you're" , "you've" , "your" , "yours" , "yourself" , "yourselves" , "zero"])
