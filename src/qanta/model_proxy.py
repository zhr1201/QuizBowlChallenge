# Copyright 2021 UMD (Haoran Zhou)

# A a model proxy class for training and evaluation and etc. 


from abc import ABC
import click

from typing import Optional
from typing import Tuple, List


from qanta.guesser import Guesser
from qanta.abs_reranker import AbsReranker
from qanta.abs_retriever import AbsRetriever
from qanta.tfidf_retriever import TfidfRetriever
from qanta.bm25_retriever import BM25Retriever
from qanta.bm25_Bags_of_words_retriever import BM25BoWRetriever
from qanta.feature_reranker import FeatureReranker
from qanta.heiarchical_attention_reranker import HeiarchicalAttentionReranker
import yaml


'''
Bad code structure for the buzzer that is not open to extension. But we won't
extend the buzzer logic in this project so just leave it for now (Haoran)
'''
BUZZ_NUM_GUESSES = 10
BUZZ_THRESHOLD = 0.3

def guess_and_buzz(model, question_text) -> Tuple[str, bool]:
    guesses = model.guess([question_text], BUZZ_NUM_GUESSES)[0]
    scores = [guess[1] for guess in guesses]
    buzz = scores[0] / sum(scores) >= BUZZ_THRESHOLD
    return guesses[0][0], buzz


def batch_guess_and_buzz(model, questions) -> List[Tuple[str, bool]]:
    question_guesses = model.guess(questions, BUZZ_NUM_GUESSES)
    outputs = []
    for guesses in question_guesses:
        scores = [guess[1] for guess in guesses]
        buzz = scores[0] / sum(scores) >= BUZZ_THRESHOLD
        outputs.append((guesses[0][0], buzz))
    return outputs

'''
for dynamically loading retriever and reranker classes
add your class to this dictionary for extending more retriever and reranker
'''

RETRIEVER_CHOICES = {'TFIDF': TfidfRetriever, 'BM25': BM25Retriever, 'BM25_BoW': BM25BoWRetriever}
RERANKER_CHOICES = {'FeatureReranker': FeatureReranker, "HAR": HeiarchicalAttentionReranker}


class ModelProxy(ABC):
    '''
    ModelProxy class shouldn't be instantiated, only class method available for training, 
    loading the model
    ''' 

    def __init__(self):
        raise RuntimeError("This class can't be instantiated.")

    @classmethod
    def load(cls, config_file: str) -> Guesser:
        ''' 
        load models for evaluation
        Args:
            config_file: str, yaml file for model config
        '''
        args = cls._load_yaml(config_file)
        retriever = cls._build_retriever(is_load=True, **args)
        if 'reranker' not in args or args['reranker'] is None:
            reranker = None
        else:
            reranker = cls._build_reranker(is_load=True, **args)
        return Guesser(retriever, reranker)

    @classmethod
    def train_retriever(cls, config_file: str):
        ''' 
        train a retriever
        Args:
            config_file: str, yaml file for model config
        '''
        args = cls._load_yaml(config_file)
        retriever = cls._build_retriever(is_load=False, **args)
        path = args['retriever_path']
        if 'retriever_train_args' in args and args['retriever_train_args'] is not None:
            train_args = args['retriever_train_args']
        else:
            train_args = {}
        retriever.train(path=path, **train_args)
    
    @classmethod
    def train_reranker(cls, config_file: str):
        '''
        train a reranker
        Args:
            config_file: str, yaml file for model config
        '''
        args = cls._load_yaml(config_file)
        retriever = cls._build_retriever(is_load=True, **args)
        reranker = cls._build_reranker(is_load=False, **args)
        path = args['reranker_path']
        if 'reranker_train_args' in args and args['reranker_train_args'] is not None:
            train_args = args['reranker_train_args']
        else:
            train_args = {}
        reranker.train(path=path, retriever=retriever, **train_args)
    
    @classmethod
    def _build_retriever(cls, is_load: bool, **args) -> AbsRetriever:
        '''
        private helper for building a retriever
        Args:
            is_load: bool, True if load from file 
            otherwise using constructor to create a new model
        '''
        retriever_class = RETRIEVER_CHOICES[args['retriever']]
        if 'retriever_conf' in args and args['retriever_conf'] is not None:
            retriever_args = args['retriever_conf']
        else:
            retriever_args = {}
        if is_load:
            path = args['retriever_path']
            return retriever_class.load(path=path, **retriever_args)
        else:
            return retriever_class(**retriever_args)
    
    @classmethod
    def _build_reranker(cls, is_load: bool, **args) -> AbsReranker:
        '''
        private helper for building a reranker
        Args:
            is_load: bool, True if load from file 
            otherwise using constructor to create a new model
        Code Smell: duplication of load_retriever (Haoran)
        '''
        reranker_class = RERANKER_CHOICES[args['reranker']]
        if 'reranker_conf' in args and args['reranker_conf'] is not None:
            reranker_args = args['reranker_conf']
        else:
            reranker_args = {}
        if is_load:
            path = args['reranker_path']
            return reranker_class.load(path=path, **reranker_args)
        else:
            return reranker_class(**reranker_args)

    @classmethod
    def _load_yaml(cls, config_file: str) -> dict:
        '''
        Args:
            config_file: str, yaml file for model config
        '''
        args = None
        with open(config_file, 'r') as stream:
            args = yaml.safe_load(stream)
        return args
