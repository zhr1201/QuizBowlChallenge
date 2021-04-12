from typing import List, Optional, Tuple
from collections import defaultdict
import pickle
import json
from os import path

from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, jsonify, request

from qanta.dataset import QuizBowlDataset
from qanta.abs_retriever import AbsRetriever

from typing import Optional
from typing import Tuple, List


class TfidfRetriever(AbsRetriever):
    def __init__(self, k: int):
        '''
        Args:
            k: int, for retrieve the top k results
        '''
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.i_to_ans = None
        self.k = k
    
    @property
    def output_size(self) -> int:
        return self.k

    def train(self, path: str) -> None:
        '''
        train the model
        Args:
            path: str, pkl model path
        '''
        dataset = QuizBowlDataset(guesser_train=True)
        training_data = dataset.training_data()

        questions = training_data[0]
        answers = training_data[1]
        answer_docs = defaultdict(str)
        for q, ans in zip(questions, answers):
            text = ' '.join(q)
            answer_docs[ans] += ' ' + text

        x_array = []
        y_array = []
        for ans, doc in answer_docs.items():
            x_array.append(doc)
            y_array.append(ans)

        self.i_to_ans = {i: ans for i, ans in enumerate(y_array)}
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 3), min_df=2, max_df=.9
        ).fit(x_array)
        self.tfidf_matrix = self.tfidf_vectorizer.transform(x_array)
        self._save(path)

    def retrieve(self, questions: List[str]) -> List[List[Tuple[str, float]]]:
        '''
        Args:
            questions: List[str], a list of input questions
        Returns:
            retrieved titles of the wiki passages
        '''       
        representations = self.tfidf_vectorizer.transform(questions)
        guess_matrix = self.tfidf_matrix.dot(representations.T).T
        guess_indices = (-guess_matrix).toarray().argsort(axis=1)[:, 0:self.k]
        guesses = []
        for i in range(len(questions)):
            idxs = guess_indices[i]
            guesses.append([(self.i_to_ans[j], guess_matrix[i, j]) for j in idxs])

        return guesses

    @classmethod
    def load(cls, path: str, k: int) -> 'TfidfRetriever':
        '''
        Args:
            path: str, path to a saved model
            k: int, number of retrieved results
        Returns:
            retrieved titles of the wiki passages
        '''  
        with open(path, 'rb') as f:
            params = pickle.load(f)
            retriever = TfidfRetriever(k)
            retriever.tfidf_vectorizer = params['tfidf_vectorizer']
            retriever.tfidf_matrix = params['tfidf_matrix']
            retriever.i_to_ans = params['i_to_ans']
            return retriever
    
    def _save(self, model_path: str):
        '''
        private helper for saving the model
        Args:
            model_path: str, pkl model path
        '''
        with open(model_path, 'wb') as f:
            pickle.dump({
                'i_to_ans': self.i_to_ans,
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'tfidf_matrix': self.tfidf_matrix
            }, f)
