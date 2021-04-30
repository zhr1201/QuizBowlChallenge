from typing import List, Optional, Tuple
from collections import defaultdict
import pickle
import json
from os import path

import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
from nltk.tokenize import word_tokenize

from tqdm import tqdm
from flask import Flask, jsonify, request
from gensim import corpora
from gensim.summarization import bm25

from qanta.dataset import QuizBowlDataset
from qanta.abs_retriever import AbsRetriever

from typing import Optional
from typing import Tuple, List 


class BM25Retriever(AbsRetriever):
    def __init__(self, k: int):
        '''
        Args:
            k: int, for retrieve the top k results
        '''
        super().__init__()
        self.bm25_model = None
        self.i_to_ans = None
        self.dictionary = None
        self.k = k
        self.stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"])
    
    @property
    def output_size(self) -> int:
        return self.k

    def train(self, path: str) -> None:
        '''
        train the model
        Args:
            path: str, pkl model path
        '''

        print("*** preprocessing ***")
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
            tokens = word_tokenize(doc)
            # remove stop words
            x_array.append([t for t in tokens if not t in self.stop_words])
            y_array.append(ans)

        self.i_to_ans = {i: ans for i, ans in enumerate(y_array)}

        print("*** building dictionary ***")
        self.dictionary = corpora.Dictionary(x_array)

        print("*** building bag of words representations ***")
        docs_bag_of_words = [self.dictionary.doc2bow(x) for x in x_array]

        print("*** building bm25 model ***")
        self.bm25_model = bm25.BM25(docs_bag_of_words)

        
        print("*** running two test questions ***")
        questions = [""" The oldest document written in this language is a letter written in 1521 in the town of Câmpulung, while more recent poets writing in this language include Carmen Sylva and Anton Pann. This language uses five cases, though the genitive and dative cases are identical, as are the nominative and accusative. Tripthongs occur frequently in this language, as in "rusaoică," while interjections in this language include "mamă-mamă. " It is more closely related to Dalmatian than to Italian or Spanish, and this language includes the pronouns "noi," "voi," and "eu" ["AY-oo"] and favors labial consonants such as "b" and "m" over velars such as "g" and "k." For 10 points, name this tongue spoken by the members of O-Zone and Nicolae Ceauşescu, an Eastern Romance language spoken in Bucharest. This tongue has direct and the oblique cases, and, unlike its related languages, maintains a long o/short u distinction. The future tense in this language is invoked by compounding the verb meaning "to wish," "a vrea. " This language's verb for "to preserve," "a p?stra," is the source of its sole loanword into English, "pastrami. " Its endangered dialects include the Megleno- and Istro- versions. The most popular regional varieties of this language are the Aro- and Daco- forms. It is identical to a language known for nationalist reasons as (*) "Moldovan" and, due to its geographic distribution, exhibits a high degree of borrowed vocabulary from Slavic tongues. For 10 points, identify this easternmost Romance language, spoken by such figures as Ion Antonescu and Constantin Brancusi. """, """  The narrator of this novel is alarmed when a ghost\'s hand reaches into his room and starts bleeding from shattered glass. Isabella and Edgar Linton, heirs to Thrushcross Grange, marry the two protagonists of this novel. Narrated in parts by Nelly Dean and Mr. Lockwood, this novel\'s tragic couple is Catherine Earnshaw and Heathcliff. For 10 points, name this only novel by Emily Bronte. The narrator of this novel is thought by the servant Joseph to have stolen a lantern, and that narrator is consequently attacked by dogs set loose by Joseph. One character is adopted from an orphanage in Liverpool and throws applesauce at another character. Hindley, whose wife Frances dies after giving birth to Hareton, hates that orphan. Nelly Dean tells the story of the house to the narrator of this novel, Mr. Lockwood, who rents a room at Thrushcross Grange. Identify this novel centering on the romance between Heathcliff and Catherine, a work by Emily BrontÃ«. This book\'s final chapter chronicles how one of its protagonists starts to starve himself to death, during which he claims to have seen the edge of hell but is now near heaven. Another chapter of this novel describes how a premature baby named Catherine is buried near the corner of a church with the Lintons. One character in this novel is freed by the housekeeper Zillah after she is imprisoned for many days. The frame story of this novel is set up between Nelly\'s tale to Lockwood about Thrushcross Grange and the titular locale. For 10 points, name this novel about Heathcliff and his relationship to Catherineforrtl """]
        print("answers: ", self.retrieve(questions))

        print("*** saving bm25 retriever ***")
        self._save(path)
        

    def retrieve(self, questions: List[str]) -> List[List[Tuple[str, float]]]:
        '''
        Args:
            questions: List[str], a list of input questions
        Returns:
            retrieved titles of the wiki passages
        '''       
        x_array = []
        for question in questions:
            tokens = word_tokenize(question)
            # remove stop words
            x_array.append([t for t in tokens if not t in self.stop_words])

        questions_bag_of_words = [self.dictionary.doc2bow(x) for x in x_array]
        scores = [self.bm25_model.get_scores(BoW) for BoW in questions_bag_of_words]
        best_matches = [sorted(range(len(score)), key=lambda i: -score[i])[:self.k] for score in scores]
        
        guesses = []
        for i in range(len(questions)):
            best_match = best_matches[i]
            score = scores[i]
            ans = [(self.i_to_ans[bm], score[bm]) for bm in best_match]
            guesses.append(ans)

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
            retriever = BM25Retriever(k)
            retriever.bm25_model = params['bm25_model']
            retriever.i_to_ans = params['i_to_ans']
            retriever.dictionary = params['dictionary']
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
                'bm25_model': self.bm25_model,
                'dictionary': self.dictionary
            }, f)
