# Copyright 2021 UMD (Haoran Zhou)

# Reranker abstract class

from typing import Optional
from typing import Tuple, List
from abc import ABC
from abc import abstractmethod
from qanta.abs_retriever import AbsRetriever
from qanta.abs_reranker import AbsReranker
import spacy

import pickle
import wikipedia
import os.path
import urllib.request
from qanta.dataset import QuizBowlDataset
from tqdm import tqdm
import logging
import json
import numpy as np
import re
from qanta.tfidf_retriever import TfidfRetriever
from qanta.bm25_retriever import BM25Retriever
from collections import defaultdict
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
WIKI_DUMP_URL = "https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/wikipedia/wiki_lookup.json"
logger = logging.getLogger(__name__)
logging.basicConfig(level = logging.INFO)


class SpacyNERNNV2Reranker(AbsReranker):
    def __init__(self,):
        '''
        Args:
            weight: int, extra score if an anchor text in a wikipage is mentioned in the original question
        '''
        super().__init__()
        #self.weight = weight
        self.wiki_page_dict = {}
        #self.tokenizer = None
        self.classifier = None
        self.ner_tagger = None
        self.retriever = None
        self.tags_to_id  = None
        self.scaler = None
        self.ner_tagger = spacy.load("en_core_web_trf")

    def rerank(
        self,
        questions: List[str],
        top_k_list: List[List[Tuple[str, float]]],
    ) -> List[List[Tuple[str, float]]]:
        '''
        Args:
            questions: List[str], a list of input questions
            top_k_list: List[List[Tuple[str, float]]], a list of input top k result of [wiki_tag, retrieve score]
        Returns:
            the reranked results [wiki_tag, reranked points]
        '''
        #self.tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        #self.model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
        
        
        ret = []
        for i in range(len(questions)):
            question = questions[i]
            top_k = top_k_list[i]
            ret.append(self._rerank_one_question(question, top_k))
        return ret
    
    def _rerank_one_question(
        self,
        question: str,
        top_k: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        '''
        Args:
            question: str, original question
            top_k: List[Tuple[str, float]], a list of input top k result of [wiki_tag, retrieve score]
        Returns:
            the reranked results [wiki_tag, reranked points]
        '''

        ret = []
        ner_labeled = self.ner_tagger(question).ents
        entities = [[n.text,n.label_] for n in ner_labeled]
        entities = np.unique(entities,axis=0).tolist()
        for result in top_k:
            new_score = self._rescore_one_page(entities, result)
            # if new_score is None:
            #     logger.warning("Skip reranking due to missing page info")
            #     return top_k
            ret.append(new_score)
        return ret

    def _rescore_one_page(
        self,
        entities: List[List[str]],
        page_info: Tuple[str, float],
    ) -> Tuple[str, float]:
        '''
        Args:
            question: str, original question
            page_info: Tuple[str, float], page info of [wiki_tag, retrieve score]
        Returns:
            the reranked results [wiki_tag, reranked points]
        '''
        #logger.info("reranking " + page_info[0] + ' score ' + str(page_info[1]))
        
        if page_info[0] in self.wiki_page_dict:
            page = self.wiki_page_dict[page_info[0]]
        else:
            page =''
            logger.warning("Page" + page_info[0] + " not in the dictionary of the model")
            #return None
        
        x = np.zeros(len(self.tags_to_id)+1)
        x[-1] = page_info[1]
        for entity in entities:
            x[self.tags_to_id[entity[1]]] += page.count(entity[0])
            
                #logger.info("matched " + entity.text)
        #x = self.scaler.transform(x.reshape(1,-1))
        #x = self.classifier.transform(x.reshape(1,-1))           
        score = self.classifier.decision_function(x.reshape(1,-1))[0]
        #score = self.classifier.decision_function(x.reshape(1,-1))[1]
            #score = page_info[1]*(1-self.weight) + mention_count/len(entities) * self.weight
        
        return (page_info[0],score)

    def train(
        self,
        path: str,
        retriever: AbsRetriever,
        wiki_dump: str,
        retriever_path: str,
        k: int,
        ner_train_dump: str,
    ):
        '''
        download all the wikipage info in the training set
        Args:
            path: str, pkl model path
            retriever: AbsRetriever, not used here
            wiki_dump: str, the wiki dump json file path at the time those page lables are created
        '''

        if not os.path.isfile(wiki_dump):
            logger.warning("Wiki dump doesn't exit, download a new one")
            urllib.request.urlretrieve(WIKI_DUMP_URL, wiki_dump)
        with open(wiki_dump, 'r') as f:
            old_wiki_dict = json.load(f)
        
        dataset = QuizBowlDataset(guesser_train=True)
        training_data = dataset.training_data()
        answers = training_data[1]
        logger.info("Start extracing wiki pages")

        # get it from the wikipedia API since it has anchor text information
        # only using page search will have disambuigation issues since page names are changed

        for ans in tqdm(np.unique(answers).tolist()):
            try:
                self.wiki_page_dict[ans] = old_wiki_dict[ans]['text']#wikipedia.page(pageid=wiki_pageid)
            except:
                logger.warning("Fail to get wikipage %s using the old wikidump " % ans)
                try:
                    logger.warning("Using direct pageid search %s "  % ans)
                    wiki_pageid = old_wiki_dict[ans]['id']
                    self.wiki_page_dict[ans] = wikipedia.page(pageid=wiki_pageid).content #
                except:
                    logger.warning("Using direct page search%s "  % ans)
                    try:
                        self.wiki_page_dict[ans] = wikipedia.page(ans, auto_suggest=False).content
                    except:
                        logger.warning("Using direct page search without space%s "  % ans)
                        try:
                            self.wiki_page_dict[ans] = wikipedia.page(re.sub(r'\s+','',ans), auto_suggest=False).content
                        except:
                            logger.warning("Fail to get " + ans)
        
        self.ner_tagger = spacy.load("en_core_web_trf")
        labels = self.ner_tagger.pipe_labels['ner']
        tags_to_id = {labels[i]:i for i in range(len(labels))}
        self.tags_to_id = tags_to_id
        try:
            data_all = pickle.load(open(ner_train_dump,"rb"))
        except:
            self.retriever = BM25Retriever(k).load(retriever_path,k)
            #dataset = QuizBowlDataset(guesser_train=True)
            ## load questions
            #training_data = dataset.training_data()
            questions = training_data[0]
            #answers = training_data[1]
            answer_docs = defaultdict(str)
            for q, ans in zip(questions, answers):
                text = ' '.join(q)
                answer_docs[ans] += ' ' + text
            x_array = []
            y_array = []
            for ans, doc in answer_docs.items():
                x_array.append(doc)
                y_array.append(ans)
            ## end load questins

            questions = [" ".join(q) for q in questions]
            guess_pages_score = self.retriever.retrieve(questions)
            guess_pages= [[x[0] for x in y] for y in guess_pages_score]
            guess_score = [[x[1] for x in y] for y in guess_pages_score]
            
            #X_train = np.zeros((len(ques),len(labels)))

            data_all = []
            for i in range(len(questions)):
                top_k = guess_pages[i]
                ner_labeled = self.ner_tagger(questions[i]).ents
                ner_tags = [[n.text,n.label_] for n in ner_labeled]
                ner_tags = np.unique(ner_tags,axis=0).tolist()
                for j in range(len(top_k)):
                    x = np.zeros(len(labels)+2)
                    x[-2] = guess_score[i][j]
                    if top_k[j] in self.wiki_page_dict:
                        for ner in ner_tags:
                            #try:
                            x[tags_to_id[ner[1]]] += self.wiki_page_dict[top_k[j]].count(ner[0])
                    if answers[i] == top_k[j]:
                            x[-1] = 1
                    data_all.append(x)
            
            data_all = np.array(data_all)
            pickle.dump(data_all,open(ner_train_dump,"wb"))
        x_train = data_all[:,:-1]           
        y_train = data_all.T[-1]
        
        #scaler = preprocessing.StandardScaler().fit(x_train)
        #self.scaler = scaler
        #X_scaled = scaler.transform(x_train)
        clf = make_pipeline(StandardScaler(), SVC(gamma='auto',verbose=True,random_state=723))
        #clf = MLPClassifier(hidden_layer_sizes=(12,8),random_state=1, max_iter=300,verbose=True,alpha=1e-3)
        #clf = Perceptron(tol=1e-3, random_state=0,n_jobs=24,verbose=1)
        clf.fit(x_train, y_train)
        self.classifier = clf
        # pages_with_tag = []
        # for top_k in range(len(guess_pages)):
        #     for ans in guess_pages[top_k]:
        #         if ans in self.wiki_page_dict:
        #             if ans==answers[top_k]:
        #                 pages_with_tag.append([self.wiki_page_dict[ans],1])
        #             else:
        #                 pages_with_tag.append([self.wiki_page_dict[ans],0])
        
        
        with open(path, 'wb') as f:
            pickle.dump({
                'wiki_page_dict': self.wiki_page_dict,
                'classifier': self.classifier,
                'tags_to_id': self.tags_to_id,
                'scaler': self.scaler,
            }, f)

    @classmethod
    def load(
        cls,
        path: str,
    ) -> 'SpacyNERNNV2Reranker':
        '''
        load the model, factory method
        Args:
            path: str, path for loading the model, not needed since no training is required
            weight: int, extra score if an anchor text in a wikipage is mentioned in the original question
        Returns:
            the loaded model
        '''
        reranker = SpacyNERNNV2Reranker()
        with open(path, 'rb') as f:
            params = pickle.load(f)
            reranker.wiki_page_dict = params['wiki_page_dict']
            reranker.classifier = params['classifier']
            reranker.tags_to_id = params['tags_to_id']
            reranker.scaler = params['scaler']
        return reranker
