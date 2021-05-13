# Copyright 2021 UMD (Haoran Zhou)

# Reranker abstract class

from typing import Optional
from typing import Tuple, List
from abc import ABC
from abc import abstractmethod
from qanta.abs_retriever import AbsRetriever
from qanta.abs_reranker import AbsReranker
import spacy
from collections import defaultdict
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
WIKI_DUMP_URL = "https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/wikipedia/wiki_lookup.json"
logger = logging.getLogger(__name__)
logging.basicConfig(level = logging.INFO)


class SpacyNERQUESTAGReranker(AbsReranker):
    def __init__(self, weight: float):
        '''
        Args:
            weight: int, extra score if an anchor text in a wikipage is mentioned in the original question
        '''
        super().__init__()
        self.weight = weight
        #self.wiki_page_dict = {}
        self.answer_docs = {}
        #self.tokenizer = None
        #self.model = None
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
        #self.ner_tagger = spacy.load("en_core_web_trf")
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
        max_score = sum(y for x,y in top_k)
        for result in top_k:
            
            #question_lower = question.lower()
            
            #entities = []
            #entity = ""
            # for i,ner_tag in enumerate(ner_results):
            #     if ner_tag['entity'].startswith("B"):
            #         entities.append(entity)
            #         entity = ner_tag['word']
            #     else:
            #         entity = entity +" " + ner_tag["word"]
            # entities = entities[1:]
            #max_score = top_k[0][1]
            new_score = self._rescore_one_page(entities, result,max_score)
            if new_score is None:
                logger.warning("Skip reranking due to missing page info")
                return top_k
            ret.append(new_score)
        # ret.sort(key = lambda x: x[1])
        # ret.reverse()
        # if (ret[0][0] != top_k[0][0]):
        #     logger.info("Reranker changed the best guess from %s to %s" % (top_k[0][0], ret[0][0]))
        #     logger.info("Original question: %s" % question)
        return ret

    def _rescore_one_page(
        self,
        entities: List[str],
        page_info: Tuple[str, float],
        max_score: float
    ) -> Tuple[str, float]:
        '''
        Args:
            question: str, original question
            page_info: Tuple[str, float], page info of [wiki_tag, retrieve score]
        Returns:
            the reranked results [wiki_tag, reranked points]
        '''
        #logger.info("reranking " + page_info[0] + ' score ' + str(page_info[1]))
        mention_count = 0
        if page_info[0] in self.answer_docs:
            page = self.answer_docs[page_info[0]]['ner']
        else:
            logger.warning("Page" + page_info[0] + " not in the dictionary of the model")
            return None
        
        
        if len(entities) == 0:
            score = page_info[1]*(1-self.weight)
        else:
            for entity in entities:
                if entity in page:
                    #logger.info("matched " + entity.text)
                    mention_count +=1
            score = page_info[1]*(1-self.weight)/max_score + mention_count/len(entities) * self.weight
        
        return (page_info[0],score)
        # try:
        #     links = [ x.lower() for x in page.links ]
        # except:
        #     return None
        # anchor_list = set(entities)  # possible duplicates
        # for anchor in anchor_list:
        #     if anchor in page.text:
        #         logger.info("matched " + anchor)
        #         mention_count += 1
        
        # score = page_info[1] + mention_count * self.weight
        # logger.info("reranked " + str(score))
        # return (page_info[0], score)

    def train(
        self,
        path: str,
        retriever: AbsRetriever,
        #wiki_dump: str,
    ):
        '''
        download all the wikipage info in the training set
        Args:
            path: str, pkl model path
            retriever: AbsRetriever, not used here
            wiki_dump: str, the wiki dump json file path at the time those page lables are created
        '''

        # if not os.path.isfile(wiki_dump):
        #     logger.warning("Wiki dump doesn't exit, download a new one")
        #     urllib.request.urlretrieve(WIKI_DUMP_URL, wiki_dump)
        # with open(wiki_dump, 'r') as f:
        #     old_wiki_dict = json.load(f)
        
        dataset = QuizBowlDataset(guesser_train=True)
        training_data = dataset.training_data()
        questions = training_data[0]
        answers = training_data[1]
        answer_docs = defaultdict(dict)
        for q, ans in zip(questions, answers):
            text = ' '.join(q)
            try:
                answer_docs[ans]['text'] += ' ' + text
            except:
                answer_docs[ans]['text'] = ' ' + text
        #self.answer_docs = answer_docs
        for ans in answer_docs:
            entities = self.ner_tagger(answer_docs[ans]['text']).ents
            answer_docs[ans]['ner'] = []
            for ent in entities:
                answer_docs[ans]['ner'].append([ent.text.lower(),ent.label_])
        self.answer_docs = answer_docs        
        # logger.info("Start extracing wiki pages")

        # # get it from the wikipedia API since it has anchor text information
        # # only using page search will have disambuigation issues since page names are changed

        # for ans in tqdm(np.unique(answers).tolist()):
        #     try:
        #         self.wiki_page_dict[ans] = old_wiki_dict[ans]['text']#wikipedia.page(pageid=wiki_pageid)
        #     except:
        #         logger.warning("Fail to get wikipage %s using the old wikidump " % ans)
        #         try:
        #             logger.warning("Using direct pageid search %s "  % ans)
        #             wiki_pageid = old_wiki_dict[ans]['id']
        #             self.wiki_page_dict[ans] = wikipedia.page(pageid=wiki_pageid).content #
        #         except:
        #             logger.warning("Using direct page search%s "  % ans)
        #             try:
        #                 self.wiki_page_dict[ans] = wikipedia.page(ans, auto_suggest=False).content
        #             except:
        #                 logger.warning("Using direct page search without space%s "  % ans)
        #                 try:
        #                     self.wiki_page_dict[ans] = wikipedia.page(re.sub(r'\s+','',ans), auto_suggest=False).content
        #                 except:
        #                     logger.warning("Fail to get " + ans)

        with open(path, 'wb') as f:
            pickle.dump({
                'answer_docs': self.answer_docs
            }, f)

    @classmethod
    def load(
        cls,
        path: str,
        weight: int,
    ) -> 'SpacyNERQUESTAGReranker':
        '''
        load the model, factory method
        Args:
            path: str, path for loading the model, not needed since no training is required
            weight: int, extra score if an anchor text in a wikipage is mentioned in the original question
        Returns:
            the loaded model
        '''
        reranker = SpacyNERQUESTAGReranker(weight)
        with open(path, 'rb') as f:
            params = pickle.load(f)
            reranker.answer_docs= params['answer_docs']
        return reranker
