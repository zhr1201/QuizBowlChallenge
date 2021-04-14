# Copyright 2021 UMD (Haoran Zhou)

# Feature based reranker utilizing 


from typing import Optional
from typing import Tuple, List
from abc import ABC
from abc import abstractmethod
from qanta.abs_reranker import AbsReranker
from qanta.abs_retriever import AbsRetriever
from qanta.dataset import QuizBowlDataset
import wikipedia
import logging
from tqdm import tqdm
import pickle


logger = logging.getLogger(__name__)
logger.setLevel(20)


class FeatureReranker(AbsReranker):
    '''
    The reranker uses wiki entities in that page to rescore the answers
    '''
    def __init__(self, weight: int):
        '''
        Args:
            weight: int, extra score if an anchor text in a wikipage is mentioned in the original question
        '''
        super().__init__()
        self.weight = weight
        self.wiki_page_dict = {}
    
    def train(
        self,
        path: str,
        retriever: AbsRetriever,
    ):
        '''
        download all the wikipage info in the training set
        Args:
            path: str, pkl model path
            retriever: AbsRetriever, not used here
        '''
        dataset = QuizBowlDataset(guesser_train=True)
        training_data = dataset.training_data()
        answers = training_data[1]
        logger.info("Start extracing wiki pages")

        for ans in tqdm(answers):
            try:
                self.wiki_page_dict[ans] = wikipedia.page(ans, auto_suggest=False)
            except:
                logger.warning("Fail to get " + ans)

        with open(path, 'wb') as f:
            pickle.dump({
                'wiki_page_dict': self.wiki_page_dict
            }, f)

    @classmethod
    def load(
        cls,
        path: str,
        weight: int,
    ) -> 'FeatureReranker':
        '''
        load the model, factory method
        Args:
            path: str, path for loading the model, not needed since no training is required
            weight: int, extra score if an anchor text in a wikipage is mentioned in the original question
        Returns:
            the loaded model
        '''
        reranker = FeatureReranker(weight)
        with open(path, 'rb') as f:
            params = pickle.load(f)
            reranker.wiki_page_dict = params['wiki_page_dict']
        return reranker
    
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
        for result in top_k:
            question_lower = question.lower()
            new_score = self._rescore_one_page(question_lower, result)
            if new_score is None:
                logger.warning("Skip reranking due to missing page info")
                return top_k
            ret.append()
        return ret

    def _rescore_one_page(
        self,
        question: str,
        page_info: Tuple[str, float],
    ) -> Tuple[str, float]:
        '''
        Args:
            question: str, original question
            page_info: Tuple[str, float], page info of [wiki_tag, retrieve score]
        Returns:
            the reranked results [wiki_tag, reranked points]
        ''' 
        
        logger.info("reranking " + page_info[0] + ' score ' + str(page_info[1]))
        mention_count = 0
        if page_info[0] in self.wiki_page_dict:
            page = self.wiki_page_dict[page_info[0]]
        else:
            logger.warning("Page" + page_info[0] + " not in the dictionary of the model")
            return None
        links = [ x.lower() for x in page.links ]
        anchor_list = set(links)  # possible duplicates
        for anchor in anchor_list:
            if anchor in question:
                logger.info("matched " + anchor)
                mention_count += 1
        
        score = page_info[1] + mention_count * self.weight
        logger.info("reranked " + str(score))
        return (page_info[0], score)
   