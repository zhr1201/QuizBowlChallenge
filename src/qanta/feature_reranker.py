# Copyright 2021 UMD (Haoran Zhou)

# Feature based reranker utilizing 


from typing import Optional
from typing import Tuple, List
from abc import ABC
from abc import abstractmethod
from qanta.abs_reranker import AbsReranker
from qanta.abs_retriever import AbsRetriever
import wikipedia
import logging


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
    
    def train(
        self,
        path: str,
        retriever: AbsRetriever,
        **args,
    ):
        '''
        doesn't require training
        '''
        pass

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
            ret.append(self._rescore_one_page(question_lower, result))
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
        
        logging.warning("reranking " + page_info[0] + ' score ' + str(page_info[1]))
        mention_count = 0
        page = wikipedia.page(page_info[0], auto_suggest=False)
        # page = self._try_get_page0(page_info[0])
        links = [ x.lower() for x in page.links ]
        anchor_list = set(links)  # possible duplicates
        for anchor in anchor_list:
            if anchor in question:
                logging.warning(anchor)
                mention_count += 1
        
        score = page_info[1] + mention_count * self.weight
        logging.warning("reranked " + str(score))
        return (page_info[0], score)
    
    # def _try_get_page0(self, page_name: str) -> wikipedia.wikipedia.WikipediaPage:
    #     '''
    #     Wrapper for trying to get the wiki page from it's nametag
    #     may not working perfectly
    #     Args:
    #         page_name: str, input page name
    #     Return:
    #         the wiki page with page_name
    #     '''
    #     try:
    #         page = wikipedia.page(page_name)
    #     except:
    #         try:
    #             page = self._try_get_page1(page_name)
    #         except:
    #             raise RuntimeError("Fail to get wikipage with tag " + page_name)
    #     return page

    # def _try_get_page1(self, page_name: str) -> str:
    #     '''
    #     search with '.' removed
    #     Args:
    #         page_name: str, input page name
    #     Return:
    #         the wiki page with page_name
    #     '''
    #     page_name = page_name.replace('.', '')
    #     try:
    #         page = wikipedia.page(page_name)
    #     except:
    #         page = self._try_get_page2(page_name)
    #     return page

    # def _try_get_page2(self, page_name: str) -> str:
    #     '''
    #     search with '_' removed
    #     Args:
    #         page_name: str, input page name
    #     Return:
    #         the wiki page with page_name
    #     '''
    #     page_name = page_name.replace('_', '')
    #     page = wikipedia.page(page_name)
    #     return page
