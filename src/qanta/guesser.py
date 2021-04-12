# Copyright 2021 UMD (Haoran Zhou)

# Guesser model consists of a retriever and a reranker

from typing import Optional
from typing import Tuple, List
from abc import ABC
from qanta.abs_retriever import AbsRetriever
from qanta.abs_reranker import AbsReranker


class Guesser(ABC):
    def __init__(
        self,
        retriever: AbsRetriever,
        reranker: Optional[AbsReranker] = None,  
    ):
        '''
        Args:
            retriever: AbsRetriever, retriever for retrieving k best results
            reranker: AbsReranker, reranker to rerank the result, if None, it won't rerank the results
        '''
        self.retriever = retriever
        self.reranker = reranker

    def guess(
        self,
        questions: List[str],
        max_n_guesses: int,
    ) -> List[List[Tuple[str, float]]]:
        '''
        Args:
            questions: List[str], a list of input questions (a batch)
            max_n_guesses: the max number of guesses
        Returns:
            max_n_guesses of titles of the wiki passages (a batch)
        '''
        top_k_list = self.retriever.retrieve(questions)
        if self.retriever.output_size > max_n_guesses:
            raise RuntimeError(
                "Can't produce max_n_guesses cause the retirever doesn't provide enough results")
        if self.reranker is not None:
            reranked_list = self.reranker.rerank(top_k_list)
        else:
            reranked_list = top_k_list
        
        guesses = []
        for i in range(len(questions)):
            rerank_qi = reranked_list[i]
            rerank_qi.sort(key = lambda x : x[1])
            rerank_qi.reverse()
            guesses.append(rerank_qi[:max_n_guesses])
        return guesses
