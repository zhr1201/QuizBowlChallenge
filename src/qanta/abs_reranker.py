# Copyright 2021 UMD (Haoran Zhou)

# Reranker abstract class

from typing import Optional
from typing import Tuple, List
from abc import ABC
from abc import abstractmethod
from qanta.abs_retriever import AbsRetriever


class AbsReranker(ABC):
    @abstractmethod
    def rerank(
        self,
        top_k_list: List[List[Tuple[str, float]]],
    ) -> List[List[Tuple[str, float]]]:
        '''
        Args:
            questions: List[List[Tuple[str, float]]], a list of input top k result of [wiki_tag, retrieve score]
        Returns:
            the reranked results [wiki_tag, reranked points]
        '''

        raise NotImplementedError

    @abstractmethod
    def train(
        self,
        path: str,
        retriever: AbsRetriever,
        **args,
    ):
        '''
        train the model
        Args:
            path: str, path for saving the model
            retriever: AbsRetriever, a trained retriever model for generating the top k result for the reranker
        '''
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(
        cls,
        path: str,
        **args,
    ) -> 'AbsReranker':
        '''
        load the model, factory method
        Args:
            path: str, path for loading the model
        Returns:
            the loaded model
        '''
        raise NotImplementedError
