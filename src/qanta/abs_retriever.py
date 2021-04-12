# Copyright 2021 UMD (Haoran Zhou)

# Retriever abstract class

from typing import Optional
from typing import Tuple, List
from abc import ABC
from abc import abstractmethod


class AbsRetriever(ABC):
    @property
    @abstractmethod
    def output_size(self) -> int:
        '''
        Returns: 
            the number of k in k top retrieved results
        '''
        raise NotImplementedError

    @abstractmethod
    def retrieve(
        self,
        questions: List[str],
    ) -> List[List[Tuple[str, float]]]:
        '''
        Args:
            questions: List[str], a list of input questions
        Returns:
            retrieved titles of the wiki passages
        '''
        raise NotImplementedError

    @abstractmethod
    def train(
        self,
        path: str,
        **args,
    ):
        '''
        train the model
        Args:
            path: str, path for saving the model
        '''
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(
        cls,
        path: str,
        **args,
    ) -> 'AbsRetriever':
        '''
        load the model
        Args:
            path: str, path for loading the model
        Returns:
            the loaded model
        '''
        raise NotImplementedError
