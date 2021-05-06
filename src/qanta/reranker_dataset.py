# Copyright 2021 UMD (Haoran Zhou)

# Pytorch Dataset Class for training a reranker
# TODO: Add batch sampler and collect_func to reorder the samples by length
#       for more efficient transformer training (Haoran)

from typing import Any
from typing import Callable
from typing import Collection
from typing import Dict
from typing import Mapping
from typing import Tuple
from typing import Union, List

from abc import ABC
from abc import abstractmethod
import logging
import numpy as np
import nltk
from tqdm import tqdm
import pickle

from torch.utils.data.dataset import Dataset
from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)
logging.basicConfig(level = logging.INFO)

MAX_SENT_N = 10


class HARDataset(Dataset):
    """Pytorch Dataset class for training heiarchical attention rerankers (HAR).
    """
    def __init__(
        self,
        data: List[Union[str, List[str], List[bool]]],
        sent_trans: SentenceTransformer,
    ):
        '''
        Args:
            data: List[str, List[str], List[bool]], List[question, List[wiki_passages], List[labels]]
            sent_trans: SentenceTransformer, pretrained sentence transformer for getting the sentence embedding

        '''
        super().__init__()
        self._get_splitter()
        self.data_list = []

        for q, sentences, labels in tqdm(data):
                
            q_last_sent = self.splitter.tokenize(q)[-1]
            try:
                q_emb = sent_trans.encode([q], show_progress_bar=False)
            except:
                logger.warning('Failed to process one page')
                continue
            q_emb = q_emb[0]
            for label, passage in zip(labels, sentences):                
                sentences = self.splitter.tokenize(passage)
                if len(sentences) > MAX_SENT_N:
                    sentences = sentences[0:MAX_SENT_N]
                sent_emb = sent_trans.encode(sentences, show_progress_bar=False)
                self.data_list.append(
                    {'question_emb': q_emb,
                     'sentence_emb': sent_emb,
                     'label': label})

    def __len__(self) -> int:
        '''
        required method for torch dataset
        Returns:
            len of the dataset
        '''
        return len(self.data_list)
        
    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        '''
        one way to implemnent a iterable
        Args:
            index: int, index for getting a item in the HARDataset
        Returns:
            one data item
        '''
        return self.data_list[index]

    def _get_splitter(self):
        '''
        create a self.splitter that can divide a paragraph into sentences
        '''
        nltk.download('punkt')
        self.splitter = nltk.data.load('tokenizers/punkt/english.pickle', verbose=False)


    
