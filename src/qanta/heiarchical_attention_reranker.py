# Copyright 2021 UMD (Haoran Zhou)

# Rescore each wikipage using neural model
# Use multiheadded attention layers to handle the embedding of paragraphs and questions got by BERT


from typing import Optional
from typing import Tuple, List, Dict, Union
from abc import ABC
from abc import abstractmethod
from qanta.abs_reranker import AbsReranker
from qanta.abs_retriever import AbsRetriever
from qanta.dataset import QuizBowlDataset
from qanta.trainer import Trainer
import wikipedia
import logging
from tqdm import tqdm
import pickle
import os.path
import urllib.request
import json
import torch
import time

from sentence_transformers import SentenceTransformer
from qanta.reranker_dataset import HARDataset
from qanta.transformer_encoder import TransformerEncoder
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch.optim

import numpy as np

import nltk
import nltk.data

from tqdm import tqdm


logger = logging.getLogger(__name__)
logging.basicConfig(level = logging.INFO)


THREASH_HOLD = 0.5


class TransformerEncoderClassifier(torch.nn.Module):
    '''
    Using transformer encoder to classify if the page is the answer to the question using Transformer encoder
    given the BERT representations of the question and sentences in the wiki page summary
    '''
    def __init__(
        self,
        input_size: int,
        attention_dim: int = 256,
        attention_heads: int = 8,
        num_blocks: int = 3):
        '''
        Args:
            input_size: int, input dim
            output_size: int, dim of attention
            attention_heads: int, number of attention heads
            num_blocks: int, number of attention layers
        '''
        super().__init__()
        self.transformer_enc = TransformerEncoder(
            input_size=input_size, output_size=attention_dim, attention_heads=attention_heads, num_blocks=num_blocks)
        self.lin = torch.nn.Linear(attention_dim, 1)
        self.act = torch.nn.Sigmoid()
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict:
        '''
        forward computation
        Args:
            xs_pad: torch.Tensor, BERT embedding of question and passage sequences(B, L, N)
            ilens: torch.Tensor, input length sqe
            labels: torch.Tensor, labels
        '''
        hs_pad, olens, _ = self.transformer_enc(xs_pad, ilens)
        hs_cls = hs_pad[:, 0, :]
        lin_out = self.lin(hs_cls)
        loss = self.criterion(lin_out, labels)
        act = self.act(lin_out)
        inference = act > THREASH_HOLD
        acc = (inference == labels).sum() / inference.shape[0]
        return {'loss': loss, 'inference': inference, 'acc': acc}

    @staticmethod
    def collate_fn(batch: List[Dict[str, np.array]]) -> Dict[str, np.array]:
        '''
        Collate function (call back function to be passed to DataLoader) 
        compatible with this model and RerankerDataset
        post processing for samples got from Dataset and BatchSampler
        Args:
            batch:  List[Dict[str, np.ndarray]], batch data
        Returns:
            training data compatible with the model and HARDataset
        '''
        xs = [np.concatenate((x['question_emb'].reshape([1, -1]), x['sentence_emb'])) for x in batch]
        ilen = [x.shape[0] for x in xs]
        labels = [x['label'] for x in batch]
        xs = [torch.from_numpy(x) for x in xs]
        xs_pad = pad_sequence(xs)
        xs_pad = xs_pad.permute([1, 0, 2])
        ilen = torch.tensor(ilen)
        labels = torch.tensor(labels)
        return {'xs_pad': xs_pad, 'ilens': ilen, 'labels': labels}


WIKI_DUMP_URL = "https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/wikipedia/wiki_lookup.json"
BERT_MODEL = 'paraphrase-distilroberta-base-v1'


class HeiarchicalAttentionReranker(AbsReranker):
    '''
    The reranker rescores the wikipage given a question using neural model.
    1. It gets the Bert representation of the questions and paragraphs in the wikipage
    2. It then feed those representations into a few multi-headed attention layers
    3. It then uses the output corresponding to the question to do a classification 
        (true: the wikipage is the answer, false: the wikipage is not the answer)
    '''
    def __init__(
        self,
        weight: int,
        input_size: int,
        attention_dim: int,
        attention_heads: int,
        num_blocks: int
    ):
        '''
        Args:
            weight: int, extra score if an anchor text in a wikipage is mentioned in the original question
        '''
        super().__init__()
        self.weight = weight
        self.wiki_page_dict = {}
        self.sent_embedder = SentenceTransformer('paraphrase-distilroberta-base-v1')
        self.model = TransformerEncoderClassifier(
            input_size, attention_dim,attention_heads, num_blocks).to('cuda:0')
    
    def train(
        self,
        path: str,
        retriever: AbsRetriever,
        wiki_dump: str,
        dict_path: str,
        batch_size: int, 
        n_epoch: int,
    ):
        '''
        download all the wikipage info in the training set
        and train the model
        Args:
            path: str, model path
            retriever: AbsRetriever, not used here
            wiki_dump: str, the wiki dump json file path at the time those page lables are created
            dict_path: str, path to the prepared wiki dict
            batch_size: int, training batch size
            n_epoch: int, number of epochs for training
        '''
        old_wiki_dict = self._prepare_wiki_dump(wiki_dump)
        self._prepare_wiki_dict(dict_path, old_wiki_dict)

        sample_list_train = self._prepare_question_list('train', retriever)

        TRAIN_DATA_DUMP = "train_data.dump"
        if not os.path.isfile(TRAIN_DATA_DUMP):
            dataset_train = HARDataset(sample_list_train, self.sent_embedder)
            with open(TRAIN_DATA_DUMP, 'wb') as f:
                pickle.dump({
                    'train_data_dump': dataset_train
                }, f)
        else:
            with open(TRAIN_DATA_DUMP, 'rb') as f:
                params = pickle.load(f)
                dataset_train = params['train_data_dump']

        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, collate_fn=self.model.collate_fn)
        
        sample_list_val = self._prepare_question_list('val', retriever)

        VAL_DATA_DUMP = "val_data.dump"
        if not os.path.isfile(VAL_DATA_DUMP):
            dataset_val = HARDataset(sample_list_val, self.sent_embedder)
            with open(VAL_DATA_DUMP, 'wb') as f:
                pickle.dump({
                    'val_data_dump': dataset_val
                }, f)
        else:
            with open(VAL_DATA_DUMP, 'rb') as f:
                params = pickle.load(f)
                dataset_val = params['val_data_dump']

        dataloader_val = DataLoader(dataset_val, batch_size=batch_size, collate_fn=self.model.collate_fn)
        
        optimizer = torch.optim.Adam(self.model.parameters())
        Trainer.run(self.model, optimizer, dataloader_train, dataloader_val, 1, path, n_epoch)

    @classmethod
    def load(
        cls,
        path: str,
    ) -> 'FeatureReranker':
        '''
        load the model, factory method
        Args:
            path: str, path for loading the model, not needed since no training is required
        Returns:
            the loaded model
        '''
        reranker = FeatureReranker(weight)
        reranker._load_wiki_dict(dict_path)
        reranker.model.load_state_dict(torch.load(path)['model'], path)
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
    
    def _load_wiki_dict(self, path: str):
        '''
        Args:
            path: str, path to the wiki dict
        '''
        with open(path, 'rb') as f:
            params = pickle.load(f)
            self.wiki_page_dict = params['wiki_page_dict']

    def _prepare_wiki_dump(self, wiki_dump: str) -> dict:
        '''
        Download the wiki dump if it hasn't been downloaded
        Args:
            wiki_dump: str, the path to the wiki dump
        Returns:
            a dictionary for the wiki dump
        '''
        if not os.path.isfile(wiki_dump):
            logger.warning("Wiki dump doesn't exit, download a new one")
            urllib.request.urlretrieve(WIKI_DUMP_URL, wiki_dump)
        with open(wiki_dump, 'r') as f:
            old_wiki_dict = json.load(f)
        return old_wiki_dict
    
    def _prepare_wiki_dict(self, path: str, old_wiki_dict: dict):
        '''
        Load the page info into a dict
        Args:
            path: str, the path to the wiki dict
            old_wiki_dict: dict, the loaded old wiki dump
        '''
        if os.path.isfile(path):
            logger.info("Wiki dict already exits, loading from file")
            self._load_wiki_dict(path)
            return
        logger.warning("Wiki dict not found, preparing the wiki dict, this might take a while")
        dataset = QuizBowlDataset(guesser_train=True)
        training_data = dataset.training_data()
        answers = training_data[1]
        logger.info("Start extracing wiki pages")

        # get it from the wikipedia API since it has anchor text information
        # only using page search will have disambuigation issues since page names are changed

        for ans in tqdm(answers):
            try:
                wiki_pageid = old_wiki_dict[ans]['id']
                self.wiki_page_dict[ans] = wikipedia.page(pageid=wiki_pageid)
            except:
                logger.warning("Fail to get wikipage %s using the id of the old wikidump " % ans)
                try:
                    logger.warning("Using direct page search %s "  % ans)
                    self.wiki_page_dict[ans] = wikipedia.page(ans, auto_suggest=False)
                except:
                    logger.warning("Fail to get " + ans)

        with open(path, 'wb') as f:
            pickle.dump({
                'wiki_page_dict': self.wiki_page_dict
            }, f)
    
    def _prepare_question_list(self, data_type: str, retriever: AbsRetriever) -> List[Union[str, List[str], List[bool]]]:
        '''
        Args:
            data_type: str, 'train' / 'val'.
            retriever: AbsRetriever
        Returns:
            the question list
        '''
        dump_path = "HAR" + data_type + ".dump"
        if os.path.isfile(dump_path):
            logger.info(dump_path + " already exits, loading from file")
            with open(dump_path, 'rb') as f:
                params = pickle.load(f)
                sample_list = params['data_dump']
            return sample_list
        dataset = QuizBowlDataset(guesser_train=True)
        
        if data_type == 'train':
            data = dataset.training_data_text()
        elif data_type == 'val':
            data = dataset.dev_data_text()
        sample_list = []
        questions = data[0]
        answers = data[1]
        # for q, ans in tqdm(zip(questions, answers)):
        #     retrieved_pages = retriever.retrieve([q])[0]
        #     passage_list = []
        #     label_list = []
        #     for page, score in retrieved_pages:
        #         try:
        #             page_item = self.wiki_page_dict[page]
        #             # Just use summary for now
        #             # time.sleep(0.1)
        #             summary = page_item.summary
        #             passage_list.append(summary)
        #         except:
        #             logger.warning("Skipping page" + page)
        #             continue
        #         label_list.append(ans == page)
        #     sample_list.append([q, passage_list, label_list])

        tot_len = len(questions)
        retrieved_pages = []
        idx = 0
        batch_size = 1000
        for idx in tqdm(range(0, tot_len, batch_size)):
            end_idx = min(idx + batch_size, tot_len)
            one_batch = retriever.retrieve(questions[idx:end_idx])
            retrieved_pages.extend(one_batch)
        
        for q, pages, ans in tqdm(zip(questions, retrieved_pages, answers)):
            passage_list = []
            label_list = []
            for page, score in pages:
                try:
                    page_item = self.wiki_page_dict[page]
                    # Just use summary for now
                    passage_list.append(page_item.summary)
                except:
                    logger.warning("Skipping page " + page)
                    continue
                label_list.append(ans == page)
            sample_list.append([q, passage_list, label_list])

        with open(dump_path, 'wb') as f:
            pickle.dump({
                'data_dump': sample_list
            }, f)
        return sample_list

