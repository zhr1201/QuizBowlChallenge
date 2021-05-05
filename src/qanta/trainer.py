# Copyright 2021 UMD (Haoran Zhou)

# Trainer class that takes care of the torch model training


from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import torch
import torch.nn
import torch.optim
import logging

import time
from qanta.torch_utils import to_device
import sys


logger = logging.getLogger(__name__)
logging.basicConfig(level = logging.INFO)


class Trainer:
    '''
    A trainer wrapper for training pytorch models
    TODO: resume not implemented
    '''
    def __init__(self):
        raise RuntimeError("This class can't be instantiated.")

    @classmethod
    def run(
        cls,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_iterator: Iterable[Dict[str, torch.Tensor]],
        valid_iterator: Iterable[Dict[str, torch.Tensor]],
        n_gpu: int,
        output_dir: str,
        max_epoch: int,
    ) -> None:
        '''
        perform training
        Args:
            model: torch.nn.Module, the model for training
            optimizer: torch.optim.Optimizer, optimizer used for training
            train_iterator: Iterable[Dict[str, torch.Tensor]], Iterable object for getting the training data
            valid_iterator: Iterable[Dict[str, torch.Tensor]], Iterable object for getting the validation data
            n_gpu: int, number of gpus for training, 0 if only cpu is available
            output_dir: str, output directory for trianing
            max_epoch: int, max num of epochs for training
        '''
        start_epoch = 1
        start_time = time.perf_counter()
        logging.info("Training started!")
        best_loss = sys.float_info.max
        for iepoch in range(start_epoch, max_epoch + 1):
            cls.train_one_epoch(model, optimizer, iterator, n_gpu)
            cur_loss = cls.validate_one_epoch(model, iterator, n_gpu)
            if cur_loss < best_loss:
                best_loss = cur_loss
                torch.save(
                    {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    },
                    output_dir,
                )
    
    @classmethod
    def train_one_epoch(
        cls,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        iterator: Iterable[Dict[str, torch.Tensor]],
        n_gpu: int,
    ) -> None:
        '''
        train one epoch
        Args:
            model: torch.nn.Module, the model for training
            optimizer: torch.optim.Optimizer, optimizer used for training
            iterator: Iterable[Dict[str, torch.Tensor]], Iterable object for getting the training data
            n_gpu: int, number of gpus for training, 0 if only cpu is available
        '''
        start_time = time.perf_counter()
        model.train()
        for iiter, batch in enumerate(iterator):
            batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")
            retval = model(**batch)
            loss = retval["loss"]
            stats = retval["stats"]
            weight = retval["weight"]
            stats_str = ""
            for k, v in stats.items():
                stats_str += k + ": " + v

            logging.info('%n th epoch: ' +  stats_str)
            loss.backward()

    @classmethod
    @torch.no_grad()
    def validate_one_epoch(
        cls,
        model: torch.nn.Module,
        iterator: Iterable[Dict[str, torch.Tensor]],
        n_gpu, int,
    ) -> float:
        '''
        validate one epoch
        Args:
            model: torch.nn.Module, the model for training
            iterator: Iterable[Dict[str, torch.Tensor]], Iterable object for getting the validation data
            n_gpu: int, number of gpus for training, 0 if only cpu is available
            output_dir: str, output directory for trianing
        Returns:
            loss of type float
        '''
        model.eval()
        loss_tot = 0
        batch_count = 0
        for (_, batch) in iterator:
            batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")
            retval = model(**batch)
            stats_str = ""
            for k, v in stats.items():
                stats_str += k + ": " + v
            loss_tot += stats['loss']
            batch_count += 1
        validation_loss = loss_tot / batch_count
        logging.info('%n th epoch validation loss: ' +  validation_loss)
        return validation_loss