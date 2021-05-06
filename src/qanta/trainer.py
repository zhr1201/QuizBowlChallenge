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
            cls.train_one_epoch(model, optimizer, train_iterator, n_gpu)
            cur_loss = cls.validate_one_epoch(model, valid_iterator, n_gpu)
            logger.info("%d th epoch validation loss: %.2f" % (iepoch, cur_loss))
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
        i = 0
        for batch in iterator:
            i += 1   
            optimizer.zero_grad()
            batch = to_device(batch, "cuda" if n_gpu > 0 else "cpu")
            retval = model(**batch)
            loss = retval["loss"]

            logging.info(str(i) + ' batch: ' +  str(loss))
            loss.backward()
            optimizer.step()

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
            batch = to_device(batch, "cuda" if n_gpu > 0 else "cpu")
            retval = model(**batch)
            stats_str = ""
            for k, v in stats.items():
                stats_str += k + ": " + v
            loss_tot += stats['loss']
            batch_count += 1
        validation_loss = loss_tot / batch_count
        return validation_loss
