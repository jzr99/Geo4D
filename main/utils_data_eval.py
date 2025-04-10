from functools import partial
import numpy as np

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

import os, sys
from lvdm.data.base import Txt2ImgIterableBaseDataset
from utils.utils import instantiate_from_config
from lvdm.data.batched_sampler import BatchedRandomSampler


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False, train_img=None,
                 test_max_n_samples=None, multi_resolution=False, multi_task=False):
        super().__init__()
        if multi_resolution or multi_task:
            self.local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('SLURM_LOCALID')))
            self.global_rank = int(os.environ.get('RANK', os.environ.get('SLURM_PROCID')))
            self.num_rank = int(os.environ.get('WORLD_SIZE', os.environ.get('SLURM_NTASKS')))
        else:
            self.local_rank = 0
            self.global_rank = 0
            self.num_rank = 1
        print("****************************************Init datmodule *************************************")
        print(f"local_rank: {self.local_rank}, global_rank: {self.global_rank}, num_rank: {self.num_rank}")

        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader

        self.multi_resolution = multi_resolution
        self.multi_task = multi_task
        print(f"multi_resolution_sampler: {self.multi_resolution}")
        self.img_loader = None
        self.wrap = wrap
        self.test_max_n_samples = test_max_n_samples
        self.collate_fn = None

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.datasets = dict((k, instantiate_from_config(self.dataset_configs[k])) for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        if self.multi_resolution:
            if self.multi_task:
                pool_size = len(self.datasets["train"]._resolutions) * len(self.datasets["train"]._tasks)
            else:
                pool_size = len(self.datasets["train"]._resolutions)
            # __init__(self, dataset, batch_size, pool_size, world_size=1, rank=0, drop_last=True)
            batchsamplers = BatchedRandomSampler(self.datasets["train"], self.batch_size, pool_size, world_size=self.num_rank, rank=self.global_rank, drop_last=True)
        loader = DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False if is_iterable_dataset or self.multi_resolution else True,
                          worker_init_fn=init_fn, collate_fn=self.collate_fn, sampler=batchsamplers if self.multi_resolution else None
                          )
        return loader

    def _val_dataloader(self, shuffle=False):
        if isinstance(self.datasets['validation'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=init_fn,
                          shuffle=shuffle, 
                          collate_fn=self.collate_fn,
                          )

    def _test_dataloader(self, shuffle=False):
        try:
            is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
            pool_size = len(self.datasets["train"]._resolutions)
            # __init__(self, dataset, batch_size, pool_size, world_size=1, rank=0, drop_last=True)
            batchsamplers = BatchedRandomSampler(self.datasets["train"], self.batch_size, pool_size, world_size=self.num_rank, rank=self.global_rank, drop_last=True)
        except:
            is_iterable_dataset = isinstance(self.datasets['test'], Txt2ImgIterableBaseDataset)

        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not (is_iterable_dataset or self.multi_resolution))
        if self.test_max_n_samples is not None:
            dataset = torch.utils.data.Subset(self.datasets["test"], list(range(self.test_max_n_samples)))
        else:
            dataset = self.datasets["test"]
        return DataLoader(dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle,
                          collate_fn=self.collate_fn,
                          sampler=batchsamplers if self.multi_resolution else None
                          )

    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets['predict'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn,
                          collate_fn=self.collate_fn,
                          )
