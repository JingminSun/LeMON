import os
import h5py
import math
import random
import numpy as np
import json
import torch
import copy
import torch.nn.functional as F
from torch.utils.data import Dataset
import io
from collections import defaultdict
from logging import getLogger
from omegaconf import ListConfig

logger = getLogger()



class MultiPDE(Dataset):
    def __init__(self, params, symbol_env,split="train",types = None, datasets= None, skip = 0):
        super().__init__()

        # general initialization, should be called by all subclasses

        self.train = split == "train"
        self.params = params
        # if meta is None:
        #     self.meta = self.params.meta
        # else:
        #     self.meta = meta
        self.reload_size = params.train_size if self.train else params.eval_size
        if datasets is not None:
            if datasets == "":
                self.datasets = [""]
            elif isinstance(datasets, (list, ListConfig)):
                self.datasets = ["_" + a if a != "reg" else "" for a in datasets]
            else:
                assert isinstance(datasets, str)
                self.datasets = ["_" + datasets]  if datasets != "reg" else [""]
        else:
            if self.train:
                print(params.data.train_data )
                if params.data.train_data is None:
                    self.datasets = [""]
                elif isinstance(params.data.train_data,(list, ListConfig)) :
                    self.datasets = ["_" + a for a in params.data.train_data]
                else:
                    assert isinstance(params.data.train_data, str)
                    self.datasets = ["_" +  params.data.train_data]
            else:
                # self.datasets = "_" + params.data.eval_data if params.data.eval_data is not None else ""
                if params.data.eval_data is None:
                    self.datasets = [""]
                elif isinstance(params.data.eval_data, (list, ListConfig)):
                    self.datasets = ["_" + a if a != "reg" else "" for a in params.data.eval_data]
                else:
                    assert isinstance(params.data.eval_data, str)
                    self.datasets = ["_" + params.data.eval_data]  if datasets != "reg" else [""]

        if self.train:
            if params.train_size_get > 0:
                self.get_size = params.train_size_get
            else:
                self.get_size = params.train_size
        else:
            if params.eval_size_get > 0:
                self.get_size = params.eval_size_get
            else:
                self.get_size = params.eval_size

        self.symbol_env = symbol_env
        self.split = split
        self.IC_per_param = params.IC_per_param
        self.skip = skip
        if types is not None:
            self.types = types
        else:
            self.types = params.data.train_types if self.train else params.data.eval_types

        self.directory = params.data.directory
        if self.types == -1:
            self.types = [name for name in os.listdir(self.directory ) if
             os.path.isdir(os.path.join(self.directory , name))]
        print(self.types)
        self.num_workers = params.num_workers
        self.local_rank = params.local_rank
        self.n_gpu_per_node = params.n_gpu_per_node

        self.t_num = params.data.t_num
        self.t_range = params.data.t_range
        self.x_range = params.data.x_range
        self.x_num = params.data.x_num

        self.rng = None
        self.data = []
        self.task_data = []
        self.task_indices = {}
        self.task_name = []
        self.load_data()



    def init_rng(self):
        """
        Initialize different random generator for each worker.
        """
        if self.rng is not None:
            return

        worker_id = self.get_worker_id()
        self.worker_id = worker_id
        params = self.params
        if self.train:
            base_seed = params.base_seed
            # base_seed = np.random.randint(1_000_000_000)

            seed = [worker_id, params.global_rank, base_seed]
            self.rng = np.random.default_rng(seed)
            # logger.info(f"Initialize random generator with seed {seed} (worker, dataset, rank, base_seed)")
        else:
            seed = [worker_id, params.global_rank, params.test_seed]
            self.rng = np.random.default_rng(seed)
            # logger.info(f"Initialize random generator with seed {seed} (worker, dataset, rank, test_seed)")

    def get_worker_id(self):
        worker_info = torch.utils.data.get_worker_info()
        return 0 if worker_info is None else worker_info.id

    def add_noise(self, data: np.ndarray):
        if self.params.noise > 0:
            # self.init_rng()
            gamma = self.params.noise
            cur_noise = self.rng.normal(size=data.shape).astype(np.single)

            if self.params.noise_type == "multiplicative":
                return data + gamma * np.abs(data) * cur_noise
            else:  # additive
                eps = 1e-6
                sigma = gamma * np.linalg.norm(data) / (np.linalg.norm(cur_noise) + eps)
                return data + sigma * cur_noise
        else:
            return data

    def load_data(self):
        self.init_rng()
        files = [
            f for f in os.listdir(self.directory)
            if (isinstance(self.types, (list, ListConfig)) and f in self.types) or
               (isinstance(self.types, str) and f == self.types)
        ]
        for file in files:
            for datasets in self.datasets:
                task_name = file + datasets
                file_name = file+ "_" + str(self.IC_per_param) + datasets + ".prefix"
                file_path = os.path.join(self.directory, file,file_name)
                task_indices_begin = len(self.data)
                self.data.extend(self.load_onetask_data(file_path,task_name))
                task_indices_end = len(self.data)
                self.task_name.append(task_name)
                self.task_indices[task_name] = np.arange(task_indices_begin,task_indices_end)

    def load_onetask_data(self,path,task_name = None):
        logger.info(f"Loading data from {path} ...")
        with io.open(path, mode="r", encoding="utf-8") as f:
            print("skipping", self.skip)
            if self.params.data.random_idx:
                reload_indices = self.rng.choice(range(self.skip, self.skip + self.reload_size), self.get_size,
                                                 replace=False)
            else:
                start = int(np.ceil(self.skip / self.IC_per_param))  * self.IC_per_param  # Ensure start is an integer
                size = self.get_size - self.get_size % self.IC_per_param
                reload_indices = np.arange(start, start + size, dtype=int)
            sorted_reload_indices = sorted(reload_indices)
                # Distribute indices among GPUs
            # Each GPU gets a slice of the sorted indices array, spaced by the number of GPUs
            local_indices = sorted_reload_indices[self.local_rank::self.n_gpu_per_node]

            lines = []
            index_set = set(local_indices)  # Convert list to set for faster lookup

            for i, line in enumerate(f):
                if i in index_set:
                    lines.append(json.loads(line.rstrip()))
        data = lines

        if self.params.separate_modality:
            filename = path[:-7] + "_data.h5"
            assert os.path.isfile(filename), "Data file {} not found".format(path)
            with h5py.File(filename, "r") as hf:
                data_matrix = hf["data"][local_indices]

            assert "dim" in data[0]
            assert len(data_matrix) == len(data), "Dataset size mismatch"

            final_data = [data,data_matrix]

            logger.info(f"Data size: {data_matrix.shape}.")

        else:
            final_data = data

        logger.info(f"Loaded {len(data)} equations from the disk.")

        processed_data= self.process_data(final_data,task_name)
        return processed_data


    def process_data(self, final_data,task_name):
        processed_data = []
        self.init_rng()
        if self.params.separate_modality:
            data, data_matrix = final_data
        else:
            data = final_data
        for idx in range(len(data)):
            cur_data = data[idx]
            x = dict()
            x["task"] = task_name

            if self.params.separate_modality:
                this_data_matrix =data_matrix[idx]
                this_data_matrix = self.add_noise(this_data_matrix)
                x["data"] = torch.from_numpy(this_data_matrix).float()
            else:
                x["data"] = torch.FloatTensor(self.add_noise(cur_data["data"]))
            # if self.meta:
            #     x["data"] = x["data"].unsqueeze(0)
            x["t"] = torch.from_numpy(np.linspace(self.t_range[0],self.t_range[1], self.t_num +1)[:-1]).float()
            dx = (self.x_range[1] -self.x_range[0])/self.x_num
            x["x"] = torch.from_numpy(np.linspace(self.x_range[0],self.x_range[1], self.x_num +1)[:-1] + 0.5 * dx).float()

            for key in cur_data.keys():
                if key != data:
                    x[key] = copy.deepcopy(cur_data[key])

            # x["tree"]=self.symbol_env.equation_encoder.decode(cur_data["tree_encoded"])
            processed_data.append(x)


        return processed_data



    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)

class MultiPDE_DeepO(MultiPDE):
    def __init__(self, params, symbol_env,split="train",types = None, datasets= None, skip = 0):
        self.input_len =params.data.input_len
        if split == "train":
            self.output_start = params.data.input_len if params.data.output_start is None else params.data.output_start
        else:
            self.output_start = params.data.input_len if params.data.output_start_eval is None else params.data.output_start_eval
        self.input_step =params.data.input_step
        self.output_step =params.data.output_step
        super().__init__(params, symbol_env,split,types, datasets, skip)


    def process_data(self, final_data,task_name):
        processed_data = []
        self.init_rng()
        if self.params.separate_modality:
            data, data_matrix = final_data
        else:
            data = final_data
        for idx in range(len(data)):
            cur_data = data[idx]
            # x = dict()
            # x["task"] = task_name

            if self.params.separate_modality:
                this_data_matrix =data_matrix[idx]
                this_data_matrix = self.add_noise(this_data_matrix)
                data_matrix_get = torch.from_numpy(this_data_matrix).float()
            else:
                data_matrix_get = torch.FloatTensor(self.add_noise(cur_data["data"]))

            t = torch.from_numpy(np.linspace(self.t_range[0],self.t_range[1], self.t_num +1)[:-1]).float()
            dx = (self.x_range[1] -self.x_range[0])/self.x_num
            x = torch.from_numpy(np.linspace(self.x_range[0],self.x_range[1], self.x_num +1)[:-1] + 0.5 * dx).float()

            # query_locations = []
            # for tt in t[self.output_start::self.output_step]:
            #     for xx in x:
            #         query_locations.append([tt, xx])

            query_matrix = t[self.output_start::self.output_step]

            # Example of how to create the final y dictionary
            y = {
                "data": data_matrix_get,
                "t": t,
                "x":x,
                "query": query_matrix,
                "task": task_name
            }
            processed_data.append(y)

        return processed_data




if __name__ == "__main__":
    import hydra
    from torch.utils.data import DataLoader
    import logging
    import sys
    from collate import custom_collate
    from symbol_utils.environment import SymbolicEnvironment
    from itertools import cycle

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    def print_sample(sample):
        for k, v in sample.items():
            if isinstance(v, dict):
                print(f"{k}")
                print_sample(v)
            else: #k in ["data","tree", "t", "data_mask"]:
                if isinstance(v, torch.Tensor):
                    print(f"{k}: {v.size()}, {v.dtype}")
                else:
                    print(f"{k}: {[len(e)for e in v]}")
            # else:
            #     print(f"{k}: {v}")
        print()

    @hydra.main(version_base=None, config_path="../configs", config_name="main")
    def test(params):
        params.base_seed = 0
        params.n_gpu_per_node = 1
        params.local_rank = 0
        params.global_rank = 0
        params.num_workers = 4
        params.batch_size = 10
        params.meta=0
        params.train_size = 512
        params.train_size_get = 512
        params.data.train_types = "burgers"
        params.eval_size = 512
        params.eval_size_get = 512
        params.data.eval_types = "burgers"

        symbol_env = SymbolicEnvironment(params.symbol)
        dataset = MultiPDE(params, symbol_env, split="eval")
        print(dataset.__len__())
        loader = DataLoader(
            dataset,
            batch_size=params.batch_size_eval,
            num_workers=params.num_workers,
            collate_fn=custom_collate(params.data.max_output_dimension,symbol_env),
            shuffle=True
        )

        data_iter = iter(loader)
        data = next(data_iter)
        print_sample(data)  # (bs, t_num, x_num, x_num, max_output_dimension)
        data2 = next(data_iter)
        print_sample(data2)
        print_sample(next(data_iter))

        params.meta=1
        params.data.num_support = 0
        params.data.num_query = 30
        dataset = MultiPDE(params, symbol_env, split="train")
        print(dataset.__len__())
        loader = DataLoader(
            dataset,
            batch_size=params.batch_size_eval,
            num_workers=params.num_workers,
            collate_fn=custom_collate(params.data.max_output_dimension,symbol_env),
            shuffle=True
        )

        data_iter = iter(loader)
        data1 = next(data_iter)
        print_sample(data1)

    test()
