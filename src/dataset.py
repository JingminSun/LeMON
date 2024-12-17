
from data_utils.all_datasets import MultiPDE, MultiPDE_DeepO
from itertools import combinations
from logging import getLogger
import os
from torch.utils.data import  ConcatDataset
logger = getLogger()


def get_non_empty_subsets(lst,min_len=1,max_len=-1):
    # Create an empty list to store non-empty subsets
    non_empty_subsets = []
    if max_len == -1:
        max_len = len(lst)
    # Iterate over all possible subset sizes (starting from 1 to exclude the empty set)
    for r in range(min_len, max_len + 1):
        # Generate all combinations of the current size
        subsets_of_size_r = combinations(lst, r)

        # Add each subset to the list of non-empty subsets
        for subset in subsets_of_size_r:
            non_empty_subsets.append(list(subset))

    return non_empty_subsets


def get_dataset(params, symbol_env, split, skip = 0,meta = True, mixed_type= False, support=1):

    if split == "train" and (mixed_type or  not meta):
        if params.model.name == "DeepONet" :
            return MultiPDE_DeepO(params, symbol_env, skip=skip)
        else:
            return MultiPDE(params,symbol_env,split = split, skip= skip)
    else:
        datasets = {}
        if params.data.diff_supp_query_type and meta:
            assert  ( isinstance(params.data.eval_types_support ,str) and isinstance(params.data.eval_types_query ,str) )  or len(params.data.eval_types_support) == len( params.data.eval_types_query )

            assert ( isinstance(params.data.train_types_support ,str) and isinstance(params.data.train_types_query ,str) )  or len(params.data.train_types_support) == len( params.data.train_types_query )
            if support:
                types = params.data.eval_types_support if split == "eval" else params.data.train_types_support
            else:
                types = params.data.eval_types_query if split == "eval" else params.data.train_types_query
            types = [types] if isinstance(types, str) else types
        else:
            if  params.data.eval_types == -1 and split == "eval":
                types = [name for name in os.listdir(params.data.directory) if
                                  os.path.isdir(os.path.join(params.data.directory, name)) and "cosflux" not in name]
            elif params.data.train_types == -1 and split == "train":
                types = [name for name in os.listdir(params.data.directory) if
                                  os.path.isdir(os.path.join(params.data.directory, name)) and "cosflux" not in name]
            else:
                types =  params.data.eval_types if split == "eval" else params.data.train_types
                types = [types] if isinstance(types,str) else types


        if split == "eval":
            datasets_list = params.data.eval_data if params.data.eval_data is not None else [""]
        else:
            datasets_list = params.data.train_data if params.data.train_data is not None else [""]
        datasets_list = [datasets_list] if isinstance(datasets_list, str) else datasets_list

        if not mixed_type:
            for t in   types:
                for d in datasets_list:
                    try:
                        if params.model.name == "DeepONet":
                            ds =  MultiPDE_DeepO(params, symbol_env, split = split,types=t,datasets=d, skip=skip)
                        else:
                            ds = MultiPDE(params,symbol_env,split = split,types=t,datasets=d, skip= skip)
                        if datasets_list == [""]:
                            name = t
                        else:
                            if d == "":
                                name = t
                            else:
                                name = t + "%" + d
                        datasets[name] = ds
                    except:
                        pass
        else:
            assert len(datasets_list)==1, "Only support one dataset for mixed type"
            assert params.model.name != "DeepONet" or params.model.name != "FNO"
            datasets_pre = {}
            for data_type in types:
                ds = MultiPDE(params, symbol_env, split="eval", types=[data_type], datasets=datasets_list[0],skip=skip)
                datasets_pre[data_type] = ds


            subsets = get_non_empty_subsets(types,min_len=params.data.eval_task_min_data,max_len=params.data.eval_task_max_data)

            for comb_type in subsets:
                combined_datasets = [datasets_pre[t] for t in comb_type]

                # Use ConcatDataset to combine the datasets
                combined_dataset = ConcatDataset(combined_datasets)

                name = '+'.join(comb_type)
                datasets[name] = combined_dataset

        return datasets


if __name__ == "__main__":
    import hydra
    from torch.utils.data import DataLoader
    import logging
    import sys
    from data_utils.collate import custom_collate
    from symbol_utils.environment import SymbolicEnvironment
    from itertools import cycle
    import torch

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

    @hydra.main(version_base=None, config_path="configs", config_name="main")
    def test(params):
        params.base_seed = 0
        params.n_gpu_per_node = 1
        params.local_rank = 0
        params.global_rank = 0
        params.num_workers = 4
        params.batch_size = 10
        params.eval_size=200
        params.data.eval_types = -1

        symbol_env = SymbolicEnvironment(params.symbol)
        dataset:dict = get_dataset(params,symbol_env,split="eval")
        print(dataset.__len__())
        loader = {k:cycle(DataLoader(
            v,
            batch_size=params.batch_size_eval,
            num_workers=params.num_workers,
            collate_fn=custom_collate(params.data.max_output_dimension,symbol_env,params.meta),
            shuffle=True
        )) for k,v in dataset.items()}

        data_iter = iter(loader["burgers"])
        data = next(data_iter)
        print_sample(data)  # (bs, t_num, x_num, x_num, max_output_dimension)
        print_sample(next(data_iter))
        print_sample(next(data_iter))

    test()