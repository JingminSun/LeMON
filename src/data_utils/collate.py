import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
import random
from collections import defaultdict


def get_padding_mask(lengths, max_len=None,  beos_added = False):
    """
    Input:
        lengths:           LongTensor (bs, )  length of each example
        max_len:           Optional[int]      if None, max_len = lengths.max()
    Output:
        key_padding_mask:  BoolTensor (bs, max_len)    (positions with value True are padding)
    """
    if max_len is None:
        max_len = lengths.max().item()

    bs = lengths.size(0)
    if beos_added:
        key_padding_mask = torch.arange(max_len, device=lengths.device).expand(bs, max_len) >= lengths.unsqueeze(1) - 1 # Excluding EOS
    else:
        key_padding_mask = torch.arange(max_len, device=lengths.device).expand(bs,max_len) >= lengths.unsqueeze(1)
    return key_padding_mask


def get_data_mask(lengths, max_len=None):
    """
    Input:
        lengths:           LongTensor (bs, )  length of each example
        max_len:           Optional[int]      if None, max_len = lengths.max()
    Output:
        data_mask:         Tensor (bs, max_len)    (positions with value 0 are padding)
    """
    if max_len is None:
        max_len = lengths.max().item()

    bs = lengths.size(0)
    mask = torch.arange(max_len, device=lengths.device).expand(bs, max_len) < lengths.unsqueeze(1)
    return mask.float()


def collate_symbols(batch_of_equations,symbol_env):
    lengths = torch.LongTensor([2 + len(eq) for eq in batch_of_equations])
    max_length = lengths.max().item()
    this = torch.LongTensor(len(batch_of_equations), max_length).fill_(symbol_env.float_word2id["<PAD>"])

    this[:, 0] = symbol_env.equation_word2id["<BOS>"]
    cur_eqs = symbol_env.word_to_idx(batch_of_equations, float_input=False)
    for i, equation in enumerate(cur_eqs):
        data_dim = len(equation) + 2
        this[i, 1:data_dim - 1].copy_(equation)
        this[i, data_dim - 1] = symbol_env.equation_word2id["<EOS>"]
    return  this, lengths,max_length

def custom_collate(max_data_dim,symbol_env):
    """
    Input:
        max_data_dim: maximum output dim

    Output:
       collate function
    """


    def my_collate(batch):

        res = {}


        keys = batch[0].keys()
        for k in keys:
            if k == "data" or k == "data_input":
                lst = []
                dims = []
                for d in batch:
                    cur_data = d[k]
                    data_dim = cur_data.size(-1)
                    dims.append(data_dim)
                    diff = max_data_dim - data_dim
                    if diff > 0:
                        lst.append(F.pad(cur_data, (0, diff), "constant"))
                    else:
                        lst.append(cur_data)
                res[k] = default_collate(lst)

                data_dims = torch.LongTensor(dims)
                res["data_mask"] = get_data_mask(data_dims, max_data_dim)  # (bs, max_data_dim)
            elif k == "tree_encoded":

                if not isinstance(batch[0][k][0],list):
                    batch_tree = []
                    for d in batch:
                        batch_tree.append(d[k])
                    collated_tree, lengths,max_length = collate_symbols(batch_tree,symbol_env)
                    res[k] = collated_tree
                    res["tree_mask"] = get_padding_mask(lengths,max_length, beos_added = True)
                    res["original_tree"] = batch_tree
                else:
                    res[k] = []
                    res["tree_mask"] = []
                    res["original_tree"] = []
                    for d in batch:
                        cur_data = d[k]
                        collated_tree, lengths,max_length = collate_symbols(cur_data,symbol_env)
                        res[k].append(collated_tree)
                        res["original_tree"].append(cur_data)
                        res["tree_mask"].append(get_padding_mask(lengths,max_length, beos_added = True))

            elif isinstance(batch[0][k],dict):
                res[k] = []
                for d in batch:
                    res[k].append(d[k])
            else:
                # if meta:
                #     res[k] = []
                #     for d in batch:
                #         res[k].append(default_collate(d[k]))
                #     if isinstance(res[k],list) and isinstance(res[k][0],torch.Tensor):
                #         res[k] = default_collate(res[k])
                # else:
                res[k] = default_collate([d[k] for d in batch])
        for k in res:
            if isinstance(res[k][0],dict) :
                res[k] = my_collate(res[k])

        return res

    #
    return my_collate
# def my_collate(batch, max_data_dim, symbol_env):
#     res = {}
#     keys = batch[0].keys()
#     for k in keys:
#         if k == "data" or k == "data_input":
#             lst = []
#             dims = []
#             for d in batch:
#                 cur_data = d[k]
#                 data_dim = cur_data.size(-1)
#                 dims.append(data_dim)
#                 diff = max_data_dim - data_dim
#                 if (diff := max_data_dim - data_dim) > 0:
#                     lst.append(F.pad(cur_data, (0, diff), "constant"))
#                 else:
#                     lst.append(cur_data)
#             res[k] = default_collate(lst)
#             data_dims = torch.LongTensor(dims)
#             res["data_mask"] = get_data_mask(data_dims, max_data_dim)  # (bs, max_data_dim)
#         elif k == "tree_encoded":
#             if not isinstance(batch[0][k][0], list):
#                 batch_tree = [d[k] for d in batch]
#                 collated_tree, lengths, max_length = collate_symbols(batch_tree, symbol_env)
#                 res[k] = collated_tree
#                 res["tree_mask"] = get_padding_mask(lengths, max_length, beos_added=True)
#             else:
#                 res[k] = []
#                 res["tree_mask"] = []
#                 for d in batch:
#                     cur_data = d[k]
#                     collated_tree, lengths, max_length = collate_symbols(cur_data, symbol_env)
#                     res[k].append(collated_tree)
#                     res["tree_mask"].append(get_padding_mask(lengths, max_length, beos_added=True))
#         elif isinstance(batch[0][k], dict):
#             res[k] = [d[k] for d in batch]
#         else:
#             res[k] = default_collate([d[k] for d in batch])
#     for k in res:
#         if isinstance(res[k][0], dict):
#             res[k] = my_collate(res[k], max_data_dim, symbol_env)
#     return res
#
# def custom_collate(max_data_dim, symbol_env):
#     def collate_fn(batch):
#         return my_collate(batch, max_data_dim, symbol_env)
#     return collate_fn

