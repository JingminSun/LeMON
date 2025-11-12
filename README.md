# LeMON:  Learning to Learn Multi-Operator Networks


## Data Generation: See detailed instruction in [gen_data.sh](scripts/gen_data.sh)
    bash scripts/gen_data.sh


### Data output Structure

#### Default Output

By default, the generated data is saved in the `Dir/type_name/` directory with the following structure:

| Component | File Path Pattern |
|-----------|-------------------|
| **Symbols** | `Dir/type_name/(type_name)_(IC_per_params).prefix` |
| **Data** | `Dir/type_name/(type_name)_(IC_per_params)_data.h5` |

#### Custom Output

To generate specific datasets, modify the `file_name` variable in [`gen_data.sh`](script/gen_data.sh). Examples are provided within the script.

When using a custom `file_name`, the output structure becomes:

| Component | File Path Pattern |
|-----------|-------------------|
| **Symbols** | `Dir/type_name/(type_name)_(IC_per_params)_(file_name).prefix` |
| **Data** | `Dir/type_name/(type_name)_(IC_per_params)_(file_name)_data.h5` |

## Run the code
### Data

Just specify your ``data.train_types`` and ``data.eval_types`` and if some specific_name needed, 
add ``data.eval_data=specific_name`` and ``data.train_data=specific_name``

Num of training and Num of evaluation:

``data.train_size`` is the num of training sample in total, and we subsample ``data.train_size_get`` for real training.
Same for ``data.eval_size``  and  ``data.eval_size_get`` 

We use ``skip = data.train_size`` (~line 54 of ``evaluator.py``) to avoid sampling the same data for training and evaluation.
However, if you want to save space/ your evaluation dataset and training dataset are different, you can comment that out.

## Experiments

### Single Operator Comparison:

    bash scripts/deeponet.sh
    bash scripts/fno.sh
    bash scripts/prose_onlyqc.sh


### LeMON experiments:
LeMON code is based on [learn2learn library](https://github.com/learnables/learn2learn/).

    bash scripts/learnable_lr.sh
