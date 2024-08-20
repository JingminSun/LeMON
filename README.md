# LeMON-PROSE:  Learning to Learn Multi-Operator Networks - Predictiong Operators and Symbolic Expressions

This work is based on previous works from our group:

- 🔭 <a href="https://github.com/felix-lyx/prose" target="_blank">PROSE</a>: Predicting Multiple Operators and Symbolic Expressions

- 🔭 <a href="https://github.com/JingminSun/prose" target="_blank">PROSE-PDE</a>: Towards a Foundation Model for Partial Differential Equations: Multi-Operator Learning and Extrapolation.

## Data Generation

The data generator is partially based on [PDEBench](https://github.com/pdebench/PDEBench), [fplanck](https://github.com/johnaparker/fplanck) and [Pseudo-spectral KdV solver](https://github.com/jundongq/Korteweg-deVries-KdV-Equation-Solution).

See detailed instruction in script/gen_data.sh

    bash scripts/gen_data.sh
    
The data is by default saving to ``your_directory/type_name/(type_name)_(IC_per_params).prefix`` for symbol part 
                        and ``your_directory/type_name/(type_name)_(IC_per_params)_data.h5`` for data part

If some specific data need to be generated, change "file_name=specific_name", samples are given in script/gen_data.sh 
and data will be stored in ``your_directory/type_name/(type_name)_(IC_per_params)_(file_name).prefix`` 
                        and ``your_directory/type_name/(type_name)_(IC_per_params)_(file_name)_data.h5``


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

### LoRA fine-tuning vs Regular fine-tuning:
LoRA code is based on [this_tutorial](https://lightning.ai/lightning-ai/studios/code-lora-from-scratch).

    bash scripts/loraexperiments.sh

### PAML experiments:
PAML code is based on [learn2learn library](https://github.com/learnables/learn2learn/).

    bash scripts/PAML.sh
