GPU=0

user=aaa
type=[burgers,conservation_cosflux,conservation_cubicflux,inviscid_burgers,inviscid_conservation_cosflux,inviscid_conservation_cubicflux]

CUDA_VISIBLE_DEVICES=$GPU python src/main.py batch_size_eval=256 exp_id=rarefactionMAML wandb.id=rarefactionMAML train_size_get=1000 eval_size_get=400 batch_size_task=1 meta_step=5 data.train_types=[burgers,conservation_cosflux,conservation_cubicflux,inviscid_burgers,inviscid_conservation_cosflux,inviscid_conservation_cubicflux] data.eval_types=[burgers,conservation_cosflux,conservation_cubicflux,inviscid_burgers,inviscid_conservation_cosflux,inviscid_conservation_cubicflux] model.meta.gd_eta=0 model.meta.name=MAML model.name=prose  data.eval_data=rarefactionv2 data.train_data=rarefactionv2 num_adapt_eval=5
CUDA_VISIBLE_DEVICES=$GPU python src/main.py batch_size_eval=256 exp_id=OneShockMAML wandb.id=OneShockMAML train_size_get=1000 eval_size_get=400 batch_size_task=1 meta_step=5 data.train_types=[burgers,conservation_cosflux,conservation_cubicflux,inviscid_burgers,inviscid_conservation_cosflux,inviscid_conservation_cubicflux] data.eval_types=[burgers,conservation_cosflux,conservation_cubicflux,inviscid_burgers,inviscid_conservation_cosflux,inviscid_conservation_cubicflux] model.meta.gd_eta=0 model.meta.name=MAML model.name=prose data.eval_data=one_shockv2 data.train_data=one_shockv2   num_adapt_eval=5

CUDA_VISIBLE_DEVICES=$GPU python src/main.py  batch_size_eval=512 exp_id=oneshockPretrain wandb.id=oneshockPretrain train_size_get=2000 batch_size=150 meta_step=5 data.train_types=[burgers,conservation_cosflux,conservation_cubicflux,inviscid_burgers,inviscid_conservation_cosflux,inviscid_conservation_cubicflux] data.eval_types=[burgers,conservation_cosflux,conservation_cubicflux,inviscid_burgers,inviscid_conservation_cosflux,inviscid_conservation_cubicflux] model.meta.gd_eta=0 model.name=prose meta=0 log_eval_plots=4 data.eval_data=one_shockv2 data.train_data=one_shockv2 &&
CUDA_VISIBLE_DEVICES=$GPU python src/main.py   batch_size_eval=512 exp_id=rarefactionPretrain wandb.id=rarefactionPretrainvtrain_size_get=2000 batch_size=150 meta_step=5 data.train_types=[burgers,conservation_cosflux,conservation_cubicflux,inviscid_burgers,inviscid_conservation_cosflux,inviscid_conservation_cubicflux] data.eval_types=[burgers,conservation_cosflux,conservation_cubicflux,inviscid_burgers,inviscid_conservation_cosflux,inviscid_conservation_cubicflux] model.meta.gd_eta=0 model.name=prose meta=0 log_eval_plots=4 data.eval_data=rarefactionv2 data.train_data=rarefactionv2


types=( burgers conservation_cosflux conservation_cubicflux inviscid_burgers inviscid_conservation_cosflux inviscid_conservation_cubicflux)
for type in "${types[@]}"
do
  echo "Finetuning $type"


  pre_train=rarefactionMAML
  NUMEXPR_MAX_THREADS=12 CUDA_VISIBLE_DEVICES=$GPU python src/main.py batch_size_eval=512 exp_name=ftone_shockv2 exp_id=ft100_MAML_${type}_new wandb.id=ft100_MAML_${type}_new data.eval_data=one_shockv2 data.train_data=one_shockv2 save_periodic=-1 reload_model=checkpoint/${user}/dumped/LeMON_PROSE/${pre_train}/checkpoint.pth finetune=1 finetune_name=reg train_size_get=100 eval_size_get=40000 zero_shot_only=1  batch_size=100  n_steps_per_epoch=1  max_epoch=100 log_periodic=1 data.train_types=$type data.eval_types=$type model.name=prose meta=0

  NUMEXPR_MAX_THREADS=12 CUDA_VISIBLE_DEVICES=$GPU python src/main.py batch_size_eval=512 exp_name=ftone_shockv2 exp_id=lora_64_100_MAML_${type}_new wandb.id=lora_64_100_MAML_${type}_new data.eval_data=one_shockv2 data.train_data=one_shockv2  save_periodic=-1 reload_model=checkpoint/${user}/dumped/LeMON_PROSE/${pre_train}/checkpoint.pth finetune=1 finetune_name=lora lora_r=64 lora_alpha=5 train_size_get=100 eval_size_get=40000 zero_shot_only=1  batch_size=100  n_steps_per_epoch=1  max_epoch=100 log_periodic=1 data.train_types=$type data.eval_types=$type model.name=prose meta=0


  pre_train=OneShockMAML
  NUMEXPR_MAX_THREADS=12 CUDA_VISIBLE_DEVICES=$GPU python src/main.py batch_size_eval=512 exp_name=ftrarefactionv2 exp_id=ft100_MAML_${type}_new wandb.id=ft100_MAML_${type}_new data.eval_data=rarefactionv2 data.train_data=rarefactionv2 save_periodic=-1 reload_model=checkpoint/${user}/dumped/LeMON_PROSE/${pre_train}/checkpoint.pth finetune=1 finetune_name=reg train_size_get=100 eval_size_get=40000 zero_shot_only=1  batch_size=100  n_steps_per_epoch=1  max_epoch=100 log_periodic=1 data.train_types=$type data.eval_types=$type model.name=prose meta=0

  NUMEXPR_MAX_THREADS=12 CUDA_VISIBLE_DEVICES=$GPU python src/main.py batch_size_eval=512 exp_name=ftrarefactionv2 exp_id=lora_64_100_MAML_${type}_new wandb.id=lora_64_100_MAML_${type}_new data.eval_data=rarefactionv2 data.train_data=rarefactionv2  save_periodic=-1 reload_model=checkpoint/${user}/dumped/LeMON_PROSE/${pre_train}/checkpoint.pth finetune=1 finetune_name=lora lora_r=64 lora_alpha=5 train_size_get=100 eval_size_get=40000 zero_shot_only=1  batch_size=100  n_steps_per_epoch=1  max_epoch=100 log_periodic=1 data.train_types=$type data.eval_types=$type model.name=prose meta=0


  pre_train=rarefactionPretrain
  NUMEXPR_MAX_THREADS=12 CUDA_VISIBLE_DEVICES=$GPU python src/main.py batch_size_eval=512 exp_name=ftone_shockv2 exp_id=ft100_TL_${type}_new wandb.id=ft100_TL_${type}_new data.eval_data=one_shockv2 data.train_data=one_shockv2 save_periodic=-1 reload_model=checkpoint/${user}/dumped/LeMON_PROSE/${pre_train}/checkpoint.pth finetune=1 finetune_name=reg train_size_get=100 eval_size_get=40000 zero_shot_only=1  batch_size=100  n_steps_per_epoch=1  max_epoch=100 log_periodic=1 data.train_types=$type data.eval_types=$type model.name=prose meta=0

  NUMEXPR_MAX_THREADS=12 CUDA_VISIBLE_DEVICES=$GPU python src/main.py batch_size_eval=512 exp_name=ftone_shockv2 exp_id=lora_64_100_TL_${type}_new wandb.id=lora_64_100_MAML_${type}_new data.eval_data=one_shockv2 data.train_data=one_shockv2  save_periodic=-1 reload_model=checkpoint/${user}/dumped/LeMON_PROSE/${pre_train}/checkpoint.pth finetune=1 finetune_name=lora lora_r=64 lora_alpha=5 train_size_get=100 eval_size_get=40000 zero_shot_only=1  batch_size=100  n_steps_per_epoch=1  max_epoch=100 log_periodic=1 data.train_types=$type data.eval_types=$type model.name=prose meta=0


  pre_train=oneshockPretrain
  NUMEXPR_MAX_THREADS=12 CUDA_VISIBLE_DEVICES=$GPU python src/main.py batch_size_eval=512 exp_name=ftrarefactionv2 exp_id=ft100_TL_${type}_new wandb.id=ft100_TL_${type}_new data.eval_data=rarefactionv2 data.train_data=rarefactionv2 save_periodic=-1 reload_model=checkpoint/${user}/dumped/LeMON_PROSE/${pre_train}/checkpoint.pth finetune=1 finetune_name=reg train_size_get=100 eval_size_get=40000 zero_shot_only=1  batch_size=100  n_steps_per_epoch=1  max_epoch=100 log_periodic=1 data.train_types=$type data.eval_types=$type model.name=prose meta=0

  NUMEXPR_MAX_THREADS=12 CUDA_VISIBLE_DEVICES=$GPU python src/main.py batch_size_eval=512 exp_name=ftrarefactionv2 exp_id=lora_64_100_TL_${type}_new wandb.id=lora_64_100_TL_${type}_new data.eval_data=rarefactionv2 data.train_data=rarefactionv2  save_periodic=-1 reload_model=checkpoint/${user}/dumped/LeMON_PROSE/${pre_train}/checkpoint.pth finetune=1 finetune_name=lora lora_r=64 lora_alpha=5 train_size_get=100 eval_size_get=40000 zero_shot_only=1  batch_size=100  n_steps_per_epoch=1  max_epoch=100 log_periodic=1 data.train_types=$type data.eval_types=$type model.name=prose meta=0


done