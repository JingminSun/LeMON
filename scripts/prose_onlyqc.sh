GPU=0

user=aaa

pre_train_types=( advection conservation_sinflux diff_bistablereact_1D fplanck heat inviscid_conservation_cubicflux Klein_Gordon)
pre_train_typevec=[advection,conservation_sinflux,diff_bistablereact_1D,fplanck,heat,inviscid_conservation_cubicflux,Klein_Gordon]
ft_types=(burgers inviscid_burgers conservation_cubicflux diff_linearreact_1D diff_squarelogisticreact_1D  cahnhilliard_1D Sine_Gordon inviscid_conservation_sinflux kdv diff_logisreact_1D  porous_medium wave)

CUDA_VISIBLE_DEVICES=$GPU python src/main.py  batch_size_eval=256 exp_id=pretrain_size1500_v2 wandb.id=pretrain_size1500_v2 train_size=10000 train_size_get=750 eval_size=10000 eval_size_get=10000 batch_size=150 meta_step=5 meta=0  data.train_types=$pre_train_typevec data.eval_types=$pre_train_typevec model.name=prose model.meta.gd_eta=0 data.eval_data=onlyqc  data.train_data=onlyqc  data.input_len=1 data.input_step=1 noise=0

pre_train=pretrain_size1500_v2
for pre_type in "${pre_train_types[@]}"
do

      echo "Finetuning $pre_type"

      NUMEXPR_MAX_THREADS=12 CUDA_VISIBLE_DEVICES=$GPU python src/main.py batch_size_eval=256 exp_name=diff_ft exp_id=${pre_type}_ood1_new wandb.id=${pre_type}_ood1_new data.eval_data=ood0.3_1  data.train_data=ood0.3_1  reload_model=checkpoint/${user}/dumped/LeMON_PROSE/${pre_train}/checkpoint.pth save_periodic=-1  finetune=1 finetune_name=reg  zero_shot_only=1 train_size_get=100 train_size=100 eval_size=500 eval_size_get=500  batch_size=100  n_steps_per_epoch=1  max_epoch=100 log_periodic=1 data.train_types=$pre_type data.eval_types=$pre_type model=prose meta=0 data.input_len=1  data.input_step=1 IC_per_param=1024 noise=0
      NUMEXPR_MAX_THREADS=12 CUDA_VISIBLE_DEVICES=$GPU python src/main.py batch_size_eval=256 exp_name=diff_ft exp_id=${pre_type}_ood2_new wandb.id=${pre_type}_ood2_new data.eval_data=ood0.3_2  data.train_data=ood0.3_2  reload_model=checkpoint/${user}/dumped/LeMON_PROSE/${pre_train}/checkpoint.pth save_periodic=-1  finetune=1 finetune_name=reg  zero_shot_only=1 train_size_get=100 train_size=100 eval_size=500 eval_size_get=500  batch_size=100  n_steps_per_epoch=1  max_epoch=100 log_periodic=1 data.train_types=$pre_type data.eval_types=$pre_type model=prose meta=0 data.input_len=1  data.input_step=1 IC_per_param=1024 noise=0
      NUMEXPR_MAX_THREADS=12 CUDA_VISIBLE_DEVICES=$GPU python src/main.py batch_size_eval=256 exp_name=diff_ft exp_id=${pre_type}_ood3_new wandb.id=${pre_type}_ood3_new data.eval_data=ood0.3_3  data.train_data=ood0.3_3  reload_model=checkpoint/${user}/dumped/LeMON_PROSE/${pre_train}/checkpoint.pth save_periodic=-1  finetune=1 finetune_name=reg  zero_shot_only=1 train_size_get=100 train_size=100 eval_size=500 eval_size_get=500  batch_size=100  n_steps_per_epoch=1  max_epoch=100 log_periodic=1 data.train_types=$pre_type data.eval_types=$pre_type model=prose meta=0 data.input_len=1  data.input_step=1 IC_per_param=1024 noise=0

done

  for type in "${ft_types[@]}"
  do

    echo "Finetuning $type"

    NUMEXPR_MAX_THREADS=12 CUDA_VISIBLE_DEVICES=$GPU python src/main.py batch_size_eval=256 exp_name=diff_ft exp_id=${type} wandb.id=${type} data.eval_data=onlyqc  data.train_data=onlyqc  reload_model=checkpoint/${user}/dumped/LeMON_PROSE/${pre_train}/checkpoint.pth save_periodic=-1  finetune=1 finetune_name=reg  zero_shot_only=1 train_size=10000 eval_size=10000 train_size_get=100 eval_size_get=500  batch_size=100  n_steps_per_epoch=1  max_epoch=100 log_periodic=1 data.train_types=$type data.eval_types=$type model=prose meta=0 data.input_len=1 data.input_step=1 noise=0
    NUMEXPR_MAX_THREADS=12 CUDA_VISIBLE_DEVICES=$GPU python src/main.py batch_size_eval=256 exp_name=evaluate eval_only=1 exp_id=eval_after_ft_on${type}_prose wandb.id=eval_after_ft_on${type}_prose zero_shot_only=1 data.eval_types=$pre_train_typevec  data.eval_data=onlyqc  eval_from_exp=checkpoint/${user}/dumped/diff_ft/${type}_5  model=prose meta=0 data.input_len=1 data.input_step=1 noise=0 train_size=10000 train_size_get=1500 eval_size=10000 eval_size_get=10000

  done