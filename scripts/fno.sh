GPU=0

user=aaa

pre_train_types=( advection conservation_sinflux diff_bistablereact_1D fplanck heat inviscid_conservation_cubicflux Klein_Gordon)
for type in "${pre_train_types[@]}"
do
#pre_train=deepo_pret_10000_${type}_nonoise
#
CUDA_VISIBLE_DEVICES=$GPU python src/main.py  batch_size_eval=128 exp_name=fno exp_id=fno_pret_10000_${type}_nonoise wandb.id=fno_pret_10000_${type}_nonoise data.eval_data=onlyqc  data.train_data=onlyqc train_size=10000 train_size_get=10000 eval_size=10000 eval_size_get=10000 batch_size=150 meta=0 model=fno data.train_types=${type} data.eval_types=${type} max_epoch=15  zero_shot_only=1 data.input_len=1 data.input_step=1 noise=0 amp=0
done

ft_types=(diff_linearreact_1D inviscid_burgers burgers diff_squarelogisticreact_1D cahnhilliard_1D conservation_cubicflux Sine_Gordon inviscid_conservation_sinflux kdv diff_logisreact_1D wave porous_medium)

for pre_type in "${pre_train_types[@]}"
do
      pre_train=fno_pret_10000_${pre_type}_nonoise
      echo "Finetuning $pre_type"

      NUMEXPR_MAX_THREADS=12 CUDA_VISIBLE_DEVICES=$GPU python src/main.py batch_size_eval=128 exp_name=fno exp_id=ft${pre_type}_ood1_new wandb.id=ft${pre_type}_ood1_new data.eval_data=ood0.3_1  data.train_data=ood0.3_1  reload_model=checkpoint/${user}/dumped/fno/${pre_train}/checkpoint.pth save_periodic=-1  finetune=1 finetune_name=reg  zero_shot_only=1 train_size_get=100 train_size=100 eval_size=500 eval_size_get=500  batch_size=100  n_steps_per_epoch=1  max_epoch=100 log_periodic=1 data.train_types=$pre_type data.eval_types=$pre_type model=fno meta=0 data.input_len=1 data.input_step=1   IC_per_param=1024 noise=0 amp=0
      NUMEXPR_MAX_THREADS=12 CUDA_VISIBLE_DEVICES=$GPU python src/main.py batch_size_eval=128 exp_name=fno exp_id=ft${pre_type}_ood2_new wandb.id=ft${pre_type}_ood2_new data.eval_data=ood0.3_2  data.train_data=ood0.3_2  reload_model=checkpoint/${user}/dumped/fno/${pre_train}/checkpoint.pth save_periodic=-1  finetune=1 finetune_name=reg  zero_shot_only=1 train_size_get=100 train_size=100 eval_size=500 eval_size_get=500  batch_size=100  n_steps_per_epoch=1  max_epoch=100 log_periodic=1 data.train_types=$pre_type data.eval_types=$pre_type model=fno meta=0 data.input_len=1  data.input_step=1  IC_per_param=1024 noise=0 amp=0
      NUMEXPR_MAX_THREADS=12 CUDA_VISIBLE_DEVICES=$GPU python src/main.py batch_size_eval=128 exp_name=fno exp_id=ft${pre_type}_ood3_new wandb.id=ft${pre_type}_ood3_new data.eval_data=ood0.3_3  data.train_data=ood0.3_3  reload_model=checkpoint/${user}/dumped/fno/${pre_train}/checkpoint.pth save_periodic=-1  finetune=1 finetune_name=reg  zero_shot_only=1 train_size_get=100 train_size=100 eval_size=500 eval_size_get=500  batch_size=100  n_steps_per_epoch=1  max_epoch=100 log_periodic=1 data.train_types=$pre_type data.eval_types=$pre_type model=fno meta=0 data.input_len=1 data.input_step=1   IC_per_param=1024 noise=0 amp=0

    for type in "${ft_types[@]}"
    do

      echo "Finetuning $type"
      NUMEXPR_MAX_THREADS=12 CUDA_VISIBLE_DEVICES=$GPU python src/main.py batch_size_eval=128 exp_name=fno exp_id=${pre_type}_${type} wandb.id=${pre_type}_${type} data.eval_data=onlyqc  data.train_data=onlyqc  reload_model=checkpoint/${user}/dumped/fno/${pre_train}/checkpoint.pth save_periodic=-1  finetune=1 finetune_name=reg  zero_shot_only=1 train_size=10000 eval_size=10000 train_size_get=100 eval_size_get=500  batch_size=100  n_steps_per_epoch=1  max_epoch=100 log_periodic=1 data.train_types=$type data.eval_types=$type model=fno meta=0 data.input_len=1 data.input_step=1  noise=0 amp=0
      NUMEXPR_MAX_THREADS=12 CUDA_VISIBLE_DEVICES=$GPU python src/main.py batch_size_eval=256 exp_name=evaluate eval_only=1 exp_id=eval_${pre_type}_after_ft_on${type}_fno wandb.id=eval_${pre_type}_after_ft_on${type}_deepo zero_shot_only=1 data.eval_types=$pre_type  data.eval_data=onlyqc  eval_from_exp=checkpoint/${user}/dumped/fno/${pre_type}_${type}  model=deeponet meta=0 data.input_len=1 data.input_step=1  noise=0 train_size=10000 train_size_get=1500 eval_size=10000 eval_size_get=10000 amp=0
    done
done
