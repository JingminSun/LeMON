GPU=0
ICs_equation=50
directory="/home/shared/prose/sep_types/"


#datasets=(burgers conservation_sinflux conservation_cubicflux inviscid_burgers inviscid_conservation_sinflux inviscid_conservation_cubicflux )
datasets=( conservation_cosflux inviscid_conservation_cosflux  )

#datasets=(burgers conservation_sinflux conservation_cubicflux inviscid_burgers inviscid_conservation_sinflux inviscid_conservation_cubicflux advection diff_bistablereact_1D fplanck heat  Klein_Gordon diff_linearreact_1D diff_squarelogisticreact_1D cahnhilliard_1D Sine_Gordon kdv diff_logisreact_1D wave)
#for dataset in "${datasets[@]}"; do
#    # Cleanup old data to make sure no mismatching of datasize
#    ### Data will stored in the following path as well:
##    rm $directory/$dataset/${dataset}_$ICs_equation.prefix
##    rm $directory/$dataset/${dataset}_${ICs_equation}_data.h5
##     ### Generate Basic Data, param range [(1 - data.param_range_gamma)* q_c, (1 + data.param_range_gamma)* q_c] change size if needed, 512K is just a larger enough number that should cover all your needs
##    CUDA_VISIBLE_DEVICES=$GPU python3 src/data_gen_pde.py num_workers=12  IC_per_param=$ICs_equation data.param_range_gamma=0.1 data.types=${dataset}  size=51200 directory=$directory
##
##    file_name=viscous0.1
##    rm $directory/$dataset/${dataset}_${ICs_equation}_${file_name}.prefix
##    rm $directory/$dataset/${dataset}_${ICs_equation}_${file_name}_data.h5
##     ### Generate Basic Data, param range [(1 - data.param_range_gamma)* q_c, (1 + data.param_range_gamma)* q_c] change size if needed, 512K is just a larger enough number that should cover all your needs
##    CUDA_VISIBLE_DEVICES=$GPU python3 src/data_gen_pde.py num_workers=12  IC_per_param=$ICs_equation data.param_range_gamma=0.5 data.use_sharedcoeff=true data.types=${dataset}  size=51200 directory=$directory file_name=${file_name}
#
##
##
##   '''
##    ## If you just want to generate data with q_c parameter, just set data.param_range_gamma=0
##
##
##    ## This generates the data with only q_c as parameter (only one operator from each family)
##    file_name=onlyqc
##    rm $directory/$dataset/${dataset}_${ICs_equation}_${file_name}.prefix
##    rm $directory/$dataset/${dataset}_${ICs_equation}_${file_name}_data.h5
##    CUDA_VISIBLE_DEVICES=$GPU python3 src/data_gen_pde.py num_workers=12  IC_per_param=$ICs_equation data.param_range_gamma=0 data.types=${dataset}  size=512000 directory=$directory file_name=$file_name
##
##
##
##    ## And if you want to generate data with q_a \neq q_c, and q_a \in [(1 - data.param_range_gamma)* q_c, (1 + data.param_range_gamma)* q_c]
##    ## It is better to generate it one by one, sry we did not support generating multiple ones for now
##
##    ## Here is an example you want to generate with three different q_as: (each 1024 data)
##    # Cleanup old data to make sure no mismatching of datasize
##    ### Data will stored in the following path as well:
##    ICs_equation=1024
##    file_name=ood0.3_1
##    rm $directory/$dataset/${dataset}_${ICs_equation}_${file_name}.prefix
##    rm $directory/$dataset/${dataset}_${ICs_equation}_${file_name}_data.h5
##    CUDA_VISIBLE_DEVICES=$GPU python3 src/data_gen_pde.py num_workers=0  IC_per_param=$ICs_equation data.param_range_gamma=0.3 data.types=${dataset}  size=1024 directory=$directory file_name=${file_name}
##
##
##    file_name=ood0.3_2
##    rm $directory/$dataset/${dataset}_${ICs_equation}_${file_name}.prefix
##    rm $directory/$dataset/${dataset}_${ICs_equation}_${file_name}_data.h5
##    CUDA_VISIBLE_DEVICES=$GPU python3 src/data_gen_pde.py num_workers=0  IC_per_param=$ICs_equation data.param_range_gamma=0.3 data.types=${dataset}  size=1024 directory=$directory file_name=${file_name}
##
##    file_name=ood0.3_3
##    rm $directory/$dataset/${dataset}_${ICs_equation}_${file_name}.prefix
##    rm $directory/$dataset/${dataset}_${ICs_equation}_${file_name}_data.h5
##    CUDA_VISIBLE_DEVICES=$GPU python3 src/data_gen_pde.py num_workers=0  IC_per_param=$ICs_equation data.param_range_gamma=0.3 data.types=${dataset}  size=1024 directory=$directory file_name=${file_name}
##  '''
#
#done
#
#   '''
#
#   #### Change for-loop within all conservation types: One Shock/ Two shocks/ rarefaction: just modify file_name and data.IC_types
#    file_name=one_shock
#    rm $directory/$dataset/${dataset}_${ICs_equation}_${file_name}.prefix
#    rm $directory/$dataset/${dataset}_${ICs_equation}_${file_name}_data.h5
#    CUDA_VISIBLE_DEVICES=$GPU python3 src/data_gen_pde.py num_workers=12  IC_per_param=$ICs_equation data.param_range_gamma=0.1 data.IC_types=one_shock data.types=${dataset}  size=512000 directory=$directory file_name=$file_name
#  '''


#### Sort and filter data for meta_learning

CUDA_VISIBLE_DEVICES=$GPU python3 src/utils/sort_data.py +base_directory=${directory}

CUDA_VISIBLE_DEVICES=$GPU python3 src/utils/valid_data.py +base_directory=${directory}
## you can also sort a filter a single file by python3 src/utils/valid_data_single.py by changing parameters in that file