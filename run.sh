#!/usr/bin/env bash


for DATASET in  "syn"   #  'syn' # "PEMS07" (larger one) #"PEMS03" #"PEMSD8" "PEMSD4"
  do
    if [ $DATASET == "syn" ]; then
      syn_seeds=(  1212 )
    else
      syn_seeds=( 0 )
    fi

    for SEED in "${syn_seeds[@]}"
    do

      for T in 300 # 500 300 200 100
      do
#        # Train
        python -W ignore model/Run.py --dataset $DATASET --syn_seed $SEED --mode 'train' --tinit 1
        # Calib
#         python -W ignore model/Run.py  --dataset $DATASET --syn_seed $SEED --mode 'calcorrection' --correctionmode 'mlp'
#        # Test Baseline: CP
#        python -W ignore model/Run.py --dataset $DATASET --syn_seed $SEED --mode 'test' --CP_MLP_test --tinit $T

#       for gamma in 0 0.01 #0.05 # 0 #0.01 0.05
#       do
##         python -W ignore model/Run.py --dataset $DATASET --syn_seed $SEED --mode 'test' --ACI_test --tinit $T --gamma $gamma
##         python -W ignore model/Run.py --dataset $DATASET --syn_seed $SEED --mode 'test' --ACI_test --tinit $T --gamma $gamma --link_pred --map_dim 9
#         python -W ignore model/Run.py --dataset $DATASET --syn_seed $SEED --mode 'test' --ACI_MLP_test --tinit $T --gamma $gamma --map_dim 9
#         python -W ignore model/Run.py --dataset $DATASET --syn_seed $SEED --mode 'test' --ACI_MLP_test --tinit $T --gamma $gamma --link_pred --map_dim 9
#       done

       done
    done
done


# -- link_pred: add graph similarity measure in score