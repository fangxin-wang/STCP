#!/usr/bin/env bash

OUTPUT_FILE="results.csv"

echo "DATASET,SEED,LMBD,gamma,COV,metric1,metric2" > $OUTPUT_FILE

for DATASET in  "PEMS03"   #  'syn_gpvar' # "PEMS07" (larger one) #"PEMS03" #"PEMS08" "PEMS04"
  do
    if [ $DATASET == "syn_tailup" ]; then
      syn_seeds=(  10 )
    else
      syn_seeds=( 0 )
    fi

    for SEED in "${syn_seeds[@]}"
    do

#        # Train: Keep using
#        python3 -W ignore model/Run.py --dataset $DATASET --syn_seed $SEED --mode 'train'
#        # Calib
#         python -W ignore model/Run.py  --dataset $DATASET --syn_seed $SEED --mode 'calcorrection' --correctionmode 'mlp'
#        # Test Baseline: CP
#        python -W ignore model/Run.py --dataset $DATASET --syn_seed $SEED --mode 'test' --CP_MLP_test --tinit $T

    for COV in  "ellip" "sphere" #"ellip" "sphere" "GT"
    do
    for LMBD in 0 #0.3 #0.5 0.8 1
      do
      for gamma in 0 0.01 #0 0.01 0.05
         do
            # Capture the output of the Python script
            python3 -W ignore model/Run.py --dataset $DATASET --syn_seed $SEED --mode 'test' --PCP_ellip_test --lmbd $LMBD --gamma $gamma --Cov_type $COV
         done
      done
    done

#  for COV in  "GT" "sphere"
#  do
#    for gamma in 0 0.01 0.05
#    do
#          OUTPUT=$(python3 -W ignore model/Run.py --dataset $DATASET --syn_seed $SEED --mode 'test' --PCP_ellip_test --gamma $gamma --Cov_type $COV)
#          echo $OUTPUT >> $OUTPUT_FILE
#    done
#  done
#
#  for COV in  "ellip"
#  do
#  for LMBD in 0 0.1 0.2 0.3 0.4 0.5 0.8 1
#    do
#    for gamma in 0 0.01 0.05
#       do
#          OUTPUT=$(python3 -W ignore model/Run.py --dataset $DATASET --syn_seed $SEED --mode 'test' --PCP_ellip_test --lmbd $LMBD --gamma $gamma --Cov_type $COV)
#          echo $OUTPUT >> $OUTPUT_FILE
#       done
#    done
#    done

    done
done


# -- link_pred: add graph similarity measure in score

#       for gamma in 0 0.01 #0.05 # 0 #0.01 0.05
#       do
##         python -W ignore model/Run.py --dataset $DATASET --syn_seed $SEED --mode 'test' --ACI_test --tinit $T --gamma $gamma
##         python -W ignore model/Run.py --dataset $DATASET --syn_seed $SEED --mode 'test' --ACI_test --tinit $T --gamma $gamma --link_pred --map_dim 9
#         python -W ignore model/Run.py --dataset $DATASET --syn_seed $SEED --mode 'test' --ACI_MLP_test --tinit $T --gamma $gamma --map_dim 9
#         python -W ignore model/Run.py --dataset $DATASET --syn_seed $SEED --mode 'test' --ACI_MLP_test --tinit $T --gamma $gamma --link_pred --map_dim 9
#       done

