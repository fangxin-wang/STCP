#!/usr/bin/env bash

OUTPUT_FILE="PEMS03_no_update_T_500.csv"
echo "dataset,syn_seed,tinit,lambda,gamma,cov_type,picp_mean,eff_mean,eff_std" > $OUTPUT_FILE

<<<<<<< HEAD
for DATASET in "PEMS03_top_12" # "PEMS03_top_12" # "syn_tailup_gen"   #  'syn_gpvar'
  do
    if [ $DATASET == "syn_tailup_gen" ]; then
      syn_seeds=( 30 31 32 33 34 35 36 37 38 39) #30 73 # 70 71 72 73 74 75 76 77 78 79 # 30 31 32 33 34 35 36 37 38 39
=======
echo "DATASET,SEED,LMBD,gamma,COV,metric1,metric2" > $OUTPUT_FILE

for DATASET in  "PEMS03"   #  'syn_gpvar' # "PEMS07" (larger one) #"PEMS03" #"PEMS08" "PEMS04"
  do
    if [ $DATASET == "syn_tailup" ]; then
      syn_seeds=(  10 )
>>>>>>> parent of 189c765 (gen_syn_tailup)
    else
      syn_seeds=( 0 )
    fi

    for SEED in "${syn_seeds[@]}"
    do
#      OUTPUT_FILE="grained_lambda_${DATASET}_seed${SEED}.csv"
#      echo "dataset,syn_seed,tinit,lambda,gamma,cov_type,picp_mean,eff_mean,eff_std" > $OUTPUT_FILE

#        ## Train: Keep using
#        python3 -W ignore model/Run.py --dataset $DATASET --syn_seed $SEED --mode 'train'
#         ## Calib
#         python -W ignore model/Run.py  --dataset $DATASET --syn_seed $SEED --mode 'calcorrection' --correctionmode 'mlp'
#         ## Test Baseline: CP
#         python -W ignore model/Run.py --dataset $DATASET --syn_seed $SEED --mode 'test' --CP_MLP_test --tinit $T

#    for TINIT  in 500 # 100 200 300 400 500
#    do
#        for COV in "ellip" # "ellip" # "square" # "ellip" # "sphere" "GT"
#        do
##          for LMBD in 0 #0 0.1 0.3 0.5 0.7 1
#          for LMBD in 0.5 #`seq 0 0.02 1`
#            do
#            for gamma in 0.01 #0.01
#               do
#                  # Capture the output of the Python script
#                  python3 -W ignore model/Run.py --dataset $DATASET --syn_seed $SEED --mode 'test' --PCP_ellip_test --lmbd $LMBD --gamma $gamma --Cov_type $COV --tinit $TINIT --weight_type 'fixed' # 'offline'
#               done
#            done
#        done
#    done

  for TINIT  in 100 200 300 400 500
  do

#          for COV in  "GT" "square" "sphere"
#          do
#            for gamma in  0 0.01
#            do
#                  OUTPUT=$(python3 -W ignore model/Run.py --dataset $DATASET --syn_seed $SEED --mode 'test' --PCP_ellip_test --gamma $gamma --Cov_type $COV --tinit $TINIT )
#                  echo $OUTPUT >> $OUTPUT_FILE
#            done
#          done

          for COV in  "ellip"
          do
          for LMBD in 0 0.5 #`seq 0 0.02 1`
            do
            for gamma in 0 0.01
               do
                  OUTPUT=$(python3 -W ignore model/Run.py --dataset $DATASET --syn_seed $SEED --mode 'test' --PCP_ellip_test --lmbd $LMBD --gamma $gamma --Cov_type $COV --tinit $TINIT --no_update)
                  echo $OUTPUT >> $OUTPUT_FILE
               done
            done
          done

  done


done
done

