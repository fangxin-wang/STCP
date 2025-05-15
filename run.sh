#!/usr/bin/env bash

for DATASET in "PEMS03_top_12" "PEMS03_w12" # "syn_tailup_gen" 'syn_gpvar'
do
  if [ $DATASET == "syn_tailup_gen" ]; then
    syn_seeds=( 30 73) #73
  else
    syn_seeds=( 0 )
  fi

  for MODEL in 'AGCRN' 'ASTGCN' 'STGODE' # "MSTGCN"
  do
    # Define the output file inside the MODEL loop so each model gets its own file
    OUTPUT_FILE="result_csv/grained_lambda_${DATASET}_${MODEL}.csv"
    echo "tinit,lambda,gamma,cov_type,picp_mean,eff_mean,eff_std" > $OUTPUT_FILE
    
    for SEED in "${syn_seeds[@]}"
    do
      # # Train the model
      python3 -W ignore model/Run.py --model $MODEL --dataset $DATASET --syn_seed $SEED --mode 'train'

    #   # Test with different parameters
      for TINIT in 100 200 300 400 500 
      do
        # Test with spherical covariance
        for COV in "sphere"
        do
          for gamma in 0 0.01
          do
            OUTPUT=$(python3 -W ignore model/Run.py --model $MODEL --dataset $DATASET --syn_seed $SEED --mode 'test' --PCP_ellip_test --gamma $gamma --Cov_type $COV --tinit $TINIT)
            echo $OUTPUT >> $OUTPUT_FILE
          done
        done

        # Test with elliptical covariance
        for COV in "ellip"
        do
          for LMBD in `seq 0 0.1 1`
          do
            for gamma in 0 0.01
            do
              OUTPUT=$(python3 -W ignore model/Run.py --model $MODEL --dataset $DATASET --syn_seed $SEED --mode 'test' --PCP_ellip_test --lmbd $LMBD --gamma $gamma --Cov_type $COV --tinit $TINIT --weight_type 'offline')
              echo $OUTPUT >> $OUTPUT_FILE
            done
          done
        done

        # Test with square covariance
        for COV in "square"
        do
          for gamma in 0 # 0.01 not available for square
          do
            OUTPUT=$(python3 -W ignore model/Run.py --model $MODEL --dataset $DATASET --syn_seed $SEED --mode 'test' --PCP_ellip_test --gamma $gamma --Cov_type $COV --tinit $TINIT)
            echo $OUTPUT >> $OUTPUT_FILE
          done
        done
      done
    done
  done
done