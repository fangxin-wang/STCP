#!/usr/bin/env bash

# Create directories for organization
mkdir -p .tmp_pids
mkdir -p result_csv

# Clear any previous tracking
> .tmp_pids/training_pids.txt

# Define datasets and models - use the same for both training and testing
DATASETS=("PEMS03_top_12" "PEMS03_w12") # "PEMS03_top_12" "PEMS03_w12"
MODELS=("STGCN" "STGODE") # "AGCRN" "ASTGCN" "STGCN" "STGODE" # "MSTGCN"

echo "Starting all training processes in parallel..."
for DATASET in "${DATASETS[@]}"
do
  if [ $DATASET == "syn_tailup_gen" ]; then
    syn_seeds=( 30 73 )
  else
    syn_seeds=( 0 )
  fi

  for MODEL in "${MODELS[@]}"
  do
    # Define the output file for this model/dataset
    OUTPUT_FILE="result_csv/grained_lambda_${DATASET}_${MODEL}.csv"
    echo "tinit,lambda,gamma,cov_type,picp_mean,eff_mean,eff_std" > $OUTPUT_FILE
    
    for SEED in "${syn_seeds[@]}"
    do
      # Start training and save the process ID
      echo "Starting training for $MODEL on $DATASET with seed $SEED"
      python3 -W ignore model/Run.py --model $MODEL --dataset $DATASET --syn_seed $SEED --mode 'train' &
      
      # Store the training PID along with model and dataset info for later reference
      echo "$! $MODEL $DATASET $SEED" >> .tmp_pids/training_pids.txt
    done
  done
done

# Wait for all training processes to complete
echo "Waiting for all training processes to complete..."
wait

# Now run testing with the same models and datasets
echo "Starting testing processes..."
for DATASET in "${DATASETS[@]}"
do
  if [ $DATASET == "syn_tailup_gen" ]; then
    syn_seeds=( 30 73 )
  else
    syn_seeds=( 0 )
  fi

  for MODEL in "${MODELS[@]}"
  do
    # Define the output file (same as above)
    OUTPUT_FILE="result_csv/grained_lambda_${DATASET}_${MODEL}.csv"
    
    for SEED in "${syn_seeds[@]}"
    do
      echo "Testing $MODEL on $DATASET with seed $SEED"
      
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

# Clean up temporary files
rm -rf .tmp_pids
echo "All training and testing complete!"
echo "Processed datasets and models: ${DATASETS[@]}, ${MODELS[@]}."