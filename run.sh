#!/usr/bin/env bash

for DATASET in "PEMS03_top_12" "PEMS03_w12" # "syn_tailup_gen" 'syn_gpvar'
do
  if [ $DATASET == "syn_tailup_gen" ]; then
    syn_seeds=( 30 73) #73
  else
    syn_seeds=( 0 )
  fi

  for MODEL in 'STGCN' 'STGODE' # 'AGCRN' 'ASTGCN' 'STGCN' 'STGODE' # "MSTGCN"
  do
    # Define the output file inside the MODEL loop so each model gets its own file
    OUTPUT_FILE="result_csv/grained_lambda_${DATASET}_${MODEL}.csv"
    echo "tinit,lambda,gamma,cov_type,picp_mean,eff_mean,eff_std" > $OUTPUT_FILE
<<<<<<< HEAD
    
=======

    for MODEL in  'AGCRN' #"DCRNN" # 'AGCRN' 'A3TGCN'
    do
>>>>>>> origin/main
    for SEED in "${syn_seeds[@]}"
    do
      # # Train the model
      # python3 -W ignore model/Run.py --model $MODEL --dataset $DATASET --syn_seed $SEED --mode 'train'

<<<<<<< HEAD
<<<<<<< HEAD
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
=======
##      # Train: Keep using
#      python3 -W ignore model/Run.py --model $MODEL --dataset $DATASET --syn_seed $SEED --mode 'train'
>>>>>>> origin/main
=======
##      # Train: Keep using
#      python3 -W ignore model/Run.py --model $MODEL --dataset $DATASET --syn_seed $SEED --mode 'train'
>>>>>>> origin/main

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

<<<<<<< HEAD
<<<<<<< HEAD
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
=======
=======
>>>>>>> origin/main
      for TINIT  in 500 # 100 200 300 400 500
      do

          for COV in  "square" # "square" # "sphere" # "GT"
            do
              for gamma in  0 # 0.01 not available for square
              do
                    python3 -W ignore model/Run.py --model $MODEL --dataset $DATASET --syn_seed $SEED --mode 'test' --PCP_ellip_test --gamma $gamma --Cov_type $COV --tinit $TINIT
              done
          done

#          for COV in "ellip" # "ellip" #
#          do
#  #          for LMBD in 0 #0 0.1 0.3 0.5 0.7 1
#            for LMBD in 0 0.5 #`seq 0 0.02 1`
#              do
#              for gamma in 0 0.01 #0.01
#                 do
#                    # Capture the output of the Python script
#                    python3 -W ignore model/Run.py --model $MODEL --dataset $DATASET --syn_seed $SEED --mode 'test' --PCP_ellip_test --lmbd $LMBD --gamma $gamma --Cov_type $COV --tinit $TINIT --weight_type 'fixed' # 'offline'
#                 done
#              done
#          done

      done



#  for TINIT  in 100 200 300 400 500 #300
#  do
#
#    for COV in  "sphere"
#    do
#      for gamma in  0 0.01
#      do
#            OUTPUT=$(python3 -W ignore model/Run.py --model $MODEL --dataset $DATASET --syn_seed $SEED --mode 'test' --PCP_ellip_test --gamma $gamma --Cov_type $COV --tinit $TINIT )
#            echo $OUTPUT >> $OUTPUT_FILE
#      done
#    done
#
#    for COV in  "ellip"
#    do
#    for LMBD in `seq 0 0.1 1`
#      do
#      for gamma in 0 0.01
#         do
#            OUTPUT=$(python3 -W ignore model/Run.py --model $MODEL --dataset $DATASET --syn_seed $SEED --mode 'test' --PCP_ellip_test --lmbd $LMBD --gamma $gamma --Cov_type $COV --tinit $TINIT --weight_type 'offline')
#            echo $OUTPUT >> $OUTPUT_FILE
#         done
#      done
#      done
#
#  done


done
done
done
>>>>>>> origin/main
