#!/usr/bin/env bash



for DATASET in "PEMS03_top_12" # "PEMS03_top_12" # "syn_tailup_gen"   #  'syn_gpvar'
  do
    if [ $DATASET == "syn_tailup_gen" ]; then
      syn_seeds=( 30 73) #73
    else
      syn_seeds=( 0 )
    fi

    OUTPUT_FILE="result_csv/grained_lambda_${DATASET}.csv"
    echo "tinit,lambda,gamma,cov_type,picp_mean,eff_mean,eff_std" > $OUTPUT_FILE

    for MODEL in  'A3TGCN' #"DCRNN" # 'AGCRN' 'A3TGCN'
    do
    for SEED in "${syn_seeds[@]}"
    do

#      # Train: Keep using
      python3 -W ignore model/Run.py --model $MODEL --dataset $DATASET --syn_seed $SEED --mode 'train'


#      for TINIT  in 500 # 100 200 300 400 500
#      do
#
#          for COV in  "sphere"
#            do
#              for gamma in  0 0.01
#              do
#                    python3 -W ignore model/Run.py --model $MODEL --dataset $DATASET --syn_seed $SEED --mode 'test' --PCP_ellip_test --gamma $gamma --Cov_type $COV --tinit $TINIT
#              done
#          done
#
#          for COV in "ellip" # "ellip" # "square" # "ellip" # "sphere" "GT"
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
#
#      done



#  for TINIT  in  500 #100 200 300 400 500 #300
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
