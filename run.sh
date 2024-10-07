#!/usr/bin/env bash

for DATASET in 'PEMS03'  # "PEMS03"  # PEMSD8 # PEMSD4 # "PEMS07" (larger one)
  do
  for T in 300 # 500 300 200 100
  do

    # Train
#    python -W ignore model/Run.py --dataset $DATASET --mode 'train'
#
#    # Calib GNN
#    python -W ignore model/Run.py --dataset $DATASET --mode 'calcorrection' --correctionmode 'gnn'
    # Test Baseline: CP
    python -W ignore model/Run.py --dataset $DATASET --mode 'testgnn' --CP_test --tinit $T
    # Test Baseline: ACI
    python -W ignore model/Run.py --dataset $DATASET --mode 'testgnn' --ACI_test --tinit $T
#    # Test: Ours
#    python -W ignore model/Run.py --dataset $DATASET --mode 'testgnn' --ACI_GNN_test --tinit $T

#    # Calib GNN
#    python -W ignore model/Run.py --dataset $DATASET --mode 'calcorrection' --correctionmode 'mlp'
#    # Test Baseline: CP + MLP
#    python -W ignore model/Run.py --dataset $DATASET --mode 'test' --CP_MLP_test --tinit $T
#    # Test Baseline: CP + MLP
#    python -W ignore model/Run.py --dataset $DATASET --mode 'test' --ACI_MLP_test --tinit $T


  done
done