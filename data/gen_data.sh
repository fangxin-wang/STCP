#!/usr/bin/env bash

#python3 -W ignore data/gen_syn_data.py --syn_seed 1107  --ar --ma --corr --node_num 5
#
#python3 -W ignore data/gen_syn_data.py --syn_seed 1108  --ar --ma --corr --node_num 10


#python3 -W ignore data/gen_syn_graph.py --syn_seed 1110  --node_num 5

#python3 -W ignore data/gen_syn_graph.py --syn_seed 1110  --node_num 5
#python3 -W ignore data/gen_syn_graph.py --syn_seed 1111  --node_num 10

python3 -W ignore data/gen_syn_graph.py --syn_seed 1212  --node_num 5
python3 -W ignore data/gen_syn_graph.py --syn_seed 1213  --node_num 10

python3 -W ignore data/gen_syn_graph_tailup.py