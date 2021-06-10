import argparse
import numpy as np
import os
from os.path import join, exists
import sys
import glob
from eval_metrics.metrics import eval_deter
from eval_metrics.metrics_sample import eval_samples
from inference import run_inference
import copy



if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    # ----------- args for load checkpoint of trained models
    parser.add_argument("--search_dir", default="ckpt/pretrained")
    parser.add_argument('--pretrained_dir', default=None, type=str,
                        help='the config dir for trained model; if set as None, will be automatically set based on {ckpt_dir},{dataset},{trial_id}')
    parser.add_argument("--config_filename", default=None, type=str, help="The config file to used; if None, automatically find the latest checkpoint under {config_dir}")
    # ----------- if config_filename and pretrained_dir are not set; please ensure the following setting the same as the 'training.py' in order to evaluate the model on test set.
    parser.add_argument("--rnn_type", type=str, default=None, help="The type of rnn architecture of rnn_flow; if None, following the setting in the config file")
    parser.add_argument("--cost", type=str, default=None, help="The type of loss function (e.g., [mae], [nll]); if None, following the setting of the config file")
    parser.add_argument("--dataset", default="PEMSD8", help="name of datasets")
    parser.add_argument("--ckpt_dir", default="./ckpt", help="the dir to store checkpoints")
    parser.add_argument("--trial_id", type=int, default=123, help="id of the trial. Used as the random seed in multiple trials training")
    # ----------- the args to load data, please keep consistent with 'training.py'
    parser.add_argument("--data_dir", default="./data", help="the dir storing dataset")
    parser.add_argument("--graph_dir", default=None, help="the dir storing the graph information; if None, will be set as the '{data_dir}/sensor_graph'.")
    parser.add_argument("--output_dir", type=str, default="./result", help="The dir to store the output result")
    parser.add_argument('--output_filename', default=None, help="the name of output file; if None, automatically set as p{rnn_type}_{cost}_prediction_{dataset}_trial_{trial_id}.npz ")
    # ----------- 
    parser.add_argument('--gpu_id', default=0, type=int, help='GPU id to use; by default using 0')
    parser.add_argument('--use_cpu_only', action="store_true", help='Add this if want to train in cpu only')
    args = parser.parse_args()

    if not args.use_cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)# the GPU number   
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(-1)# the GPU number

    if args.graph_dir is None:
        args.graph_dir = join(args.data_dir, "sensor_graph")

    args.dataset = args.dataset.lower()

    dirs = glob.glob(args.search_dir + "/*/*/")
    args.pre_set = True


    bad_files = []
    bad_reasons = []
    for d in dirs:
        args.pretrained_dir = d
        config = copy.deepcopy(args)
        print(d)
        try:
            run_inference(config)
        except Exception as e:
            print("\n-----------------------\n {} \n {} \n-----------------------   ".format(d, e))
            bad_files.append(d)
            bad_reasons.append(e)
        
    
    print("\n\n\n\n----------------------- bad_files -----------------------   ")
    if len(bad_files) == 0:
        print("no bad files")
    for i in range(len(bad_files)):
        print(bad_files[i])
        print(bad_reasons[i])
        print("-------")



