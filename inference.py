import argparse
import numpy as np
import os
from os.path import join, exists
import sys
import glob
from eval_metrics.metrics import eval_deter
from eval_metrics.metrics_sample import eval_samples
import re

if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    # ----------- args for load checkpoint of trained models
    parser.add_argument('--pretrained_dir', default=None, type=str,
                        help='the config dir for trained model; if set as None, will be automatically set based on {ckpt_dir},{dataset},{trial_id}')
    parser.add_argument("--config_filename", default=None, type=str, help="The config file to used; if None, automatically find the latest checkpoint under {config_dir}")
    # ----------- if config_filename and pretrained_dir are not set; please ensure the following setting the same as the 'training.py' in order to evaluate the model on test set.
    parser.add_argument("--rnn_type", type=str, default=None, help="The type of rnn architecture of rnn_flow; if None, following the setting in the config file")
    parser.add_argument("--cost", type=str, default=None, help="The type of loss function (e.g., [mae], [nll]); if None, following the setting of the config file")
    parser.add_argument("--dataset", default="PEMSD8", help="name of datasets")
    parser.add_argument("--ckpt_dir", default="./ckpt", help="the dir to store checkpoints")
    parser.add_argument("--trial_id", type=int, default=1, help="id of the trial. Used as the random seed in multiple trials training")
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


    if args.pretrained_dir is None:
        args.pre_set = False
        if args.rnn_type is None or args.cost is None:
            raise NotImplementedError("[pretrained_dir] and [rnn_type & cost] cannot be None at the same time")
        args.pretrained_dir = join(args.ckpt_dir, args.dataset, args.rnn_type + "_" + args.cost, "trial_{}".format(args.trial_id))
    else:
        args.pre_set = True

import tensorflow as tf
import yaml, os

from lib.utils import load_graph_data
from model.prnn_supervisor import PRNNSupervisor

def run_inference(args):
    tf.reset_default_graph()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)


    
    if args.config_filename is None:
        args.config_file = config_finder(args.pretrained_dir)
    else:
        args.config_file = join(args.pretrained_dir, args.config_filename)



    with open(args.config_file) as f:
        config = yaml.load(f)
    

    tf_config = tf.ConfigProto()
    if args.use_cpu_only:
        tf_config = tf.ConfigProto(device_count={'GPU': 0})
    tf_config.gpu_options.allow_growth = True

    args.cost = config["train"]["cost"]
    args.rnn_type = config["model"]["rnn_type"]
    try:
        dataset = config["data"]["dataset_name"]
        args.dataset = dataset
    except Exception:
        print("")


    graph_pkl_filename = join(args.graph_dir, "adj_" + args.dataset + ".pkl")
    config['data']["graph_pkl_filename"] = graph_pkl_filename

    adj_mx = load_graph_data(graph_pkl_filename).astype('float32')


    if args.output_filename is None:
        args.output_filename = get_output_filename(config, args)

    args.output_filename = join(args.output_dir, args.output_filename)



    print("Start evaluation on Test set")
    with tf.Session(config=tf_config) as sess:
        supervisor = PRNNSupervisor(adj_mx=adj_mx, args=args, inference=True, pretrained_dir=args.pretrained_dir, **config)
        supervisor.load(sess, config['train']['model_filename'])
        outputs = supervisor.evaluate(sess)
        np.savez_compressed(args.output_filename, **outputs)
        print('Predictions saved as {}.'.format(args.output_filename))
    
    deter_metrics = eval_deter(args)
    sample_metrics = eval_samples(args)
    deter_lines = "\n".join(deter_metrics)
    sample_lines = "\n".join(sample_metrics)
    lines = "Rnn_type: {}   Cost: {}   Dataset: {}\n".format(args.rnn_type, args.cost, args.dataset)
    lines = lines + "\n-------------------------------\n" + deter_lines
    lines = lines + "\n-------------------------------\n" + sample_lines

    with open(args.output_filename.replace(".npz", "_result.txt"), "w") as f:
        f.writelines(lines)

    

def config_finder(path):
    files = glob.glob(join(path, "**.yaml"))
    # print(files)
    if len(files) == 0:
        raise NotImplementedError("cannot find checkpoint files")

    index = [int(i.replace(join(path, "config_"), "").replace(".yaml","")) for i in files]
    max_id = np.argmax(index)
    config_file = files[max_id]
    return config_file

def get_output_filename(config, args):
    rnn_type = config["model"]["rnn_type"]
    loss_type = config["train"]["cost"]
    dataset = args.dataset

    if args.pre_set:
        pre_set_dir = args.pretrained_dir.replace("/","_").replace("\\", "_").replace(".","").replace("ckpt","")
        pre_set_dir = re.sub("__*", "_", pre_set_dir)
        filename = "{}_{}_predictions_{}_{}.npz".format(rnn_type, loss_type, dataset, pre_set_dir)
    else:
        filename = "{}_{}_predictions_{}_trial_{}.npz".format(rnn_type, loss_type, dataset, args.trial_id)

    return filename




if __name__ == '__main__':
    sys.path.append(os.getcwd())

    run_inference(args)


