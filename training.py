from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
from os.path import join, exists
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="PEMSD8", help="name of datasets")
    parser.add_argument("--data_dir", default="./data", help="the dir storing dataset")
    parser.add_argument("--ckpt_dir", default="./ckpt", help="the dir to store checkpoints")
    parser.add_argument("--graph_dir", default=None, help="the dir storing the graph information; if None, will be set as the '{data_dir}/sensor_graph'.")
    parser.add_argument('--config_dir', default=None, type=str,
                        help="The dir storing the detailed configuration of model (including hyperparameters); if None, will be set as the '{data_dir}/model_config'.")
    parser.add_argument('--gpu_id', default=0, type=int, help='GPU id to use; by default using 0')
    parser.add_argument('--use_cpu_only', action="store_true", help='Add this if want to train in cpu only')
    parser.add_argument("--trial_id", type=int, default=1, help="id of the trial. Used as the random seed in multiple trials training")
    # -------------- setting of the models: 
    #                      - recommend to set this two hyperparmeters here; which is essential to find the saved checkpoitns in 'inference.py'
    parser.add_argument("--rnn_type", type=str, default=None, help="The type of rnn architecture of rnn_flow; if None, following the setting in the config file")
    parser.add_argument("--cost", type=str, default=None, help="The type of loss function (e.g., [mae], [nll]); if None, following the setting of the config file")
    args = parser.parse_args()

    if args.graph_dir is None:
        args.graph_dir = join(args.data_dir, "sensor_graph")

    if args.config_dir is None:
        args.config_dir = join(args.data_dir, "model_config")


    if args.use_cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(-1)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    if args.graph_dir is None:
        args.graph_dir = join(args.data_dir, "sensor_graph")

    args.dataset = args.dataset.lower()




import tensorflow.compat.v1 as tf
import yaml
from scipy.sparse import csr_matrix
from lib.utils import load_graph_data
from model.prnn_supervisor import PRNNSupervisor
import inference


def main(args):
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    # if not tf.

    args.config_filename = join(args.config_dir, "prnn_" + args.dataset + ".yaml")
    with open(args.config_filename) as f:

        supervisor_config = yaml.load(f)

        if args.rnn_type is not None:
            supervisor_config["model"]["rnn_type"] = args.rnn_type
        if args.cost is not None:
            supervisor_config["train"]["cost"] = args.cost


        graph_pkl_filename = join(args.graph_dir, "adj_" + args.dataset + ".pkl")
        supervisor_config['data']["graph_pkl_filename"] = graph_pkl_filename

        adj_mx = load_graph_data(graph_pkl_filename).astype('float32')
        
        tf_config = tf.ConfigProto()
        if args.use_cpu_only:
            tf_config = tf.ConfigProto(device_count={'GPU': 0})
        else:
            tf_config = tf.ConfigProto(device_count={'GPU': 1})

        tf_config.gpu_options.allow_growth = True

        with tf.Session(config=tf_config) as sess:
            supervisor = PRNNSupervisor(adj_mx=adj_mx, args=args, inference=False, pretrained_dir=None, **supervisor_config)
            args.pretrained_dir = supervisor._log_dir
            supervisor.train(sess=sess)


    print("the checkpoint files are saved in :", args.pretrained_dir)


if __name__ == '__main__':
    main(args)
