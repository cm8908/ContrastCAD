import os
from utils import ensure_dirs
import argparse
import json
import shutil
from cadlib.macro import *


class ConfigCL(object):
    def __init__(self, phase):
        self.is_train = phase == "train"

        self.set_configuration()

        # init hyperparameters and parse from command-line
        parser, args = self.parse()
        self.args = args

        # set as attributes
        print("----Experiment Configuration-----")
        for k, v in args.__dict__.items():
            print("{0:20}".format(k), v)
            self.__setattr__(k, v)

        # experiment paths
        self.exp_dir = os.path.join(self.proj_dir, self.exp_name)
        if phase == "train" and args.cont is not True and os.path.exists(self.exp_dir):
            response = input(f'Experiment {self.exp_dir} already exists, overwrite? (y/n) ')
            if response != 'y':
                exit()
            shutil.rmtree(self.exp_dir)

        self.log_dir = os.path.join(self.exp_dir, 'log')
        self.model_dir = os.path.join(self.exp_dir, 'model')
        ensure_dirs([self.log_dir, self.model_dir])

        # GPU usage
        if args.gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)

        # create soft link to experiment log directory
        # if not os.path.exists('train_log'):
            # os.symlink(self.exp_dir, 'train_log')

        # save this configuration
        if self.is_train:
            with open('{}/config.txt'.format(self.exp_dir), 'w') as f:
                json.dump(args.__dict__, f, indent=2)

    def set_configuration(self):
        self.args_dim = ARGS_DIM # 256
        self.n_args = N_ARGS
        self.n_commands = len(ALL_COMMANDS)  # line, arc, circle, EOS, SOS

        self.n_layers = 4                # Number of Encoder blocks
        self.n_layers_decode = 4         # Number of Decoder blocks
        self.n_heads = 8                 # Transformer config: number of heads
        self.dim_feedforward = 512       # Transformer config: FF dimensionality
        self.d_model = 256               # Transformer config: model dimensionality
        self.dropout = 0.1                # Dropout rate used in basic layers and Transformers
        self.dim_z = 256                 # Latent vector dimensionality
        self.use_group_emb = True

        self.max_n_ext = MAX_N_EXT
        self.max_n_loops = MAX_N_LOOPS
        self.max_n_curves = MAX_N_CURVES

        self.max_num_groups = 30
        self.max_total_len = MAX_TOTAL_LEN

        self.loss_weights = {
            "loss_cmd_weight": 1.0,
            "loss_args_weight": 2.0,
            "loss_cl_weight": 2.0,
        }

        self.keep_seq_len = False


    def parse(self):
        """initiaize argument parser. Define default hyperparameters and collect from command-line arguments."""
        parser = argparse.ArgumentParser()
        parser.add_argument('--random_seed', type=int, default=2023)
        parser.add_argument('--proj_dir', type=str, default="proj_log", help="path to project folder where models and logs will be saved")
        parser.add_argument('--data_root', type=str, default="../datasets/cad_data/", help="path to source data folder")
        parser.add_argument('--cmd_weight', type=float, default=1.0)
        parser.add_argument('--args_weight', type=float, default=2.0)
        parser.add_argument('--exp_name', type=str, default=os.getcwd().split('/')[-1], help="name of this experiment")
        parser.add_argument('-g', '--gpu_ids', type=str, default='3', help="gpu to use, e.g. 0  0,1,2. CPU not supported.")

        parser.add_argument('--batch_size', type=int, default=1024, help="batch size")
        parser.add_argument('--num_workers', type=int, default=8, help="number of workers for data loading")

        parser.add_argument('--nr_epochs', type=int, default=1000, help="total number of epochs to train")
        parser.add_argument('--lr', type=float, default=1e-3, help="initial learning rate")
        parser.add_argument('--grad_clip', type=float, default=1.0, help="initial learning rate")
        parser.add_argument('--temperature', type=float, default=0.07, help="initial learning rate")
        parser.add_argument('--warmup_step', type=int, default=2000, help="step size for learning rate warm up")
        parser.add_argument('--continue', dest='cont',  action='store_true', help="continue training from checkpoint")
        parser.add_argument('--ckpt', type=str, default='latest', required=False, help="desired checkpoint to restore")
        parser.add_argument('--vis', action='store_true', default=False, help="visualize output in training")
        parser.add_argument('--save_frequency', type=int, default=500, help="save models every x epochs")
        parser.add_argument('--val_frequency', type=int, default=10, help="run validation every x iterations")
        parser.add_argument('--vis_frequency', type=int, default=2000, help="visualize output every x iterations")
        parser.add_argument('--augment', action='store_true', default=False, help="use random data augmentation")
        parser.add_argument('--dataset_augment_type', type=str, default='default', choices=['default', 're-extrude', 'replace', 'redraw', 'corrupt2', 'replace_arc', 'replace_p_arc', 'replace_arc2', 'replace_p_arc2', 'arc', 'arc2', 'rre'])
        parser.add_argument('--dataset_augment_prob', type=float, default=0.5)

        parser.add_argument('--fp32', dest='fp16', action='store_false', default=True)
        parser.add_argument('--latent_dropout', type=float, default=0.3)
        parser.add_argument('--cl_loss', type=str, default='simclr', choices=['infonce', 'simclr'])
        parser.add_argument('--n_phead_layers', type=int, default=1)
        parser.add_argument('--phead_type', type=str, default='multilayer', choices=['multilayer', 'legacy'])
        if not self.is_train:
            parser.add_argument('-m', '--mode', type=str, choices=['rec', 'enc', 'dec'])
            parser.add_argument('-o', '--outputs', type=str, default=None)
            parser.add_argument('--z_path', type=str, default=None)
            parser.add_argument('--tag', type=str, default=None)
            parser.add_argument('--use_ext_mask', action='store_true')
            parser.add_argument('--force_close_loop', action='store_true')
        
        args = parser.parse_args()
        return parser, args

