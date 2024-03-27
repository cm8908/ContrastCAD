import os
from trainer.finetuner import CLASSIFIERS
from utils import ensure_dirs
import argparse
import json
import shutil
from cadlib.macro import *
from dataset.augmentations import AUG2FN


class ConfigCL(object):
    def __init__(self, phase):
        self.is_finetune = phase == 'finetune'
        self.is_train = phase == "train"
        self.is_test = phase == 'test'

        self.set_configuration()

        # init hyperparameters and parse from command-line
        parser, args = self.parse()
        
        # args.proj_dir = 'debugs'
        # args.gpu_ids = '1'
        # args.keep_seq_len = True

        # set as attributes
        print("----Experiment Configuration-----")
        for k, v in args.__dict__.items():
            print("{0:20}".format(k), v)
            self.__setattr__(k, v)

        # experiment paths
        self.exp_dir = os.path.join(self.proj_dir, self.exp_name)
        if phase == "train" and args.cont is not True and os.path.exists(self.exp_dir) and self.exp_name != 'debug':
            response = input(f'Experiment {self.exp_dir} already exists, overwrite? (y/n) ')
            if response != 'y':
                exit()
            shutil.rmtree(self.exp_dir)

        self.log_dir = os.path.join(self.exp_dir, 'log')
        self.model_dir = os.path.join(self.exp_dir, 'model')
        ensure_dirs([self.log_dir, self.model_dir])
        if self.is_test:
            self.result_dir = os.path.join(self.exp_dir, 'results')
            ensure_dirs(self.result_dir)

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
        
        self.dim_phead = self.d_model

        self.max_n_ext = MAX_N_EXT
        self.max_n_loops = MAX_N_LOOPS
        self.max_n_curves = MAX_N_CURVES

        self.max_num_groups = 30
        # self.max_total_len = MAX_TOTAL_LEN

        self.loss_weights = {
            "loss_cmd_weight": 1.0,
            "loss_args_weight": 2.0
        }

        self.num_code = 500


    def parse(self):
        """initiaize argument parser. Define default hyperparameters and collect from command-line arguments."""
        parser = argparse.ArgumentParser()
        parser.add_argument('--max_total_len', type=int, default=MAX_TOTAL_LEN)

        parser.add_argument('--proj_dir', type=str, default="proj_log", help="path to project folder where models and logs will be saved")
        parser.add_argument('--data_root', type=str, default="../datasets/cad_data", help="path to source data folder")
        parser.add_argument('--exp_name', type=str, default=os.getcwd().split('/')[-1], help="name of this experiment")
        parser.add_argument('-g', '--gpu_ids', type=str, default='0', help="gpu to use, e.g. 0  0,1,2. CPU not supported.")
        parser.add_argument('--use_clean_dataset', action='store_true')

        parser.add_argument('--batch_size', type=int, default=512, help="batch size")
        parser.add_argument('--num_workers', type=int, default=8, help="number of workers for data loading")

        parser.add_argument('--nr_epochs', type=int, default=100, help="total number of epochs to train")
        parser.add_argument('--lr', type=float, default=1e-3, help="initial learning rate")
        parser.add_argument('--grad_clip', type=float, default=1.0, help="initial learning rate")
        parser.add_argument('--warmup_step', type=int, default=2000, help="step size for learning rate warm up")
        parser.add_argument('--continue', dest='cont',  action='store_true', help="continue training from checkpoint")
        parser.add_argument('--ckpt', type=str, default='latest', required=False, help="desired checkpoint to restore")
        parser.add_argument('--vis', action='store_true', default=False, help="visualize output in training")
        parser.add_argument('--save_frequency', type=int, default=500, help="save models every x epochs")
        parser.add_argument('--val_frequency', type=int, default=10, help="run validation every x iterations")
        parser.add_argument('--vis_frequency', type=int, default=2000, help="visualize output every x iterations")
        parser.add_argument('--augment', action='store_true', help="use random data augmentation")

        parser.add_argument('--fp32', dest='fp16', action='store_false', default=True)

        parser.add_argument('--keep_seq_len', action='store_true', help="keep z variable's length as original sequence length. (1 if set False)")
        # parser.add_argument('--dim_phead', type=int, default=512)
        parser.add_argument('--temperature', type=float, default=0.07)
        parser.add_argument('--phead_activation', type=str, choices=ACT2FN.keys(), default='relu')
        parser.add_argument('--cl_augment1', type=str, choices=AUG2FN.keys(), default='identity')
        parser.add_argument('--cl_augment2', type=str, choices=AUG2FN.keys(), default='identity')

        parser.add_argument('--corruption_prob', type=float, default=None)  # For corrupt_v3
        parser.add_argument('--deletion_prob', type=float, default=None)  # For deletion
        parser.add_argument('--insertion_prob', type=float, default=None)  # For insertion

        parser.add_argument('--add_recon_loss', action='store_true')
        parser.add_argument('--no_phead', action='store_true')
        parser.add_argument('--lamda', type=float, default=1., help='weight for contrastive loss')
        parser.add_argument('--kappa', type=float, default=1., help='weight for reconstructive loss')
        parser.add_argument('-t', '--tag', type=str, default=None)

        parser.add_argument('--cl_loss', type=str, choices=['simclr', 'supcon'], default='simclr')
        parser.add_argument('--base_temperature', type=float, default=0.07)
        parser.add_argument('--contrast_mode', type=str, choices=['one', 'all'], default='all')

        parser.add_argument('--augment_method', type=str, choices=['manual', 'randaug', 'randaug_oneside', 'scale_transform',
                                                                   'random_transform', 'random_flip', 'flip_sketch'], default='manual')
        parser.add_argument('--n_augment', dest='n_augment', type=int, default=None)  # For RandAug
        parser.add_argument('--m_augment', dest='m_augment', type=int, default=None)  # For RandAug

        parser.add_argument('--encoder_type', type=str, default='DeepCAD', choices=['DeepCAD', 'SkexGen'])  # 
        # parser.add_argument('--fp16', action='store_true')

        parser.add_argument('--no_scale_transform_seq', dest='scale_transform_seq', action='store_false', default=True)
        parser.add_argument('--no_scale_transform_profile', dest='scale_transform_profile', action='store_false', default=True)
        parser.add_argument('--scale_factor', type=float, default=None)
        parser.add_argument('--flip_axis', type=str, default='xy', choices=['x', 'y', 'xy'])
        
        # if not self.is_train:
        #     parser.add_argument('-m', '--mode', type=str, choices=['rec', 'enc', 'dec'])
        #     parser.add_argument('-o', '--outputs', type=str, default=None)
        #     parser.add_argument('--z_path', type=str, default=None)
        if self.is_test:
            parser.add_argument('-m', '--mode', type=str, choices=['rec', 'enc', 'dec'])
            parser.add_argument('-o', '--outputs', type=str, default=None)
            parser.add_argument('--z_path', type=str, default=None)
        if self.is_finetune:
            parser.add_argument('--ft_lr', type=float, default=1e-4)
            parser.add_argument('--ft_nr_epochs', type=int, default=100)
        if self.is_finetune or self.is_test:
            parser.add_argument('--classifier_type', type=str, choices=list(CLASSIFIERS.keys()), default='transformer_decoder')
        
        args = parser.parse_args()
        return parser, args
