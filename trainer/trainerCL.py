import torch
import torch.optim as optim
from tqdm import tqdm
from model.cl_dropout_autoencoder import CADContrastiveDropoutTransformer
from .base import BaseTrainer
from .cl_loss import *
from .scheduler import GradualWarmupScheduler
from cadlib.macro import *
from torch.cuda.amp import GradScaler


class TrainerCL(BaseTrainer):
    def build_net(self, cfg):
        self.net = CADContrastiveDropoutTransformer(cfg).cuda()

    def set_optimizer(self, cfg):
        """set optimizer and lr scheduler used in training"""
        self.optimizer = optim.Adam(self.net.parameters(), cfg.lr)
        self.scheduler = GradualWarmupScheduler(self.optimizer, 1.0, cfg.warmup_step)
        # self.scheduler = None
        self.scaler = GradScaler(enabled=self.cfg.fp16)

    def set_loss_function(self):
        self.device = 'cuda'
        self.loss_func = CADContrastiveLoss(self.cfg, self.device, self.cfg.batch_size, self.cfg.temperature).to(self.device)

    def forward(self, data):
        commands = data['command'].cuda() # (N, S)
        args = data['args'].cuda()  # (N, S, N_ARGS)
        # commands2 = data['command1'].cuda() # (N, S)
        # args2 = data['args1'].cuda()  # (N, S, N_ARGS)
        
        # print("commands.shape: ", commands.shape)
        # print("args.shape: ", args.shape)
        # print("commands2.shape: ", commands2.shape)
        # print("args2.shape: ", args2.shape)

        outputs = self.net(commands, args)
        # outputs = self.net(commands, args, commands2.detach(), args2.detach())
        loss_dict = self.loss_func(outputs)

        return outputs, loss_dict

    def encode(self, data, is_batch=False):
        """encode into latent vectors"""
        commands = data['command'].cuda()
        args = data['args'].cuda()
        if not is_batch:
            commands = commands.unsqueeze(0)
            args = args.unsqueeze(0)
        z = self.net(commands, args, encode_mode=True)
        return z

    def decode(self, z):
        """decode given latent vectors"""
        outputs = self.net(None, None, z=z, return_tgt=False)
        return outputs

    def logits2vec(self, outputs, refill_pad=True, to_numpy=True, use_mask=False, force_close_loop=False):
        """network outputs (logits) to final CAD vector"""
        out_command = torch.argmax(torch.softmax(outputs['command_logits'], dim=-1), dim=-1)  # (N, S)
        if use_mask:
            end_mask = torch.zeros(*out_command.shape).bool().to(out_command.device)
            for b in range(out_command.shape[0]):
                if EOS_IDX in out_command[b]:
                    end_idx = out_command[b].cpu().numpy().tolist().index(EOS_IDX)
                    end_mask[b, end_idx] = True
                else:
                    end_mask[b, 0] = True
            last_token_mask = torch.roll(end_mask, shifts=-1, dims=1)
            without_ext_indices = out_command[last_token_mask] != EXT_IDX
            mask = end_mask * without_ext_indices[:,None]
            mask[:,2] *= False
            out_command = torch.masked_fill(out_command, mask, EXT_IDX)
        out_args = torch.argmax(torch.softmax(outputs['args_logits'], dim=-1), dim=-1) - 1  # (N, S, N_ARGS)
        if force_close_loop:
            vec_cat = torch.cat([out_command.unsqueeze(-1), out_args], dim=-1)
            out_vecs = []
            for b in range(vec_cat.shape[0]):
                vec_ = vec_cat[b].cpu().numpy()
                if EOS_IDX in vec_[:,0]:
                    end_idx = vec_[:,0].tolist().index(EOS_IDX)
                    vec_ = vec_[:end_idx]
                sol_indices = np.where(vec_[:,0] == SOL_IDX)[0]
                split_vecs = np.split(vec_, sol_indices)[1:]
                out_vector = []
                for vec in split_vecs:
                    if vec.shape[0] < 2:
                        continue
                    last_idx = -1 if vec[-1,0] != EXT_IDX else -2
                    if CIRCLE_IDX not in vec:
                        if vec[1,1] != vec[last_idx,1] and vec[1,2] != vec[last_idx,2]:
                            d_x = abs(vec[1,1] - vec[last_idx,1])
                            d_y = abs(vec[1,2] - vec[last_idx,2])
                            idx_to_update = np.argmin([d_x,d_y]) + 1
                            vec[last_idx, idx_to_update] = vec[1, idx_to_update]
                    out_vector.append(vec)
                out_vector = np.concatenate(out_vector)
                # add EOS VEC
                pad_len = MAX_TOTAL_LEN - out_vector.shape[0]
                out_vector = np.concatenate([out_vector, EOS_VEC[None].repeat(pad_len, axis=0)], axis=0)
                out_vecs.append(out_vector)
            out_vecs = np.stack(out_vecs)
            out_vecs = torch.LongTensor(out_vecs).cuda()
            out_command, out_args = out_vecs[:,:,0], out_vecs[:,:,1:]
            
        if refill_pad: # fill all unused element to -1
            mask = ~torch.tensor(CMD_ARGS_MASK).bool().cuda()[out_command.long()]
            out_args[mask] = -1

        out_cad_vec = torch.cat([out_command.unsqueeze(-1), out_args], dim=-1)
        if to_numpy:
            out_cad_vec = out_cad_vec.detach().cpu().numpy()
        return out_cad_vec

    def evaluate(self, test_loader):
        """evaluatinon during training"""
        self.net.eval()
        pbar = tqdm(test_loader)
        pbar.set_description("EVALUATE[{}]".format(self.clock.epoch))

        all_ext_args_comp = []
        all_line_args_comp = []
        all_arc_args_comp = []
        all_circle_args_comp = []

        for i, data in enumerate(pbar):
            with torch.no_grad():
                commands = data['command'].cuda() # (N, S)
                args = data['args'].cuda()  # (N, S, N_ARGS)
                outputs = self.net(commands, args)
                out_args = torch.argmax(torch.softmax(outputs['args_logits'], dim=-1), dim=-1) - 1
                out_args = out_args.long().detach().cpu().numpy()  # (N, S, n_args)

            gt_commands = commands.squeeze(1).long().detach().cpu().numpy() # (N, S)
            gt_args = args.squeeze(1).long().detach().cpu().numpy() # (N, S, n_args)

            ext_pos = np.where(gt_commands == EXT_IDX)
            line_pos = np.where(gt_commands == LINE_IDX)
            arc_pos = np.where(gt_commands == ARC_IDX)
            circle_pos = np.where(gt_commands == CIRCLE_IDX)

            args_comp = (gt_args == out_args).astype(np.int)
            all_ext_args_comp.append(args_comp[ext_pos][:, -N_ARGS_EXT:])
            all_line_args_comp.append(args_comp[line_pos][:, :2])
            all_arc_args_comp.append(args_comp[arc_pos][:, :4])
            all_circle_args_comp.append(args_comp[circle_pos][:, [0, 1, 4]])

        all_ext_args_comp = np.concatenate(all_ext_args_comp, axis=0)
        sket_plane_acc = np.mean(all_ext_args_comp[:, :N_ARGS_PLANE])
        sket_trans_acc = np.mean(all_ext_args_comp[:, N_ARGS_PLANE:N_ARGS_PLANE+N_ARGS_TRANS])
        extent_one_acc = np.mean(all_ext_args_comp[:, -N_ARGS_EXT_PARAM])
        line_acc = np.mean(np.concatenate(all_line_args_comp, axis=0))
        arc_acc = np.mean(np.concatenate(all_arc_args_comp, axis=0))
        circle_acc = np.mean(np.concatenate(all_circle_args_comp, axis=0))

        self.val_tb.add_scalars("args_acc",
                                {"line": line_acc, "arc": arc_acc, "circle": circle_acc,
                                 "plane": sket_plane_acc, "trans": sket_trans_acc, "extent": extent_one_acc},
                                global_step=self.clock.epoch)
