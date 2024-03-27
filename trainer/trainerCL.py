from functools import partial
import torch
import torch.optim as optim
from tqdm import tqdm
from model.autoencoder import CADDecoder, CADEncoder,  ProjectionHead
from model.my_skex_encoders import SkexEncoder
from .base import BaseTrainer
from .loss import CADLoss, contrastive_loss, sup_con_loss
from .scheduler import GradualWarmupScheduler
from cadlib.macro import *
from torch.cuda.amp import GradScaler


class TrainerCL(BaseTrainer):
    def build_net(self, cfg):
        # self.net = CADTransformer(cfg).cuda()
        if cfg.encoder_type == 'DeepCAD':
            self.net = CADEncoder(cfg).cuda()
        elif cfg.encoder_type == 'SkexGen':
            self.net = SkexEncoder(cfg).cuda()
        
        if cfg.no_phead:
            self.phead = lambda x: x
        else:
            self.phead = ProjectionHead(cfg).cuda()
            
        if cfg.add_recon_loss:
            self.decoder = CADDecoder(cfg).cuda()

    def set_optimizer(self, cfg):
        """set optimizer and lr scheduler used in training"""
        self.optimizer = optim.Adam(self.net.parameters(), cfg.lr)
        self.scheduler = GradualWarmupScheduler(
            self.optimizer, 1.0, cfg.warmup_step)
        self.scaler = GradScaler(enabled=self.cfg.fp16)

    def set_loss_function(self):
        if self.cfg.add_recon_loss:
            self.reconstruction_loss = CADLoss(self.cfg).cuda()

    def forward(self, data):

        commands1 = data['command'].cuda()  # (N, S)
        args1 = data['args'].cuda()  # (N, S, N_ARGS)
        commands2 = data['command_aug'].cuda()  # (N, S)
        args2 = data['args_aug'].cuda()  # (N, S, N_ARGS)

        kwargs = {}
        if self.cfg.encoder_type == 'SkexGen':
            kwargs['epoch'] = self.clock.epoch
            
        # mixed precision
        with torch.autocast(device_type='cuda', enabled=self.cfg.fp16, dtype=torch.float16):
            output1 = self.net(commands1, args1, **kwargs)  # Note: `output` could possibly contain loss (if SkexEncoder is used)
            projected_z1 = self.phead(output1['representation'])

            output2 = self.net(commands2, args2, **kwargs)
            projected_z2 = self.phead(output2['representation'])

            if self.cfg.cl_loss == 'simclr':
                cont_loss = contrastive_loss(projected_z1, projected_z2, self.cfg)
            elif self.cfg.cl_loss == 'supcon':
                labels = (commands1 == EXT_IDX).sum(dim=1)  # (N,)
                cont_loss = sup_con_loss(projected_z1, projected_z2, self.cfg, labels)
            loss_dict = {'cl_loss': cont_loss}

            if self.cfg.add_recon_loss:

                cmd_logits1, args_logits1 = self.decoder(output1['representation'])
                recon_loss1 = self.reconstruction_loss({
                    'tgt_commands': commands1,
                    'tgt_args': args1,
                    'command_logits': cmd_logits1,
                    'args_logits': args_logits1
                })
                loss_dict.update({
                    'rec_loss_cmd1': recon_loss1['loss_cmd'],
                    'rec_loss_args1': recon_loss1['loss_args']
                })

                cmd_logits2, args_logits2 = self.decoder(output2['representation'])
                recon_loss2 = self.reconstruction_loss({
                    'tgt_commands': commands2,
                    'tgt_args': args2,
                    'command_logits': cmd_logits2,
                    'args_logits': args_logits2
                })
                loss_dict.update({
                    'rec_loss_cmd2': recon_loss2['loss_cmd'],
                    'rec_loss_args2': recon_loss2['loss_args']
                })
                
                loss_dict['cl_loss'] *= self.cfg.lamda
                for key in loss_dict.keys():
                    if key != 'cl_loss':
                        loss_dict[key] *= self.cfg.kappa

        return (projected_z1, projected_z2), loss_dict

    def encode(self, data, is_batch=False):
        """encode into latent vectors"""
        commands = data['command'].cuda()
        args = data['args'].cuda()
        if not is_batch:
            commands = commands.unsqueeze(0)
            args = args.unsqueeze(0)
        z = self.net(commands, args)['representation']
        return z

    def decode(self, z):
        """decode given latent vectors"""
        raise NotImplementedError
        outputs = self.net(None, None, z=z, return_tgt=False)
        return outputs

    def logits2vec(self, outputs, refill_pad=True, to_numpy=True):
        command_logits = outputs['command_logits']  # (N, S, N_CMD)
        args_logits = outputs['args_logits']  # (N, S, N_ARGS, ARGS_DIM)

        """network outputs (logits) to final CAD vector"""
        out_command = torch.argmax(torch.softmax(
            command_logits, dim=-1), dim=-1)  # (N, S)
        out_args = torch.argmax(torch.softmax(
            args_logits, dim=-1), dim=-1) - 1  # (N, S, N_ARGS)
        if refill_pad:  # fill all unused command element to -1
            mask = ~torch.tensor(CMD_ARGS_MASK).bool().cuda()[
                out_command.long()]
            out_args[mask] = -1

        out_cad_vec = torch.cat(
            [out_command.unsqueeze(-1), out_args], dim=-1)  # (N, S, N_ARGS+1)
        if to_numpy:
            out_cad_vec = out_cad_vec.detach().cpu().numpy()
        return out_cad_vec

    def evaluate(self, test_loader):
        """evaluatinon during training"""
        raise NotImplementedError
        self.net.eval()
        pbar = tqdm(test_loader)
        pbar.set_description("EVALUATE[{}]".format(self.clock.epoch))

        all_ext_args_comp = []
        all_line_args_comp = []
        all_arc_args_comp = []
        all_circle_args_comp = []

        for i, data in enumerate(pbar):
            with torch.no_grad():
                commands = data['command'].cuda()
                args = data['args'].cuda()
                outputs = self.net(commands, args)
                out_args = torch.argmax(torch.softmax(
                    outputs['args_logits'], dim=-1), dim=-1) - 1
                out_args = out_args.long().detach().cpu().numpy()  # (N, S, n_args)

            gt_commands = commands.squeeze(
                1).long().detach().cpu().numpy()  # (N, S)
            gt_args = args.squeeze(1).long().detach(
            ).cpu().numpy()  # (N, S, n_args)

            ext_pos = np.where(gt_commands == EXT_IDX)
            line_pos = np.where(gt_commands == LINE_IDX)
            arc_pos = np.where(gt_commands == ARC_IDX)
            circle_pos = np.where(gt_commands == CIRCLE_IDX)

            args_comp = (gt_args == out_args).astype(int)
            all_ext_args_comp.append(args_comp[ext_pos][:, -N_ARGS_EXT:])
            all_line_args_comp.append(args_comp[line_pos][:, :2])
            all_arc_args_comp.append(args_comp[arc_pos][:, :4])
            all_circle_args_comp.append(args_comp[circle_pos][:, [0, 1, 4]])

        all_ext_args_comp = np.concatenate(all_ext_args_comp, axis=0)
        sket_plane_acc = np.mean(all_ext_args_comp[:, :N_ARGS_PLANE])
        sket_trans_acc = np.mean(
            all_ext_args_comp[:, N_ARGS_PLANE:N_ARGS_PLANE+N_ARGS_TRANS])
        extent_one_acc = np.mean(all_ext_args_comp[:, -N_ARGS_EXT_PARAM])
        line_acc = np.mean(np.concatenate(all_line_args_comp, axis=0))
        arc_acc = np.mean(np.concatenate(all_arc_args_comp, axis=0))
        circle_acc = np.mean(np.concatenate(all_circle_args_comp, axis=0))

        self.val_tb.add_scalars("args_acc",
                                {"line": line_acc, "arc": arc_acc, "circle": circle_acc,
                                 "plane": sket_plane_acc, "trans": sket_trans_acc, "extent": extent_one_acc},
                                global_step=self.clock.epoch)
