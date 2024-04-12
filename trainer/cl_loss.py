import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model_utils import _get_padding_mask, _get_visibility_mask
from cadlib.macro import CMD_ARGS_MASK


class CADContrastiveLoss(nn.Module):
    def __init__(
        self, 
        cfg,
        device,
        batch_size,
        temperature: float = 0.07,
    ):
        super().__init__()

        self.n_commands = cfg.n_commands
        self.args_dim = cfg.args_dim + 1
        self.weights = cfg.loss_weights
        self.device= device
        self.temperature = temperature
        self.batch_size = batch_size
        self.cl_loss_type = cfg.cl_loss

        self.register_buffer("cmd_args_mask", torch.tensor(CMD_ARGS_MASK))

    def forward(self, output):
        # Target & predictions
        tgt_commands, tgt_args = output["tgt_commands"], output["tgt_args"]
        
        # print("tgt_commands.shape: ", tgt_commands.shape)
        # print("tgt_args.shape: ", tgt_args.shape)
        
        tgt_commands = torch.swapaxes(tgt_commands, 0, 1)
        tgt_args = torch.swapaxes(tgt_args, 0, 1)
        # print("tgt_commands.shape: ", tgt_commands.shape)
        # print("tgt_args.shape: ", tgt_args.shape)

        visibility_mask = _get_visibility_mask(tgt_commands, seq_dim=-1)
        padding_mask = _get_padding_mask(tgt_commands, seq_dim=-1, extended=True) * visibility_mask.unsqueeze(-1)

        command_logits, args_logits = output["command_logits"], output["args_logits"]

        mask = self.cmd_args_mask[tgt_commands.long()]

        loss_cmd = F.cross_entropy(
            command_logits[padding_mask.bool()].reshape(-1, self.n_commands), 
            tgt_commands[padding_mask.bool()].reshape(-1).long()
        )
        loss_args = F.cross_entropy(
            args_logits[mask.bool()].reshape(-1, self.args_dim), 
            tgt_args[mask.bool()].reshape(-1).long() + 1
        )  # shift due to -1 PAD_VAL

        loss_cmd = self.weights["loss_cmd_weight"] * loss_cmd
        loss_args = self.weights["loss_args_weight"] * loss_args
        
        if self.cl_loss_type == 'infonce':
            logit, label = self._info_nce_loss(output["proj_z1"],output["proj_z2"])
            loss_contrastive = self.weights["loss_cl_weight"] * torch.nn.CrossEntropyLoss()(logit, label)
        elif self.cl_loss_type == 'simclr':
            loss_contrastive = self.weights['loss_cl_weight'] * \
                self._contrastive_loss(output["proj_z1"],output["proj_z2"])
        res = {
            "loss_cmd": loss_cmd, 
            "loss_args": loss_args,
            "loss_contrastive": loss_contrastive,
        }
        return res
    
    def _info_nce_loss(self, f1,f2):
        labels = torch.cat([torch.arange(f1.size(1)) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        # print("f1, f2 shape: ", f1.shape, f2.shape)
        f1 = f1.squeeze(0)
        f2 = f2.squeeze(0)
        # print("f1, f2 shape: ", f1.shape, f2.shape)
        f1 = F.normalize(f1, dim=1)
        f2 = F.normalize(f2, dim=1)
        features = torch.cat([f1,f2], dim=0)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.temperature
        return logits, labels

    def _contrastive_loss(self, z1, z2):
        z1 = F.normalize(z1.mean(1))  # (N, D)
        z2 = F.normalize(z2.mean(1))  # (N, D)

        batch_size = z1.size(0)
        labels = F.one_hot(torch.arange(batch_size), batch_size * 2).float().cuda()
        masks = F.one_hot(torch.arange(batch_size), batch_size).cuda()
        
        logits_aa = torch.matmul(z1, z1.T) / self.temperature
        logits_aa = logits_aa - masks * 1e9
        logits_bb = torch.matmul(z2, z2.T) / self.temperature
        logits_bb = logits_bb - masks * 1e9
        logits_ab = torch.matmul(z1, z2.T) / self.temperature
        logits_ba = torch.matmul(z2, z1.T) / self.temperature

        loss_a = F.cross_entropy(torch.cat([logits_ab, logits_aa], 1), labels)
        loss_b = F.cross_entropy(torch.cat([logits_ba, logits_bb], 1), labels)
        loss = (loss_a + loss_b).mean()
        return loss