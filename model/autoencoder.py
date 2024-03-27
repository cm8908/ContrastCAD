import sys
sys.path.append('..')
from .layers.transformer import *
from .layers.improved_transformer import *
from .layers.positional_encoding import *
from .model_utils import _make_seq_first, _make_batch_first, \
    _get_padding_mask, _get_key_padding_mask, _get_group_mask, _logits2vec
from cadlib.macro import *


class CADEmbedding(nn.Module):
    """Embedding: positional embed + command embed + parameter embed + group embed (optional)"""
    def __init__(self, cfg, seq_len, use_group=False, group_len=None):
        super().__init__()

        n_commands = cfg.n_commands
        args_dim = cfg.args_dim + 1
        if cfg.proj_dir == 'maskedlm':
            n_commands += 1
            args_dim += 1

        self.command_embed = nn.Embedding(n_commands, cfg.d_model)
        self.arg_embed = nn.Embedding(args_dim, 64, padding_idx=0)
        self.embed_fcn = nn.Linear(64 * cfg.n_args, cfg.d_model)

        # use_group: additional embedding for each sketch-extrusion pair
        self.use_group = use_group
        if use_group:
            if group_len is None:
                group_len = cfg.max_num_groups
            self.group_embed = nn.Embedding(group_len + 2, cfg.d_model)

        self.pos_encoding = PositionalEncodingLUT(cfg.d_model, max_len=seq_len+2)

    def forward(self, commands, args, groups=None):
        """
        commands: (S, N)
        args: (S, N, N_ARGS)
        groups: (S, N)
        """
        S, N = commands.shape

        #  (S, N, D) + (S, N, N_ARGS*64=>D)
        src = self.command_embed(commands.long()) + \
              self.embed_fcn(
                  self.arg_embed((args + 1).long()).view(S, N, -1)
                  )  # shift due to -1 PAD_VAL
              # (S, N, D)

        if self.use_group:
            src = src + self.group_embed(groups.long())  # (S, N, D)

        src = self.pos_encoding(src)

        return src


class ConstEmbedding(nn.Module):
    """learned constant embedding"""
    def __init__(self, cfg, seq_len):
        super().__init__()

        self.d_model = cfg.d_model
        self.seq_len = seq_len

        self.PE = PositionalEncodingLUT(cfg.d_model, max_len=seq_len)

    def forward(self, z):
        N = z.size(1)
        src = self.PE(z.new_zeros(self.seq_len, N, self.d_model))
        return src


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        seq_len = cfg.max_total_len
        self.keep_seq_len = cfg.keep_seq_len
        self.use_group = cfg.use_group_emb
        self.embedding = CADEmbedding(cfg, seq_len, use_group=self.use_group)

        encoder_layer = TransformerEncoderLayerImproved(cfg.d_model, cfg.n_heads, cfg.dim_feedforward, cfg.dropout)
        encoder_norm = LayerNorm(cfg.d_model)
        self.encoder = TransformerEncoder(encoder_layer, cfg.n_layers, encoder_norm)

    def forward(self, commands, args):
        padding_mask, key_padding_mask = _get_padding_mask(commands, seq_dim=0), _get_key_padding_mask(commands, seq_dim=0)
        #  (S, N, 1), (N, S)
        group_mask = _get_group_mask(commands, seq_dim=0) if self.use_group else None  # (S, N)

        src = self.embedding(commands, args, group_mask)  # (S, N, D)

        memory = self.encoder(src, mask=None, src_key_padding_mask=key_padding_mask)  # (S, N, dim_z) where dim_z = D = 256

        if not self.keep_seq_len:
            z = (memory * padding_mask).sum(dim=0, keepdim=True) / padding_mask.sum(dim=0, keepdim=True) # (1, N, dim_z) average pooled
        else:
            z = memory * padding_mask  # (S, N, dim_z)
        return z


class FCN(nn.Module):
    def __init__(self, d_model, n_commands, n_args, args_dim=256):
        super().__init__()

        self.n_args = n_args
        self.args_dim = args_dim

        self.command_fcn = nn.Linear(d_model, n_commands)
        self.args_fcn = nn.Linear(d_model, n_args * args_dim)

    def forward(self, out):
        S, N, _ = out.shape

        command_logits = self.command_fcn(out)  # Shape [S, N, n_commands]

        args_logits = self.args_fcn(out)  # Shape [S, N, n_args * args_dim]
        args_logits = args_logits.reshape(S, N, self.n_args, self.args_dim)  # Shape [S, N, n_args, args_dim]

        return command_logits, args_logits


class Decoder(nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()

        self.embedding = ConstEmbedding(cfg, cfg.max_total_len)

        decoder_layer = TransformerDecoderLayerGlobalImproved(cfg.d_model, cfg.dim_z, cfg.n_heads, cfg.dim_feedforward, cfg.dropout)
        decoder_norm = LayerNorm(cfg.d_model)
        self.decoder = TransformerDecoder(decoder_layer, cfg.n_layers_decode, decoder_norm)

        args_dim = cfg.args_dim + 1
        self.fcn = FCN(cfg.d_model, cfg.n_commands, cfg.n_args, args_dim)

    def forward(self, z):
        # z: (1 | S, N, D)
        src = self.embedding(z)  # (S, N, D)
        out = self.decoder(src, z, tgt_mask=None, tgt_key_padding_mask=None)  # (S, N, D)

        command_logits, args_logits = self.fcn(out)

        out_logits = (command_logits, args_logits)
        return out_logits

class Bottleneck(nn.Module):
    def __init__(self, cfg):
        super(Bottleneck, self).__init__()

        self.bottleneck = nn.Sequential(nn.Linear(cfg.d_model, cfg.dim_z),
                                        nn.Tanh())

    def forward(self, z):
        return self.bottleneck(z)


class CADTransformer(nn.Module):
    def __init__(self, cfg):
        super(CADTransformer, self).__init__()

        self.args_dim = cfg.args_dim + 1

        self.encoder = Encoder(cfg)

        self.bottleneck = Bottleneck(cfg)

        self.decoder = Decoder(cfg)

    def forward(self, commands_enc, args_enc,
                z=None, return_tgt=True, encode_mode=False):
        """
        N = batch size, S = sequence length, N_ARGS = number of params (=16)
        commands_enc: (N, S)
        args_enc: (N, S, N_ARGS)
        """
        commands_enc_, args_enc_ = _make_seq_first(commands_enc, args_enc)  # Possibly None, None
        # (S, N), (S, N, N_ARGS)
        if z is None:
            z = self.encoder(commands_enc_, args_enc_)
            z = self.bottleneck(z)  # (1, N, D)
        else:
            z = _make_seq_first(z)

        if encode_mode: return _make_batch_first(z)

        out_logits = self.decoder(z)
        out_logits = _make_batch_first(*out_logits)

        res = {
            "command_logits": out_logits[0],  # (N, S, n_commands)
            "args_logits": out_logits[1],  # (N, S, n_args, args_dim)
            "representation": _make_batch_first(z)  # (N, S, D)
        }

        if return_tgt:
            res["tgt_commands"] = commands_enc
            res["tgt_args"] = args_enc

        return res

class ProjectionHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = Encoder(cfg)
        
        self.linear1 = nn.Linear(cfg.dim_z, cfg.dim_phead)
        self.linear2 = nn.Linear(cfg.dim_phead, cfg.dim_z)
        self.act = ACT2FN[cfg.phead_activation]()
    def forward(self, z):
        # Note that `z` here actually equals to `h` in the paper "SimCLR"
        return self.linear2(self.act(self.linear1(z)))

class CADEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        seq_len = cfg.max_total_len
        self.keep_seq_len = cfg.keep_seq_len
        self.use_group = cfg.use_group_emb
        self.embedding = CADEmbedding(cfg, seq_len, use_group=self.use_group)

        encoder_layer = TransformerEncoderLayerImproved(cfg.d_model, cfg.n_heads, cfg.dim_feedforward, cfg.dropout)
        encoder_norm = LayerNorm(cfg.d_model)
        self.encoder = TransformerEncoder(encoder_layer, cfg.n_layers, encoder_norm)

    def forward(self, commands, args):
        commands, args = _make_seq_first(commands, args)
        padding_mask, key_padding_mask = _get_padding_mask(commands, seq_dim=0), _get_key_padding_mask(commands, seq_dim=0)
        #  (S, N, 1), (N, S)
        group_mask = _get_group_mask(commands, seq_dim=0) if self.use_group else None  # (S, N)

        src = self.embedding(commands, args, group_mask)  # (S, N, D)

        memory = self.encoder(src, mask=None, src_key_padding_mask=key_padding_mask)  # (S, N, dim_z) where dim_z = D = 256

        if not self.keep_seq_len:
            z = (memory * padding_mask).sum(dim=0, keepdim=True) / padding_mask.sum(dim=0, keepdim=True) # (1, N, dim_z) average pooled
        else:
            z = memory * padding_mask  # (S, N, dim_z)
        z = _make_batch_first(z)
        return {
            "representation": z
        }

class CADDecoder(nn.Module):
    def __init__(self, cfg):
        super(CADDecoder, self).__init__()

        self.embedding = ConstEmbedding(cfg, cfg.max_total_len)

        decoder_layer = TransformerDecoderLayerGlobalImproved(cfg.d_model, cfg.dim_z, cfg.n_heads, cfg.dim_feedforward, cfg.dropout)
        decoder_norm = LayerNorm(cfg.d_model)
        self.decoder = TransformerDecoder(decoder_layer, cfg.n_layers_decode, decoder_norm)

        args_dim = cfg.args_dim + 1
        self.fcn = FCN(cfg.d_model, cfg.n_commands, cfg.n_args, args_dim)

    def forward(self, z):
        z = _make_seq_first(z)
        src = self.embedding(z)  # (S, N, D)
        out = self.decoder(src, z, tgt_mask=None, tgt_key_padding_mask=None)  # (S, N, D)

        command_logits, args_logits = self.fcn(out)

        out_logits = (_make_batch_first(command_logits, args_logits))
        return out_logits