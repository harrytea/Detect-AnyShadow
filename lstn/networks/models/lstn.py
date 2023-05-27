import torch.nn as nn

from networks.encoders import build_encoder
from networks.layers.transformer import LongShortTermTransformer
from networks.decoders import build_decoder
from networks.layers.position import PositionEmbeddingSine


class LSTN(nn.Module):
    def __init__(self, cfg, encoder='mobilenetv2', decoder='fpn'):
        super().__init__()
        self.cfg = cfg
        self.max_obj_num = cfg.MODEL_MAX_OBJ_NUM
        self.epsilon = cfg.MODEL_EPSILON

        self.encoder = build_encoder(encoder,
                                     frozen_bn=cfg.MODEL_FREEZE_BN,
                                     freeze_at=cfg.TRAIN_ENCODER_FREEZE_AT)
        self.encoder_projector = nn.Conv2d(cfg.MODEL_ENCODER_DIM[-1], cfg.MODEL_ENCODER_EMBEDDING_DIM, kernel_size=1)

        self.LSAB = LongShortTermTransformer(cfg.MODEL_LSAB_NUM,
                                            cfg.MODEL_ENCODER_EMBEDDING_DIM,
                                            cfg.MODEL_SELF_HEADS,
                                            cfg.MODEL_ATT_HEADS,
                                            emb_dropout=cfg.TRAIN_LSAB_EMB_DROPOUT,
                                            droppath=cfg.TRAIN_LSAB_DROPPATH,
                                            lt_dropout=cfg.TRAIN_LSAB_LT_DROPOUT,
                                            st_dropout=cfg.TRAIN_LSAB_ST_DROPOUT,
                                            droppath_lst=cfg.TRAIN_LSAB_DROPPATH_LST,
                                            droppath_scaling=cfg.TRAIN_LSAB_DROPPATH_SCALING,
                                            intermediate_norm=cfg.MODEL_DECODER_INTERMEDIATE_LSAB,
                                            return_intermediate=True)

        decoder_indim = cfg.MODEL_ENCODER_EMBEDDING_DIM * (cfg.MODEL_LSAB_NUM + 1) \
            if cfg.MODEL_DECODER_INTERMEDIATE_LSAB else cfg.MODEL_ENCODER_EMBEDDING_DIM

        self.decoder = build_decoder(decoder,
                                    in_dim=decoder_indim,
                                    out_dim=cfg.MODEL_MAX_OBJ_NUM + 1,
                                    decode_intermediate_input=cfg.MODEL_DECODER_INTERMEDIATE_LSAB,
                                    hidden_dim=cfg.MODEL_ENCODER_EMBEDDING_DIM,
                                    shortcut_dims=cfg.MODEL_ENCODER_DIM,
                                    align_corners=cfg.MODEL_ALIGN_CORNERS)

        if cfg.MODEL_ALIGN_CORNERS:
            self.patch_wise_id_bank = nn.Conv2d(cfg.MODEL_MAX_OBJ_NUM+1, cfg.MODEL_ENCODER_EMBEDDING_DIM, kernel_size=17, stride=16, padding=8)
        else:
            self.patch_wise_id_bank = nn.Conv2d(cfg.MODEL_MAX_OBJ_NUM+1, cfg.MODEL_ENCODER_EMBEDDING_DIM, kernel_size=16, stride=16, padding=0)

        self.id_dropout = nn.Dropout(cfg.TRAIN_LSAB_ID_DROPOUT, True)
        self.pos_generator = PositionEmbeddingSine(cfg.MODEL_ENCODER_EMBEDDING_DIM // 2, normalize=True)
        self._init_weight()

    def get_pos_emb(self, x):
        pos_emb = self.pos_generator(x)  # [1, 256, 30, 30]
        return pos_emb

    def get_id_emb(self, x):
        id_emb = self.patch_wise_id_bank(x)  # conv(11, 256, k17, s16, p8)
        id_emb = self.id_dropout(id_emb)  # [2, 256, 30, 30]
        return id_emb

    def encode_image(self, img):
        # [10, 24, 117, 117]
        # [10, 32, 59, 59]
        # [10, 96, 30, 30]
        # [10, 1280, 30, 30]
        xs = self.encoder(img)
        xs[-1] = self.encoder_projector(xs[-1])
        return xs

    def decode_id_logits(self, lsab_emb, shortcuts):
        n, c, h, w = shortcuts[-1].size()  # [2, 256, 30, 30]
        decoder_inputs = [shortcuts[-1]]  # [2, 256, 30, 30]
        for emb in lsab_emb:
            decoder_inputs.append(emb.view(h, w, n, c).permute(2, 3, 0, 1))  # [2, 256, 30, 30]
        pred_logit = self.decoder(decoder_inputs, shortcuts)  # [2, 11, 117, 117]
        return pred_logit

    def LSAB_forward(self,curr_embs, long_term_memories, short_term_memories, curr_id_emb=None, pos_emb=None, size_2d=(30, 30)):
        n, c, h, w = curr_embs[-1].size()  # [2, 256, 30, 30]
        curr_emb = curr_embs[-1].view(n, c, h*w).permute(2, 0, 1)  # [900, 2, 256]
        lsab_embs, lsab_memories = self.LSAB(curr_emb, long_term_memories, short_term_memories, curr_id_emb, pos_emb, size_2d)
        lsab_curr_memories, lsab_long_memories, lsab_short_memories = zip(*lsab_memories)
        return lsab_embs, lsab_curr_memories, lsab_long_memories, lsab_short_memories

    def _init_weight(self):
        nn.init.xavier_uniform_(self.encoder_projector.weight)
        nn.init.orthogonal_(self.patch_wise_id_bank.weight.view(
                            self.cfg.MODEL_ENCODER_EMBEDDING_DIM, -1).permute(0, 1),
                            gain=17**-2 if self.cfg.MODEL_ALIGN_CORNERS else 16**-2)
