class DefaultModelConfig():
    def __init__(self):
        self.MODEL_NAME = 'LSTNDefault'

        self.MODEL_VSD = 'lstn'
        self.MODEL_ENGINE = 'lstnengine'
        self.MODEL_ALIGN_CORNERS = True
        self.MODEL_ENCODER = 'mobilenetv2'
        self.MODEL_ENCODER_PRETRAIN = './pretrain_models/mobilenet_v2-b0353104.pth'
        self.MODEL_ENCODER_DIM = [24, 32, 96, 1280]  # 4x, 8x, 16x, 16x
        self.MODEL_ENCODER_EMBEDDING_DIM = 256
        self.MODEL_DECODER_INTERMEDIATE_LSAB = True
        self.MODEL_FREEZE_BN = True
        self.MODEL_FREEZE_BACKBONE = False
        self.MODEL_MAX_OBJ_NUM = 1  # 1
        self.MODEL_SELF_HEADS = 8
        self.MODEL_ATT_HEADS = 8
        self.MODEL_LSAB_NUM = 1
        self.MODEL_EPSILON = 1e-5
        self.MODEL_USE_PREV_PROB = False

        self.TRAIN_LONG_TERM_MEM_GAP = 9999
        self.TEST_LONG_TERM_MEM_GAP = 9999
        self.TEST_SHORT_TERM_MEM_SKIP = 1
