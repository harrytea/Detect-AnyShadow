from networks.models.lstn import LSTN


def build_vos_model(name, cfg, **kwargs):
    if name == 'lstn':
        return LSTN(cfg, encoder=cfg.MODEL_ENCODER, **kwargs)
    else:
        raise NotImplementedError
