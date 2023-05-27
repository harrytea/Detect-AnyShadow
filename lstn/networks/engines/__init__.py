from lstn.networks.engines.lstn_engine import LSTNEngine, LSTNInferEngine


def build_engine(name, phase='train', **kwargs):
    if name == 'lstnengine':
        if phase == 'train':
            return LSTNEngine(**kwargs)
        elif phase == 'eval':
            return LSTNInferEngine(**kwargs)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
