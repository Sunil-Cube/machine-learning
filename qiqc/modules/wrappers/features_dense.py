import torch
from torch import nn
from qiqc.modules.wrappers.base import NNModuleWrapperBase
from qiqc.registry import FEATURES_DENSE_REGISTRY

class FeaturesWrapper(NNModuleWrapperBase):

    default_config = None
    registry = FEATURES_DENSE_REGISTRY

    def __init__(self, config):
        super().__init__()
        in_size = config.fd_in_size
        self.config = config
        self.module = self.registry[config.aggregator]()
        assert isinstance(config.fd_n_hiddens, list)
        layers = []
        if config.fd_bn0:
            layers.append(nn.BatchNorm1d(in_size))
        if config.fd_dropout0 > 0:
            layers.append(nn.Dropout(config.fd_dropout0))
        for n_hidden in config.fd_n_hiddens:
            layers.append(nn.Linear(in_size, n_hidden))
            if config.fd_bn:
                layers.append(nn.BatchNorm1d(n_hidden))
            if config.fd_actfun is not None:
                layers.append(config.fd_actfun)
            if config.fd_dropout > 0:
                layers.append(nn.Dropout(config.fd_dropout))
            in_size = n_hidden
        self.layers = nn.Sequential(*layers)


    @classmethod
    def add_args(cls, parser):
        assert isinstance(cls.default_config, dict)
        parser.add_argument('--fd-n-hiddens', type=list)
        parser.add_argument('--fd-bn', type=bool)
        parser.add_argument('--fd-bn0', type=bool)
        parser.add_argument('--fd-dropout', type=float, default=0.)
        parser.add_argument('--fd-dropout0', type=float, default=0.)
        parser.add_argument('--fd-actfun', default=0.)
        parser.set_defaults(**cls.default_config)

    @classmethod
    def add_extra_args(cls, parser, config):
        pass

    def forward(self, X, mask):
        h = self.module(X, mask)
        # h = X
        # if X.shape[1] + X2.shape[1] == self.in_size:
        #     h = torch.cat([h, X2], dim=1)
        # h = self.layers(h)
        return h
