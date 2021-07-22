import torch.nn as nn

import models.hyp_layers as hyp_layers
from geoopt import Lorentz

class Encoder(nn.Module):
    def __init__(self, c):
        super(Encoder, self).__init__()
        self.c = c

    def encode(self, x, adj):
        if self.encode_graph:
            input = (x, adj)
            output, _ = self.layers.forward(input)
        else:
            output = self.layers.forward(x)
        return output


class HG(Encoder):
    def __init__(self, c, args):
        super(HG, self).__init__(c)
        self.manifold = Lorentz(c)
        assert args.num_layers > 1

        hgc_layers = []
        in_dim = out_dim = args.embedding_dim
        hgc_layers.append(
            hyp_layers.HyperbolicGraphConvolution(
                self.manifold, in_dim, out_dim, self.c, args.network, args.num_layers
            )
        )
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True

    def encode(self, x, adj):
        x_hyp = self.manifold.projx(x)
        return super(HG, self).encode(x_hyp, adj)
