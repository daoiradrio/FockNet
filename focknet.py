import e3x

from flax import linen as nn



class FockNet(nn.Module):

    n_features: int

    @nn.compact
    def __call__(self, x_dftb):
        x = e3x.nn.TensorDense(features=self.n_features)(x)
