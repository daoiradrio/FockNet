import e3x

from flax import linen as nn
from jax import numpy as jnp



class E3EqAtomWeave(nn.Module):

    num_features: int

    @nn.compact
    def __call__(self, FA, FP, pair_split):
        AA = e3x.nn.TensorDense()(FA)
        AA = e3x.nn.relu(AA)

        PA = e3x.nn.TensorDense()(FP)
        PA = e3x.nn.relu(PA)
        PA = e3x.ops.indexed_sum(PA, dst_idx=pair_split, num_segments=FA.shape[-4])
        
        A = jnp.concatenate((AA, PA), axis=-1)
        A = e3x.nn.TensorDense(features=self.num_features)(A)
        A = e3x.nn.relu(A)

        return A



class FockNet(nn.Module):

    num_features: int
    num_blocks: int

    @nn.compact
    def __call__(self, x_atom, x_pair, pair_split):

        # ATOM PART

        FA = e3x.nn.TensorDense(features=self.num_features)(x_atom)
        FP = e3x.nn.TensorDense(features=self.num_features)(x_pair)

        for block in range(self.num_blocks):
            A = E3EqAtomWeave(num_features=self.num_features)(FA, FP, pair_split)

        A = e3x.nn.TensorDense(features=1, include_pseudotensors=False)(A)

        # PAIR PART

        # ...

        return A
