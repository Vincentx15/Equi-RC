import torch
import torch.nn as nn

from pytorch_rclayers import RegToRegConv, RegToIrrepConv, IrrepToIrrepConv, IrrepActivationLayer, \
    IrrepConcatLayer, IrrepBatchNorm, RegBatchNorm, RegConcatLayer, ToKmerLayer


class CustomRCPS(nn.Module):

    def __init__(self,
                 filters=(16, 16, 16),
                 kernel_sizes=(15, 14, 14),
                 pool_size=40,
                 pool_strides=20,
                 out_size=1,
                 placeholder_bn=False,
                 kmers=1):
        """
        This is an example of use of the equivariant layers :

        The network takes as inputs windows of 1000 base pairs one hot encoded and outputs a binary prediction
        The architecture follows the paper of Avanti Shrikumar : Reverse Complement Parameter Sharing
        We reimplement everything with equivariant layers and add the possibility to start the encoding with
        a K-Mer encoding layer.
        """
        super(CustomRCPS, self).__init__()

        self.kmers = int(kmers)
        self.to_kmer = ToKmerLayer(k=self.kmers)
        reg_in = self.to_kmer.features // 2
        filters = [reg_in] + list(filters)

        # Now add the intermediate layer : sequence of conv, BN, activation
        self.reg_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.activation_layers = nn.ModuleList()
        for i in range(len(filters) - 1):
            prev_reg = filters[i]
            next_reg = filters[i + 1]
            self.reg_layers.append(RegToRegConv(
                reg_in=prev_reg,
                reg_out=next_reg,
                kernel_size=kernel_sizes[i],
            ))
            self.bn_layers.append(RegBatchNorm(reg_dim=next_reg, placeholder=placeholder_bn))
            # Don't add activation if it's the last layer
            placeholder = (i == len(filters) - 1)
            self.activation_layers.append(nn.ReLU())

        self.concat = RegConcatLayer(reg=filters[-1])
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_strides)
        self.flattener = nn.Flatten()
        self.dense = nn.Linear(in_features=752, out_features=out_size)

    def forward(self, inputs):
        x = self.to_kmer(inputs)
        for reg_layer, bn_layer, activation_layer in zip(self.reg_layers, self.bn_layers, self.activation_layers):
            x = reg_layer(x)
            x = bn_layer(x)
            x = activation_layer(x)

        # Average two strands predictions, pool and go through Dense
        x = self.concat(x)
        x = self.pool(x)
        x = self.flattener(x)
        x = self.dense(x)
        outputs = torch.sigmoid(x)
        return outputs


class EquiNetBinary(nn.Module):

    def __init__(self,
                 filters=((16, 16), (16, 16), (16, 16)),
                 kernel_sizes=(15, 14, 14),
                 pool_size=40,
                 pool_length=20,
                 out_size=1,
                 placeholder_bn=False,
                 kmers=1):
        """
        This network takes as inputs windows of 1000 base pairs one hot encoded and outputs a binary prediction

        First maps the regular representation to irrep setting
        Then goes from one setting to another.
        """
        super(EquiNetBinary, self).__init__()

        self.kmers = int(kmers)
        self.to_kmer = ToKmerLayer(k=self.kmers)

        # First mapping goes from the input to an irrep feature space
        reg_in = self.to_kmer.features // 2
        first_kernel_size = kernel_sizes[0]
        first_a, first_b = filters[0]
        self.last_a, self.last_b = filters[-1]
        self.reg_irrep = RegToIrrepConv(reg_in=reg_in,
                                        a_out=first_a,
                                        b_out=first_b,
                                        kernel_size=first_kernel_size)
        self.first_bn = IrrepBatchNorm(a=first_a, b=first_b, placeholder=placeholder_bn)
        self.first_act = IrrepActivationLayer(a=first_a, b=first_b)

        # Now add the intermediate layers : sequence of conv, BN, activation
        self.irrep_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.activation_layers = nn.ModuleList()
        for i in range(1, len(filters)):
            prev_a, prev_b = filters[i - 1]
            next_a, next_b = filters[i]
            self.irrep_layers.append(IrrepToIrrepConv(
                a_in=prev_a,
                b_in=prev_b,
                a_out=next_a,
                b_out=next_b,
                kernel_size=kernel_sizes[i],
            ))
            self.bn_layers.append(IrrepBatchNorm(a=next_a, b=next_b, placeholder=placeholder_bn))
            self.activation_layers.append(IrrepActivationLayer(a=next_a,
                                                               b=next_b))

        self.concat = IrrepConcatLayer(a=self.last_a, b=self.last_b)
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_length)
        self.flattener = nn.Flatten()
        self.dense = nn.Linear(in_features=1472, out_features=out_size)
        self.final_activation = nn.Sigmoid()

    def forward(self, inputs):
        x = self.to_kmer(inputs)
        x = self.reg_irrep(x)
        x = self.first_bn(x)
        x = self.first_act(x)

        for irrep_layer, bn_layer, activation_layer in zip(self.irrep_layers, self.bn_layers, self.activation_layers):
            x = irrep_layer(x)
            x = bn_layer(x)
            x = activation_layer(x)

        # Average two strands predictions, pool and go through Dense
        x = x.float()
        x = self.concat(x)
        x = self.pool(x)
        x = self.flattener(x)
        x = self.dense(x)
        outputs = self.final_activation(x)
        return outputs


if __name__ == '__main__':
    inputs = torch.ones(size=(1, 4, 1000)).double()
    model = EquiNetBinary(kmers=2, filters=((24, 8), (24, 8), (24, 8)))
    outputs = model(inputs)
