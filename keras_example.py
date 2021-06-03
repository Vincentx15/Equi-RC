import keras
import keras.layers as kl

from keras_rclayers import RegToIrrepConv, IrrepToIrrepConv, IrrepActivationLayer,\
    IrrepConcatLayer, IrrepBatchNorm, ToKmerLayer


class EquiNetBinary:

    def __init__(self,
                 filters=((16, 16), (16, 16), (16, 16)),
                 kernel_sizes=(15, 14, 14),
                 pool_size=40,
                 pool_length=20,
                 out_size=1,
                 placeholder_bn=False,
                 kmers=1):
        """
        First map the regular representation to irrep setting
        Then goes from one setting to another.
        """

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
        self.irrep_layers = []
        self.bn_layers = []
        self.activation_layers = []
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
        self.pool = kl.MaxPooling1D(pool_length=pool_size, strides=pool_length)
        self.flattener = kl.Flatten()
        self.dense = kl.Dense(out_size, activation='sigmoid')

    def func_api_model(self):
        inputs = keras.layers.Input(shape=(1000, 4), dtype="float32")

        x = self.to_kmer(inputs)
        x = self.reg_irrep(x)
        x = self.first_bn(x)
        x = self.first_act(x)

        for irrep_layer, bn_layer, activation_layer in zip(self.irrep_layers, self.bn_layers, self.activation_layers):
            x = irrep_layer(x)
            x = bn_layer(x)
            x = activation_layer(x)

        # Average two strands predictions, pool and go through Dense
        x = self.concat(x)
        x = self.pool(x)
        x = self.flattener(x)
        outputs = self.dense(x)
        model = keras.Model(inputs, outputs)
        return model

if __name__=='__main__':
    model = EquiNetBinary(kmers=2, filters=((24, 8), (24, 8), (24, 8)))
    model = model.func_api_model()
    model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
                  loss="binary_crossentropy", metrics=["accuracy"])

