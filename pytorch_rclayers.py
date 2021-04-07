import torch
import numpy as np


def create_xavier_convparameter(shape):
    tensor = torch.zeros(size=shape, dtype=torch.double)
    torch.nn.init.xavier_normal_(tensor, gain=1.0)
    return torch.nn.Parameter(tensor)


class RegToRegConv(torch.nn.Module):
    """
    Mapping from one reg layer to another
    """

    def __init__(self, reg_in, reg_out, kernel_size,
                 dilatation=1,
                 padding=0):
        super(RegToRegConv, self).__init__()
        self.reg_in = reg_in
        self.reg_out = reg_out
        self.input_dim = 2 * reg_in
        self.filters = 2 * reg_out

        self.kernel_size = kernel_size
        self.padding = padding
        self.dilatation = dilatation

        if self.kernel_size > 1:
            self.left_kernel = create_xavier_convparameter(shape=(self.filters, self.input_dim, self.kernel_size // 2))

        # odd size : To get the equality for x=0, we need to have transposed columns + flipped bor the b_out dims
        if self.kernel_size % 2 == 1:
            self.half_center = create_xavier_convparameter(shape=(self.filters, self.input_dim, 1))

    def forward(self, inputs):
        # Build the right part of the kernel from the left one
        if self.kernel_size > 1:
            right_kernel = torch.flip(self.left_kernel, [0, 1, 2])

        # Extra steps are needed for building the middle part when using the odd size
        # We build the missing parts by transposing everything and averaging to get a symetric tensor
        if self.kernel_size % 2 == 1:
            other_half = torch.flip(self.half_center, [0, 1])
            center_kernel = (other_half + self.half_center) / 2
            if self.kernel_size > 1:
                kernel = torch.cat((self.left_kernel, center_kernel, right_kernel), dim=2)
            else:
                kernel = center_kernel
        else:
            if self.kernel_size > 1:
                kernel = torch.cat((self.left_kernel, right_kernel), dim=2)
            else:
                raise ValueError('The kernel size should be bigger than one')
        outputs = torch.nn.functional.conv1d(inputs,
                                             kernel,
                                             padding=self.padding,
                                             dilation=self.dilatation)
        return outputs


class RegToIrrepConv(torch.nn.Module):
    """
    Mapping from one irrep layer to another
    """

    def __init__(self, reg_in, a_out, b_out, kernel_size,
                 dilatation=1,
                 padding=0):
        super(RegToIrrepConv, self).__init__()
        self.reg_in = reg_in
        self.a_out = a_out
        self.b_out = b_out
        self.input_dim = 2 * reg_in
        self.filters = a_out + b_out

        self.kernel_size = kernel_size
        self.padding = padding
        self.dilatation = dilatation

        self.left_kernel = create_xavier_convparameter(shape=(self.filters, self.input_dim, self.kernel_size // 2))

        # odd size : To get the equality for x=0, we need to have transposed columns + flipped bor the b_out dims
        if self.kernel_size % 2 == 1:
            if self.a_out > 0:
                self.top_left = create_xavier_convparameter(shape=(self.a_out, self.reg_in, 1))
            if self.b_out > 0:
                self.bottom_right = create_xavier_convparameter(shape=(self.b_out, self.reg_in, 1))

    def forward(self, inputs):
        # Build the right part of the kernel from the left one
        # Columns are transposed, the b lines are flipped
        if self.kernel_size > 1:
            if self.a_out == 0:
                right_kernel = - torch.flip(self.left_kernel[self.a_out:, :, :], [1, 2])
            elif self.b_out == 0:
                right_kernel = torch.flip(self.left_kernel[:self.a_out, :, :], [1, 2])
            else:
                right_top = torch.flip(self.left_kernel[:self.a_out, :, :], [1, 2])
                right_bottom = - torch.flip(self.left_kernel[self.a_out:, :, :], [1, 2])
                right_kernel = torch.cat((right_top, right_bottom), dim=0)

        # Extra steps are needed for building the middle part when using the odd size
        # We build the missing parts by transposing and flipping the sign of b_parts.
        if self.kernel_size % 2 == 1:
            if self.a_out == 0:
                bottom_left = - torch.flip(self.bottom_right, [1])
                bottom = torch.cat((bottom_left, self.bottom_right), dim=1)
                if self.kernel_size > 1:
                    kernel = torch.cat((self.left_kernel, bottom, right_kernel), dim=2)
                else:
                    kernel = bottom
            elif self.b_out == 0:
                top_right = torch.flip(self.bottom_right, [1])
                top = torch.cat((self.top_left, top_right), dim=1)
                if self.kernel_size > 1:
                    kernel = torch.cat((self.left_kernel, top, right_kernel), dim=2)
                else:
                    kernel = top
            else:
                bottom_left = - torch.flip(self.bottom_right, [1])
                bottom = torch.cat((bottom_left, self.bottom_right), dim=1)
                top_right = torch.flip(self.top_left, [1])
                top = torch.cat((self.top_left, top_right), dim=1)
                center_kernel = torch.cat((top, bottom), dim=0)
                if self.kernel_size > 1:
                    kernel = torch.cat((self.left_kernel, center_kernel, right_kernel), dim=2)
                else:
                    kernel = center_kernel
        else:
            kernel = torch.cat((self.left_kernel, right_kernel), dim=2)
        outputs = torch.nn.functional.conv1d(inputs,
                                             kernel,
                                             padding=self.padding,
                                             dilation=self.dilatation)
        return outputs


class IrrepToRegConv(torch.nn.Module):
    """
    Mapping from one irrep layer to another
    """

    def __init__(self, reg_out, a_in, b_in, kernel_size,
                 dilatation=1,
                 padding=0):
        super(IrrepToRegConv, self).__init__()
        self.reg_out = reg_out
        self.a_in = a_in
        self.b_in = b_in
        self.input_dim = a_in + b_in
        self.filters = 2 * reg_out

        self.kernel_size = kernel_size
        self.padding = padding
        self.dilatation = dilatation

        if self.kernel_size > 1:
            self.left_kernel = create_xavier_convparameter(shape=(self.filters, self.input_dim, self.kernel_size // 2))
        # odd size : To get the equality for x=0, we need to have transposed columns + flipped bor the b_in dims
        if self.kernel_size % 2 == 1:
            if self.a_in > 0:
                self.top_left = create_xavier_convparameter(shape=(self.reg_out, self.a_in, 1))
            if self.b_in > 0:
                self.bottom_right = create_xavier_convparameter(shape=(self.reg_out, self.b_in, 1))

    def forward(self, inputs):
        # Build the right part of the kernel from the left one
        # Rows are transposed, the b columns are flipped
        if self.kernel_size > 1:
            if self.a_in == 0:
                right_kernel = - torch.flip(self.left_kernel[:, self.a_in:, :], [0, 2])
            elif self.b_in == 0:
                right_kernel = torch.flip(self.left_kernel[:, :self.a_in, :], [0, 2])
            else:
                right_right = - torch.flip(self.left_kernel[:, self.a_in:, :], [0, 2])
                right_left = torch.flip(self.left_kernel[:, :self.a_in, :], [0, 2])
                right_kernel = torch.cat((right_left, right_right), dim=1)

        # Extra steps are needed for building the middle part when using the odd size
        # We build the missing parts by transposing and flipping the sign of b_parts.
        if self.kernel_size % 2 == 1:
            if self.a_in == 0:
                top_right = - torch.flip(self.bottom_right, [0])
                right = torch.cat((top_right, self.bottom_right), dim=0)
                if self.kernel_size > 1:
                    kernel = torch.cat((self.left_kernel, right, right_kernel), dim=0)
                else:
                    kernel = right
            elif self.b_in == 0:
                bottom_left = torch.flip(self.top_left, [0])
                left = torch.cat((self.top_left, bottom_left), dim=0)
                if self.kernel_size > 1:
                    kernel = torch.cat((self.left_kernel, left, right_kernel), dim=0)
                else:
                    kernel = left
            else:
                top_right = - torch.flip(self.bottom_right, [0])
                right = torch.cat((top_right, self.bottom_right), dim=0)
                bottom_left = torch.flip(self.top_left, [0])
                left = torch.cat((self.top_left, bottom_left), dim=0)
                center_kernel = torch.cat((left, right), dim=1)
                if self.kernel_size > 1:
                    kernel = torch.cat((self.left_kernel, center_kernel, right_kernel), dim=2)
                else:
                    kernel = center_kernel
        else:
            kernel = torch.cat((self.left_kernel, right_kernel), dim=2)

        outputs = torch.nn.functional.conv1d(inputs,
                                             kernel,
                                             padding=self.padding,
                                             dilation=self.dilatation)
        return outputs


class IrrepToIrrepConv(torch.nn.Module):
    """
    Mapping from one irrep layer to another
    """

    def __init__(self, a_in, a_out, b_in, b_out, kernel_size,
                 dilatation=1,
                 padding=0):
        super(IrrepToIrrepConv, self).__init__()

        self.a_in = a_in
        self.b_in = b_in
        self.a_out = a_out
        self.b_out = b_out
        self.input_dim = a_in + b_in
        self.filters = a_out + b_out

        self.kernel_size = kernel_size
        self.padding = padding
        self.dilatation = dilatation

        if self.kernel_size > 1:
            self.left_kernel = create_xavier_convparameter(shape=(self.filters, self.input_dim, self.kernel_size // 2))
        # odd size
        if self.kernel_size % 2 == 1:
            # For the kernel to be anti-symmetric, we need to have zeros on the anti-symmetric parts
            # Here we initialize the non zero blocks
            if self.a_out > 0 and self.a_in > 0:
                self.top_left = create_xavier_convparameter(shape=(self.a_out, self.a_in, 1))
            if self.b_out > 0 and self.b_in > 0:
                self.bottom_right = create_xavier_convparameter(shape=(self.b_out, self.b_in, 1))

    def forward(self, inputs):
        # Build the right part of the kernel from the left one
        # Here being on a 'b' part means flipping so the block diagonal is flipped
        if self.kernel_size > 1:
            # going from as ->
            if self.b_in == 0:
                # going from as -> bs
                if self.a_out == 0:
                    right_kernel = - torch.flip(self.left_kernel, [2])
                # going from as -> as
                elif self.b_out == 0:
                    right_kernel = torch.flip(self.left_kernel, [2])
                # going from as -> abs
                else:
                    right_top = torch.flip(self.left_kernel[:self.a_out], [2])
                    right_bottom = - torch.flip(self.left_kernel[self.a_out:], [2])
                    right_kernel = torch.cat((right_top, right_bottom), dim=0)

            # going from bs ->
            elif self.a_in == 0:
                # going from bs -> bs
                if self.a_out == 0:
                    right_kernel = torch.flip(self.left_kernel, [2])
                # going from bs -> as
                elif self.b_out == 0:
                    right_kernel = - torch.flip(self.left_kernel, [2])
                # going from bs -> abs
                else:
                    right_top = - torch.flip(self.left_kernel[:self.a_out], [2])
                    right_bottom = torch.flip(self.left_kernel[self.a_out:], [2])
                    right_kernel = torch.cat((right_top, right_bottom), dim=0)

            # going to -> bs
            elif self.a_out == 0:
                # going from bs -> bs
                if self.a_in == 0:
                    right_kernel = torch.flip(self.left_kernel, [2])
                # going from as -> bs
                elif self.b_in == 0:
                    right_kernel = - torch.flip(self.left_kernel, [2])
                # going from abs -> bs
                else:
                    right_left = - torch.flip(self.left_kernel[:, :self.a_in, :], [2])
                    right_right = torch.flip(self.left_kernel[:, self.a_in:, :], [2])
                    right_kernel = torch.cat((right_left, right_right), dim=1)

            # going to -> as
            elif self.b_out == 0:
                # going from bs -> as
                if self.a_in == 0:
                    right_kernel = - torch.flip(self.left_kernel, [2])
                # going from as -> as
                elif self.b_in == 0:
                    right_kernel = torch.flip(self.left_kernel, [2])
                # going from abs -> as
                else:
                    right_left = torch.flip(self.left_kernel[:, :self.a_in, :], [2])
                    right_right = - torch.flip(self.left_kernel[:, self.a_in:, :], [2])
                    right_kernel = torch.cat((right_left, right_right), dim=1)

            else:
                right_top_left = torch.flip(self.left_kernel[:self.a_out, :self.a_in, :], [2])
                right_top_right = -torch.flip(self.left_kernel[:self.a_out, self.a_in:, :self.a_out], [2])
                right_bottom_left = -torch.flip(self.left_kernel[self.a_out:, :self.a_in, :], [2])
                right_bottom_right = torch.flip(self.left_kernel[self.a_out:, self.a_in:, :], [2])

                right_left = torch.cat((right_top_left, right_bottom_left), dim=0)
                right_right = torch.cat((right_top_right, right_bottom_right), dim=0)
                right_kernel = torch.cat((right_left, right_right), dim=1)

        # Extra steps are needed for building the middle part when using the odd size
        if self.kernel_size % 2 == 1:

            # We only have the left part
            # going from as ->
            if self.b_in == 0:
                # going from as -> bs
                if self.a_out == 0:
                    center_kernel = torch.zeros(size=(self.b_out, self.a_in, 1))
                # going from as -> as
                elif self.b_out == 0:
                    center_kernel = self.top_left
                # going from as -> abs
                else:
                    bottom_left = torch.zeros(size=(self.b_out, self.a_in, 1))
                    center_kernel = torch.cat((self.top_left, bottom_left), dim=0)

            # We only have the right part
            # going from bs ->
            elif self.a_in == 0:
                # going from bs -> bs
                if self.a_out == 0:
                    center_kernel = self.bottom_right
                # going from bs -> as
                elif self.b_out == 0:
                    center_kernel = torch.zeros(size=(self.a_out, self.b_in, 1))
                # going from bs -> abs
                else:
                    top_right = torch.zeros(size=(self.a_out, self.b_in, 1))
                    center_kernel = torch.cat((top_right, self.bottom_right), dim=0)

            # in <=> left/right, out <=> top/bottom

            # We only have the bottom
            # going to -> bs
            elif self.a_out == 0:
                # going from bs -> bs
                if self.a_in == 0:
                    center_kernel = self.bottom_right
                # going from as -> bs
                elif self.b_in == 0:
                    center_kernel = torch.zeros(size=(self.b_out, self.a_in, 1))
                # going from abs -> bs
                else:
                    bottom_left = torch.zeros(size=(self.b_out, self.a_in, 1))
                    center_kernel = torch.cat((bottom_left, self.bottom_right), dim=1)

            # We only have the top
            # going to -> as
            elif self.b_out == 0:
                # going from bs -> as
                if self.a_in == 0:
                    center_kernel = torch.zeros(size=(self.a_out, self.b_in, 1))
                # going from as -> as
                elif self.b_in == 0:
                    center_kernel = self.top_left
                # going from abs -> as
                else:
                    top_right = torch.zeros(size=(self.a_out, self.b_in, 1))
                    center_kernel = torch.cat((self.top_left, top_right), dim=1)

            else:
                # For the kernel to be anti-symmetric, we need to have zeros on the anti-symmetric parts:
                bottom_left = torch.zeros(size=(self.b_out, self.a_in, 1))
                top_right = torch.zeros(size=(self.a_out, self.b_in, 1))
                left = torch.cat((self.top_left, bottom_left), dim=0)
                right = torch.cat((top_right, self.bottom_right), dim=0)
                center_kernel = torch.cat((left, right), dim=1)
            if self.kernel_size > 1:
                kernel = torch.cat((self.left_kernel, center_kernel, right_kernel), dim=2)
            else:
                kernel = center_kernel
        else:
            kernel = torch.cat((self.left_kernel, right_kernel), dim=2)

        outputs = torch.nn.functional.conv1d(inputs,
                                             kernel,
                                             padding=self.padding,
                                             dilation=self.dilatation)
        return outputs


class IrrepActivationLayer(torch.nn.Module):
    """
    BN layer for a_n, b_n feature map
    """

    def __init__(self, a, b, placeholder=False):
        super(IrrepActivationLayer, self).__init__()
        self.a = a
        self.b = b
        self.placeholder = placeholder

    def call(self, inputs):
        if self.placeholder:
            return inputs
        a_outputs = None
        if self.a > 0:
            a_inputs = inputs[:, :, :self.a]
            a_outputs = torch.relu(a_inputs)
        if self.b > 0:
            b_inputs = inputs[:, :, self.a:]
            b_outputs = torch.tanh(b_inputs)
            if a_outputs is not None:
                return torch.cat((a_outputs, b_outputs), dim=-1)
            else:
                return b_outputs
        return a_outputs


class RegBatchNorm(torch.nn.Module):
    """
    BN layer for regular layers
    """

    def __init__(self, reg_dim, momentum=0.99, use_momentum=True, placeholder=False):
        super(RegBatchNorm, self).__init__()
        self.reg_dim = reg_dim
        self.placeholder = placeholder
        self.momentum = momentum
        self.use_momentum = use_momentum

        if not placeholder:
            self.passed = 0
            self.mu = torch.nn.Parameter(torch.zeros(size=reg_dim))
            self.sigma = torch.nn.Parameter(torch.ones(size=reg_dim))
            self.running_mu = torch.nn.Parameter(torch.zeros(size=reg_dim), requires_grad=False)
            self.running_sigma = torch.nn.Parameter(torch.ones(size=reg_dim), requires_grad=False)

    def forward(self, inputs):
        if self.placeholder:
            return inputs
        a = inputs.shape
        batch_size = a[0]
        length = a[1]
        division_over = batch_size * length
        modified_inputs = torch.cat(
            tensors=[inputs[:, :, :self.reg_dim],
                     inputs[:, :, self.reg_dim:][:, :, ::-1]],
            dim=1)

        if self.training:
            mu_batch = torch.mean(modified_inputs, dim=(0, 1))
            sigma_batch = torch.std(modified_inputs, dim=(0, 1)) + 0.0001
            train_normed_inputs = (modified_inputs - mu_batch) / sigma_batch * self.sigma + self.mu

            # Only update the running means in train mode
            if self.use_momentum:
                self.running_mu = self.running_mu * self.momentum + mu_batch * (1 - self.momentum)
                self.running_sigma = self.running_sigma * self.momentum + sigma_batch * (1 - self.momentum)
            else:
                self.running_mu = (self.running_mu * self.passed + division_over * mu_batch) / (
                        self.passed + division_over)
                self.running_sigma = (self.running_sigma * self.passed + division_over * sigma_batch) / (
                        self.passed + division_over)
                self.passed = self.passed + division_over
            train_true_normed_inputs = torch.cat(
                tensors=[train_normed_inputs[:, :length, :],
                         train_normed_inputs[:, length:, :][:, :, ::-1]],
                dim=2)
            return train_true_normed_inputs

        else:
            # Test mode
            test_normed_inputs = (modified_inputs - self.running_mu) / self.running_sigma * self.sigma + self.mu
            test_true_normed_inputs = torch.cat(
                tensors=[test_normed_inputs[:, :length, :],
                         test_normed_inputs[:, length:, :][:, :, ::-1]],
                dim=2)

            return test_true_normed_inputs


class IrrepBatchNorm(torch.nn.Module):
    """
    BN layer for a_n, b_n feature map
    """

    def __init__(self, a, b, placeholder=False, use_momentum=True, momentum=0.99):
        super(IrrepBatchNorm, self).__init__()
        self.a = a
        self.b = b
        self.placeholder = placeholder
        self.momentum = momentum
        self.use_momentum = use_momentum

        if not placeholder:
            self.passed = 0
            if a > 0:
                self.mu_a = torch.nn.Parameter(torch.zeros(size=a))
                self.sigma_a = torch.nn.Parameter(torch.ones(size=a))
                self.running_mu_a = torch.nn.Parameter(torch.zeros(size=a), requires_grad=False)
                self.running_sigma_a = torch.nn.Parameter(torch.ones(size=a), requires_grad=False)
            if b > 0:
                self.sigma_b = torch.nn.Parameter(torch.ones(size=b))
                self.running_sigma_b = torch.nn.Parameter(torch.ones(size=b), requires_grad=False)

    def forward(self, inputs):
        if self.placeholder:
            return inputs

        a = inputs.shape
        batch_size = a[0]
        length = a[1]
        division_over = batch_size * length

        if self.training:
            # We have to compute statistics and update the running means if in train
            train_a_outputs = None
            if self.a > 0:
                a_inputs = inputs[:, :, :self.a]
                mu_a_batch = torch.mean(a_inputs, dim=(0, 1))
                sigma_a_batch = torch.std(a_inputs, dim=(0, 1)) + 0.0001

                # print('inbatch mu', mu_a_batch.numpy())
                # print('inbatch sigma', sigma_a_batch.numpy())
                train_a_outputs = (a_inputs - mu_a_batch) / sigma_a_batch * self.sigma_a + self.mu_a

                # Momentum version :
                if self.use_momentum:
                    self.running_mu_a = self.running_mu_a * self.momentum + mu_a_batch * (1 - self.momentum)
                    self.running_sigma_a = self.running_sigma_a * self.momentum + sigma_a_batch * (1 - self.momentum)
                else:
                    self.running_mu_a = (self.running_mu_a * self.passed + division_over * mu_a_batch) / (
                            self.passed + division_over)
                    self.running_sigma_a = (self.running_sigma_a * self.passed + division_over * sigma_a_batch) / (
                            self.passed + division_over)

            # For b_dims, the problem is that we cannot compute a std from the mean as we include as a prior
            # that the mean is zero
            # We compute some kind of averaged over group action mean/std : std with a mean of zero.
            if self.b > 0:
                b_inputs = inputs[:, :, self.a:]
                numerator = torch.sqrt(torch.sum(torch.square(b_inputs), dim=(0, 1)))
                sigma_b_batch = numerator / torch.sqrt(division_over) + 0.0001
                train_b_outputs = b_inputs / sigma_b_batch * self.sigma_b

                # Uncomment and call with RC batch to see that the mean is actually zero
                # and the estimated std is the right one
                # mu_b_batch = tf.reduce_mean(b_inputs, axis=(0, 1))
                # sigma_b_batch_emp = tf.math.reduce_std(b_inputs, axis=(0, 1)) + 0.0001
                # print('inbatch mu', mu_b_batch.numpy())
                # print('inbatch sigma', sigma_b_batch_emp.numpy())
                # print('computed sigman', sigma_b_batch.numpy())

                # Momentum version
                if self.use_momentum:
                    self.running_sigma_b = self.running_sigma_b * self.momentum + sigma_b_batch * (1 - self.momentum)
                else:
                    self.running_sigma_b = (self.running_sigma_b * self.passed + division_over * sigma_b_batch) / (
                            self.passed + division_over)

                if train_a_outputs is not None:
                    train_outputs = torch.cat((train_a_outputs, train_b_outputs), dim=-1)
                else:
                    train_outputs = train_b_outputs

            else:
                train_outputs = train_a_outputs

            self.passed += division_over
            return train_outputs
        else:
            # ============== Compute test values ====================
            test_a_outputs = None
            if self.a > 0:
                a_inputs = inputs[:, :, :self.a]
                test_a_outputs = (a_inputs - self.running_mu_a) / self.running_sigma_a * self.sigma_a + self.mu_a

            # For b_dims, we compute some kind of averaged over group action mean/std
            if self.b > 0:
                b_inputs = inputs[:, :, self.a:]
                test_b_outputs = b_inputs / self.running_sigma_b * self.sigma_b
                if test_a_outputs is not None:
                    test_outputs = torch.cat((test_a_outputs, test_b_outputs), axis=-1)
                else:
                    test_outputs = test_b_outputs
            else:
                test_outputs = test_a_outputs

            return test_outputs


class IrrepConcatLayer(torch.nn.Module):
    """
    Concatenation layer to average both strands outputs
    """

    def __init__(self, a, b):
        super(IrrepConcatLayer, self).__init__()
        self.a = a
        self.b = b

    def forward(self, inputs):
        a_outputs = None
        if self.a > 0:
            a_outputs = (inputs[:, :self.a, :] + torch.flip(inputs[:, :self.a, :], [2])) / 2
        if self.b > 0:
            b_outputs = (inputs[:, self.a:, :] - torch.flip(inputs[:, self.a:, :], [2])) / 2
            if a_outputs is not None:
                return torch.cat((a_outputs, b_outputs), dim=1)
            else:
                return b_outputs
        return a_outputs


class RegConcatLayer(torch.nn.Module):
    """
    Concatenation layer to average both strands outputs
    """

    def __init__(self, reg):
        super(RegConcatLayer, self).__init__()
        self.reg = reg

    def forward(self, inputs):
        outputs = (inputs[:, :self.reg, :] + torch.flip(inputs[:, self.reg:, :], [1, 2])) / 2
        return outputs


class ToKmerLayer(torch.nn.Module):
    """
        to go from 1-hot strand in regular representation to another one
        """

    def __init__(self, k=3):
        super(ToKmerLayer, self).__init__()
        self.k = k
        # We have extra features in case of even filters because of palindromic k-mers
        self.features = 4 ** k if k % 2 else 4 ** k + (4 ** (k // 2))
        self.kernel = self.build_kernel()

    def build_kernel(self):
        """
        Build pattern extractor :
        We build ordered filters and then perform convolution.
        If the full pattern is detected ie we get a specific k_mer, we have a score of k so we can use thresholding
        to get the one hot representation.
        """
        import collections
        all_kernels = collections.deque()
        hashing_set = set()

        # In order to get the filters enumeration, we use the base 4 decomposition.
        # We then add the RC filter at the right position using a deque.
        # We have to hash our results because the filters are not contiguous i
        for i in range(4 ** self.k):
            # Get the string and array representation in base 4 for both forward and rc
            tempencoding = np.base_repr(i, 4, padding=self.k)[-self.k:]
            padded_split = [int(char) for char in tempencoding]
            rc_padded_split = [4 - (i + 1) for i in reversed(padded_split)]
            rc_hash = ''.join([str(i) for i in rc_padded_split])

            if rc_hash not in hashing_set:
                # If they are not yet in the filter bank, one-hot encode them and add them
                hashing_set.add(rc_hash)
                hashing_set.add(tempencoding)
                np_forward = np.array(padded_split)
                np_rc = np.array(rc_padded_split)

                one_hot_forward = np.eye(4)[np_forward].T
                one_hot_rc = np.eye(4)[np_rc].T
                all_kernels.append(one_hot_forward)
                all_kernels.appendleft(one_hot_rc)

        # We get (self.k, 4, n_kmers_filters) shape that checks the equivariance condition
        # n_kmers = 4**k for odd k and 4**k + 4**(k//2) for even because we repeat palindromic units
        all_kernels = list(all_kernels)
        kernel = np.stack(all_kernels, axis=0)
        kernel = torch.from_numpy(kernel)
        kernel.requires_grad = False
        return kernel

    def forward(self, inputs):
        if self.k == 1:
            return inputs
        with torch.no_grad():
            outputs = torch.nn.functional.conv1d(inputs,
                                                 self.kernel,
                                                 padding=0)
        outputs = outputs >= self.k
        outputs = outputs.int()
        return outputs


def reg_action(sequence_tensor):
    """
    apply the regular action on the input tensor : flips along the channels and length axis
    :param sequence_tensor: an input torch tensor of shape (batch, features, length)
    :return: the modified tensor
    """
    return torch.flip(sequence_tensor, [1, 2])


def a_action(sequence_tensor):
    """
    apply the a action on the input tensor : flips along the length axis
    :param sequence_tensor: an input torch tensor of shape (batch, features, length)
    :return: the modified tensor
    """
    return torch.flip(sequence_tensor, [2])


def b_action(sequence_tensor):
    """
    apply the b action on the input tensor : flips along the length axis and change sign
    :param sequence_tensor: an input torch tensor of shape (batch, features, length)
    :return: the modified tensor
    """
    return -torch.flip(sequence_tensor, [2])


# class EquiNetBinary:
#
#     def __init__(self,
#                  filters=((2, 2), (2, 2), (2, 2), (1, 1)),
#                  kernel_sizes=(5, 5, 7, 7),
#                  pool_size=40,
#                  pool_length=20,
#                  out_size=1,
#                  placeholder_bn=False):
#         """
#         First map the regular representation to irrep setting
#         Then goes from one setting to another.
#         """
#
#         # assert len(filters) == len(kernel_sizes)
#         # self.input_dense = 1000
#         # successive_shrinking = (i - 1 for i in kernel_sizes)
#         # self.input_dense = 1000 - sum(successive_shrinking)
#
#         # First mapping goes from the input to an irrep feature space
#         first_kernel_size = kernel_sizes[0]
#         first_a, first_b = filters[0]
#         self.last_a, self.last_b = filters[-1]
#         self.reg_irrep = RegToIrrepConv(reg_in=2,
#                                         a_out=first_a,
#                                         b_out=first_b,
#                                         kernel_size=first_kernel_size)
#         self.first_bn = IrrepBatchNorm(a=first_a, b=first_b, placeholder=placeholder_bn)
#         self.first_act = IrrepActivationLayer(a=first_a, b=first_b)
#
#         # Now add the intermediate layer : sequence of conv, BN, activation
#         self.irrep_layers = []
#         self.bn_layers = []
#         self.activation_layers = []
#         for i in range(1, len(filters)):
#             prev_a, prev_b = filters[i - 1]
#             next_a, next_b = filters[i]
#             self.irrep_layers.append(IrrepToIrrepConv(
#                 a_in=prev_a,
#                 b_in=prev_b,
#                 a_out=next_a,
#                 b_out=next_b,
#                 kernel_size=kernel_sizes[i],
#             ))
#             self.bn_layers.append(IrrepBatchNorm(a=next_a, b=next_b, placeholder=placeholder_bn))
#             # Don't add activation if it's the last layer
#             placeholder = (i == len(filters) - 1)
#             self.activation_layers.append(IrrepActivationLayer(a=next_a,
#                                                                b=next_b,
#                                                                placeholder=placeholder))
#
#         self.pool = kl.MaxPooling1D(pool_length=pool_size, strides=pool_length)
#         self.flattener = kl.Flatten()
#         self.dense = kl.Dense(out_size, activation='sigmoid')
#
#     def func_api_model(self):
#         inputs = keras.layers.Input(shape=(1000, 4), dtype="float32")
#         x = self.reg_irrep(inputs)
#         x = self.first_bn(x)
#         x = self.first_act(x)
#
#         for irrep_layer, bn_layer, activation_layer in zip(self.irrep_layers, self.bn_layers, self.activation_layers):
#             x = irrep_layer(x)
#             x = bn_layer(x)
#             x = activation_layer(x)
#
#         # Average two strands predictions, pool and go through Dense
#         x = IrrepConcatLayer(a=self.last_a, b=self.last_b)(x)
#         x = self.pool(x)
#         x = self.flattener(x)
#         outputs = self.dense(x)
#         model = keras.Model(inputs, outputs)
#         return model
#
#     def eager_call(self, inputs):
#         rcinputs = inputs[:, ::-1, ::-1]
#
#         x = self.reg_irrep(inputs)
#         x = self.first_bn(x)
#         x = self.first_act(x)
#
#         rcx = self.reg_irrep(rcinputs)
#         rcx = self.first_bn(rcx)
#         rcx = self.first_act(rcx)
#
#         # print(x.numpy()[0, :5, :])
#         # print('reversed')
#         # print(rcx[:, ::-1, :].numpy()[0, :5, :])
#         # print()
#
#         for irrep_layer, bn_layer, activation_layer in zip(self.irrep_layers, self.bn_layers, self.activation_layers):
#             x = irrep_layer(x)
#             x = bn_layer(x)
#             x = activation_layer(x)
#
#             rcx = irrep_layer(rcx)
#             rcx = bn_layer(rcx)
#             rcx = activation_layer(rcx)
#
#         # Print the beginning of both strands to see it adds up in concat
#         # print(x.shape)
#         # print(x.numpy()[0, :5, :])
#         # print('end')
#         # print(rcx.numpy()[0, :5, :])
#         # print('reversed')
#         # print(rcx[:, ::-1, :].numpy()[0, :5, :])
#         # print()
#
#         # Average two strands predictions
#         x = IrrepConcatLayer(a=self.last_a, b=self.last_b)(x)
#         rcx = IrrepConcatLayer(a=self.last_a, b=self.last_b)(rcx)
#
#         # print(x.numpy()[0, :5, :])
#         # print('reversed')
#         # print(rcx.numpy()[0, :5, :])
#         # print()
#
#         x = self.pool(x)
#         rcx = self.pool(rcx)
#
#         # print(x.shape)
#         # print(x.numpy()[0, :5, :])
#         # print('reversed')
#         # print(rcx.numpy()[0, :5, :])
#         # print()
#
#         x = self.flattener(x)
#         rcx = self.flattener(rcx)
#
#         # print(x.shape)
#         # print(x.numpy()[0, :5])
#         # print('reversed')
#         # print(rcx.numpy()[0, :5])
#         # print()
#
#         outputs = self.dense(x)
#         rcout = self.dense(rcx)
#
#         # print(outputs.shape)
#         # print(outputs.numpy()[0, :5])
#         # print('reversed')
#         # print(rcout.numpy()[0, :5])
#         # print()
#         return outputs
#
#
# # Loss Function
# def multinomial_nll(true_counts, logits):
#     """Compute the multinomial negative log-likelihood
#     Args:
#       true_counts: observed count values
#       logits: predicted logit values
#     """
#     counts_per_example = tf.reduce_sum(true_counts, axis=-1)
#     dist = tf.compat.v1.distributions.Multinomial(total_count=counts_per_example,
#                                                   logits=logits)
#     return (-tf.reduce_sum(dist.log_prob(true_counts)) /
#             tf.cast((tf.shape(true_counts)[0]), tf.float32))
#
#
# class MultichannelMultinomialNLL(object):
#     def __init__(self, n=2):
#         self.__name__ = "MultichannelMultinomialNLL"
#         self.__class_name__ = "MultichannelMultinomialNLL"
#         self.n = n
#
#     def __call__(self, true_counts, logits):
#         total = 0
#         for i in range(self.n):
#             loss = multinomial_nll(true_counts[..., i], logits[..., i])
#             if i == 0:
#                 total = loss
#             else:
#                 total += loss
#         return total
#
#     def get_config(self):
#         return {"n": self.n}
#
#
# class EquiNetBP(torch.nn.Module):
#     def __init__(self,
#                  dataset,
#                  input_seq_len=1346,
#                  c_task_weight=0,
#                  p_task_weight=1,
#                  filters=((64, 64), (64, 64), (64, 64), (64, 64), (64, 64), (64, 64), (64, 64)),
#                  kernel_sizes=(21, 3, 3, 3, 3, 3, 3, 75),
#                  outconv_kernel_size=75,
#                  weight_decay=0.01,
#                  optimizer='Adam',
#                  lr=0.001,
#                  kernel_initializer="glorot_uniform",
#                  seed=42,
#                  is_add=True,
#                  kmers=1,
#                  **kwargs):
#         super(EquiNetBP, self).__init__()
#
#         self.dataset = dataset
#         self.input_seq_len = input_seq_len
#         self.c_task_weight = c_task_weight
#         self.p_task_weight = p_task_weight
#         self.filters = filters
#         self.kernel_sizes = kernel_sizes
#         self.outconv_kernel_size = outconv_kernel_size
#         self.optimizer = optimizer
#         self.weight_decay = weight_decay
#         self.lr = lr
#         self.learning_rate = lr
#         self.kernel_initializer = kernel_initializer
#         self.seed = seed
#         self.is_add = is_add
#         self.n_dil_layers = len(filters) - 1
#
#         # Add k-mers, if k=1, it's just a placeholder
#         self.kmers = int(kmers)
#         self.to_kmer = ToKmerLayer(k=self.kmers)
#         self.conv1_kernel_size = kernel_sizes[0] - self.kmers + 1
#         reg_in = self.to_kmer.features // 2
#         first_a, first_b = filters[0]
#         self.first_conv = RegToIrrepConv(reg_in=reg_in,
#                                          a_out=first_a,
#                                          b_out=first_b,
#                                          kernel_size=self.conv1_kernel_size,
#                                          padding=0)
#         self.first_act = IrrepActivationLayer(a=first_a, b=first_b)
#
#         # Now add the intermediate layer : sequence of conv, activation
#         self.irrep_layers = []
#         self.activation_layers = []
#         self.croppings = []
#         for i in range(1, len(filters)):
#             prev_a, prev_b = filters[i - 1]
#             next_a, next_b = filters[i]
#             dilation_rate = 2 ** i
#             self.irrep_layers.append(IrrepToIrrepConv(
#                 a_in=prev_a,
#                 b_in=prev_b,
#                 a_out=next_a,
#                 b_out=next_b,
#                 kernel_size=kernel_sizes[i],
#                 dilatation=dilation_rate
#             ))
#             self.croppings.append((kernel_sizes[i] - 1) * dilation_rate)
#             # self.bn_layers.append(IrrepBatchNorm(a=next_a, b=next_b, placeholder=placeholder_bn))
#             self.activation_layers.append(IrrepActivationLayer(a=next_a, b=next_b))
#
#         self.last_a, self.last_b = filters[-1]
#         self.prebias = IrrepToRegConv(reg_out=1,
#                                       a_in=self.last_a,
#                                       b_in=self.last_b,
#                                       kernel_size=self.outconv_kernel_size,
#                                       padding=0)
#         self.last = RegToRegConv(reg_in=3,
#                                  reg_out=1,
#                                  kernel_size=1,
#                                  padding=0)
#
#         self.last_count = IrrepToRegConv(a_in=self.last_a + 2,
#                                          b_in=self.last_b,
#                                          reg_out=1,
#                                          kernel_size=1,
#                                          kernel_initializer=self.kernel_initializer)
#
#     def get_output_profile_len(self):
#         embedding_len = self.input_seq_len
#         embedding_len -= (self.conv1_kernel_size - 1)
#         for cropping in self.croppings:
#             embedding_len -= cropping
#         out_profile_len = embedding_len - (self.outconv_kernel_size - 1)
#         return out_profile_len
#
#     def trim_flanks_of_inputs(self, inputs, output_len, width_to_trim, filters):
#         layer = keras.layers.Lambda(
#             function=lambda x: x[:, int(0.5 * (width_to_trim)):-(width_to_trim - int(0.5 * (width_to_trim)))],
#             output_shape=(output_len, filters))(inputs)
#         return layer
#
#     def get_inputs(self):
#         out_pred_len = self.get_output_profile_len()
#
#         inp = kl.Input(shape=(self.input_seq_len, 4), name='sequence')
#         if self.dataset == "SPI1":
#             bias_counts_input = kl.Input(shape=(1,), name="control_logcount")
#             bias_profile_input = kl.Input(shape=(out_pred_len, 2),
#                                           name="control_profile")
#         else:
#             bias_counts_input = kl.Input(shape=(2,), name="patchcap.logcount")
#             # if working with raw counts, go from logcount->count
#             bias_profile_input = kl.Input(shape=(1000, 2),
#                                           name="patchcap.profile")
#         return inp, bias_counts_input, bias_profile_input
#
#     def get_names(self):
#         if self.dataset == "SPI1":
#             countouttaskname = "task0_logcount"
#             profileouttaskname = "task0_profile"
#         elif self.dataset == 'NANOG':
#             countouttaskname = "CHIPNexus.NANOG.logcount"
#             profileouttaskname = "CHIPNexus.NANOG.profile"
#         elif self.dataset == "OCT4":
#             countouttaskname = "CHIPNexus.OCT4.logcount"
#             profileouttaskname = "CHIPNexus.OCT4.profile"
#         elif self.dataset == "KLF4":
#             countouttaskname = "CHIPNexus.KLF4.logcount"
#             profileouttaskname = "CHIPNexus.KLF4.profile"
#         elif self.dataset == "SOX2":
#             countouttaskname = "CHIPNexus.SOX2.logcount"
#             profileouttaskname = "CHIPNexus.SOX2.profile"
#         else:
#             raise ValueError("The dataset asked does not exist")
#         return countouttaskname, profileouttaskname
#
#     def get_keras_model(self):
#         """
#         Make a first convolution, then use skip connections with dilatations (that shrink the input)
#         to get 'combined_conv'
#
#         Then create two heads :
#          - one is used to predict counts (and has a weight of zero in the loss)
#          - one is used to predict the profile
#         """
#         sequence_input, bias_counts_input, bias_profile_input = self.get_inputs()
#
#         kmer_inputs = self.to_kmer(sequence_input)
#         curr_layer_size = self.input_seq_len - (self.conv1_kernel_size - 1) - (self.kmers - 1)
#         prev_layers = self.first_conv(kmer_inputs)
#
#         for i, (conv_layer, activation_layer, cropping) in enumerate(zip(self.irrep_layers,
#                                                                          self.activation_layers,
#                                                                          self.croppings)):
#
#             conv_output = conv_layer(prev_layers)
#             conv_output = activation_layer(conv_output)
#             curr_layer_size = curr_layer_size - cropping
#
#             trimmed_prev_layers = self.trim_flanks_of_inputs(inputs=prev_layers,
#                                                              output_len=curr_layer_size,
#                                                              width_to_trim=cropping,
#                                                              filters=self.filters[i][0] + self.filters[i][1])
#             if self.is_add:
#                 prev_layers = kl.add([trimmed_prev_layers, conv_output])
#             else:
#                 prev_layers = kl.average([trimmed_prev_layers, conv_output])
#
#         combined_conv = prev_layers
#
#         countouttaskname, profileouttaskname = self.get_names()
#
#         # ============== Placeholder for counts =================
#         count_out = kl.Lambda(lambda x: x, name=countouttaskname)(bias_counts_input)
#
#         # gap_combined_conv = kl.GlobalAvgPool1D()(combined_conv)
#         # stacked = kl.Reshape((1, -1))(kl.concatenate([
#         #     # concatenation of the bias layer both before and after
#         #     # is needed for rc symmetry
#         #     kl.Lambda(lambda x: x[:, ::-1])(bias_counts_input),
#         #     gap_combined_conv,
#         #     bias_counts_input], axis=-1))
#         # convout = self.last_count(stacked)
#         # count_out = kl.Reshape((-1,), name=countouttaskname)(convout)
#
#         # ============== Profile prediction ======================
#         profile_out_prebias = self.prebias(combined_conv)
#
#         # # concatenation of the bias layer both before and after is needed for rc symmetry
#         concatenated = kl.concatenate([kl.Lambda(lambda x: x[:, :, ::-1])(bias_profile_input),
#                                        profile_out_prebias,
#                                        bias_profile_input], axis=-1)
#         profile_out = self.last(concatenated)
#         profile_out = kl.Lambda(lambda x: x, name=profileouttaskname)(profile_out)
#
#         model = keras.models.Model(
#             inputs=[sequence_input, bias_counts_input, bias_profile_input],
#             outputs=[count_out, profile_out])
#         model.compile(keras.optimizers.Adam(lr=self.lr),
#                       loss=['mse', MultichannelMultinomialNLL(2)],
#                       loss_weights=[self.c_task_weight, self.p_task_weight])
#         # print(model.summary())
#         return model
#
#     def eager_call(self, sequence_input, bias_counts_input, bias_profile_input):
#         """
#         Testing only
#         """
#         kmer_inputs = self.to_kmer(sequence_input)
#         curr_layer_size = self.input_seq_len - (self.conv1_kernel_size - 1) - (self.kmers - 1)
#         prev_layers = self.first_conv(kmer_inputs)
#
#         for i, (conv_layer, activation_layer, cropping) in enumerate(zip(self.irrep_layers,
#                                                                          self.activation_layers,
#                                                                          self.croppings)):
#
#             conv_output = conv_layer(prev_layers)
#             conv_output = activation_layer(conv_output)
#             curr_layer_size = curr_layer_size - cropping
#
#             trimmed_prev_layers = self.trim_flanks_of_inputs(inputs=prev_layers,
#                                                              output_len=curr_layer_size,
#                                                              width_to_trim=cropping,
#                                                              filters=self.filters[i][0] + self.filters[i][1])
#             if self.is_add:
#                 prev_layers = trimmed_prev_layers + conv_output
#             else:
#                 prev_layers = (trimmed_prev_layers + conv_output) / 2
#
#         combined_conv = prev_layers
#
#         # Placeholder for counts
#         count_out = bias_counts_input
#
#         # Profile prediction
#         profile_out_prebias = self.prebias(combined_conv)
#
#         # concatenation of the bias layer both before and after is needed for rc symmetry
#         rc_profile_input = bias_profile_input[:, :, ::-1]
#         concatenated = torch.cat([rc_profile_input,
#                                   profile_out_prebias,
#                                   bias_profile_input], dim=-1)
#
#         profile_out = self.last(concatenated)
#
#         return count_out, profile_out
#
#
# class RCNetBinary:
#
#     def __init__(self,
#                  filters=(2, 2, 2),
#                  kernel_sizes=(5, 5, 7),
#                  pool_size=40,
#                  pool_length=20,
#                  out_size=1,
#                  placeholder_bn=False):
#         """
#         First map the regular representation to irrep setting
#         Then goes from one setting to another.
#         """
#
#         # First mapping goes from the input to an irrep feature space
#         first_kernel_size = kernel_sizes[0]
#         first_reg = filters[0]
#         self.last_reg = filters[-1]
#
#         self.first_conv = RegToRegConv(reg_in=2,
#                                        reg_out=first_reg,
#                                        kernel_size=first_kernel_size)
#         self.first_bn = RegBatchNorm(reg_dim=first_reg, placeholder=placeholder_bn)
#         self.first_act = kl.core.Activation("relu")
#
#         # Now add the intermediate layer : sequence of conv, BN, activation
#         self.irrep_layers = []
#         self.bn_layers = []
#         self.activation_layers = []
#         for i in range(1, len(filters)):
#             prev = filters[i - 1]
#             next = filters[i]
#             self.irrep_layers.append(RegToRegConv(
#                 reg_in=prev,
#                 reg_out=next,
#                 kernel_size=kernel_sizes[i],
#             ))
#             self.bn_layers.append(RegBatchNorm(reg_dim=next, placeholder=placeholder_bn))
#             self.activation_layers.append(kl.core.Activation("relu"))
#
#         self.pool = kl.MaxPooling1D(pool_length=pool_size, strides=pool_length)
#         self.flattener = kl.Flatten()
#         self.dense = kl.Dense(out_size, activation='sigmoid')
#
#     def func_api_model(self):
#         inputs = keras.layers.Input(shape=(1000, 4), dtype="float32")
#         x = self.first_conv(inputs)
#         x = self.first_bn(x)
#         x = self.first_act(x)
#
#         for conv_layer, bn_layer, activation_layer in zip(self.irrep_layers, self.bn_layers, self.activation_layers):
#             x = conv_layer(x)
#             x = bn_layer(x)
#             x = activation_layer(x)
#
#         # Average two strands predictions, pool and go through Dense
#         x = RegConcatLayer(reg=self.last_reg)(x)
#         x = self.pool(x)
#         x = self.flattener(x)
#         outputs = self.dense(x)
#         model = keras.Model(inputs, outputs)
#         return model
#
#     def eager_call(self, inputs):
#         rcinputs = inputs[:, ::-1, ::-1]
#
#         x = self.first_conv(inputs)
#         x = self.first_bn(x)
#         x = self.first_act(x)
#
#         rcx = self.first_conv(rcinputs)
#         rcx = self.first_bn(rcx)
#         rcx = self.first_act(rcx)
#
#         # print(x.numpy()[0, :5, :])
#         # print('reversed')
#         # print(rcx[:, ::-1, :].numpy()[0, :5, :])
#         # print()
#
#         for conv_layer, bn_layer, activation_layer in zip(self.irrep_layers, self.bn_layers, self.activation_layers):
#             x = conv_layer(x)
#             x = bn_layer(x)
#             x = activation_layer(x)
#
#             rcx = conv_layer(rcx)
#             rcx = bn_layer(rcx)
#             rcx = activation_layer(rcx)
#
#         # Print the beginning of both strands to see it adds up in concat
#         # print(x.shape)
#         # print(x.numpy()[0, :5, :])
#         # print('end')
#         # print(rcx.numpy()[0, :5, :])
#         # print('reversed')
#         # print(rcx[:, ::-1, :].numpy()[0, :5, :])
#
#         # Average two strands predictions
#         x = RegConcatLayer(reg=self.last_reg)(x)
#         rcx = RegConcatLayer(reg=self.last_reg)(rcx)
#
#         # print(x.numpy()[0, :5, :])
#         # print('reversed')
#         # print(rcx.numpy()[0, :5, :])
#         # print()
#
#         x = self.pool(x)
#         rcx = self.pool(rcx)
#
#         # print(x.shape)
#         # print(x.numpy()[0, :5, :])
#         # print('reversed')
#         # print(rcx.numpy()[0, :5, :])
#         # print()
#
#         x = self.flattener(x)
#         rcx = self.flattener(rcx)
#
#         # print(x.numpy()[0, :5])
#         # print('reversed')
#         # print(rcx.numpy()[0, :5])
#         # print()
#
#         outputs = self.dense(x)
#         rcout = self.dense(rcx)
#
#         # print(outputs.shape)
#         # print(outputs.numpy()[0, :5])
#         # print('reversed')
#         # print(rcout.numpy()[0, :5])
#         # print()
#         return outputs


if __name__ == '__main__':
    pass

    curr_seed = 42
    np.random.seed(curr_seed)
    torch.random.manual_seed(curr_seed)


    def random_one_hot(size=(2, 100)):
        """
        This is copied from the keras use case, so we have to flip the data in the end
        The output has shape (batch, features, num_nucleotides)
        :param size:
        :return:
        """
        bs, len = size
        numel = bs * len
        randints_np = np.random.randint(0, 3, size=numel)
        one_hot_np = np.eye(4)[randints_np]
        one_hot_np = np.reshape(one_hot_np, newshape=(bs, len, 4))
        one_hot_torch = torch.from_numpy(one_hot_np)
        one_hot_torch = torch.transpose(one_hot_torch, 1, 2)
        one_hot_torch.requires_grad = False
        return one_hot_torch


    x = random_one_hot(size=(1, 40))
    rcx = torch.flip(x, [1, 2])
    # print(x.shape)

    a_1 = 2
    b_1 = 1
    a_2 = 2
    b_2 = 1
    reg_out = 3

    regreg = RegToRegConv(reg_in=2, reg_out=reg_out, kernel_size=5, dilatation=1, padding=0)
    concat_reg = RegConcatLayer(reg=reg_out)
    regirrep = RegToIrrepConv(reg_in=reg_out, a_out=a_1, b_out=b_1, kernel_size=5, dilatation=1, padding=0)
    irrepirrep = IrrepToIrrepConv(a_in=a_1, b_in=b_1, a_out=a_2, b_out=b_2, kernel_size=5, dilatation=1, padding=0)
    irrepreg = IrrepToRegConv(a_in=a_2, b_in=b_2, reg_out=1, kernel_size=5, dilatation=1, padding=0)
    concat_irrep = IrrepConcatLayer(a=a_2, b=b_2)

    with torch.no_grad():
        out = regreg(x)
        rc_out = regreg(rcx)
        # print(out.shape)
        assert torch.allclose(out, reg_action(rc_out))

        concat_reg_out = concat_reg(out)
        rc_concat_reg_out = concat_reg(rc_out)
        assert torch.allclose(concat_reg_out, rc_concat_reg_out)

        out = regirrep(out)
        rc_out = regirrep(rc_out)
        assert torch.allclose(out[:, :a_1], a_action(rc_out[:, :a_1]))
        assert torch.allclose(out[:, a_1:], b_action(rc_out[:, a_1:]))

        out = irrepirrep(out)
        rc_out = irrepirrep(rc_out)
        assert torch.allclose(out[:, :a_2], a_action(rc_out[:, :a_2]))
        assert torch.allclose(out[:, a_2:], b_action(rc_out[:, a_2:]))

        concat_irrep_out = concat_irrep(out)
        rc_concat_irrep_out = concat_irrep(rc_out)
        assert torch.allclose(concat_irrep_out, rc_concat_irrep_out)

        out = irrepreg(out)
        rc_out = irrepreg(rc_out)
        assert torch.allclose(out, reg_action(rc_out))

        # print(out)
        # print(rcout2)

        # TEST K-MER
        tokmer = ToKmerLayer(2)
        kmer_x = tokmer(x)
        kmer_rcx = tokmer(rcx)
        # Should be full ones with a few 2 because of palindromic representation.
        # print(kmer_x.sum(axis=1))
        # The flipped version should revert correctly
        assert torch.allclose(kmer_x, reg_action(kmer_rcx))
