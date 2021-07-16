import numpy as np
import torch
from torch import nn

"""
    The reg_in, reg_out arguments should be understood as the number of cycles.
    This correspond to half the input dimension (so for the input, we have 4 nucleotides, so reg=2)

    The a_n are of type +1, the b_n of type -1
"""


def create_xavier_convparameter(shape):
    """
    Small util function to create and initialize parameters in a custom layer.
    :param shape:
    :return:
    """
    tensor = torch.zeros(size=shape, dtype=torch.float32)
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
    Mapping from one regular layer to an irrep one
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

        if self.kernel_size > 1:
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
                top_right = torch.flip(self.top_left, [1])
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
    Mapping from one irrep layer to a regular type one
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
                    kernel = torch.cat((self.left_kernel, right, right_kernel), dim=2)
                else:
                    kernel = right
            elif self.b_in == 0:
                bottom_left = torch.flip(self.top_left, [0])
                left = torch.cat((self.top_left, bottom_left), dim=0)
                if self.kernel_size > 1:
                    kernel = torch.cat((self.left_kernel, left, right_kernel), dim=2)
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
    Mapping from one irrep layer to another irrep one
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
                    center_kernel = torch.zeros(size=(self.b_out, self.a_in, 1), dtype=torch.float32)
                # going from as -> as
                elif self.b_out == 0:
                    center_kernel = self.top_left
                # going from as -> abs
                else:
                    bottom_left = torch.zeros(size=(self.b_out, self.a_in, 1), dtype=torch.float32)
                    center_kernel = torch.cat((self.top_left, bottom_left), dim=0)

            # We only have the right part
            # going from bs ->
            elif self.a_in == 0:
                # going from bs -> bs
                if self.a_out == 0:
                    center_kernel = self.bottom_right
                # going from bs -> as
                elif self.b_out == 0:
                    center_kernel = torch.zeros(size=(self.a_out, self.b_in, 1), dtype=torch.float32)
                # going from bs -> abs
                else:
                    top_right = torch.zeros(size=(self.a_out, self.b_in, 1), dtype=torch.float32)
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
                    center_kernel = torch.zeros(size=(self.b_out, self.a_in, 1), dtype=torch.float32)
                # going from abs -> bs
                else:
                    bottom_left = torch.zeros(size=(self.b_out, self.a_in, 1), dtype=torch.float32)
                    center_kernel = torch.cat((bottom_left, self.bottom_right), dim=1)

            # We only have the top
            # going to -> as
            elif self.b_out == 0:
                # going from bs -> as
                if self.a_in == 0:
                    center_kernel = torch.zeros(size=(self.a_out, self.b_in, 1), dtype=torch.float32)
                # going from as -> as
                elif self.b_in == 0:
                    center_kernel = self.top_left
                # going from abs -> as
                else:
                    top_right = torch.zeros(size=(self.a_out, self.b_in, 1), dtype=torch.float32)
                    center_kernel = torch.cat((self.top_left, top_right), dim=1)

            else:
                # For the kernel to be anti-symmetric, we need to have zeros on the anti-symmetric parts:
                bottom_left = torch.zeros(size=(self.b_out, self.a_in, 1), dtype=torch.float32)
                top_right = torch.zeros(size=(self.a_out, self.b_in, 1), dtype=torch.float32)
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
    Activation layer for a_n, b_n feature map
    """

    def __init__(self, a, b, placeholder=False):
        super(IrrepActivationLayer, self).__init__()
        self.a = a
        self.b = b
        self.placeholder = placeholder

    def forward(self, inputs):
        if self.placeholder:
            return inputs
        a_outputs = None
        if self.a > 0:
            a_inputs = inputs[:, :self.a, :]
            a_outputs = torch.relu(a_inputs)
        if self.b > 0:
            b_inputs = inputs[:, self.a:, :]
            b_outputs = torch.tanh(b_inputs)
            if a_outputs is not None:
                return torch.cat((a_outputs, b_outputs), dim=1)
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
            self.mu = torch.nn.Parameter(torch.zeros(size=(1, reg_dim, 1)))
            self.sigma = torch.nn.Parameter(torch.ones(size=(1, reg_dim, 1)))
            self.register_buffer('running_mu', torch.zeros(size=(1, reg_dim, 1), requires_grad=False))
            self.register_buffer('running_sigma', torch.zeros(size=(1, reg_dim, 1), requires_grad=False))

    def forward(self, inputs):
        if self.placeholder:
            return inputs
        a = inputs.shape
        batch_size = a[0]
        length = a[2]
        division_over = batch_size * length
        modified_inputs = torch.cat(
            tensors=(inputs[:, :self.reg_dim, :],
                     torch.flip(inputs[:, self.reg_dim:, :], [1])),
            dim=2)

        if self.training:
            mu_batch = torch.mean(modified_inputs, dim=(0, 2))
            sigma_batch = torch.std(modified_inputs, dim=(0, 2)) + 0.0001
            mu_batch, sigma_batch = mu_batch[None, :, None], sigma_batch[None, :, None]
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
                tensors=(train_normed_inputs[:, :, :length],
                         torch.flip(train_normed_inputs[:, :, length:], [1])),
                dim=1)

            return train_true_normed_inputs

        else:
            # Test mode
            test_normed_inputs = (modified_inputs - self.running_mu) / self.running_sigma * self.sigma + self.mu
            test_true_normed_inputs = torch.cat(
                tensors=(test_normed_inputs[:, :, :length],
                         torch.flip(test_normed_inputs[:, :, length:], [1])),
                dim=1)
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
                self.mu_a = torch.nn.Parameter(torch.zeros(size=(1, a, 1)))
                self.sigma_a = torch.nn.Parameter(torch.ones(size=(1, a, 1)))

                self.register_buffer('running_mu_a', torch.zeros(size=(1, a, 1), requires_grad=False))
                self.register_buffer('running_sigma_a', torch.zeros(size=(1, a, 1), requires_grad=False))
            if b > 0:
                self.sigma_b = torch.nn.Parameter(torch.ones(size=(1, b, 1)))
                self.register_buffer('running_sigma_b', torch.zeros(size=(1, b, 1), requires_grad=False))

    def forward(self, inputs):
        if self.placeholder:
            return inputs

        a = inputs.shape
        batch_size = a[0]
        length = a[2]
        division_over = torch.tensor(batch_size * length, dtype=torch.float32)

        if self.training:
            # We have to compute statistics and update the running means if in train
            train_a_outputs = None
            if self.a > 0:
                a_inputs = inputs[:, :self.a, :]
                mu_a_batch = torch.mean(a_inputs, dim=(0, 2))
                sigma_a_batch = torch.std(a_inputs, dim=(0, 2)) + 0.0001
                mu_a_batch, sigma_a_batch = mu_a_batch[None, :, None], sigma_a_batch[None, :, None]
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
                b_inputs = inputs[:, self.a:, :]
                numerator = torch.sqrt(torch.sum(torch.square(b_inputs), dim=(0, 2)))
                sigma_b_batch = numerator / torch.sqrt(division_over) + 0.0001
                sigma_b_batch = sigma_b_batch[None, :, None]
                train_b_outputs = b_inputs / sigma_b_batch * self.sigma_b

                # Momentum version
                if self.use_momentum:
                    self.running_sigma_b = self.running_sigma_b * self.momentum + sigma_b_batch * (1 - self.momentum)
                else:
                    self.running_sigma_b = (self.running_sigma_b * self.passed + division_over * sigma_b_batch) / (
                            self.passed + division_over)

                if train_a_outputs is not None:
                    train_outputs = torch.cat((train_a_outputs, train_b_outputs), dim=1)
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
                a_inputs = inputs[:, :self.a, :]
                test_a_outputs = (a_inputs - self.running_mu_a) / self.running_sigma_a * self.sigma_a + self.mu_a

            # For b_dims, we compute some kind of averaged over group action mean/std
            if self.b > 0:
                b_inputs = inputs[:, self.a:, :]
                test_b_outputs = b_inputs / self.running_sigma_b * self.sigma_b
                if test_a_outputs is not None:
                    test_outputs = torch.cat((test_a_outputs, test_b_outputs), dim=1)
                else:
                    test_outputs = test_b_outputs
            else:
                test_outputs = test_a_outputs

            return test_outputs


class RegConcatLayer(torch.nn.Module):
    """
    Concatenation layer to average both strands outputs for a regular feature map
    """

    def __init__(self, reg):
        super(RegConcatLayer, self).__init__()
        self.reg = reg

    def forward(self, inputs):
        outputs = (inputs[:, :self.reg, :] + torch.flip(inputs[:, self.reg:, :], [1, 2])) / 2
        return outputs


class IrrepConcatLayer(torch.nn.Module):
    """
    Concatenation layer to average both strands outputs for an irrep feature map
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


class ToKmerLayer(torch.nn.Module):
    """
    To go from a one hot encoding of single nucleotides to one using k-mers one hot encoding.
    This new encoding still can be acted upon using regular representation : symmetric kmers have a symmetric indexing
    To do so, we build an ordered one hot pattern matching kernel, and use thresholding to get the one hot encoding.

    Careful : the RC palindromic kmers (TA for instance) are encoded twice,
    resulting in some dimensions having two ones activated.
    However this avoids us having to make special cases and using '+1' type feature maps and
    enables the use of just using regular representation on the output
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
        kernel = torch.from_numpy(kernel).float()
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
        outputs = outputs.int().float()
        return outputs


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


def random_one_hot(size=(2, 100)):
    """
    This is copied from the keras use case, so we have to flip the data in the end
    The output has shape (batch, features, num_nucleotides)
    :param size: the size of first dimensions (batch_size, length)
    :return: a one hot, nucleotide like tensor of shape (batch_size, 4, length)
    """
    bs, len = size
    numel = bs * len
    randints_np = np.random.randint(0, 3, size=numel)
    one_hot_np = np.eye(4, dtype=np.float32)[randints_np]
    one_hot_np = np.reshape(one_hot_np, newshape=(bs, len, 4))
    one_hot_torch = torch.from_numpy(one_hot_np)
    one_hot_torch = torch.transpose(one_hot_torch, 1, 2)
    one_hot_torch.requires_grad = False
    return one_hot_torch


def test_layers(a_1=2,
                b_1=1,
                a_2=2,
                b_2=3,
                reg_out=3,
                k_1=1,
                k_2=1,
                k_3=1,
                k_4=1):
    """
    Test the main layers by making a forward with a fixed layer parameters settings and checking equivariance
    """
    x = random_one_hot(size=(1, 40))
    rcx = torch.flip(x, [1, 2])

    regreg = RegToRegConv(reg_in=2, reg_out=reg_out, kernel_size=k_1, dilatation=1, padding=0)
    concat_reg = RegConcatLayer(reg=reg_out)
    regirrep = RegToIrrepConv(reg_in=reg_out, a_out=a_1, b_out=b_1, kernel_size=k_2, dilatation=1, padding=0)
    irrepirrep = IrrepToIrrepConv(a_in=a_1, b_in=b_1, a_out=a_2, b_out=b_2, kernel_size=k_3, dilatation=1, padding=0)
    activ = IrrepActivationLayer(a=a_2, b=b_2)
    irrepreg = IrrepToRegConv(a_in=a_2, b_in=b_2, reg_out=1, kernel_size=k_4, dilatation=1, padding=0)
    concat_irrep = IrrepConcatLayer(a=a_2, b=b_2)

    with torch.no_grad():
        out = regreg(x)
        rc_out = regreg(rcx)
        assert torch.allclose(out, reg_action(rc_out), atol=1e-5)

        concat_reg_out = concat_reg(out)
        rc_concat_reg_out = concat_reg(rc_out)
        assert torch.allclose(concat_reg_out, rc_concat_reg_out, atol=1e-5)

        out = regirrep(out)
        rc_out = regirrep(rc_out)
        assert torch.allclose(out[:, :a_1], a_action(rc_out[:, :a_1]), atol=1e-5)
        assert torch.allclose(out[:, a_1:], b_action(rc_out[:, a_1:]), atol=1e-5)

        out = irrepirrep(out)
        rc_out = irrepirrep(rc_out)
        assert torch.allclose(out[:, :a_2], a_action(rc_out[:, :a_2]), atol=1e-5)
        assert torch.allclose(out[:, a_2:], b_action(rc_out[:, a_2:]), atol=1e-5)

        out = activ(out)
        rc_out = activ(rc_out)
        assert torch.allclose(out[:, :a_2], a_action(rc_out[:, :a_2]), atol=1e-5)
        assert torch.allclose(out[:, a_2:], b_action(rc_out[:, a_2:]), atol=1e-5)

        concat_irrep_out = concat_irrep(out)
        rc_concat_irrep_out = concat_irrep(rc_out)
        assert torch.allclose(concat_irrep_out, rc_concat_irrep_out, atol=1e-5)

        out = irrepreg(out)
        rc_out = irrepreg(rc_out)
        assert torch.allclose(out, reg_action(rc_out), atol=1e-5)


def test_kmers(k=2):
    """
    Test k-mers layers creation for a fixed value of k
    :param k:
    :return:
    """
    tokmer = ToKmerLayer(k)
    x = random_one_hot(size=(1, 40))
    rcx = torch.flip(x, [1, 2])
    kmer_x = tokmer(x)
    kmer_rcx = tokmer(rcx)
    # Should be full ones with a few 2 because of palindromic representation.
    assert torch.allclose(kmer_x.int().float(), kmer_x)
    assert torch.any(torch.logical_and(3 > kmer_x.sum(axis=1), 0 < kmer_x.sum(axis=1)))
    # The flipped version should revert correctly
    assert torch.allclose(kmer_x, reg_action(kmer_rcx))


def test_bn(a=2, b=2):
    """
    Test BN layers :

    The test consists in checking everything runs and we get equivariant outputs.
    One can also uncomment printing lines to see running means evolve in training mode and not in testing mode
    :param a:
    :param b:
    :return:
    """
    x = random_one_hot(size=(1, 10))
    rcx = torch.flip(x, [1, 2])
    bnreg = RegBatchNorm(reg_dim=2)
    regirrep = RegToIrrepConv(reg_in=2, a_out=a, b_out=b, kernel_size=4, dilatation=1, padding=0)
    bnirrep = IrrepBatchNorm(a=a, b=b)

    # print(bnreg.running_mu)
    out = bnreg(x)
    # print(bnreg.running_mu)
    rc_out = bnreg(rcx)
    assert torch.allclose(out, reg_action(rc_out))

    out = regirrep(out)
    rc_out = regirrep(rc_out)
    assert torch.allclose(out[:, :a], a_action(rc_out[:, :a]))
    assert torch.allclose(out[:, a:], b_action(rc_out[:, a:]))

    out = bnirrep(out)
    rc_out = bnirrep(rc_out)
    assert torch.allclose(out[:, :a], a_action(rc_out[:, :a]))
    assert torch.allclose(out[:, a:], b_action(rc_out[:, a:]))

    # Now test in test mode
    bnreg.eval()
    bnirrep.eval()
    out = bnreg(x)
    rc_out = bnreg(rcx)
    # print(bnreg.running_mu)
    assert torch.allclose(out, reg_action(rc_out))

    out = regirrep(out)
    rc_out = regirrep(rc_out)
    assert torch.allclose(out[:, :a], a_action(rc_out[:, :a]))
    assert torch.allclose(out[:, a:], b_action(rc_out[:, a:]))

    out = bnirrep(out)
    rc_out = bnirrep(rc_out)
    assert torch.allclose(out[:, :a], a_action(rc_out[:, :a]))
    assert torch.allclose(out[:, a:], b_action(rc_out[:, a:]))


def test_all():
    """
        Test all equivariant layers

    Using itertools.product, we ensure that all combinations of layers are tested.
    We then ensure that all outputs are equivariant.
    :return:
    """
    import itertools
    import sys

    a_1 = range(0, 2)
    b_1 = range(0, 2)
    a_2 = range(0, 2)
    b_2 = range(0, 2)
    reg_out = range(1, 3)
    k_1 = range(1, 4)
    k_2 = range(1, 4)
    k_3 = range(1, 4)
    k_4 = range(1, 4)

    kmers = range(1, 4)

    bns_a = range(2)
    bns_b = range(2)

    # Test Main Layers
    settings_to_test = itertools.product(a_1, b_1, a_2, b_2, reg_out, k_1, k_2, k_3, k_4)
    for i, elt in enumerate(settings_to_test):
        if not (i + 1) % 100 or i == 2591:
            sys.stdout.write(f'{i + 1}/2592 tests passed for the main test\r')
            sys.stdout.flush()
        a_1, b_1, a_2, b_2, reg_out, k_1, k_2, k_3, k_4 = elt
        if a_1 == 0 and b_1 == 0:
            continue
        if a_2 == 0 and b_2 == 0:
            continue
        try:
            test_layers(*elt)
        except RuntimeError:
            print(elt)
            raise RuntimeError
    print()
    # Test K-Mers
    for k in kmers:
        test_kmers(k=k)

    # Test BN layers
    settings_to_test = itertools.product(bns_a, bns_b)
    for elt in settings_to_test:
        a, b = elt
        if a == 0 and b == 0:
            continue
        test_bn(a=a, b=b)
    print("All layers passed the tests")


if __name__ == '__main__':
    pass

    curr_seed = 42
    np.random.seed(curr_seed)
    torch.random.manual_seed(curr_seed)

    test_all()
