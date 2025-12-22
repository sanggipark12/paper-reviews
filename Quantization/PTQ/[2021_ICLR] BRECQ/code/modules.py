import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.functional as F

def round_ste(x):
    return (x.round() - x).detach() + x

class UniformQuantizer(nn.Module):
    def __init__(self, n_bits=4, symmetric=True, channel_wise=False, is_weight=False):
        super().__init__()

        self.n_bits = n_bits
        self.symmetric = symmetric
        self.channel_wise = channel_wise
        self.is_weight = is_weight # weight or activaition

        self.max_q = None
        self.min_q = None
        self.delta = None # scale
        self.zero_point = None

    def init_quantization_params(self, x):

        x_clone = x.clone().detach()

        # channel_wise는 언제 사용할까 -> weight를 양자화 할 때, weight는 채널 마다 제각각이기 때문에
        if self.channel_wise:
            max_w = x_clone.abs().view(x_clone.shape[0], -1).max(1)[0]
            max_w = max_w.view(-1, 1, 1, 1)
        else:
            max_w = torch.max(torch.abs(x_clone))

        if self.symmetric:
            self.min_q = -2 ** (self.n_bits - 1)
            self.max_q = 2 ** (self.n_bits - 1) - 1

            self.delta = max_w / self.max_q
            self.zero_point = 0
        else:
            pass

    def forward(self, x):

        if self.delta is None:
            self.init_quantization_params(x)

        x_int = round_ste(x / self.delta) + self.zero_point
        x_quant = torch.clamp(x_int, self.min_q, self.max_q)

        return self.delta * (x_quant - self.zero_point)


class AdaRoundQuantizer(nn.Module):
    def __init__(self, uq :UniformQuantizer, x):
        super().__init__()

        self.n_bits = uq.n_bits
        self.sym = uq.symmetric
        self.delta = uq.delta
        self.zero_point = uq.zero_point

        self.min_q = uq.min_q
        self.max_q = uq.max_q

        self.alpha = None # AdaRound parameter (v)
        self.soft_targets = True # adaround 여부

        self.gamma, self.zeta = -0.1, 1.1

        self.init_alpha(x.clone())

    def init_alpha(self, x):
        w_floor = torch.floor(x / self.delta)

        rest = (x / self.delta) - w_floor
        rest = torch.clamp(rest, 0.01, 0.99)

        sig_inv = (rest - self.gamma) / (self.zeta - self.gamma)
        sig_inv = torch.clamp(sig_inv, 0.01, 0.99)

        alpha = torch.log(sig_inv / (1 - sig_inv))
        self.alpha = nn.Parameter(alpha)

    def rectified_sigmoid(self):
        # Equation 23
        x = torch.clamp(torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma, 0, 1)

        return x

    def forward(self, x):
        # Soft Quantization
        if self.soft_targets:
            w_floor = torch.floor(x / self.delta)

            w_soft = w_floor + self.rectified_sigmoid()

            w_quant = self.delta * torch.clamp(w_soft - self.zero_point, self.min_q, self.max_q)
            return w_quant

        # Hard Quantization
        else:
            return torch.round(w_quant)
        
        

class QuantModule(nn.Module):
    def __init__(self, org_module: nn.Conv2d, weight_quantizer):
        super().__init__()
        self.org_module = org_module
        self.weight_quantizer = weight_quantizer # 아까 만든 AdaRoundQuantizer
        self.use_quantization = True # 스위치
        self.conv_type = dict(bias = self.org_module.bias,
                              stride = self.org_module.stride, padding = self.org_module.padding,
                                dilation=self.org_module.dilation, groups=self.org_module.groups)

    def forward(self, x):
        if self.use_quantization:
            # 가중치 가져오기
            w = self.org_module.weight

            # 가중치 양자화 (여기서 AdaRound가 작동)
            w_q = self.weight_quantizer(w)

            # 양자화된 가중치로 Conv 연산
            # bias, stride, padding 등은 원본 모듈의 설정을 그대로 
            out = F.conv2d(x, w_q, **self.conv_type)
            return out
        else:
            return self.org_module(x)
        
