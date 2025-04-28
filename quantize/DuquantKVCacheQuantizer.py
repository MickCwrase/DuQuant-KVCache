import torch
import torch.nn as nn

def duquant_get_qmin_qmax(bits, sym=True):
    if sym:
        qmax = torch.tensor(2 ** (bits - 1) - 1)
        qmin = -qmax - 1
    else:
        qmax = torch.tensor(2 ** bits - 1)
        qmin = 0
    return qmin, qmax

def duquant_sym_quant(x, scale, qmax):
    scale = scale.to(x.device)
    q = torch.clamp((x / scale).round(), -(qmax + 1), qmax)
    return q

def duquant_sym_dequant(q, scale):
    return q * scale

def duquant_sym_quant_dequant(x, scale, qmax):
    return duquant_sym_dequant(duquant_sym_quant(x, scale, qmax), scale)

class DuquantKVCacheQuantizer(nn.Module):
    def __init__(self, bits=4, sym=True):
        super().__init__()
        self.bits = bits
        self.sym = sym
        self.qmin, self.qmax = duquant_get_qmin_qmax(bits, sym)
        self.enable = True

    def forward(self, x):
        if not self.enable or self.bits == 16:
            return x
        fq_x = self.fake_quant(x)
        return fq_x

    def fake_quant(self, x):
        x_dtype = x.dtype
        scale = self.get_scale(x)
        return duquant_sym_quant_dequant(x, scale, self.qmax).to(x_dtype)

    def get_scale(self, x):
        x_absmax = x.abs().max(dim=-1, keepdim=True)[0]
        x_absmax = torch.clamp(x_absmax, min=1e-5)
        scale = x_absmax / self.qmax
        return scale

    def quantize(self, x):
        scale = self.get_scale(x)
        q = duquant_sym_quant(x, scale, self.qmax)
        return q, scale

    def dequantize(self, q, scale):
        return duquant_sym_dequant(q, scale)


