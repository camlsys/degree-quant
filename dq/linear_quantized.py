import torch.nn as nn
import torch.nn.functional as F


class LinearQuantized(nn.Linear):
    """A quantizable linear layer"""
    def __init__(
        self, in_features, out_features, layer_quantizers, bias=True
    ):
        # create quantization modules for this layer
        self.layer_quant_fns = layer_quantizers
        super(LinearQuantized, self).__init__(in_features, out_features, bias)

    def reset_parameters(self):
        super().reset_parameters()
        self.layer_quant = nn.ModuleDict()
        for key in ["inputs", "features", "weights"]:
            self.layer_quant[key] = self.layer_quant_fns[key]()

    def forward(self, input):
        input_q = self.layer_quant["inputs"](input)
        w_q = self.layer_quant["weights"](self.weight)
        out = F.linear(input_q, w_q, self.bias)
        out = self.layer_quant["features"](out)

        return out
