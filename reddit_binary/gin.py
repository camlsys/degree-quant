import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, Identity, BatchNorm1d as BN
from torch_geometric.nn import global_mean_pool

from dq.quantization import IntegerQuantizer
from dq.linear_quantized import LinearQuantized
from dq.baseline_quant import GINConvQuant
from dq.multi_quant import evaluate_prob_mask, GINConvMultiQuant


def create_quantizer(qypte, ste, momentum, percentile, signed, sample_prop):
    if qypte == "FP32":
        return Identity
    else:
        return lambda: IntegerQuantizer(
            4 if qypte == "INT4" else 8,
            signed=signed,
            use_ste=ste,
            use_momentum=momentum,
            percentile=percentile,
            sample=sample_prop,
        )


def make_quantizers(qypte, dq, sign_input, ste, momentum, percentile, sample_prop):
    if dq:
        # GIN doesn't apply DQ to the LinearQuantize layers so we keep the 
        # default inputs, weights, features keys.
        # See NOTE in the multi_quant.py file
        layer_quantizers = {
            "inputs": create_quantizer(
                qypte, ste, momentum, percentile, sign_input, sample_prop
            ),
            "weights": create_quantizer(
                qypte, ste, momentum, percentile, True, sample_prop
            ),
            "features": create_quantizer(
                qypte, ste, momentum, percentile, True, sample_prop
            ),
        }
        mp_quantizers = {
            "message_low": create_quantizer(
                qypte, ste, momentum, percentile, True, sample_prop
            ),
            "message_high": create_quantizer(
                "FP32", ste, momentum, percentile, True, sample_prop
            ),
            "update_low": create_quantizer(
                qypte, ste, momentum, percentile, True, sample_prop
            ),
            "update_high": create_quantizer(
                "FP32", ste, momentum, percentile, True, sample_prop
            ),
            "aggregate_low": create_quantizer(
                qypte, ste, momentum, percentile, True, sample_prop
            ),
            "aggregate_high": create_quantizer(
                "FP32", ste, momentum, percentile, True, sample_prop
            ),
        }
    else:
        layer_quantizers = {
            "inputs": create_quantizer(
                qypte, ste, momentum, percentile, sign_input, sample_prop
            ),
            "weights": create_quantizer(
                qypte, ste, momentum, percentile, True, sample_prop
            ),
            "features": create_quantizer(
                qypte, ste, momentum, percentile, True, sample_prop
            ),
        }
        mp_quantizers = {
            "message": create_quantizer(
                qypte, ste, momentum, percentile, True, sample_prop
            ),
            "update_q": create_quantizer(
                qypte, ste, momentum, percentile, True, sample_prop
            ),
            "aggregate": create_quantizer(
                qypte, ste, momentum, percentile, True, sample_prop
            ),
        }
    return layer_quantizers, mp_quantizers


class ResettableSequential(Sequential):
    def reset_parameters(self):
        for child in self.children():
            if hasattr(child, "reset_parameters"):
                child.reset_parameters()


class GIN(torch.nn.Module):
    def __init__(
        self,
        dataset,
        num_layers,
        hidden,
        dq,
        qypte,
        ste,
        momentum,
        percentile,
        sample_prop,
    ):
        super(GIN, self).__init__()

        self.is_dq = dq
        gin_layer = GINConvMultiQuant if dq else GINConvQuant 

        lq, mq = make_quantizers(
            qypte,
            dq,
            False,
            ste=ste,
            momentum=momentum,
            percentile=percentile,
            sample_prop=sample_prop,
        )
        lq_signed, _ = make_quantizers(
            qypte,
            dq,
            True,
            ste=ste,
            momentum=momentum,
            percentile=percentile,
            sample_prop=sample_prop,
        )

        # NOTE: see comment in multi_quant.py on the use of 
        # "mask-aware" MLPs.
        self.conv1 = gin_layer(
            ResettableSequential(
                Linear(dataset.num_features, hidden),
                ReLU(),
                LinearQuantized(hidden, hidden, layer_quantizers=lq),
                ReLU(),
                BN(hidden),
            ),
            train_eps=True,
            mp_quantizers=mq,
        )
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                gin_layer(
                    ResettableSequential(
                        LinearQuantized(hidden, hidden, layer_quantizers=lq_signed),
                        ReLU(),
                        LinearQuantized(hidden, hidden, layer_quantizers=lq),
                        ReLU(),
                        BN(hidden),
                    ),
                    train_eps=True,
                    mp_quantizers=mq,
                )
            )

        self.lin1 = LinearQuantized(hidden, hidden, layer_quantizers=lq_signed)
        self.lin2 = LinearQuantized(hidden, dataset.num_classes, layer_quantizers=lq)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        # NOTE: It is possible to use the same mask consistently or generate a 
        # new mask per layer. For other experiments we used a per-layer mask
        # We did not observe major differences but we expect the impact will
        # be layer and dataset dependent. Extensive experiments assessing the
        # difference were not run, however, due to the high cost.
        if hasattr(data, "prob_mask") and data.prob_mask is not None:
            mask = evaluate_prob_mask(data)
        else:
            mask = None

        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index, mask)
        for conv in self.convs:
            x = conv(x, edge_index, mask)

        x = global_mean_pool(x, batch)
        # NOTE: the linear layers from here do not contribute significantly to run-time
        # Therefore you probably don't want to quantize these as it will likely have 
        # an impact on performance.
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        # NOTE: This is a quantized final layer. You probably don't want to be
        # this aggressive in practice.
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)
