import argparse

from reddit_binary.dataset import get_dataset
from reddit_binary.train_eval import cross_validation_with_val_set
from reddit_binary.gin import GIN
import reddit_binary.utils as utils

# This code is substantially derived from PyTorch Geometric's repo
# see: https://github.com/rusty1s/pytorch_geometric/tree/master/benchmark/kernel


################################################################################################
# REACH INT8-DQ acc of 91.8%
# python main.py --int8 --gc_per --lr 0.005 --DQ --low 0.0 --change 0.1 --wd 0.0002 --epochs 200
# REACH INT8 acc of 76.0%
# python main.py --int8 --ste_mom --lr 0.005 --wd 0.0002 --epochs 200
################################################################################################
# REACH INT4-DQ acc of 81.3%
# python main.py --int4 --gc_per --lr 0.001 --DQ --low 0.1 --change 0.1 --wd 4e-5 --epochs 200
# REACH INT4 acc of 54.1%
# python main.py --int4 --ste_mom --lr 0.05 --epochs 200
################################################################################################


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_layers", type=int, default=5)
parser.add_argument("--hidden", type=int, default=64)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--wd", type=float, default=4e-5)
parser.add_argument("--noise", type=float, default=1.0)
parser.add_argument("--lr_decay_factor", type=float, default=0.5)
parser.add_argument("--lr_decay_step_size", type=int, default=50)
parser.add_argument(
    "--path", type=str, default="/datasets/", help="where all datasets live"
)
parser.add_argument("--outdir", type=str, default="/output/redditBINexps")

parser.add_argument("--DQ", action="store_true", help="enables DegreeQuant")
parser.add_argument("--low", type=float, default=0.0)
parser.add_argument("--change", type=float, default=0.1)
parser.add_argument("--sample_prop", type=float, default=None)

quant_mode = parser.add_mutually_exclusive_group(required=True)
quant_mode.add_argument("--fp32", action="store_true", help="no quantization")
quant_mode.add_argument("--int8", action="store_true", help="INT8 quant")
quant_mode.add_argument("--int4", action="store_true", help="INT4 quant")

ste_mode = parser.add_mutually_exclusive_group(required=True)
ste_mode.add_argument("--ste_abs", action="store_true", help="STE-ABS")
ste_mode.add_argument("--ste_mom", action="store_true", help="STE-MOM")
ste_mode.add_argument("--gc_abs", action="store_true", help="GC-ABS")
ste_mode.add_argument("--gc_mom", action="store_true", help="GC-MOM")
ste_mode.add_argument("--ste_per", action="store_true", help="STE-PER")
ste_mode.add_argument("--gc_per", action="store_true", help="GC-PER")

args = parser.parse_args()

dataset_name = "REDDIT-BINARY"


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# generating the qConfig
if args.fp32:
    qypte = "FP32"
elif args.int8:
    qypte = "INT8"
elif args.int4:
    qypte = "INT4"
else:
    raise NotImplementedError

ste = False
momentum = False
percentile = None

# ste quant
if args.ste_abs:
    ste = True
elif args.ste_mom:
    ste = True
    momentum = True
elif args.gc_abs:
    pass
elif args.gc_mom:
    momentum = True
elif args.ste_per:
    ste = True
    percentile = 0.01 if args.int4 else 0.001
elif args.gc_per:
    percentile = 0.01 if args.int4 else 0.001
else:
    raise NotImplementedError


DQ = None
if args.DQ:
    DQ = {"prob_mask_low": args.low, "prob_mask_change": args.change}

print(args)
# below is just the default from PyG
dataset = get_dataset(args.path, dataset_name, sparse=True, DQ=DQ)

model = GIN(
    dataset,
    num_layers=args.num_layers,
    hidden=args.hidden,
    dq=args.DQ,
    qypte=qypte,
    ste=ste,
    momentum=momentum,
    percentile=percentile,
    sample_prop=args.sample_prop,
)
print(f"model has {count_parameters(model)} parameters")

# output dir and tensorboard writer
dir, writer = utils.set_outputdir_and_writer(
    "GIN",
    args.outdir,
    args.num_layers,
    args.hidden,
    args.lr,
    qypte,
    ste,
    momentum,
    percentile,
    args.DQ,
    args.wd,
    args.low,
    args.change,
)

loss, acc, std = cross_validation_with_val_set(
    dataset,
    model,
    folds=10,
    epochs=args.epochs,
    batch_size=args.batch_size,
    lr=args.lr,
    lr_decay_factor=args.lr_decay_factor,
    lr_decay_step_size=args.lr_decay_step_size,
    weight_decay=args.wd,
    writer=writer,
    logger=None,
)
best_result = (loss, acc, std)

desc = "{:.3f} ± {:.3f}".format(best_result[1], best_result[2])
print("Result - {}".format(desc))
