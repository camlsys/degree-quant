from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from pathlib import Path


def append_date_and_time_to_string(string):
    now = datetime.utcnow().strftime("%m_%d_%H_%M_%S")

    return Path(string) / now


def set_outputdir_and_writer(
    model_name,
    outdir,
    num_layers,
    hidden,
    lr,
    quant_mode,
    ste,
    momentum,
    percentile,
    is_DQ,
    w_decay,
    low,
    change,
):

    layers = "layers_" + str(num_layers)
    hidden = "hidden_" + str(hidden)

    ste_config = "STE_" if ste else "GC_"
    if momentum:
        ste_config += "MOM"
    elif percentile is not None:
        ste_config += "PER"
    else:
        ste_config += "ABS"

    if is_DQ:
        quant_mode += "_DQ_low" + str(low) + "_chng" + str(change)

    dir = (
        Path(outdir)
        / model_name
        / layers
        / hidden
        / str(quant_mode)
        / ste_config
        / str("lr_" + str(lr))
        / str("wd_" + str(w_decay))
    )

    dir = append_date_and_time_to_string(dir)

    writer = SummaryWriter(dir)
    print(f"Output dir:{dir}")

    return dir, writer
