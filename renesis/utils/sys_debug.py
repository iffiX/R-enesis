import os
import math
import psutil
import traceback

millnames = ["", " K", " M", " B", " T"]


def _millify(n):
    n = float(n)
    millidx = max(
        0,
        min(
            len(millnames) - 1,
            int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3)),
        ),
    )

    return "{:.0f}{}".format(n / 10 ** (3 * millidx), millnames[millidx])


def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(
        f"Current memory usage of process {process.pid}: {process.memory_info().rss/1024**2:.1f} MB\n"
        f"Last step stacktrace:"
    )
    for line in traceback.format_stack()[-3:]:
        print(line.strip())
    print("\n")


def print_model_size(model):
    parameter_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"*********************************************\n"
        f"Parameter number: {_millify(parameter_num)}\n"
        f"*********************************************\n",
    )
