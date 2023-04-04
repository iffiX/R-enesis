import os
import math
import psutil
import traceback

_debugger_enabled = False
_millnames = ["", " K", " M", " B", " T"]


def _millify(n):
    n = float(n)
    millidx = max(
        0,
        min(
            len(_millnames) - 1,
            int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3)),
        ),
    )

    return "{:.0f}{}".format(n / 10 ** (3 * millidx), _millnames[millidx])


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


def enable_debugger(debug_ip="localhost", debug_port=8223):
    global _debugger_enabled
    if not _debugger_enabled:
        import pydevd_pycharm

        print("Debugging enabled")
        pydevd_pycharm.settrace(
            debug_ip, port=debug_port, stdoutToServer=True, stderrToServer=True,
        )
        _debugger_enabled = True
