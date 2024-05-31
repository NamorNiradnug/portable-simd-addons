"""
Reads output of `cargo bench` from stdin
Prints markdown-formatter tables on stdout
"""

from collections import defaultdict
from tabulate import tabulate
from sys import argv

BENCHMARKED_LIBS = argv[1:]

all_bench_data: defaultdict = defaultdict(lambda: defaultdict(defaultdict))

while True:
    try:
        bench_res_line = input()
    except EOFError:
        break
    if not bench_res_line.startswith("test bench_"):
        continue
    single_bench_res = bench_res_line.split()
    [func, ftype, lib] = single_bench_res[1].removeprefix("bench_").rsplit("_", 2)
    if lib not in BENCHMARKED_LIBS:
        continue
    runtime = single_bench_res[4]
    all_bench_data[ftype][func][lib] = int(runtime.replace(",", ""))


for ftype, bench_data in all_bench_data.items():
    all_functions = sorted(bench_data.keys())
    table = [
        [f"`{func}`"] + [bench_data[func].get(lib, None) for lib in BENCHMARKED_LIBS]
        for func in all_functions
    ]
    table = [
        row[:1]
        + [
            f"`{res:,}`" if res == min(bench_data[func].values()) else f"{res:,}"
            for res in row[1:]
        ]
        for func, row in zip(all_functions, table)
    ]
    print(
        tabulate(table, headers=[ftype] + BENCHMARKED_LIBS, tablefmt="github"),
        end="\n\n",
    )
