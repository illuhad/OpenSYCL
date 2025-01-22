import csv
import sys
import os
import json

if len(sys.argv) < 3:
    print(f"usage: {os.path.basename(__file__)} sycl-bench.csv new_output.json")
    exit(1)

with open(sys.argv[1], newline='') as csvfile:
    reader = csv.reader(csvfile)
    
    # Find out which entries are run-time-mean and run-time-stddev
    first_row = next(reader)
    mean_idx   = -1
    stddev_idx = -1
    curr_idx   = 0
    for entry in first_row:
        if entry == 'run-time-mean':
            mean_idx = curr_idx
        if entry == 'run-time-stddev':
            stddev_idx = curr_idx
        curr_idx += 1
    assert(mean_idx != -1)
    assert(stddev_idx != -1)

    benchmarks = list()
    for row in reader:
        if row[0][0] == '#':
            continue

        benchmark = dict()
        benchmark['name']  = row[0]
        benchmark['unit']  = 's'
        benchmark['value'] = row[mean_idx]
        benchmark['range'] = row[stddev_idx]
    
        benchmarks.append(benchmark)
    
    with open(sys.argv[2], "w") as outfile:
        json.dump(benchmarks, outfile, indent=2)
