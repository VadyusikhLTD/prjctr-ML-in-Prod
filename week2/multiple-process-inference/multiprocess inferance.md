#

Task - PR run inference with csv with single and multiple processes, report time difference (model & data for inference you might take from test task OR your own ML problem)

Time measurement output is below:

    Single process evaluation takes 6220.61 ms
    Multiple process (1 workers) evaluation takes 6541.00 ms
    Multiple process (2 workers) evaluation takes 3442.78 ms
    Multiple process (3 workers) evaluation takes 2514.23 ms
    Multiple process (4 workers) evaluation takes 2062.01 ms
    Multiple process (5 workers) evaluation takes 1824.92 ms
    Multiple process (6 workers) evaluation takes 1701.54 ms
    Multiple process (7 workers) evaluation takes 1674.25 ms
    Multiple process (8 workers) evaluation takes 1669.01 ms
    Multiple process (9 workers) evaluation takes 1705.95 ms
    Multiple process (10 workers) evaluation takes 1764.10 ms
    Multiple process (11 workers) evaluation takes 1860.82 ms
    Multiple process (12 workers) evaluation takes 1840.23 ms
    Multiple process (13 workers) evaluation takes 2030.68 ms
    Multiple process (14 workers) evaluation takes 1958.07 ms

As we see multiple processing helps to increase inference speed up to _x3.9_ times. 
Optimal number of workers is 8, same as number of logical CPU on device.