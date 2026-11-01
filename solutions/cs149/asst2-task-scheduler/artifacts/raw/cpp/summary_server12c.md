# Summary of server12c (+_r2, _r3), medians over all repetitions

## Launch overhead, µs per run(), 12 threads, empty tasks (overhead.jsonl)

| tasks | serial | spawn | spinning | sleeping | async_spinning | async_sleeping |
|---|---|---|---|---|---|---|
| 1 | 0.0114 | 53.3 | 0.783 | 10.8 | 9.05 | 12.4 |
| 4 | 0.0289 | 155 | 1.97 | 10.5 | 12.7 | 14.9 |
| 16 | 0.0876 | 215 | 13.2 | 10.7 | 15.8 | 17.9 |
| 64 | 0.328 | 204 | 34 | 12.7 | 32.8 | 27.7 |
| 256 | 1.33 | 189 | 132 | 25.4 | 107 | 61.3 |
| 1024 | 5.18 | 192 | 521 | 78.3 | 393 | 190 |
| 4096 | 20.6 | 204 | 2.04e+03 | 284 | 1.53e+03 | 698 |
| 16384 | 82.3 | 558 | 8.1e+03 | 1.14e+03 | 5.27e+03 | 2.75e+03 | 

## Launch overhead vs threads, 64 empty tasks

| threads | serial | spawn | spinning | sleeping | async_spinning | async_sleeping |
|---|---|---|---|---|---|---|
| 1 | 0.0936 | 28.6 | 6.24 | 5.95 | 3.14 | 7.69 |
| 2 | 0.328 | 87.1 | 10.9 | 6.43 | 7.56 | 11.8 |
| 4 | 0.328 | 114 | 12.9 | 7.21 | 14.9 | 19.5 |
| 6 | 0.329 | 151 | 14.5 | 10.7 | 14.9 | 21.9 |
| 8 | 0.328 | 159 | 15.8 | 11.7 | 15.8 | 23.4 |
| 12 | 0.328 | 204 | 34 | 12.7 | 32.8 | 27.7 | 

## cpu_per_wall during overhead loop, 64 tasks

| threads | serial | spawn | spinning | sleeping | async_spinning | async_sleeping |
|---|---|---|---|---|---|---|
| 1 | 1 | 0.98 | 1.95 | 0.948 | 2.27 | 0.96 |
| 2 | 1 | 1.24 | 2.83 | 1.72 | 2.94 | 1.74 |
| 4 | 1 | 1.55 | 4.72 | 3 | 4.6 | 3.03 |
| 6 | 1 | 1.7 | 6.79 | 4.53 | 6.46 | 4.41 |
| 8 | 1 | 1.99 | 9.11 | 5.75 | 8.4 | 5.9 |
| 12 | 1 | 1.82 | 12 | 7.09 | 11.6 | 8.04 | 

## Official tests, ms, 12 threads (tests.jsonl); all correct = True

| test | serial | spawn | spinning | sleeping | async_spinning | async_sleeping |
|---|---|---|---|---|---|---|
| mandelbrot_chunked | 329 | 44 | 64.9 | 48.4 | 64 | 46.5 |
| mandelbrot_chunked_async | 329 | 41.7 | 64.9 | 46.5 | 64 | 44.3 |
| math_operations_in_tight_for_loop | 5.14 | 387 | 9.35 | 33.6 | 51.3 | 57.4 |
| math_operations_in_tight_for_loop_async | 1.49 | 380 | 6.69 | 32.6 | 62.2 | 43.8 |
| math_operations_in_tight_for_loop_fan_in | 25.8 | 354 | 53.4 | 48.8 | 60 | 28 |
| math_operations_in_tight_for_loop_fewer_tasks | 1.45 | 130 | 3.96 | 29.1 | 29 | 41.3 |
| math_operations_in_tight_for_loop_fewer_tasks_async | 1.46 | 143 | 3.96 | 28.4 | 40.3 | 25.9 |
| math_operations_in_tight_for_loop_reduction_tree | 23.8 | 111 | 28.1 | 21.3 | 7.91 | 8.7 |
| ping_pong_equal | 133 | 21.3 | 29.4 | 14 | 29.8 | 14.2 |
| ping_pong_equal_async | 129 | 20.5 | 29.5 | 14 | 13.9 | 13.9 |
| ping_pong_unequal | 143 | 22.9 | 33.8 | 15.6 | 32.3 | 16 |
| ping_pong_unequal_async | 143 | 21.9 | 32.5 | 24.1 | 31.4 | 23.1 |
| recursive_fibonacci | 77.1 | 7.95 | 7.9 | 10.7 | 7.29 | 12.8 |
| recursive_fibonacci_async | 77.5 | 7.91 | 7.58 | 10.3 | 7.1 | 10.9 |
| spin_between_run_calls | 6.36 | 6.87 | 6.28 | 6.37 | 6.39 | 9.22 |
| spin_between_run_calls_async | 6.53 | 6.95 | 6.18 | 6.39 | 6.37 | 7.26 |
| super_light | 7.63 | 78.4 | 3.02 | 10 | 30 | 18.5 |
| super_light_async | 7.09 | 78.4 | 3.04 | 9.94 | 30.4 | 16.6 |
| super_super_light | 7.26 | 84.4 | 3.1 | 9.98 | 32.8 | 18.4 |
| super_super_light_async | 5.56 | 82.3 | 3.08 | 10 | 56 | 16.7 | 

## Official tests, CPU s / wall s

| test | serial | spawn | spinning | sleeping | async_spinning | async_sleeping |
|---|---|---|---|---|---|---|
| mandelbrot_chunked | 1.00 | 8.51 | 11.85 | 9.46 | 12.00 | 9.56 |
| mandelbrot_chunked_async | 1.00 | 8.90 | 11.84 | 9.55 | 12.01 | 9.43 |
| math_operations_in_tight_for_loop | 1.00 | 1.75 | 11.81 | 6.67 | 11.66 | 6.70 |
| math_operations_in_tight_for_loop_async | 1.00 | 1.77 | 10.56 | 6.42 | 11.69 | 7.62 |
| math_operations_in_tight_for_loop_fan_in | 1.00 | 1.80 | 12.54 | 4.60 | 11.77 | 8.65 |
| math_operations_in_tight_for_loop_fewer_tasks | 1.00 | 1.71 | 12.11 | 5.92 | 11.59 | 5.36 |
| math_operations_in_tight_for_loop_fewer_tasks_async | 1.00 | 1.77 | 12.08 | 5.71 | 11.72 | 6.21 |
| math_operations_in_tight_for_loop_reduction_tree | 1.00 | 1.85 | 11.97 | 3.30 | 11.39 | 8.57 |
| ping_pong_equal | 1.00 | 7.94 | 12.18 | 10.29 | 11.81 | 10.18 |
| ping_pong_equal_async | 1.00 | 7.96 | 12.92 | 10.30 | 11.74 | 10.30 |
| ping_pong_unequal | 1.00 | 7.41 | 12.29 | 9.78 | 11.94 | 9.74 |
| ping_pong_unequal_async | 1.00 | 7.66 | 11.83 | 8.83 | 11.87 | 9.47 |
| recursive_fibonacci | 1.00 | 10.37 | 12.08 | 11.61 | 12.43 | 11.56 |
| recursive_fibonacci_async | 1.00 | 10.37 | 12.57 | 11.62 | 10.88 | 11.90 |
| spin_between_run_calls | 1.00 | 1.12 | 10.09 | 1.27 | 12.44 | 1.23 |
| spin_between_run_calls_async | 1.00 | 1.12 | 15.22 | 1.31 | 11.07 | 1.28 |
| super_light | 1.00 | 1.94 | 15.49 | 8.92 | 11.66 | 9.20 |
| super_light_async | 1.00 | 1.94 | 10.56 | 8.98 | 11.65 | 9.83 |
| super_super_light | 1.00 | 1.88 | 13.91 | 8.92 | 11.61 | 9.10 |
| super_super_light_async | 1.00 | 1.83 | 14.00 | 8.93 | 11.83 | 9.86 | 

## Serial ms (scaling.jsonl)

test
mandelbrot_chunked     332.82
ping_pong_equal        132.04
ping_pong_unequal      145.00
recursive_fibonacci     76.46 

## Speedup vs serial

|  | async_sleeping | sleeping | spawn | spinning |
|---|---|---|---|---|
| ('mandelbrot_chunked', 1) | 0.88 | 0.88 | 0.84 | 1.00 |
| ('mandelbrot_chunked', 2) | 1.53 | 1.53 | 1.39 | 1.92 |
| ('mandelbrot_chunked', 4) | 2.36 | 2.33 | 1.87 | 3.58 |
| ('mandelbrot_chunked', 6) | 2.62 | 2.73 | 2.62 | 5.02 |
| ('mandelbrot_chunked', 8) | 2.89 | 2.92 | 2.89 | 6.26 |
| ('mandelbrot_chunked', 12) | 5.45 | 5.95 | 7.75 | 5.10 |
| ('ping_pong_equal', 1) | 0.96 | 0.92 | 0.91 | 1.01 |
| ('ping_pong_equal', 2) | 1.84 | 1.86 | 0.97 | 1.98 |
| ('ping_pong_equal', 4) | 3.14 | 3.64 | 2.07 | 3.69 |
| ('ping_pong_equal', 6) | 3.82 | 4.80 | 2.99 | 4.74 |
| ('ping_pong_equal', 8) | 3.82 | 6.91 | 3.05 | 5.48 |
| ('ping_pong_equal', 12) | 5.88 | 9.33 | 4.58 | 4.50 |
| ('ping_pong_unequal', 1) | 1.01 | 0.98 | 0.93 | 1.01 |
| ('ping_pong_unequal', 2) | 1.90 | 1.89 | 0.98 | 1.97 |
| ('ping_pong_unequal', 4) | 3.36 | 3.52 | 2.10 | 3.75 |
| ('ping_pong_unequal', 6) | 3.94 | 4.18 | 3.02 | 5.40 |
| ('ping_pong_unequal', 8) | 4.23 | 5.38 | 3.16 | 6.97 |
| ('ping_pong_unequal', 12) | 5.87 | 7.29 | 6.07 | 4.57 |
| ('recursive_fibonacci', 1) | 0.97 | 0.99 | 0.97 | 1.00 |
| ('recursive_fibonacci', 2) | 1.91 | 1.92 | 1.15 | 1.94 |
| ('recursive_fibonacci', 4) | 3.80 | 3.73 | 2.55 | 3.63 |
| ('recursive_fibonacci', 6) | 4.92 | 5.41 | 4.19 | 5.01 |
| ('recursive_fibonacci', 8) | 5.64 | 5.71 | 4.91 | 7.57 |
| ('recursive_fibonacci', 12) | 6.71 | 7.24 | 9.16 | 8.89 | 

## Idle pool: cores burned (idle.jsonl)

| threads | serial | spawn | spinning | sleeping | async_spinning | async_sleeping |
|---|---|---|---|---|---|---|
| 1 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 |
| 2 | 0.00 | 0.00 | 1.99 | 0.00 | 2.00 | 0.00 |
| 4 | 0.00 | 0.00 | 3.99 | 0.00 | 3.99 | 0.00 |
| 6 | 0.00 | 0.00 | 5.99 | 0.00 | 5.99 | 0.00 |
| 8 | 0.00 | 0.00 | 7.98 | 0.00 | 7.99 | 0.00 |
| 12 | 0.00 | 0.00 | 11.96 | 0.00 | 11.98 | 0.00 | 

## Oversubscription, ms (scale 0.25) (oversub.jsonl)

|  | async_sleeping | async_spinning | sleeping | spinning |
|---|---|---|---|---|
| ('math_operations_in_tight_for_loop', 1) | 10.7 | 18.8 | 6.06 | 3.81 |
| ('math_operations_in_tight_for_loop', 2) | 15.3 | 1.44e+04 | 10.2 | 3.42 |
| ('math_operations_in_tight_for_loop', 4) | 24 | 1.53e+04 | 23.7 | 2.37 |
| ('ping_pong_equal', 1) | 3.95 | 4.03 | 5.32 | 7.93 |
| ('ping_pong_equal', 2) | 3.94 | 217 | 3.92 | 20 |
| ('ping_pong_equal', 4) | 5.02 | 201 | 9.76 | 31.6 |
| ('recursive_fibonacci', 1) | 1.85 | 2.03 | 2.37 | 2.67 |
| ('recursive_fibonacci', 2) | 1.76 | 1.86 | 1.9 | 2.48 |
| ('recursive_fibonacci', 4) | 2.59 | 1.84 | 4.21 | 1.98 |
| ('spin_between_run_calls', 1) | 1.65 | 1.61 | 1.64 | 1.67 |
| ('spin_between_run_calls', 2) | 1.69 | 1.66 | 1.6 | 1.61 |
| ('spin_between_run_calls', 4) | 1.74 | 1.6 | 1.72 | 1.57 | 

## Oversubscription, CPU s

|  | async_sleeping | async_spinning | sleeping | spinning |
|---|---|---|---|---|
| ('math_operations_in_tight_for_loop', 1) | 0.0699 | 0.221 | 0.0359 | 0.0462 |
| ('math_operations_in_tight_for_loop', 2) | 0.0947 | 173 | 0.0657 | 0.0307 |
| ('math_operations_in_tight_for_loop', 4) | 0.139 | 183 | 0.18 | 0.00298 |
| ('ping_pong_equal', 1) | 0.0402 | 0.0473 | 0.0554 | 0.0844 |
| ('ping_pong_equal', 2) | 0.0373 | 2.61 | 0.0407 | 0.24 |
| ('ping_pong_equal', 4) | 0.0391 | 2.4 | 0.0381 | 0.383 |
| ('recursive_fibonacci', 1) | 0.0211 | 0.0118 | 0.0272 | 0.0365 |
| ('recursive_fibonacci', 2) | 0.0198 | 0.00752 | 0.0217 | 0.0251 |
| ('recursive_fibonacci', 4) | 0.0201 | 0.00824 | 0.0204 | 0.00228 |
| ('spin_between_run_calls', 1) | 0.00209 | 0.0141 | 0.00208 | 0.00171 |
| ('spin_between_run_calls', 2) | 0.0024 | 0.0129 | 0.00218 | 0.00168 |
| ('spin_between_run_calls', 4) | 0.00304 | 0.0116 | 0.00284 | 0.00158 | 

## Granularity, ms per run(), total 2e8 LCG steps (granularity.jsonl)

| tasks | async_sleeping | sleeping | spawn | spinning |
|---|---|---|---|---|
| 1 | 357 | 353 | 308 | 308 |
| 2 | 162 | 188 | 147 | 160 |
| 4 | 135 | 113 | 72.3 | 72.2 |
| 8 | 64.6 | 76.9 | 42.7 | 33.8 |
| 12 | 49.4 | 55.8 | 36.8 | 34.5 |
| 16 | 41.9 | 43.9 | 31.5 | 32 |
| 24 | 31.7 | 31.2 | 28.8 | 32 |
| 32 | 27.9 | 26.2 | 29 | 32 |
| 64 | 21.8 | 20.7 | 23.5 | 32 |
| 128 | 18.5 | 18.5 | 21.7 | 32 |
| 256 | 17.5 | 17.4 | 20.9 | 32 |
| 1024 | 17.2 | 17.2 | 20.3 | 32 |
| 4096 | 17.3 | 17.2 | 20.3 | 32 |
| 16384 | 17.5 | 17.1 | 20.3 | 32 |
| 65536 | 29.6 | 17.3 | 20.2 | 32 |
| 262144 | 129 | 19.2 | 20.9 | 64 | 

## Granularity with 11 workers (granularity_11workers.jsonl)

| tasks | sleeping | spinning |
|---|---|---|
| 1 | 307 | 308 |
| 2 | 161 | 147 |
| 4 | 107 | 71.9 |
| 8 | 101 | 33.8 |
| 12 | 49.1 | 34.5 |
| 16 | 42 | 25.9 |
| 24 | 30.8 | 21.5 |
| 32 | 25.1 | 23.2 |
| 64 | 21.1 | 21.1 |
| 128 | 20.3 | 20.3 |
| 256 | 19 | 18.9 |
| 1024 | 18.7 | 18.7 |
| 4096 | 18.8 | 18.8 |
| 16384 | 18.7 | 18.7 |
| 65536 | 18.9 | 19.1 |
| 262144 | 21.2 | 21.7 | 

## DAG shapes, ms (dag.jsonl)

|  | async_sleeping | async_spinning | serial | sleeping |
|---|---|---|---|---|
| ('chain', 64) | 26.9 | 26.6 | nan | 27.1 |
| ('fanin', 2) | 14.6 | 22.3 | nan | 27.1 |
| ('indep', 1) | 16.6 | 32 | 171 | 27.1 |
| ('indep_sync', 1) | 27.3 | 54.4 | nan | 27.8 |
| ('random', 8) | 14.8 | 32 | nan | 27.1 |
| ('tree', 7) | 14.8 | 32 | nan | 27.1 | 

## Sync-per-launch vs async, ms (dag_tasks.jsonl)

| tasks | indep | indep_sync | speedup |
|---|---|---|---|
| 1 | 5.11 | 42.4 | 8.3 |
| 6 | 20.1 | 54.9 | 2.73 |
| 11 | 17.5 | 29 | 1.66 |
| 12 | 15.9 | 27.3 | 1.72 |
| 13 | 16.3 | 27.3 | 1.67 |
| 23 | 25.3 | 27.4 | 1.08 |
| 25 | 27.5 | 42.5 | 1.54 |
| 49 | 53.8 | 67.4 | 1.25 | 

## False sharing, ns per increment (false_sharing.jsonl)

|  | ('atomic', 'packed') | ('atomic', 'padded') | ('store', 'packed') | ('store', 'padded') |
|---|---|---|---|---|
| ('cores', 1) | 4.62 | 4.63 | 0.284 | 0.28 |
| ('cores', 2) | 25 | 4.63 | 0.55 | 0.461 |
| ('cores', 4) | 53.5 | 4.68 | 0.562 | 0.465 |
| ('cores', 6) | 83 | 5.67 | 0.55 | 0.471 |
| ('cores', 8) | 113 | 5.94 | 0.549 | 0.475 |
| ('cores', 12) | 113 | 5.89 | 0.679 | 0.47 |
| ('smt_siblings', 2) | 12.6 | 13 | 4.22 | 1.23 | 

## Dispatch, ns per item (dispatch.jsonl)

|  | ('atomic', 1) | ('atomic_chunk', 8) | ('atomic_chunk', 64) | ('mutex', 1) | ('static', 0) |
|---|---|---|---|---|---|
| (0, 1) | 4.63 | 0.939 | 0.789 | 13.7 | 0.52 |
| (0, 2) | 13.7 | 2.21 | 0.8 | 58.5 | 0.267 |
| (0, 4) | 14.2 | 2.39 | 0.693 | 89.8 | 0.144 |
| (0, 6) | 14.7 | 2.55 | 0.691 | 97.3 | 0.107 |
| (0, 8) | 14.7 | 2.75 | 0.709 | 99.1 | 0.116 |
| (0, 12) | 15 | 2.62 | 0.683 | 93.5 | 0.124 |
| (50, 1) | 58 | 32.9 | 31.1 | 39.4 | 30.5 |
| (50, 2) | 61.3 | 21.2 | 17.3 | 104 | 16.6 |
| (50, 4) | 47.1 | 11.8 | 9.78 | 162 | 9.58 |
| (50, 6) | 46.2 | 8.87 | 7.51 | 183 | 7.57 |
| (50, 8) | 46.4 | 7.39 | 6.17 | 187 | 6.3 |
| (50, 12) | 45.5 | 6.76 | 4.26 | 173 | 4.66 | 

## caller_core, ms (caller_core.jsonl)

| test | ('async_sleeping', 11) | ('async_sleeping', 12) | ('async_spinning', 11) | ('async_spinning', 12) | ('sleeping', 11) | ('sleeping', 12) | ('spinning', 11) | ('spinning', 12) |
|---|---|---|---|---|---|---|---|---|
| mandelbrot_chunked | 92.7 | 37.6 | 39.1 | 63.9 | 73.4 | 45.2 | 42.1 | 64.2 |
| math_operations_in_tight_for_loop | 67.3 | 42.9 | 22.6 | 52.8 | 28.1 | 30.4 | 11.9 | 10.4 |
| ping_pong_equal | 15.1 | 14.2 | 13.9 | 29.7 | 19.4 | 14.2 | 19.7 | 32.3 |
| recursive_fibonacci | 7.67 | 6.79 | 7.31 | 7.45 | 9.3 | 6.85 | 9.48 | 7.36 |
| super_light | 18.2 | 19.8 | 14.2 | 31.7 | 8.45 | 8.14 | 7.4 | 6.28 | 

## smt, ms (smt.jsonl)

| test | ('sleeping', 12) | ('sleeping', 24) | ('spinning', 12) | ('spinning', 24) |
|---|---|---|---|---|
| mandelbrot_chunked | 57.9 | 33.5 | 40 | 128 |
| math_operations_in_tight_for_loop | 42.9 | 50.9 | 7.88 | 54.8 |
| ping_pong_equal | 24.8 | 18.7 | 21.7 | 34.2 |
| recursive_fibonacci | 11.1 | 8.27 | 8.64 | 30.6 | 

