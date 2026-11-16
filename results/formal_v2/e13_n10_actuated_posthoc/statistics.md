# E13 n=10 framework vs actuated post-hoc analysis

> Positive oriented effects favor the framework. All 25 tests share one Holm family.
> A Holm-corrected difference requires adjusted p<0.05 and both bootstrap CIs to exclude zero.

| Scenario | Metric | Oriented mean | Seed 95% CI | Hierarchical 95% CI | Signs +/−/0 | Raw p | Holm p | Claim |
|---|---|---:|---|---|---:|---:|---:|---|
| low | throughput | -1.0100 | [-1.0100, -1.0100] | [-1.2700, -0.7400] | 0/10/0 | 0.001953 | 0.048828 | yes |
| low | unfinished | -1.0100 | [-1.0100, -1.0100] | [-1.2700, -0.7400] | 0/10/0 | 0.001953 | 0.048828 | yes |
| low | co2_kg | -0.2325 | [-0.2326, -0.2324] | [-0.2680, -0.1974] | 0/10/0 | 0.001953 | 0.048828 | yes |
| low | time_average_queue_vehicles | -1.0631 | [-1.0633, -1.0629] | [-1.1194, -1.0059] | 0/10/0 | 0.001953 | 0.048828 | yes |
| low | total_waiting_increment_s | -318.9190 | [-318.9840, -318.8740] | [-335.8321, -301.7580] | 0/10/0 | 0.001953 | 0.048828 | yes |
| medium | throughput | -1.0180 | [-1.1730, -0.8840] | [-1.3500, -0.6890] | 0/10/0 | 0.001953 | 0.048828 | yes |
| medium | unfinished | -1.0180 | [-1.1730, -0.8840] | [-1.3500, -0.6890] | 0/10/0 | 0.001953 | 0.048828 | yes |
| medium | co2_kg | -0.1339 | [-0.1696, -0.1036] | [-0.1901, -0.0793] | 0/10/0 | 0.001953 | 0.048828 | yes |
| medium | time_average_queue_vehicles | -1.0537 | [-1.1502, -0.9712] | [-1.1622, -0.9539] | 0/10/0 | 0.001953 | 0.048828 | yes |
| medium | total_waiting_increment_s | -316.1130 | [-345.0560, -291.3490] | [-348.6671, -286.1850] | 0/10/0 | 0.001953 | 0.048828 | yes |
| oversaturated | throughput | -1.5790 | [-1.8200, -1.3050] | [-2.0130, -1.1560] | 0/10/0 | 0.001953 | 0.048828 | yes |
| oversaturated | unfinished | -1.5790 | [-1.8200, -1.3050] | [-2.0130, -1.1560] | 0/10/0 | 0.001953 | 0.048828 | yes |
| oversaturated | co2_kg | -0.3098 | [-0.3491, -0.2672] | [-0.3771, -0.2434] | 0/10/0 | 0.001953 | 0.048828 | yes |
| oversaturated | time_average_queue_vehicles | -1.3715 | [-1.4937, -1.2227] | [-1.5137, -1.2107] | 0/10/0 | 0.001953 | 0.048828 | yes |
| oversaturated | total_waiting_increment_s | -411.4460 | [-448.1095, -366.8020] | [-454.1100, -363.1968] | 0/10/0 | 0.001953 | 0.048828 | yes |
| asymmetric | throughput | -2.8510 | [-3.3210, -2.3480] | [-3.5140, -2.2380] | 0/10/0 | 0.001953 | 0.048828 | yes |
| asymmetric | unfinished | -2.8510 | [-3.3210, -2.3480] | [-3.5140, -2.2380] | 0/10/0 | 0.001953 | 0.048828 | yes |
| asymmetric | co2_kg | -0.8741 | [-0.9492, -0.7942] | [-0.9626, -0.7847] | 0/10/0 | 0.001953 | 0.048828 | yes |
| asymmetric | time_average_queue_vehicles | -2.0893 | [-2.2642, -1.9074] | [-2.2747, -1.8998] | 0/10/0 | 0.001953 | 0.048828 | yes |
| asymmetric | total_waiting_increment_s | -626.7920 | [-679.2632, -572.2165] | [-682.4153, -569.9358] | 0/10/0 | 0.001953 | 0.048828 | yes |
| pulse | throughput | -0.4300 | [-0.4300, -0.4300] | [-0.6300, -0.2500] | 0/10/0 | 0.001953 | 0.048828 | yes |
| pulse | unfinished | -0.4300 | [-0.4300, -0.4300] | [-0.6300, -0.2500] | 0/10/0 | 0.001953 | 0.048828 | yes |
| pulse | co2_kg | -0.0003 | [-0.0003, -0.0003] | [-0.0045, +0.0039] | 0/10/0 | 0.001953 | 0.048828 | no |
| pulse | time_average_queue_vehicles | -0.0441 | [-0.0441, -0.0441] | [-0.0620, -0.0278] | 0/10/0 | 0.001953 | 0.048828 | yes |
| pulse | total_waiting_increment_s | -13.2200 | [-13.2200, -13.2200] | [-18.6002, -8.3300] | 0/10/0 | 0.001953 | 0.048828 | yes |

## Descriptive safety audit

| Scenario | Controller | Safe/runs | Rate |
|---|---|---:|---:|
| low | fixed_prior_framework | 1000/1000 | 100.0% |
| low | actuated | 95/100 | 95.0% |
| medium | fixed_prior_framework | 897/1000 | 89.7% |
| medium | actuated | 94/100 | 94.0% |
| oversaturated | fixed_prior_framework | 897/1000 | 89.7% |
| oversaturated | actuated | 99/100 | 99.0% |
| asymmetric | fixed_prior_framework | 829/1000 | 82.9% |
| asymmetric | actuated | 94/100 | 94.0% |
| pulse | fixed_prior_framework | 1000/1000 | 100.0% |
| pulse | actuated | 100/100 | 100.0% |
