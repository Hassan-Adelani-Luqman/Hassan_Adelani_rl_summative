| run | changed | lr | gamma | episodes_per_update | value_baseline | entropy_coef | normalize_returns | held_out_return | infections_averted | stranded_rate | episodes_to_converge | final_train_return |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | baseline | 0.001 | 0.99 | 16 | True | 0 | True | -787.5 ± 210.7 | 544.4 | 0.04 | 9236 | -789.4 |
| lr_3e-4 | lr=0.0003 | 0.0003 | 0.99 | 16 | True | 0 | True | -916.5 ± 221.3 | 459.6 | 0 | 4178 | -983.8 |
| lr_3e-3 | lr=0.003 | 0.003 | 0.99 | 16 | True | 0 | True | -1370.5 ± 322.7 | 199.3 | 0 | 137 | -1513.5 |
| gamma_0.95 | gamma=0.95 | 0.001 | 0.95 | 16 | True | 0 | True | -760.1 ± 181.7 | 574 | 0 | 9193 | -819.8 |
| gamma_0.999 | gamma=0.999 | 0.001 | 0.999 | 16 | True | 0 | True | -889.7 ± 194.5 | 511.6 | 0 | 8966 | -833.6 |
| episodes_4 | episodes_per_update=4 | 0.001 | 0.99 | 4 | True | 0 | True | -1381.8 ± 310.2 | 223.1 | 0 | 294 | -1365.9 |
| episodes_64 | episodes_per_update=64 | 0.001 | 0.99 | 64 | True | 0 | True | -845.6 ± 221.6 | 515.1 | 0 | 7778 | -916.1 |
| no_baseline | value_baseline=False | 0.001 | 0.99 | 16 | False | 0 | True | -748.2 ± 195.5 | 541.5 | 0 | 11553 | -777.7 |
| ★ entropy_0.01 | entropy_coef=0.01 | 0.001 | 0.99 | 16 | True | 0.01 | True | -714.0 ± 214.5 | 592 | 0 | 12821 | -711 |
| entropy_0.05 | entropy_coef=0.05 | 0.001 | 0.99 | 16 | True | 0.05 | True | -1126.8 ± 291.7 | 356.7 | 0.44 | 1034 | -1124.9 |
| no_return_norm | normalize_returns=False | 0.001 | 0.99 | 16 | True | 0 | False | -800.6 ± 204.0 | 533.1 | 0 | 11511 | -782.3 |
| no_baseline_no_norm | value_baseline=False, normalize_returns=False | 0.001 | 0.99 | 16 | False | 0 | False | -1720.8 ± 278.3 | 0 | 0 | — | -1744.2 |
