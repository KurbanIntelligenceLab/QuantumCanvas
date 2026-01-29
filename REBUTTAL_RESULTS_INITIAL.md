# QuantumCanvas Rebuttal: Experimental Results

---

## 1. Modality Comparison (Test MAE ± std)

| Model | dipole_mag_d | e_g_ev | e_homo_ev | e_lumo_ev | total_energy_ev |
|-------|--------------|--------|-----------|-----------|-----------------|
| film_cnn | 0.5487±0.0157 | 0.6302±0.1058 | 0.3727±0.0104 | 0.3842±0.0137 | 3.3669±0.7435 |
| geometry_only | 0.5622±0.0231 | 0.5240±0.0452 | 0.2616±0.0182 | 0.3268±0.0162 | 1.7532±0.4752 |
| multimodal_v2 | 0.4929±0.1031 | 0.8354±0.0318 | 0.2193±0.0078 | 0.3102±0.0108 | 4.6904±0.2563 |
| **qsn_v2** | **0.4084±0.0155** | **0.1960±0.0183** | **0.1619±0.0116** | **0.1631±0.0170** | **1.1611±0.1620** |
| tabular_mlp | 0.2126±0.0158 | 0.3470±0.1969 | 0.2666±0.0207 | 0.2791±0.0123 | 2.2856±0.2950 |
| tabular_transformer | 0.1735±0.0669 | 0.5487±0.0704 | 0.1985±0.0155 | 0.2251±0.0343 | 1.1112±0.1206 |
| vision_only | 0.1290±0.0602 | 0.5866±0.0717 | 0.5338±0.0183 | 0.5760±0.0515 | 11.1097±0.1247 |

### Model Comparison (averaged across targets)

| Model | avg MAE | avg RMSE | Params |
|-------|---------|----------|--------|
| film_cnn | 1.0605 | 1.4746 | 346,485 |
| geometry_only | 0.6855 | 1.0786 | 318,529 |
| multimodal_v2 | 1.3096 | 1.6284 | 373,955 |
| **qsn_v2** | **0.4181** | **0.7037** | 357,145 |
| tabular_mlp | 0.6782 | 1.2872 | 326,145 |
| tabular_transformer | 0.4514 | 0.8308 | 328,257 |
| vision_only | 2.5870 | 4.0710 | 357,601 |

---

## 2. Element ID Shuffle / Mask Ablation (100 epochs)

**Summary: % MAE increase when Z shuffled (vs baseline)**

| Model | dipole_mag_d | e_g_ev | e_homo_ev | e_lumo_ev | total_energy_ev |
|-------|--------------|--------|-----------|-----------|-----------------|
| film_cnn | +484.4% | +105.1% | +1185.4% | +772.9% | +1994.6% |
| geometry_only | +524.5% | +25.0% | +555.9% | +443.0% | +3175.6% |
| multimodal_v2 | **+7.1%** | **+12.7%** | +801.3% | +471.3% | +911.5% |
| qsn_v2 | +641.7% | +393.5% | +1238.2% | +1333.8% | +5542.4% |
| tabular_mlp | +368.1% | +455.7% | +1144.3% | +1030.3% | +3125.7% |
| vision_only | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% |

*vision_only (ViT) does not use element IDs.*

### Example: dipole_mag_d baseline vs ablated MAE

| Model | Normal (baseline) | shuffle | shuffle_within_sample | mask | random | constant |
|-------|-------------------|---------|------------------------|------|--------|----------|
| film_cnn | 0.2892±0.0462 | 1.6068±0.2128 (+484.4%) | 0.8136±0.0494 (+192.6%) | 1.7392±0.2286 (+533.3%) | 1.6330±0.2193 (+494.2%) | 1.8662±0.2294 (+578.6%) |
| geometry_only | 0.4259±0.0097 | 2.6584±0.0347 (+524.5%) | 0.5971±0.0049 (+40.3%) | 2.0347±0.0381 (+377.9%) | 2.5080±0.0322 (+489.4%) | 2.2292±0.0714 (+423.5%) |
| multimodal_v2 | 0.4989±0.1002 | 0.5316±0.0962 (**+7.1%**) | **0.4989±0.1002 (+0.0%)** | 0.5254±0.1045 (+5.4%) | 0.5324±0.0996 (+7.1%) | 0.4333±0.0574 (−11.4%) |
| qsn_v2 | 0.2550±0.0103 | 1.8869±0.0531 (+641.7%) | **0.2550±0.0103 (+0.0%)** | 2.3283±0.0514 (+814.2%) | 1.9554±0.0719 (+667.8%) | 2.0801±0.0257 (+717.5%) |
| tabular_mlp | 0.1443±0.0165 | 0.6647±0.0167 (+368.1%) | 0.3155±0.0104 (+122.0%) | 0.7958±0.1548 (+451.0%) | 0.7206±0.0194 (+405.5%) | 0.8870±0.0187 (+524.2%) |
| vision_only | 0.1279±0.0570 | 0.1279±0.0570 (+0.0%) | 0.1279±0.0570 (+0.0%) | 0.1279±0.0570 (+0.0%) | 0.1279±0.0570 (+0.0%) | 0.1279±0.0570 (+0.0%) |

---

## 3. OOD Composition Split Results (100 epochs)

Splits: **by_electronegativity**, **by_period**, **held_out_pairs**. Targets: dipole_mag_d, e_g_ev, total_energy_ev.

### 3.1 by_electronegativity

- Train: 2044 | Val: 228 | Test (OOD): 568

| Target | film_cnn | geometry_only | multimodal_v2 | qsn_v2 | tabular_mlp | vision_only |
|--------|----------|---------------|---------------|--------|-------------|-------------|
| dipole_mag_d | ID 0.550±0.018 / OOD 0.815±0.068 / +47.8% | 0.533±0.030 / 0.932±0.046 / +75.3% | 0.493±0.111 / 0.505±0.102 / **+2.9%** | 0.421±0.024 / 0.756±0.069 / +80.5% | 0.203±0.017 / 0.286±0.029 / +41.2% | 0.107±0.022 / 0.106±0.020 / **−0.9%** |
| e_g_ev | 0.793±0.106 / 1.139±0.060 / +45.1% | 0.476±0.106 / 0.877±0.095 / +90.5% | 0.809±0.095 / 1.146±0.058 / +43.0% | 0.245±0.056 / 0.437±0.114 / +77.4% | 0.364±0.063 / 0.780±0.080 / +118.0% | 0.447±0.026 / 0.835±0.042 / +87.8% |
| total_energy_ev | 3.51±0.18 / 12.3±1.9 / +250.4% | 1.46±0.04 / 2.85±0.13 / +95.1% | 5.31±0.26 / 8.01±2.41 / **+51.8%** | 1.03±0.10 / 3.27±0.42 / +221.6% | 2.33±0.30 / 9.84±0.42 / +329.8% | 9.11±0.51 / 21.1±0.71 / +132.4% |

### 3.2 by_period

- Train: 594 | Val: 66 | Test (OOD): 2180

| Target | film_cnn | geometry_only | multimodal_v2 | qsn_v2 | tabular_mlp | vision_only |
|--------|----------|---------------|---------------|--------|-------------|-------------|
| dipole_mag_d | 0.932±0.138 / 1.84±0.07 / +103.0% | 0.478±0.047 / 1.90±0.01 / +301.4% | 0.442±0.034 / 0.478±0.082 / **+10.4%** | 0.405±0.049 / 1.85±0.04 / +366.0% | 0.320±0.058 / 0.583±0.01 / +89.6% | 0.116±0.026 / 0.158±0.022 / **+39.8%** |
| e_g_ev | 0.667±0.093 / 0.709±0.084 / +8.5% | 0.502±0.020 / 0.901±0.095 / +79.9% | 0.713±0.151 / 0.849±0.066 / +24.0% | 0.335±0.029 / 0.568±0.041 / +70.2% | 0.625±0.106 / 0.590±0.097 / **−5.1%** | 0.686±0.091 / 0.563±0.063 / **−16.8%** |
| total_energy_ev | 5.29±0.26 / 31.0±1.72 / +486.5% | 2.49±0.67 / 31.1±0.47 / +1233.6% | 4.13±0.25 / 34.0±5.34 / +717.5% | 2.04±0.35 / 31.0±4.69 / +1490.1% | 4.45±0.60 / 28.7±1.48 / +557.9% | 14.5±0.48 / 17.0±0.52 / **+17.5%** |

### 3.3 held_out_pairs

- Train: 2044 | Val: 228 | Test (OOD): 568

| Target | film_cnn | geometry_only | multimodal_v2 | qsn_v2 | tabular_mlp | vision_only |
|--------|----------|---------------|---------------|--------|-------------|-------------|
| dipole_mag_d | 0.600±0.084 / 0.569±0.078 / **−4.5%** | 0.578±0.021 / 0.558±0.046 / **−3.4%** | 0.499±0.025 / 0.500±0.035 / **+0.2%** | 0.390±0.028 / 0.395±0.003 / **+1.9%** | 0.216±0.018 / 0.212±0.026 / **−2.2%** | 0.149±0.082 / 0.144±0.077 / **−0.9%** |
| e_g_ev | 0.724±0.051 / 0.727±0.057 / **+0.4%** | 0.590±0.189 / 0.582±0.141 / +1.7% | 0.851±0.046 / 0.861±0.031 / +1.4% | 0.209±0.041 / 0.202±0.029 / **−2.6%** | 0.460±0.188 / 0.467±0.134 / +9.2% | 0.562±0.080 / 0.559±0.015 / +1.0% |
| total_energy_ev | 3.79±0.55 / 3.86±0.25 / **+3.0%** | 1.54±0.06 / 1.57±0.15 / +2.2% | 5.06±0.37 / 5.18±0.34 / **+2.5%** | 1.15±0.04 / 1.15±0.12 / **−0.3%** | 2.75±0.30 / 2.86±0.06 / +5.1% | 11.2±1.24 / 10.4±0.52 / **−6.4%** |

### Average generalization gap by model (across OOD splits)

| Model | avg ID MAE | avg OOD MAE | avg Gap |
|-------|------------|-------------|---------|
| film_cnn | 1.8735 | 5.8864 | +104.5% |
| geometry_only | 0.9603 | 4.5868 | +208.5% |
| **multimodal_v2** | 2.0348 | 5.7247 | **+94.9%** |
| qsn_v2 | 0.6925 | 4.4089 | +256.1% |
| tabular_mlp | 1.3015 | 4.9275 | +127.1% |
| **vision_only** | 4.0976 | 5.6547 | **+28.2%** |

---

## 4. Channel Permutation Importance (100 epochs)

**Source:** `rebuttal_results_100_epochs/channel_permutation_report.txt`  
Quantumshellnet; metrics averaged over 3 seeds. Reported: baseline MAE/RMSE and mean|ΔMAE| / mean|ΔRMSE| when input channels are permuted (sensitivity per channel).

### Channel importance (% of total sensitivity, averaged over targets and over MAE/RMSE)

Per target, each channel’s share of total |ΔMAE| (and |ΔRMSE|) is computed so the 10 channels sum to 100%; then averaged over all 20 targets and over both losses.

| Channel | MAE % | RMSE % | Avg % |
|---------|-------|--------|-------|
| ch_0 | 10.8 | 11.3 | 11.1 |
| ch_1 | 0.0 | 0.0 | 0.0 |
| ch_2 | 12.7 | 12.0 | 12.4 |
| ch_3 | 12.3 | 13.2 | 12.8 |
| ch_4 | 17.3 | 17.8 | 17.6 |
| ch_5 | 2.4 | 1.6 | 2.0 |
| ch_6 | 11.5 | 11.0 | 11.2 |
| ch_7 | 20.7 | 22.6 | 21.6 |
| ch_8 | 6.0 | 5.4 | 5.7 |
| ch_9 | 6.2 | 5.1 | 5.7 |

ch_7 is most important (~22%), ch_4 next (~18%); ch_1 is negligible (~0%).

### Summary: mean|ΔMAE| and mean|ΔRMSE| per target (quantumshellnet)

| Target | baseline MAE | baseline RMSE | mean\|ΔMAE\| | mean\|ΔRMSE\| |
|--------|--------------|---------------|--------------|---------------|
| dipole_mag_d | 0.8353 | 0.9154 | 0.1359 | 0.1788 |
| e_g_ev | 1.6142 | 2.0061 | 0.0255 | 0.0375 |
| e_homo_ev | 1.0323 | 1.2749 | 0.0566 | 0.0627 |
| e_lumo_ev | 1.2655 | 1.5648 | 0.0749 | 0.0988 |
| total_energy_ev | 18.7593 | 28.7858 | 3.5862 | 3.2946 |

Full per-target and per-channel importance (all 20 targets, ch_0–ch_9) are in `rebuttal_results_100_epochs/channel_permutation_report.txt`.

---

## 4. Channel Permutation Importance (100 epochs)


**Source:** `rebuttal_results_100_epochs/channel_permutation_report.txt`  
Quantumshellnet; metrics averaged over 3 seeds. Reported: baseline MAE/RMSE and mean|ΔMAE| / mean|ΔRMSE| when input channels are permuted (sensitivity per channel).

### Channel importance (% of total sensitivity, averaged over targets and over MAE/RMSE)

Per target, each channel’s share of total |ΔMAE| (and |ΔRMSE|) is computed so the 10 channels sum to 100%; then averaged over all 20 targets and over both losses.

| Channel | MAE % | RMSE % | Avg % |
|---------|-------|--------|-------|
| ch_0 | 10.8 | 11.3 | 11.1 |
| ch_1 | 0.0 | 0.0 | 0.0 |
| ch_2 | 12.7 | 12.0 | 12.4 |
| ch_3 | 12.3 | 13.2 | 12.8 |
| ch_4 | 17.3 | 17.8 | 17.6 |
| ch_5 | 2.4 | 1.6 | 2.0 |
| ch_6 | 11.5 | 11.0 | 11.2 |
| ch_7 | 20.7 | 22.6 | 21.6 |
| ch_8 | 6.0 | 5.4 | 5.7 |
| ch_9 | 6.2 | 5.1 | 5.7 |

ch_7 is most important (~22%), ch_4 next (~18%); ch_1 is negligible (~0%).

### Summary: mean|ΔMAE| and mean|ΔRMSE| per target (quantumshellnet)

| Target | baseline MAE | baseline RMSE | mean\|ΔMAE\| | mean\|ΔRMSE\| |
|--------|--------------|---------------|--------------|---------------|
| dipole_mag_d | 0.8353 | 0.9154 | 0.1359 | 0.1788 |
| e_g_ev | 1.6142 | 2.0061 | 0.0255 | 0.0375 |
| e_homo_ev | 1.0323 | 1.2749 | 0.0566 | 0.0627 |
| e_lumo_ev | 1.2655 | 1.5648 | 0.0749 | 0.0988 |
| total_energy_ev | 18.7593 | 28.7858 | 3.5862 | 3.2946 |

Full per-target and per-channel importance (all 20 targets, ch_0–ch_9) are in `rebuttal_results_100_epochs/channel_permutation_report.txt`.

---

## 4. Inference Timing

**Source:** `rebuttal_results/timing_e_g_ev.json` (target: e_g_ev, 2840 samples, 1000 runs, CUDA).

Per-sample inference time and throughput (batch inference over full dataset, averaged).

| Model | ms/sample | samples/sec | Params |
|-------|-----------|-------------|--------|
| schnet | 0.072 | 13,913 | 470,497 |
| gatv2 | 0.075 | 13,287 | 384,577 |
| egnn | 0.083 | 12,105 | 460,997 |
| vit | 0.122 | 8,167 | 458,689 |
| faenet | 0.175 | 5,724 | 435,593 |
| quantumshellnet | 0.218 | 4,586 | 343,542 |
| dimenet | 0.249 | 4,012 | 482,958 |

*Vision model (quantumshellnet) runs at **4,586 samples/sec**; GNNs are faster but image-based inference remains practical for screening.*

---

## QuantumshellNet Channel Permutation Summary

- Targets: dipole_mag_d, e_g_ev, e_homo_ev, e_lumo_ev, total_energy_ev
- Seeds: 123, 42, 456

### Average % ΔMAE per Channel (across targets)

| Channel | Avg % ΔMAE |
| --- | --- |
| ch_4 | 15.935% |
| ch_6 | 9.666% |
| ch_7 | 2.632% |
| ch_2 | 2.309% |
| ch_9 | 1.886% |
| ch_3 | 1.135% |
| ch_8 | 0.802% |
| ch_0 | 0.216% |
| ch_5 | 0.152% |
| ch_1 | 0.001% |
