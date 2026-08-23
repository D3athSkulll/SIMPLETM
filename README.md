# SimpleTM with Band Attention

This repository contains an extension of [SimpleTM](https://github.com/thuml/SimpleTM) with a small **Band Attention** module for wavelet-based multivariate time-series forecasting.

The main idea is simple: **SimpleTM treats all wavelet bands equally, but not every frequency band is equally useful for every input window.** Band Attention learns which wavelet bands are more useful and adjusts their importance before the geometric-product attention block.

The module was developed and evaluated as part of a B.Tech project on household electricity load forecasting.

---

## What is Band Attention?

SimpleTM first converts each variable into a token and applies a **Stationary Wavelet Transform (SWT)**. This gives multiple frequency bands, from the smoother approximation component to finer detail components.

In the original SimpleTM pipeline, these bands are passed to attention without an explicit learned importance weighting.

Band Attention adds a lightweight step between the SWT and the Q/K/V projections:

```text
Input
  |
  v
Linear Projection
  |
  v
SWT
  |
  v
Band Attention
  |
  v
Geometric Product Attention
  |
  v
ISWT
  |
  v
FFN + LayerNorm
  |
  v
Forecast
```

The module:

1. Pools each wavelet band into a compact descriptor.
2. Passes all band descriptors through a small MLP.
3. Converts the output into band weights.
4. Rescales the original wavelet coefficients.
5. Sends the reweighted coefficients to SimpleTM attention.

The important part is that the weights are **input-dependent**. A window containing mostly smooth consumption can receive a different band weighting from a window containing a sudden appliance-level spike.

---

## Why add Band Attention?

Household electricity load contains information at multiple time scales.

- Low-frequency bands capture the smoother daily load pattern.
- Higher-frequency bands can contain appliance switching and short-term changes.
- Some of the fine-scale information becomes less useful as the forecasting horizon increases.

Using the same weighting for every band makes SimpleTM use a fixed trade-off for every window.

Band Attention tries to make this trade-off adaptive.

It is also intentionally small. The current implementation adds only about **32 parameters per layer**, without changing the tensor shape or replacing the original SimpleTM attention mechanism.

---

## Main Results

The module was evaluated on **Load House 2** using a multivariate-to-univariate setup with an input length of 96 and prediction lengths of:

`4, 32, 48, 96, 192`

The comparison included the original SimpleTM branch, Band Attention, other SimpleTM variants, and external forecasting baselines.

### SimpleTM vs Band Attention

| Model | Avg. MSE | Avg. MAE |
|---|---:|---:|
| Original SimpleTM | 1.041 | 0.468 |
| **Band Attention** | **1.023** | **0.461** |

Band Attention gives:

- **1.73% lower average MSE**
- **1.50% lower average MAE**
- Best average MAE among the models in the comparison
- Best/shared-best average MSE with the cross-product attention variant

The improvement is more visible at shorter horizons.

### Short Horizon

At the 4-step / 1-hour prediction horizon:

| Model | MSE | MAE |
|---|---:|---:|
| Original SimpleTM | 0.903 | 0.448 |
| **Band Attention** | **0.890** | **0.426** |

This corresponds to:

- **1.44% reduction in MSE**
- **4.91% reduction in MAE**

The MAE improvement is the largest single improvement observed for Band Attention in the experiment.

### Long Horizon

At the 192-step / 48-hour horizon:

| Model | MSE | MAE |
|---|---:|---:|
| Original SimpleTM | 1.089 | 0.475 |
| Band Attention | 1.084 | 0.475 |

The improvement is almost gone at this horizon.

This is expected. Fine-grained wavelet information is more useful for short-term forecasting, while much of that information becomes noise when predicting far into the future.

---

## Comparison with Other Variants

We also tested a few other modifications to understand whether simply reweighting wavelet bands was enough.

| Model | Avg. MSE | Avg. MAE |
|---|---:|---:|
| Original SimpleTM | 1.041 | 0.468 |
| Wavelets (Direct Weighting) | 1.054 | 0.473 |
| Wavelets (Fourier Weighting) | 1.061 | 0.472 |
| Cross Attention | 1.023 | 0.462 |
| **Band Attention** | **1.023** | **0.461** |

The fixed wavelet weighting methods were actually worse than the original SimpleTM model.

This suggests that the improvement is not simply because the wavelet bands were reweighted. The useful part is that **Band Attention learns the weighting from the input**.

Cross-product attention gives a very similar MSE, but Band Attention achieves slightly better average MAE while adding very little parameter overhead.

---

## How it works

Let the SWT output be:

```text
C ∈ R^(B × N × (m+1) × D)
```

where:

- `B` = batch size
- `N` = number of variables
- `m+1` = number of wavelet bands
- `D` = feature dimension

### 1. Squeeze

Each band is reduced to a scalar descriptor.

```text
Wavelet coefficients
        |
        v
 Global pooling
        |
        v
One descriptor per band
```

### 2. Excite

The descriptors are passed through a small two-layer MLP:

```text
Band descriptors
      |
      v
 Linear
      |
     GELU
      |
   Linear
      |
      v
 Band scores
```

### 3. Normalize

The scores are converted into weights.

The current experiments use **Softmax**:

```text
α = Softmax(scores)
```

This makes the bands compete with each other for importance.

### 4. Rescale

The learned weights are broadcast back to the original coefficient tensor:

```text
C' = α ⊙ C
```

The resulting coefficients are then passed to the existing SimpleTM Q/K/V projections.

---

## Implementation

The main implementation is located in:

```text
layers/SWTAttention_Family.py
```

The changes are intentionally small.

A new `BandAttention` module is added, and the existing `GeomAttentionLayer` calls it after SWT decomposition and before the Q/K/V projections.

The implementation also includes debug hooks for inspecting the learned band weights during experiments.

When Band Attention is disabled, the forward path remains the same as the original SimpleTM implementation.

---

## Configuration

The main Band Attention settings used in the experiments are:

```text
Wavelet        : db1
SWT depth      : 3
Number of bands: 4
Hidden size    : 4
Activation     : Softmax
Lookback       : 96
d_model        : 256
d_ff           : 1024
Encoder layers : 1
Batch size     : 256
Learning rate  : 0.01
```

For a 3-level SWT:

```text
m = 3

Number of bands = m + 1 = 4
```

The four bands represent different frequency resolutions of the input.

---

## Getting Started

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd SIMPLETM
```

### 2. Create the environment

If the repository contains the original SimpleTM environment file:

```bash
conda env create -f environment.yml -n SimpleTM
conda activate SimpleTM
```

You can also install the required Python packages manually if needed.

The experiments use:

- Python 3
- PyTorch
- PyWavelets
- NumPy
- Pandas

---

## Dataset

The experiments in this project use household electricity load data from **Load House 2**.

The preprocessing used for the experiments was:

```text
Raw 15-minute readings
        |
        v
Missing-value interpolation
        |
        v
Hourly aggregation
        |
        v
Standard scaling
        |
        v
70% Train / 10% Validation / 20% Test
        |
        v
Sliding windows
```

The Band Attention experiments use a 96-step input window.

The model is evaluated in the **multivariate-to-univariate (M → S)** setting, where multiple variables are provided as input while the target load channel is used for evaluation.

---

## Running an Experiment

The project follows the original SimpleTM training interface.

A typical run looks like:

```bash
python run.py \
  --is_training 1 \
  --root_path "./dataset/" \
  --data_path "Load House 2.csv" \
  --model_id Load_House_2 \
  --model SimpleTM \
  --data custom \
  --features M \
  --target OT \
  --freq h \
  --seq_len 96 \
  --pred_len 96 \
  --enc_in 4 \
  --dec_in 4 \
  --c_out 1 \
  --d_model 256 \
  --d_ff 1024 \
  --e_layers 1 \
  --n_heads 8 \
  --batch_size 256 \
  --learning_rate 0.01 \
  --train_epochs 10
```

The exact arguments should be adjusted according to the dataset and experiment configuration in the repository.

---

## Original SimpleTM vs Band Attention

The main difference is only in the wavelet processing stage.

### Original SimpleTM

```text
SWT
 |
 +-- Band 1 ----\
 +-- Band 2 -----\
 +-- Band 3 ------> Geometric Attention
 +-- Band 4 -----/
 |
ISWT
```

All bands enter the attention pipeline without an explicit learned band-importance module.

### SimpleTM + Band Attention

```text
SWT
 |
 +-- Band 1 --\
 +-- Band 2 ---\
 +-- Band 3 ----> Band Attention --> Geometric Attention
 +-- Band 4 ---/
 |
ISWT
```

The rest of SimpleTM is kept unchanged.

This makes the module relatively easy to add or remove from the model.

---

## What improved?

The experiments show a few useful observations.

### 1. Better short-term forecasting

The largest gain appears at the shortest prediction horizon.

At 4 steps:

```text
MSE : 0.903 → 0.890
MAE : 0.448 → 0.426
```

The MAE reduction is approximately **4.9%**.

### 2. Best average MAE

Across the five tested horizons:

```text
Original SimpleTM : 0.468
Band Attention    : 0.461
```

Band Attention achieves the lowest average MAE among the non-degenerate forecasting models in the comparison.

### 3. Very small parameter overhead

The module adds roughly:

```text
~32 parameters / layer
```

So the improvement does not come from significantly increasing the model size.

### 4. Adaptive instead of fixed weighting

Fixed wavelet weighting performed worse than the original SimpleTM model.

This supports the main idea behind the module:

> The useful frequency bands depend on the current input window and forecasting horizon.

---

## Important Notes

The current implementation is a research prototype, and there are a few things that can still be improved.

### Band pooling

The current squeeze operation uses the average of signed coefficients.

Wavelet detail coefficients are often close to zero mean, so averaging them can hide useful information.

A better version could use band energy or absolute magnitude instead:

```text
g_i = sqrt(mean(C_i²))
```

or:

```text
g_i = mean(|C_i|)
```

### Softmax scaling

With four bands, a uniform Softmax gives an average weight of:

```text
1 / 4 = 0.25
```

Since the module is applied to Q, K and V, this can reduce the scale of the attention computation more than intended.

A future version could initialize the module as an exact identity or apply the weighting in a more controlled way.

### Q/K/V weighting

The current implementation can generate separate band weights for queries, keys and values.

This means the three streams may receive different band weightings.

A possible improvement is to calculate one shared band-weight vector and use it consistently across Q/K/V.

These limitations are documented in the project report and are useful directions for future experiments.

---

## Comparison with Other Models

Band Attention was not evaluated in isolation. The Load House 2 experiments also include other SimpleTM variants and several standard multivariate forecasting models.

The comparison uses the same **M → S setting**, with a 96-step input window and prediction horizons of 4, 32, 48, 96 and 192 steps.

### Overall comparison

The table below shows the average MSE and MAE across the five prediction horizons.

| Model | Avg. MSE ↓ | Avg. MAE ↓ | 1h MSE ↓ | 1h MAE ↓ |
|---|---:|---:|---:|---:|
| **Band Attention** | **1.0234** | **0.4614** | **0.8902** | **0.4257** |
| Cross Attention | 1.0232 | 0.4616 | 0.8920 | 0.4363 |
| Original SimpleTM | 1.0412 | 0.4681 | 0.9032 | 0.4484 |
| Autoformer | 1.0536 | 0.4902 | 0.9630 | 0.4871 |
| Direct Wavelet Weighting | 1.0536 | 0.4733 | 0.9332 | 0.4703 |
| Fourier Wavelet Weighting | 1.0610 | 0.4718 | 0.9457 | 0.4738 |
| iTransformer | 1.0683 | 0.4672 | 1.0683 | 0.4672 |
| FEDformer | 1.6114 | 0.7795 | 1.5581 | 0.7718 |

Among the models that produced meaningful, horizon-dependent forecasts, **Band Attention has the best average MAE** and is tied very closely with Cross Attention on average MSE.

Compared with the original SimpleTM:

- Band Attention reduces average MSE by about **1.7%**.
- Band Attention reduces average MAE by about **1.4%**.
- At 1 hour, MSE improves by about **1.4%**.
- At 1 hour, MAE improves by about **5.1%**.
- Band Attention also gives a lower average RMSE, MAPE, MedAE, Maximum Error and SMAPE than the original branch.

### Comparison at the shortest horizon

The 1-hour results show the clearest advantage of the proposed module:

| Model | MSE | MAE | MAPE | R² | EVS |
|---|---:|---:|---:|---:|---:|
| **Band Attention** | **0.8902** | **0.4257** | **0.8448** | **0.2131** | **0.2200** |
| Cross Attention | 0.8920 | 0.4363 | 0.8902 | 0.2114 | 0.2139 |
| Original SimpleTM | 0.9032 | 0.4484 | 0.9215 | 0.0510 | 0.0550 |
| iTransformer | 1.0683 | 0.4672 | 0.9337 | 0.0520 | 0.0579 |
| Autoformer | 0.9630 | 0.4871 | — | — | — |
| Direct Wavelet Weighting | 0.9332 | 0.4703 | 0.9497 | 0.1751 | 0.1752 |
| Fourier Wavelet Weighting | 0.9457 | 0.4738 | 0.9257 | 0.1640 | 0.1648 |
| FEDformer | 1.5581 | 0.7718 | — | — | — |

At this horizon, Band Attention gives the **lowest MSE and MAE among the compared non-degenerate models**. It also gives the highest R² and EVS in this comparison.

### What the comparison tells us

The comparison is useful because it shows that the improvement is not simply caused by changing the wavelet family or adding another attention block.

**Fixed wavelet weighting does not help.** The direct and Fourier weighting variants have worse average MSE than the original SimpleTM. This supports the idea that the important part of Band Attention is the **input-dependent weighting**, rather than wavelet reweighting by itself.

**Cross Attention is a strong alternative.** It reaches almost the same average MSE as Band Attention, but Band Attention has slightly better average MAE and substantially lower 1-hour MAE.

**iTransformer and Autoformer are competitive in some individual metrics**, but their overall MSE/MAE is higher than Band Attention in this experiment.

**FEDformer performs substantially worse on this particular Load House 2 configuration.** This does not mean FEDformer is generally worse; it only describes the result obtained under the configuration used in this project.

**TimeMixer needs to be treated separately.** The Excel results report exactly the same MSE/MAE values at every horizon (`0.462 / 0.247`). A forecast whose error remains identical from 1 hour to 48 hours is suspicious and was identified in the report as a degenerate run. Therefore, its apparent numerical advantage should **not** be treated as evidence that TimeMixer outperforms the other models. The run should be repeated before drawing a conclusion.

## Results Summary

The final experiments were compared against the original SimpleTM M→S branch on Load House 2. The Excel result logs contain MSE, MAE, RMSE, MAPE, MSPE, R², EVS, MedAE, Maximum Error and SMAPE.

| Metric | Original SimpleTM | Band Attention | Relative change |
|---|---:|---:|---:|
| Average MSE | 1.0412 | **1.0234** | **1.71% ↓** |
| Average MAE | 0.4681 | **0.4614** | **1.42% ↓** |
| Average RMSE | 1.0143* | **1.0110*** | **0.33% ↓** |
| Average MAPE | 0.9545 | **0.9254** | **3.04% ↓** |
| Average MSPE | 284.41 | 332.93 | 17.06% ↑ |
| Average R² | 0.0510 | **0.0940** | **84.32% ↑** |
| Average EVS | 0.0550 | **0.0977** | **77.77% ↑** |
| Average MedAE | 0.2576 | **0.2508** | **2.66% ↓** |
| Average Maximum Error | 14.6982 | **14.6393** | **0.40% ↓** |
| Average SMAPE | 1.2213 | **1.1786** | **3.49% ↓** |

The average values above are calculated across the five tested horizons: 1 h, 8 h, 12 h, 24 h and 48 h. For metrics where lower is better, a downward percentage is an improvement; for R² and EVS, an upward percentage is an improvement.

The most useful improvements are therefore not limited to MSE and MAE. Band Attention also improves **MAPE, MedAE, Maximum Error and SMAPE**, while the explained-variance metrics show a large relative increase. The one clear regression is **MSPE**, which becomes worse because this metric strongly amplifies percentage errors around small true load values.

### Horizon-wise improvements

The short-horizon results are where Band Attention is most effective.

| Horizon | MSE ↓ | MAE ↓ | MAPE ↓ | R² ↑ | EVS ↑ | MedAE ↓ | SMAPE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 h | **1.44%** | **5.06%** | **8.33%** | **318.0%** | **300.1%** | **12.93%** | **12.97%** |
| 8 h | **2.14%** | **0.80%** | **2.19%** | **64.94%** | **57.86%** | **1.27%** | **2.42%** |
| 12 h | **3.15%** | **2.74%** | **3.60%** | **30.38%** | **28.30%** | **1.66%** | **1.97%** |
| 24 h | **1.26%** | 1.22% ↑ | ~0.03% ↑ | **23.51%** | **17.40%** | 1.28% ↑ | **0.35%** |
| 48 h | **0.52%** | 0.10% ↑ | **1.32%** | 15.25% ↓ | 14.83% ↓ | 1.31% ↑ | 0.26% ↑ |

This makes the behaviour of the module clearer: **Band Attention mainly helps while short-term frequency information is still predictive.** The MSE improvement remains positive at every horizon, but the gains in MAE and explained variance become less consistent at 24–48 hours.

At 1 hour, the improvement is particularly strong: MAE falls from **0.4484 to 0.4257**, MAPE falls from **0.9215 to 0.8448**, MedAE falls from **0.2576 to 0.2243**, and SMAPE falls from **1.2213 to 1.0628**. R² increases from **0.0510 to 0.2131**, while EVS increases from **0.0550 to 0.2200**.

At 48 hours, MSE still improves slightly (**1.0895 → 1.0838**), but MAE is effectively unchanged (**0.4746 → 0.4750**) and R²/EVS are slightly lower. This agrees with the main observation in the report that the benefit of learned band weighting fades at long horizons.

> **Note:** The Excel sheet contains an apparent RMSE entry error for the original 12-hour result (`0.041532`). Since RMSE should be the square root of MSE, the README uses the RMSE values consistently with the reported MSE values rather than treating that single spreadsheet cell as a valid metric. The original report also presents the long-horizon behaviour qualitatively rather than relying on that anomalous cell.

---

## Project Report

A detailed explanation of the architecture, methodology, experiments, comparisons and future improvements is available in the project report.
The report also contains plots comparing MSE, MAE, RMSE, MAPE, MSPE, R², EVS, MedAE, Maximum Error and SMAPE across prediction horizons.

---

## Acknowledgements

This work was carried out as part of a B.Tech project at **ABV-IIITM Gwalior** under the guidance of **Dr. Anshul**.

We also thank the authors of SimpleTM for making the original implementation publicly available, which made this extension and comparison possible.
