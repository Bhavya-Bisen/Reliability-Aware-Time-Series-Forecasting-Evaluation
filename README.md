# Reliability-Aware Multivariate Household Energy Forecasting
## Forecast Failure Analysis, Gradient Sensitivity, Explainability Reliability, and LLM Evaluation

> **Research focus:** This project studies not only whether a multivariate LSTM can forecast household electricity consumption, but also **where it fails, whether its errors contain structure, whether post-hoc explanations are numerically trustworthy, and how explanation reliability affects downstream LLM-generated interpretations**.

---

## Abstract

Deep learning models are often evaluated primarily through aggregate predictive metrics. This project takes a different approach: it treats **forecast reliability, failure analysis, and explanation validity as first-class research questions**.

Using the UCI *Individual Household Electric Power Consumption* dataset, a PyTorch LSTM is trained to forecast `Global_active_power` over a 30-step horizon from a 120-step multivariate historical window. The model is compared against persistence, seasonal-naive, and XGBoost baselines and is evaluated across forecast horizon, peak-demand regimes, temporal regions, residual structure, feature ablation, robustness conditions, and computational cost.

A major finding is that the trained LSTM does **not** outperform the persistence baseline on aggregate MAE, while its errors increase with forecast horizon and become substantially larger during peak-consumption events. Residual autocorrelation further indicates that predictable temporal structure remains in the forecasting errors.

The project also re-audits its original Captum Integrated Gradients (IG) explainability pipeline. The trained model exhibits extremely large input-gradient magnitudes for many validation windows, with sensitivity strongly concentrated in earlier historical timesteps. In a 100-window audit, IG completeness fails catastrophically for every tested sample under the original-like CUDA/cuDNN configuration. Therefore, the original attribution magnitudes and feature/timestep rankings are **not treated as reliable evidence of model importance**. Feature ablation remains independently useful because it does not depend on the failed attribution values.

The project therefore argues for a broader principle: **an explanation method should itself be validated before its outputs are interpreted or passed downstream to an LLM**.

---

# 1. Research Motivation

Forecasting systems are often presented through a single performance metric and a small number of example predictions. This can hide important questions:

- Does the model outperform a trivial but strong forecasting rule?
- Does error increase with forecast distance?
- Are rare high-demand events substantially harder than ordinary periods?
- Do residuals still contain predictable temporal structure?
- Are apparent feature attributions numerically valid?
- Does the explanation depend on computational backend details?
- Can an LLM produce trustworthy interpretations if its explanation inputs are themselves unreliable?

This project is organized around these questions rather than around model architecture alone.

---

# 2. Research Questions

### RQ1 — Forecasting value
**Does the trained LSTM outperform simpler forecasting baselines?**

### RQ2 — Forecast reliability
**How does forecasting error change across horizon, peak/non-peak regimes, and temporal regions?**

### RQ3 — Residual structure
**Does predictable temporal structure remain after LSTM forecasting?**

### RQ4 — Feature dependence
**Which input variables materially affect predictive performance when ablated?**

### RQ5 — Explanation reliability
**Are Integrated Gradients attributions numerically valid for this trained recurrent model?**

### RQ6 — Backend sensitivity
**How sensitive are model outputs and input gradients to cuDNN versus fallback LSTM implementations?**

### RQ7 — LLM explanation validity
**What can an LLM safely explain when the upstream attribution signal itself may be unreliable?**

---

# 3. Dataset

The project uses the **Individual Household Electric Power Consumption** dataset from the UCI Machine Learning Repository.

**Source:** [UCI Machine Learning Repository — Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)

**Dataset citation:**

> Hebrail, G. & Berard, A. (2006). *Individual Household Electric Power Consumption*. UCI Machine Learning Repository. https://doi.org/10.24432/C58K54

The raw measurements are transformed into multivariate historical windows with additional calendar-derived temporal features.

---

# 4. Forecasting Problem Formulation

The forecasting problem is formulated as:

```text
Historical multivariate sequence
        (120 timesteps)
               |
               v
        preprocessing
               |
               v
          PyTorch LSTM
               |
               v
       30-step forecast
               |
               v
      Global_active_power
```

### Model configuration

| Component | Value |
|---|---|
| Input features | 13 |
| Lookback | 120 timesteps |
| LSTM hidden size | 64 |
| Forecast horizon | 30 |
| Target | `Global_active_power` |
| Architecture | `LSTM(13 -> 64)` + `Linear(64 -> 30)` |
| Training batch size | 32 |
| Optimizer | Adam |
| Main saved-run loss | Huber |
| Framework | PyTorch |

---

# 5. Features

### Electrical variables

- `Global_active_power`
- `Global_reactive_power`
- `Voltage`
- `Global_intensity`
- `Sub_metering_1`
- `Sub_metering_2`
- `Sub_metering_3`

### Temporal features

- `year`
- `weekday`
- `quarter_sin`
- `quarter_cos`
- `month_sin`
- `month_cos`
- `day_of_year_sin`
- `day_of_year_cos`

These features are intended to expose both recent electrical state and recurring calendar structure.

---

# 6. Experimental Evaluation Framework

The project evaluates the forecasting system across several complementary dimensions.

| Category | Evaluation |
|---|---|
| Baseline | Persistence |
| Baseline | Seasonal naive |
| Classical ML | XGBoost |
| Deep learning | LSTM |
| Accuracy | MAE |
| Accuracy | RMSE |
| Relative error | sMAPE / MASE |
| Forecast reliability | Horizon-wise error |
| Regime analysis | Peak vs non-peak |
| Temporal analysis | Day/hour error |
| Diagnostics | Residual autocorrelation |
| Sensitivity | Feature ablation |
| Sensitivity | Lookback sensitivity |
| Robustness | Missingness / noise |
| Explainability | Integrated Gradients audit |
| Numerical reliability | Backend comparison |
| LLM evaluation | Grounding / hallucination / causal overclaim |
| Engineering | Inference/training cost |

---

# 7. Baseline Comparison

A central result is that the trained LSTM does **not** outperform the persistence baseline.

| Model | MAE | RMSE | sMAPE % | MASE |
|---|---:|---:|---:|---:|
| Persistence | 0.350385 | 0.670852 | 33.7408 | 4.03063 |
| Seasonal Naive | 0.689125 | 1.05273 | 62.5639 | 7.9273 |
| LSTM | 0.561946 | 0.854072 | 59.8693 | 6.46431 |
| XGBoost | 0.791852 | 1.12713 | 110.782 | 9.10901 |

### Finding

The persistence baseline achieves substantially lower MAE than the trained LSTM.

This is treated as a **research result rather than hidden as a failed model outcome**. The result suggests that short-horizon household power consumption contains strong local persistence that the trained recurrent model does not exploit effectively enough to justify its added complexity.

---

# 8. Forecast-Horizon Degradation

Aggregate metrics conceal the effect of forecasting distance.

The LSTM's MAE increases across the 30-step horizon:

- Horizon 1: approximately **0.525**
- Horizon 30: approximately **0.584**

<a href="evaluation_outputs/Figure_3_error_vs_horizon.png"><img src="evaluation_outputs/Figure_3_error_vs_horizon.png" alt="Error vs forecast horizon" width="700"></a>

### Finding

Predictive accuracy degrades gradually as the forecast extends further into the future.

---

# 9. Peak vs Non-Peak Forecasting

Peak demand is defined using the 90th percentile of the target distribution.

| Model | Regime | MAE | RMSE | sMAPE % | MASE |
|---|---|---:|---:|---:|---:|
| LSTM | Peak (>= P90) | 1.89631 | 2.1353 | 93.7955 | 21.8141 |
| LSTM | Non-peak | 0.413541 | 0.550879 | 56.0961 | 4.75714 |
| Persistence | Peak (>= P90) | 1.05645 | 1.44924 | 43.8137 | 12.1528 |
| Persistence | Non-peak | 0.271858 | 0.516241 | 32.6205 | 3.1273 |

<a href="evaluation_outputs/Figure_4_peak_vs_nonpeak.png"><img src="evaluation_outputs/Figure_4_peak_vs_nonpeak.png" alt="Peak vs non-peak forecasting error" width="700"></a>

### Finding

Peak consumption is substantially harder to forecast than ordinary consumption. The LSTM peak MAE is approximately **4.6x** its non-peak MAE.

Persistence also outperforms the LSTM during peak periods.

This indicates that the neural model's largest errors occur precisely in the regime where accurate prediction may be most operationally valuable.

---

# 10. Temporal Error Structure

Error is also evaluated across day-of-week and hour-of-day combinations.

<a href="evaluation_outputs/Figure_5_temporal_error_heatmap.png"><img src="evaluation_outputs/Figure_5_temporal_error_heatmap.png" alt="Temporal error heatmap" width="700"></a>

The current evaluation contains a high-error region around the late-morning period, including a particularly strong cell around Friday late morning.

### Finding

Forecast error is not uniformly distributed in time; some temporal regimes are more difficult than others.

---

# 11. Residual Diagnostics

Residual autocorrelation is used to test whether predictable temporal structure remains after forecasting.

The residual series shows strong positive autocorrelation:

- lag-1 residual autocorrelation ≈ **0.52**
- positive autocorrelation persists across multiple subsequent lags

<a href="evaluation_outputs/Figure_residual_autocorrelation.png"><img src="evaluation_outputs/Figure_residual_autocorrelation.png" alt="Residual autocorrelation" width="700"></a>

### Finding

The LSTM has not removed all predictable temporal structure from the forecasting error.

This suggests that additional predictive structure remains available but is not being captured by the current model/configuration.

---

# 12. Representative Forecast and Calibration Behaviour

A representative 30-step forecast shows a visibly compressed prediction trajectory relative to the true target.

<a href="evaluation_outputs/Figure_2_actual_vs_predicted.png"><img src="evaluation_outputs/Figure_2_actual_vs_predicted.png" alt="Actual vs predicted trajectory" width="700"></a>

The actual sequence rises above 4 in the displayed example, while the model prediction remains substantially lower.

### Finding

The current model appears to under-represent high-amplitude variation in this example.

Possible explanations include:

- target scaling / inverse transformation issues,
- target-output alignment,
- training convergence,
- insufficient capacity,
- objective-function effects,
- or distributional mismatch.

These possibilities remain hypotheses rather than established causes.

---

# 13. Feature Ablation

Feature ablation evaluates predictive dependence without relying on gradient-based explainability.

| Feature | Base MAE | Ablated MAE | Relative MAE change |
|---|---:|---:|---:|
| `Global_active_power` | 0.570728 | 0.630692 | +10.51% |
| `Global_intensity` | 0.570728 | 0.617784 | +8.24% |
| `Sub_metering_3` | 0.570728 | 0.578319 | +1.33% |
| `Sub_metering_1` | 0.570728 | 0.568401 | -0.41% |
| `weekday` | 0.570728 | 0.567725 | -0.53% |

### Finding

Removing historical `Global_active_power` or `Global_intensity` causes the largest observed performance degradation among the tested features.

Importantly, these ablation results remain interpretable even after the later Integrated Gradients reliability failure, because ablation does not depend on those attribution values.

---

# 14. Lookback Sensitivity

| Lookback | MAE | RMSE | sMAPE % | MASE | Note |
|---:|---:|---:|---:|---:|---|
| 60 | 0.572454 | 0.872414 | 60.7255 | 6.58518 | same trained weights |
| 120 | 0.569603 | 0.855295 | 60.6998 | 6.55238 | same trained weights |
| 240 | 0.561673 | 0.846871 | 59.8297 | 6.46117 | same trained weights |

These measurements indicate sensitivity to historical context length, but because the same trained weights are reused, they should not be interpreted as a fully controlled retraining comparison between different lookback architectures.

---

# 15. Explainability Reliability Audit

The original project used Captum **Integrated Gradients (IG)** to compute feature and temporal attributions.

The original configuration used:

- a training-mean historical-window baseline,
- approximately 50 IG integration steps,
- CUDA/cuDNN,
- `model.train()`,
- multiple validation examples,
- all 30 forecast outputs.

The resulting attribution magnitudes reached approximately:

- feature attribution: order **1e17**
- temporal attribution: order **1e18**

These values were originally interpreted relatively as feature/timestep importance.

A later audit questioned whether the attribution values themselves were numerically valid.

The historical notebook is retained for provenance:

```text
notebooks/Scratch/ExplainableAI.ipynb
```

The reliability investigation is maintained separately:

```text
Explainability_Reliability_Audit.ipynb
```

This preserves the original experiment while making the later methodological correction explicit.

---

# 16. Model Integrity Checks

Before attributing the problem to Captum or model implementation, the audit verified the saved model itself.

### 16.1 Evaluation mode

- `model.eval()` behaves correctly.
- all submodules are placed in evaluation mode.

### 16.2 Deterministic inference

Repeated forward passes under a fixed backend produce identical predictions:

```text
maximum repeated-prediction difference = 0
```

### 16.3 Manual forward reconstruction

The model forward pass was reconstructed manually:

```python
lstm_output, _ = model.lstm(x)
last_hidden = lstm_output[:, -1, :]
manual_output = model.fc(last_hidden)
```

The result matched `model(x)` exactly:

```text
max absolute difference = 0
```

### Finding

No hidden transformation inside the model's normal forward method explains the strange gradient behaviour.

---

# 17. cuDNN Constraint and Experimental Conditions

Attempting backward-gradient computation with a cuDNN LSTM in evaluation mode produced:

```text
RuntimeError:
cudnn RNN backward can only be called in training mode
```

This is treated as a backend constraint, not evidence of exploding gradients.

Two experimental conditions were therefore examined.

### Condition A — historical-like

```text
model.train()
CUDA
cuDNN enabled
```

### Condition B — reliability fallback

```text
model.eval()
cuDNN disabled
```

The saved model has:

- one LSTM layer,
- LSTM dropout = 0,
- no explicit Dropout modules.

Within the same backend, `train()` and `eval()` produce the same forward predictions.

---

# 18. Backend Numerical Sensitivity

For a frozen validation input, cuDNN and fallback LSTM implementations produced different numerical outputs.

### Example — horizon 1

```text
cuDNN output    ≈ 0.037474
fallback output ≈ 0.101023
```

Maximum difference across the 30 outputs:

```text
≈ 0.0705
```

Within each backend:

```text
eval_cudnn     == train_cudnn
eval_fallback  == train_fallback
```

### Finding

The discrepancy is associated with the computational LSTM implementation rather than train/eval mode.

The project does **not** claim that either backend is definitively mathematically correct or incorrect. The low-level cause was not investigated to kernel-level depth.

---

# 19. Raw Input-Gradient Sensitivity

Input gradients were computed independently of Captum using PyTorch autograd:

```python
torch.autograd.grad(model(x)[0, 0], x)
```

### Historical-like cuDNN/train condition

```text
gradient L2      ≈ 2.895e12
max |gradient|   ≈ 2.306e12
```

### eval/fallback condition

```text
gradient L2      ≈ 2.448e14
max |gradient|   ≈ 1.950e14
```

Despite the magnitude difference:

```text
cosine similarity ≈ 0.999954
fallback/cuDNN gradient-norm ratio ≈ 84.6
```

### Finding

Both implementations identify almost the same gradient direction, while the exact gradient scale is strongly backend-dependent.

A conservative interpretation is:

> The trained LSTM exhibits severe input-gradient sensitivity under both computational implementations, while the exact magnitude of that sensitivity is backend-dependent.

This does **not** establish that training-time parameter gradients exploded.

---

# 20. Temporal Gradient Amplification

The audit computes:

```text
|| dF / dx_t ||_2
```

for each of the 120 historical timesteps.

For individual validation windows:

- early historical timesteps can reach approximately **1e12–1e14**,
- recent historical timesteps can fall below **1**.

The pattern was then evaluated over 100 validation windows.

### Distribution of input-gradient L2 norms

| Statistic | Gradient L2 |
|---|---:|
| Minimum | 4.07e-1 |
| 25th percentile | 1.37e10 |
| Median | 2.94e12 |
| 75th percentile | 1.75e14 |
| 90th percentile | 2.78e16 |
| 95th percentile | 7.25e16 |
| Maximum | 7.35e17 |

The median timestep-wise curve decreases by many orders of magnitude from early to recent historical observations.

### Finding

Input-gradient sensitivity increases substantially for earlier historical observations, consistent with severe gradient amplification during backward propagation through the recurrent computation.

This statement is intentionally limited to **input-gradient behaviour during the audit**. Training-time parameter gradients from the original run were not recorded.

---

# 21. Integrated Gradients Completeness Test

Integrated Gradients has a fundamental completeness relationship:

```text
sum(IG) ≈ F(x) - F(baseline)
```

A controlled single-example experiment kept the model, input, and baseline fixed.

```text
F(x)              ≈ 0.1125
F(baseline)       ≈ 0.0871
F(x)-F(baseline)  ≈ 0.0254
```

However:

```text
IG attribution sum        ≈ -1.98e16
Captum convergence delta  ≈ -1.98e16
normalized completeness error ≈ 7.77e17
```

### Finding

The attribution is numerically invalid for this case because it catastrophically violates the expected completeness relationship.

---

# 22. Original-Like cuDNN Reproduction

To test whether the failure was introduced only by the fallback implementation, the experiment was repeated under the condition closest to the original README-era Captum setup:

```text
model.train()
CUDA
cuDNN enabled
n_steps = 50
```

For one canonical example:

```text
F(x)                 ≈ 0.037474
F(baseline)          ≈ 0.147329
output difference    ≈ -0.109855
IG sum               ≈ -1.613e13
Captum delta         ≈ -1.613e13
max |IG|             ≈ 4.10e13
normalized error     ≈ 1.47e14
```

### Finding

The severe completeness failure is present under the original-like cuDNN condition as well.

Therefore, the failure cannot be dismissed as an artifact created only by disabling cuDNN.

---

# 23. 100-Sample Explainability Reliability Audit

A systematic audit was performed over 100 validation windows distributed across the validation period.

### Configuration

```text
model.train()
CUDA/cuDNN
training-mean baseline
target horizon = 1
IG n_steps = 50
```

### Normalized IG completeness error

| Statistic | Error |
|---|---:|
| Minimum | 6.74e12 |
| 25th percentile | 2.77e15 |
| Median | 1.80e16 |
| 75th percentile | 3.43e17 |
| 90th percentile | 5.47e18 |
| 95th percentile | 1.54e19 |
| Maximum | 1.31e20 |

Most importantly:

```text
100 / 100 samples > 0.01
100 / 100 samples > 0.1
100 / 100 samples > 1
100 / 100 samples > 10
100 / 100 samples > 100
100 / 100 samples > 1000
```

### Finding

The completeness failure is **systematic across the tested validation subset**, not a single pathological example.

Consequently, the original README's Integrated Gradients magnitudes and derived feature/timestep rankings are not currently treated as trustworthy evidence of feature importance.

---

# 24. Gradient Magnitude vs Attribution Failure

Across the 100 audited validation samples:

```text
Pearson correlation:
log10(input-gradient L2)
vs
log10(normalized IG completeness error)
≈ 0.518
```

```text
Spearman correlation:
raw gradient magnitude
vs
completeness error
≈ 0.507
```

### Finding

Larger input-gradient magnitudes are moderately associated with larger Integrated Gradients completeness errors.

This supports the hypothesis that gradient instability contributes to attribution failure, but the substantial scatter shows that gradient norm alone does not fully determine IG reliability.

No causal claim is made from this correlation.

---

# 25. Revised Interpretation of Explainability Results

The original README reported global and temporal Integrated Gradients rankings.

Following the reliability audit, those rankings are retained only as **historical experimental outputs**, not as current evidence.

### Currently unsupported

The project should **not** claim that the original IG rankings establish:

- which feature is globally most important,
- which historical timestep is most important,
- which physical variable causes power consumption,
- or that the attribution magnitudes are meaningful simply because they can be ranked.

### Still independently useful

Feature ablation remains informative because it directly tests predictive degradation under input removal and does not rely on the invalid IG values.

This distinction is important:

```text
Ablation = predictive sensitivity experiment
Integrated Gradients = gradient-based attribution method
```

Failure of the latter does not automatically invalidate the former.

---

# 26. LLM Explanation Layer

The project also includes an LLM-based explanation stage:

```text
Forecast model
      |
      v
Prediction
      |
      v
Explainability signal
      |
      v
Structured evidence
      |
      v
LLM explanation
```

The LLM itself is **not** the forecasting model.

It acts as an interpretation layer over structured model outputs.

The evaluation checks:

- feature agreement,
- directional agreement,
- numerical consistency,
- feature-level hallucination,
- causal overclaiming.

### Important reliability implication

Because the original Integrated Gradients outputs fail the later reliability audit, LLM explanations conditioned on those attribution values should not be interpreted as evidence of faithful model explanation.

They remain useful for studying **LLM grounding behaviour relative to the provided structured evidence**, but the trustworthiness of the upstream evidence must be considered separately.

This motivates a general pipeline principle:

```text
Validate explanation signal first
        ↓
Then evaluate language explanation
```

---

# 27. Robustness Evaluation

| Condition | MAE | RMSE | sMAPE % | MASE | MAE degradation % |
|---|---:|---:|---:|---:|---:|
| Clean | 0.570728 | 0.859323 | 61.0452 | 6.56532 | 0 |
| 1% missing | 0.560269 | 0.857600 | 59.5477 | 6.44501 | -1.83 |
| 5% missing | 0.561285 | 0.854170 | 59.9885 | 6.45670 | -1.65 |
| Gaussian noise | 0.564109 | 0.849427 | 60.3713 | 6.48919 | -1.16 |

The observed small improvements under perturbation should not automatically be interpreted as beneficial noise regularization. They may also reflect sample variation or interactions with the current model and preprocessing pipeline.

---

# 28. Computational Cost

| Device | Parameters | Benchmark batch size | Inference seconds | Samples / second | Training seconds | Peak GPU memory |
|---|---:|---:|---:|---:|---:|---:|
| CUDA | 22,174 | 1024 | 0.006378 | 160,546 | 5895.51 | 704.161 MB |

Computational cost is included because model selection should consider not only predictive accuracy but also complexity and deployment cost.

Given that persistence currently outperforms the LSTM, this comparison is especially relevant.

---

# 29. Main Research Findings

### Finding 1 — The simple baseline wins

Persistence outperforms the trained LSTM on aggregate forecasting error.

### Finding 2 — Forecast uncertainty grows with horizon

LSTM error increases as the prediction extends further into the future.

### Finding 3 — Peak events are substantially harder

Peak-consumption MAE is far larger than non-peak MAE, and persistence remains stronger during peaks.

### Finding 4 — Residual temporal structure remains

Lagged residual correlation shows that the forecasting model leaves predictable temporal structure unexplained.

### Finding 5 — Some features matter under ablation

Historical `Global_active_power` and `Global_intensity` produce the largest observed MAE increases when removed.

### Finding 6 — Input gradients are extremely large for many windows

The trained LSTM exhibits severe input-gradient sensitivity over a substantial fraction of the validation data.

### Finding 7 — Earlier timesteps are dramatically more gradient-sensitive

The median gradient-by-timestep curve spans many orders of magnitude from early to recent observations.

### Finding 8 — Backend choice changes numerical magnitude

cuDNN and fallback implementations produce similar gradient directions but materially different forward and gradient magnitudes.

### Finding 9 — Integrated Gradients fails completeness systematically

All 100 audited validation windows show catastrophic normalized completeness error under the original-like configuration.

### Finding 10 — Original IG rankings are not reliable evidence

The historical attribution magnitudes and feature/timestep rankings should not currently be used to support claims about model importance.

### Finding 11 — Explanation validation must precede language interpretation

An LLM explanation can only be as trustworthy as the structured evidence supplied to it. Upstream attribution reliability therefore becomes part of downstream explanation reliability.

---

# 30. What the Project Can Currently Claim

The following statements are supported by the completed experiments:

1. The trained model produces deterministic predictions under a fixed computational backend.
2. The normal forward method matches manual reconstruction from the saved LSTM and FC layers.
3. The LSTM does not outperform persistence in the current evaluation.
4. Peak demand and longer horizons are substantially harder forecasting regimes.
5. Residuals retain strong temporal dependence.
6. `Global_active_power` and `Global_intensity` materially affect performance in feature-ablation experiments.
7. The trained model exhibits extremely large input gradients for many validation windows.
8. Earlier historical observations systematically exhibit much larger gradient sensitivity than recent ones.
9. Severe input-gradient sensitivity occurs under both cuDNN and fallback implementations.
10. Exact forward and gradient magnitudes are backend-sensitive.
11. Integrated Gradients catastrophically violates completeness under the original-like configuration.
12. The IG failure occurs across all 100 audited validation windows.
13. The original IG-based feature/timestep rankings should not currently be treated as reliable evidence.
14. Feature ablation remains independently informative.

---

# 31. What Has Not Been Established

The project intentionally avoids stronger claims that are not supported by the experiments.

### Not established: "training gradients exploded"

Training-time parameter gradients from the original training run were not recorded.

### Not established: "Captum is broken"

The observed failure is specific to the interaction between this trained model, its gradient field, and the tested attribution configuration.

### Not established: "cuDNN caused the problem"

Severe gradient sensitivity and IG completeness failure occur under both cuDNN and fallback implementations.

### Not established: "fallback is the mathematically correct backend"

A numerical discrepancy exists, but its low-level cause has not been resolved.

### Not established: "IG proves physical causality"

Attribution is not causality.

### Not established: "the backend mechanism is fully solved"

Kernel-level numerical analysis is outside the current scope.

---

# 32. Threats to Validity

### Single dataset

Results are currently derived from one household-energy dataset and may not generalize to other demand profiles.

### Single trained LSTM instance

The explainability audit focuses on the saved trained model. Additional random seeds and retrained models are required to determine how often the same gradient pathology occurs.

### Limited architecture coverage

The current audit does not compare recurrent alternatives such as GRU, TCN, Transformers, or simpler autoregressive neural models.

### No original training-gradient logs

Because original training-time parameter gradients were not retained, the project cannot retroactively determine whether the training process itself exhibited exploding gradients.

### Backend discrepancy not fully resolved

cuDNN and fallback differences are empirically demonstrated but not explained at the kernel/numerical-analysis level.

### LLM explanation dependence

LLM grounding results depend on the reliability of the structured evidence passed to the language model.

---

# 33. Future Research

The next useful experiments are not simply larger models. They should directly test the mechanisms exposed by the current findings.

### 33.1 Retraining with gradient instrumentation

Record during training:

- parameter-gradient norms,
- clipping events,
- hidden-state statistics,
- cell-state statistics,
- per-layer activation distributions.

This would allow training instability to be tested directly rather than inferred retrospectively.

### 33.2 Gradient clipping comparison

Retrain controlled models with and without gradient clipping to test whether input-gradient behaviour and attribution reliability improve.

### 33.3 Multi-seed reliability

Repeat training across multiple seeds to determine whether the observed pathology is model-instance specific or reproducible.

### 33.4 Architecture comparison

Compare:

- LSTM,
- GRU,
- Temporal Convolutional Network,
- Transformer-based sequence model,
- strong persistence/autoregressive baselines.

The focus should include both forecasting accuracy and gradient/explanation reliability.

### 33.5 Attribution-method comparison

Only after establishing numerically stable models, compare:

- Integrated Gradients,
- GradientSHAP,
- DeepLIFT,
- occlusion,
- perturbation-based methods,
- feature ablation.

Each method should be checked against method-specific validity conditions before interpretation.

### 33.6 Explanation-aware evaluation

A stronger LLM study would separate:

1. predictive correctness,
2. attribution reliability,
3. structured-evidence quality,
4. language-model grounding.

This would create an end-to-end explanation reliability framework rather than evaluating only linguistic fluency.

---

# 34. Reproducibility and Experiment Tracking

The project uses:

- Python
- PyTorch
- Captum
- MLflow
- Docker
- Jupyter
- NVIDIA GPU acceleration where available

MLflow tracks experiments, parameters, metrics, and artifacts.

The repository separates reusable implementation code, notebooks, outputs, and infrastructure.

```text
project/
├── config/
│   └── config.yaml
│
├── data/
│
├── notebooks/
│   └── Scratch/
│       ├── Data_Preprocessing.ipynb
│       ├── Model_Training.ipynb
│       ├── Model_Evaluation.ipynb
│       ├── ExplainableAI.ipynb
│       ├── Explainability_Reliability_Audit.ipynb
│       └── LLM.ipynb
│
├── src/
│   └── Pytorch/
│       └── ...
│
├── evaluation_outputs/
│   └── ...
│
├── mlruns/
│   └── ...
│
├── mlflow.db
├── Dockerfile
├── startup.sh
└── requirements.txt
```

---

# 35. Reproducing the Forecasting Pipeline

```text
Raw Dataset
     ↓
Data_Preprocessing.ipynb
     ↓
Model_Training.ipynb
     ↓
Model_Evaluation.ipynb
     ↓
ExplainableAI.ipynb
     ↓
Explainability_Reliability_Audit.ipynb
     ↓
LLM.ipynb
     ↓
evaluation_outputs/
```

The historical explainability notebook is preserved intentionally. The reliability-audit notebook should be treated as the current source of truth regarding whether the original Integrated Gradients results are trustworthy.

---

# 36. Docker Environment

A containerized environment is recommended for reproducibility.

```bash
docker run --gpus all -it \
    -p 5000:5000 \
    -p 8080:8080 \
    -v .:/workspace \
    --name Scratch_learn \
    base_workspace:pytorch-v.1
```

Enter the container:

```bash
docker exec -it Scratch_learn bash
```

Start MLflow:

```bash
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./artifacts \
    --host 0.0.0.0 \
    --port 5000
```

Jupyter and MLflow can then be used together for experimentation and run tracking.

---

# 37. Conclusion

The forecasting experiments show that a complex model can underperform a trivial baseline. The explainability audit shows that an attribution method can return highly structured-looking values while violating its own numerical consistency requirement by many orders of magnitude.

For this reason, the project treats **negative results, numerical diagnostics, failed assumptions, and methodological corrections as research outputs rather than implementation defects to hide**.

---

# References

1. Hebrail, G. & Berard, A. (2006). *Individual Household Electric Power Consumption*. UCI Machine Learning Repository. https://doi.org/10.24432/C58K54
2. Sundararajan, M., Taly, A., & Yan, Q. (2017). *Axiomatic Attribution for Deep Networks*. Proceedings of the 34th International Conference on Machine Learning.
3. PyTorch documentation — LSTM and autograd.
4. Captum documentation — Integrated Gradients.

