# Actionable Urban Heat Mitigation from High-Resolution LST

## Idea Document

## 1. Objective

The goal is to extract **actionable urban cooling insights** from the generated **daily 30 m Land Surface Temperature (LST)** dataset.
Rather than only analyzing temperature patterns, the system aims to **identify persistent heat hotspots and simulate realistic interventions** (such as albedo modification or vegetation addition) to estimate potential temperature reduction.

The final outcome should produce **policy-ready spatial recommendations** indicating where specific interventions would yield the highest cooling impact.

---

# 2. Core Problem

Urban heat islands are not static phenomena. Daily LST maps contain:

* short-term heat spikes
* seasonal heat patterns
* structural urban heat accumulation

Many hot pixels appear only temporarily. Acting on those would produce misleading insights.

Therefore the system must first **identify persistent heat hotspots**, meaning areas that are **consistently hotter than their surroundings over long periods**.

These persistent hotspots are the correct targets for intervention simulation.

---

# 3. Overall Pipeline

```
Daily High-Resolution LST (30 m)
            │
            ▼
City Baseline Temperature Estimation
            │
            ▼
Daily Heat Anomaly Detection
            │
            ▼
Hotspot Persistence Analysis
            │
            ▼
Spatial Hotspot Region Extraction
            │
            ▼
Intervention Scenario Simulation
            │
            ▼
Cooling Impact Estimation
            │
            ▼
Actionable Urban Cooling Insights
```

---

# 4. Step 1 – Establish Daily City Baseline

For each day, a **city-wide reference temperature** must be computed.

Using a robust estimator prevents extreme hotspots from biasing the baseline.

Example:

```
baseline_t = median(LST_t)
spread_t = MAD(LST_t)
```

This baseline represents the **background urban thermal state** for that day.

---

# 5. Step 2 – Daily Heat Anomaly Detection

For each pixel:

```
anomaly = LST_pixel - baseline_day
```

A pixel is considered **hot** if it exceeds a defined threshold:

Examples:

• Z-score threshold
• Top percentile (e.g., top 10% hottest pixels)

This produces a **binary daily hotspot map**.

---

# 6. Step 3 – Persistent Hotspot Identification

A hotspot must occur **frequently across time** to be considered meaningful.

For each pixel:

```
Hotspot Frequency = (# days pixel is hotspot) / (# valid observations)
```

This identifies areas that repeatedly experience excessive heat.

Typical interpretation:

| Frequency | Interpretation  |
| --------- | --------------- |
| <10%      | occasional heat |
| 10–30%    | semi-persistent |
| >30%      | chronic hotspot |

Only **chronic hotspots** are retained for intervention analysis.

---

# 7. Step 4 – Heat Intensity Measurement

Frequency alone is insufficient.

We also measure **how much hotter the hotspot is** when it occurs.

Example metric:

```
Mean Excess Heat = average(LST - baseline) during hotspot days
```

This allows ranking hotspots by **severity**.

Combined hotspot score:

```
HotspotScore = Frequency × MeanExcessHeat
```

---

# 8. Step 5 – Spatial Region Formation

Hotspots occur as clusters rather than isolated pixels.

The binary hotspot map is therefore converted into **connected spatial regions**.

Each region will have:

* area
* mean temperature excess
* hotspot frequency
* hotspot score

Small isolated regions are removed to reduce noise.

This produces **interpretable urban heat zones**.

---

# 9. Step 6 – Intervention Scenario Simulation

Each hotspot region becomes a **candidate intervention zone**.

Possible interventions include:

### Albedo Increase

Simulate reflective roofs or pavements.

Example modification:

```
albedo_new = albedo_original + 0.05
```

### Urban Greening

Introduce vegetation or tree canopy.

### Surface Material Change

Replace asphalt or dark surfaces.

---

# 10. Step 7 – Counterfactual Temperature Prediction

After modifying surface variables (such as albedo), the trained LST model is run again to estimate the new temperature.

Cooling effect is computed as:

```
Cooling = LST_original − LST_modified
```

Key metrics:

* mean cooling
* maximum cooling
* reduction in hotspot frequency

---

# 11. Step 8 – Intervention Ranking

Each hotspot region is evaluated based on:

* cooling potential
* hotspot severity
* spatial extent
* intervention feasibility

This produces a ranked list of locations where mitigation strategies would produce the greatest benefit.

---

# 12. Final Outputs

The system produces the following artifacts:

### Heat Persistence Maps

* hotspot frequency map
* hotspot intensity map

### Hotspot Regions

Spatial polygons representing chronic heat zones.

### Intervention Impact Maps

Predicted cooling for simulated interventions.

### Actionable Insight Dataset

Structured table listing:

| Region | Area | Hotspot Score | Intervention | Expected Cooling |

---

# 13. Expected Insights

Examples of outputs include:

* identification of **major heat-retaining zones**
* quantification of **cool roof potential**
* detection of **high-impact greening zones**
* ranking of **urban heat mitigation opportunities**

---

# 14. Scientific Value

This framework moves beyond simple UHI mapping by:

* using **temporal persistence analysis**
* applying **counterfactual intervention simulation**
* producing **decision-support outputs**

It transforms high-resolution LST data into **practical urban climate strategies**.

---

# 15. Future Extensions

Possible improvements include:

* population exposure analysis
* building-level intervention simulation
* optimization-based cooling planning
* integration with urban planning datasets

---

# Summary

This approach uses **daily high-resolution LST data** to detect persistent urban heat hotspots and evaluate the potential impact of cooling interventions through simulation. The resulting insights can guide urban planning decisions aimed at mitigating urban heat island effects.
W