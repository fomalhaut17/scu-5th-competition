# [Gemini] Structural Breakthrough Plan: Beyond the Incremental Wall

## 1. The Problem: The "Incremental Wall"
Current efforts (blending weights, adding more GBDT models, minor feature tweaks) are all **additive**. In a highly optimized or synthetic environment, the marginal gain of adding another similar model (e.g., 4-way vs 3-way) approaches zero or becomes negative due to overfitting on the small OOT set. 

To jump 20+ points, we must move from **Additive Modeling** to **Structural/Functional Modeling**.

---

## 2. Three Strategic Axes for a Structural Jump

### Axis 1: Hybrid Functional Modeling (DGP Reverse Engineering)
**The Insight**: Synthetic data is almost certainly generated via a multiplicative process: $Price = 	ext{Base} 	imes 	ext{Area}^{\alpha} 	imes 	ext{Floor}^{\beta} \dots$. 
GBDTs approximate this with step functions (piecewise constant), which is inherently inefficient for smooth, multiplicative relations.

**The Breakthrough**: Implement a **Log-Linear Functional Skeleton**.
1.  **The Model**: A highly regularized Ridge regression on $\log(	ext{Target})$ using $\log(	ext{Features})$ (e.g., $\log(	ext{Area})$, $\log(	ext{YearBuilt})$).
2.  **The Role**: This is not a "model" to be blended, but a **"Structural Prior"**.
3.  **The Implementation**: 
    *   $\hat{y}_{	ext{base}} = \exp(	ext{Ridge}(\log(X)))$
    *   $\hat{y}_{	ext{residual}} = 	ext{GBDT}(X, 	ext{Target} - \hat{y}_{	ext{base}})$
    *   **Final Prediction**: $\hat{y}_{	ext{base}} + \hat{y}_{	ext{residual}}$ (or a weighted version in log-space).
4.  **Why it jumps**: It forces the model to capture the global "physics" of the data, leaving the GBDT to focus solely on the local non-linearities/residuals.

### Axis 2: Error-Aware Segmented Stacking (RMSE Optimization)
**The Insight**: The error is not uniformly distributed. The error analysis shows a heavy concentration in **Large Area ($\ge 120	ext{m}^2$)** and **High Price ($\ge 7	ext{억}$)** segments. A global Ridge meta-learner treats a 5% error on a 3억 apartment the same as a 5% error on a 15억 apartment, but RMSE penalizes the latter much more heavily.

**The Breakthrough**: **Segmented Meta-Learning**.
1.  **The Strategy**: Instead of one global Ridge stack, we use **Segment-Specific Meta-Weights**.
2.  **The Implementation**:
    *   Define segments: `High_Value` (Price/Area criteria) and `Standard`.
    *   Train two separate Ridge meta-learners on the OOF predictions: one optimized for the `High_Value` segment and one for `Standard`.
    *   Apply the corresponding weight set to the test predictions based on the test sample's features.
3.  **Why it jumps**: It directly optimizes the RMSE metric by forcing the ensemble to prioritize accuracy in the high-magnitude error zones.

### Axis 3: Distribution-Shift Adaptive Calibration (Temporal Robustness)
**The Insight**: The shift from 2025 (Train) to 2026 (Test) is likely more than just a linear trend. There might be a change in the *distribution* of features (e.g., more large apartments in 2026).

**The Breakthrough**: **Feature-Density Based Trend Calibration**.
1.  **The Strategy**: Replace the current `Gu-average` trend correction (which is static) with a **Dynamic Calibration Factor**.
2.  **The Implementation**:
    *   Calculate the density ratio $	ext{Ratio}(f) = \frac{P_{	ext{Test}}(f)}{P_{	ext{Train}}(f)}$ for key features (e.g., `Area`, `Year_Built`).
    *   Adjust the Gu-trend $\alpha$ per segment based on these ratios. If 2026 has a higher concentration of large apartments, the "Large Apartment Trend" gets more weight.
3.  **Why it jumps**: It addresses the "Out-of-Distribution" (OOD) problem head-on, rather than trying to "average out" the shift.

---

## 3. Execution Roadmap & Priority

| Priority | Task | Type | Expected Impact | Risk |
| :--- | :--- | :--- | :--- | :--- |
| **1 (Critical)** | **Segmented Stacking** | Implementation | High (Direct RMSE target) | Low (Uses existing models) |
| **2 (High)** | **Hybrid Functional Model** | Structural | Very High (Captures DGP) | Medium (Requires careful log-space handling) |
| **3 (Medium)** | **Adaptive Calibration** | Robustness | Medium (Temporal stability) | Medium (Requires density estimation) |

## 4. Validation Mandate
For all new structures, **OOT must be used to verify "Directional Correctness"**. 
* If a new structure improves OOF but fails to improve the OOT (specifically in the high-error segments), **it is discarded immediately**.
* We will not fall into the "4-way blending" trap of over-parameterizing on a small OOT. We only accept structures that show a **significant, consistent shift in the error distribution** in OOT.
