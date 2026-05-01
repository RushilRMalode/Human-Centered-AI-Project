# Corrections Log

All changes made to `latex_report/main.tex`, `writeup/report.md`, and `writeup/theorem.md`
after cross-checking every claimed number against the saved `.npz` experiment results.

---

## 1. Abstract — Exp 1 claim (all three files)

**Was:** "every parameter's MLE error decays as 1/√N"  
**Now:** "decays toward zero with N, consistent with 1/√N scaling"  
**Reason:** The abstract implied strict pointwise 1/√N scaling; the data shows an overall downward trend but with noise-driven reversals at a few N values (only 4 seeds). The weaker phrasing is accurate.

---

## 2. Abstract — Exp 3 unmarked plateau value (all three files)

**Was:** "the unmarked model plateaus at error ~0.32"  
**Now:** "plateaus at error ~0.22–0.26"  
**Reason:** Actual mean unmarked α-errors in Exp 3 are 0.259, 0.205, 0.253, 0.235, 0.197 — the claimed value of 0.32 is from Exp 2 (β-collapse) and was incorrectly carried over. The true range is 0.20–0.26.

---

## 3. Abstract — Exp 4 z-accuracy (all three files)

**Was:** "z-label classification accuracy decays smoothly from 98% to chance"  
**Now:** "decays from 98.1% at full separation toward chance at s=0"  
**Reason:** The abstract had the direction backwards (it said "from 98% to chance" when the natural reading is increasing s → higher accuracy). Also, the s=0 value is 54.7%, not exactly chance (50%), so "toward chance" is more precise.

---

## 4. Exp 1 caption — "monotonically" (main.tex, report.md)

**Was:** "Every parameter family decays *monotonically* to zero."  
**Now:** "Every parameter family shows an overall downward trend consistent with MLE consistency. Due to the small number of seeds (n=4), point-by-point monotonicity is not guaranteed…"  
**Reason:** The data shows non-monotone steps for multiple parameters (e.g., α error goes from 0.013 at N=794 to 0.030 at N=1630; nb_mean similarly reverses). With 4 seeds the seed variance is high enough to cause this. The word "monotonically" is factually wrong.

---

## 5. Exp 2 caption — unmarked starting value and marked range (main.tex, report.md)

**Was:** "climbs monotonically from ~0.08 at large β-gap … Marked model's error stays flat at 0.03–0.06"  
**Now:** "rises from ~0.06 at large β-gap … Marked model's error remains low (mean 0.03–0.08); at large β-gap both models perform similarly"  
**Reason (a):** At β₂=0.1 the actual unmarked mean is 0.061, not 0.08.  
**Reason (b):** The marked error at β₂=0.1 is 0.082, above the claimed ceiling of 0.06. The marked model is actually slightly *worse* than the unmarked at this setting (0.082 vs 0.061), which is physically expected when timing already works well.  
**Reason (c):** "Monotonically" is wrong — the unmarked curve dips from 0.339 at β₂=0.90 to 0.285 at β₂=0.95 before rising again to 0.305 at β₂=1.0.

---

## 6. Exp 3 caption — unmarked plateau value and N range (main.tex, report.md)

**Was:** "plateaus at ~0.30–0.40 across N ∈ [38, 745]"  
**Now:** "remains elevated at ~0.20–0.26 across mean N ∈ [38, 768]"  
**Reason (a):** Actual unmarked errors range 0.197–0.259. The value 0.30–0.40 is a significant overstatement.  
**Reason (b):** N range: the individual-seed max is 833, mean N max is 768. The reported [38, 745] mixes individual-seed minimum with a rounded mean maximum; [38, 768] (mean N range) is cleaner.

---

## 7. Exp 4 caption — z-accuracy starting value and "smooth" claim (main.tex, report.md)

**Was:** "Rises smoothly from 52.6% (chance) at s=0 to 98.1% at s=1"  
**Now:** "Rises from near-chance (54.7%, vs. theoretical 50%) at s=0 to 98.1% at s=1 … (Small positive offset at s=0 is sampling noise with 4 seeds.)"  
**Reason (a):** The actual s=0 value is 54.7%, not 52.6%.  
**Reason (b):** 52.6% (or 54.7%) is not "chance" — theoretical chance is 50%. Labelling it "chance" is misleading; it should be "near-chance" with sampling noise acknowledged.  
**Reason (c):** The curve is not smooth/monotone — it dips at s=0.1 (53.5%) below s=0 (54.7%) before rising. "Smooth" removed.

---

## 8. Summary table — Exp 4 row (main.tex, report.md)

**Was:** "z-acc ramps from 52.6% to 98.1%"  
**Now:** "z-acc ramps from ~55% (chance) to 98.1%"  
**Reason:** Corrects the 52.6% error and acknowledges it is only approximately chance.

---

## 9. Discussion — marks vs. timing tradeoff (main.tex, report.md)

**Added new paragraph:** "When do marks help vs. hurt?"  
**Reason:** The original text implied marks are always better. Exp 2 shows the marked model has slightly *higher* error than the unmarked at β₂=0.1 (mean 0.082 vs 0.061). A correct discussion acknowledges this: marks add estimation variance and help most when the timing channel is degraded. This is theoretically expected and should be stated explicitly.

---

## 10. Proof Step 3 — rigour gap (main.tex, theorem.md)

**Was:** Informal statement "the conditional label probability tends to α^j/(α^1+α^2)" with no justification for why the limit is well-defined or why the ergodic theorem applies to the subsequence.  
**Now:** Added explanation that in the β₁=β₂ case the ratio is exactly α^j/(α^1+α^2) regardless of history (key algebraic fact), noted that short-gap events exist a.s. because μ>0, and added a Remark explicitly flagging that a fully rigorous treatment would invoke the conditional ergodic theorem and is omitted.  
**Reason:** The original proof could mislead a reader into thinking the subsequence argument is rigorous when it is an informal sketch for the β₁=β₂ case.

---

## 11. Consistency Corollary — honest about marked extension (main.tex, theorem.md, report.md)

**Was:** Stated the Gao et al. (2018) ergodicity result applies "via the marked-TPP martingale framework" as if this were an established fact.  
**Now:** Added Remark noting that Gao et al. (2018) prove ergodicity for the *unmarked* case; the extension to marks is expected but is not proved in that paper — it is treated as an assumption.  
**Reason:** The original phrasing attributed a proved result to Gao et al. that they did not prove. This is an honesty/attribution issue.

---

## 12. NegBin parameterisation (main.tex, theorem.md)

**Was:** NegBin(λ_j, r_j) used without defining the parameterisation.  
**Now:** Added: "We use the NB2 parameterisation: NegBin(λ_j, r_j) denotes mean λ_j and dispersion r_j (variance = λ_j + λ_j²/r_j)."  
**Reason:** NegBin has several common parameterisations. The code uses NB2 (mean + dispersion); without stating this the reader cannot verify the log-PMF formula in marks.py.

---

## Files changed

| File | Changes |
|---|---|
| `latex_report/main.tex` | Items 1–9, 10 (Remark added), 11 (Remark added), 12 |
| `writeup/report.md` | Items 1–9, 11, 12 |
| `writeup/theorem.md` | Items 10, 11, 12 |
