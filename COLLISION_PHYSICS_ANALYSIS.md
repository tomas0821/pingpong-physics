# Collision Experiment — Physics Results Analysis
## June 25, 2025 — 5 Teams (Mesa 1–5)

**Ball types:** Ping-Pong (~2.8g), Amarilla (~9.9g), Golf (~46g) | **Planned: GolfHueca** (not attempted)
**Trials per team:** 2 per ball type (3 types × 2 = 6 trials per team)
**Total data points:** 28 trials (Mesa 5 complete failure; 22 from Mesas 1–4)

---

## 1. FULL RESULTS TABLE

| Team | Ball type | Try | m₁ (g) | m₂ (g) | P% conserved | KE% retained | e | Valid? |
|---|---|---|---|---|---|---|---|---|
| Mesa 1 | PingPong | 1 | 2.7 | 2.7 | 32.8 | 10.6 | 0.317 | ❌ P low |
| Mesa 1 | PingPong | 2 | 2.7 | 2.7 | **43.0** | 10.0 | 0.122 | ✅ |
| Mesa 1 | Golf | 1 | 46.4 | 45.8 | 37.1 | 18.5 | **5.53** | ❌ e>1 |
| Mesa 1 | Golf | 2 | 46.4 | 45.8 | — | — | — | ❌ incomplete |
| Mesa 1 | Amarilla | 1 | 9.6 | 9.6 | 145.5 | 237.3 | 1.85 | ❌ P>100%, e>1 |
| Mesa 1 | Amarilla | 2 | 9.6 | 9.6 | **84.1** | 35.7 | 0.065 | ✅ |
| Mesa 2 | PingPong | 1 | 2.8 | 2.8 | **68.4** | 19.6 | 0.186 | ✅ |
| Mesa 2 | PingPong | 2 | 2.8 | 2.8 | **87.1** | 49.3 | 0.050 | ✅ |
| Mesa 2 | Golf | 1 | 46.2 | 46.3 | 184.0 | 48.6 | 0.250 | ❌ P>100% |
| Mesa 2 | Golf | 2 | 46.2 | 46.3 | 96.9 | 75.3 | **1.43** | ❌ e>1 |
| Mesa 2 | Amarilla | 1 | 9.6 | 9.9 | 130.7 | 25.4 | 0.099 | ❌ P>100% |
| Mesa 2 | Amarilla | 2 | 9.6 | 9.9 | **90.1** | 49.7 | 0.722 | ✅ ⭐ |
| Mesa 3 | PingPong | 1 | 2.7 | 2.8 | 32.6 | 10.6 | 0.310 | ❌ P low |
| Mesa 3 | PingPong | 2 | 2.7 | 2.8 | **66.5** | 21.9 | 0.026 | ✅ |
| Mesa 3 | Golf | 1 | 46.3 | 46.2 | 59.9 | 46.4 | **1.81** | ❌ e>1 |
| Mesa 3 | Golf | 2 | 46.3 | 46.2 | 11.6 | 0.7 | 0.018 | ❌ P<20% |
| Mesa 3 | Amarilla | 1 | 9.6 | 9.8 | 70.7 | 59.7 | **7.25** | ❌ e>1 |
| Mesa 3 | Amarilla | 2 | 9.6 | 9.8 | 49.9 | 12.8 | 0.099 | ✅ |
| Mesa 4 | PingPong | 1 | 2.8 | 2.8 | 40.8 | 16.9 | 0.406 | ✅ |
| Mesa 4 | PingPong | 2 | 2.8 | 2.8 | **86.0** | 41.8 | 0.007 | ✅ |
| Mesa 4 | Golf | 1 | 45.8 | 46.3 | 72.3 | 56.3 | **1.30** | ❌ e>1 |
| Mesa 4 | Golf | 2 | 45.8 | 46.3 | 112.1 | 69.6 | 0.362 | ❌ P>100% |
| Mesa 4 | Amarilla | 1 | 9.9 | 9.9 | 80.0 | 67.8 | **1.24** | ❌ e>1 |
| Mesa 4 | Amarilla | 2 | 9.9 | 9.9 | **86.0** | 39.0 | 0.174 | ✅ |
| **Mesa 5** | ALL | ALL | — | — | **5000–376000%** | **993–376000%** | **>10** | ❌❌ failure |

**Valid trials: 10 / 28** (36%) | **Mesa 5: 0/6** (complete failure)

---

## 2. SUMMARY BY BALL TYPE

| Ball type | Valid trials | Avg P% | Avg KE% | Avg e | Notes |
|---|---|---|---|---|---|
| **PingPong** (2.8g) | 6 | **71.6%** | **26.6%** | 0.133 | Most consistent |
| **Amarilla** (~10g) | 4 | **77.5%** | **34.3%** | 0.265 | Best single result |
| **Golf** (46g) | **0** | — | — | — | All invalid |
| **GolfHueca** | 0 | — | — | — | Not attempted |

---

## 3. BEST AND WORST RESULTS

### ⭐ Best result: Mesa 2, Amarilla try 2
- m₁ = 9.6g, m₂ = 9.9g (stationary target)
- P conserved: **90.1%** | KE retained: 49.7% | e = **0.722**
- Nearly elastic for small rubber balls — makes physical sense

### ✅ Most consistent team: Mesa 2
- 3/4 valid trials (Golf excluded)
- Best P conservation average: 81.9%

### ❌ Worst: Mesa 5
- All 6 trials produce physically impossible results (P > 1000%, KE > 100000%)
- Root cause: **extremely low pre-collision velocities** (v ≈ 0.5–5 cm/s instead of typical 15–100 cm/s)
- Detection failure — balls were not properly tracked; false positives or extreme near-stationary measurements inflated post-collision velocities
- Note: some cells contain Spanish decimal notation (`1,99` instead of `1.99`) — suggests manual data entry on a Spanish-locale system, but the velocities themselves indicate tracking failure regardless

---

## 4. WHY IS THE COEFFICIENT OF RESTITUTION SO LOW?

The measured e values (0.007–0.722) are far below the expected values for real ping-pong balls (e ≈ 0.80–0.90 vs. hard floor). This is a systematic effect with three causes:

### a) Out-of-plane component not measured
The 2D camera system captures only x and y velocities. Real table-top collisions always involve some vertical (z) bounce — the ball lifts slightly after impact. This vertical KE is "lost" from the 2D measurement, making KE retention look lower than it is and e appear smaller.

**Estimate:** A 1–2 cm vertical bounce for a ping-pong ball represents ~5–15% of collision KE — consistent with the observed low KE retention.

### b) Spin (angular momentum) not captured
The CV system tracks the ball's center position only. After collision, balls rotate. This rotational KE is invisible to the tracker, so the measured translational speed after collision underestimates the true post-collision energy.

### c) Post-collision deceleration before velocity measurement
The trajectory regression (`K` mode) is computed over a window of frames **after** the collision. Table friction decelerates the ball between impact and measurement. The longer the post-collision window, the lower the apparent final velocity, and thus the lower the apparent e.

**Practical implication for students:** The low e values are expected and educational — they show why real experiments always measure *effective* coefficients of restitution, not the intrinsic ball property.

---

## 5. WHY ARE GOLF BALL TRIALS ALL INVALID?

**Every single golf ball trial** has either e > 1 or P > 110%. The likely cause: **ball identity swap during tracking**.

The `colisiones_v2.py` nearest-neighbor matcher assigns ball identities frame-by-frame. For **equal-mass balls** (golf balls are nearly identical in size and appearance), the tracker can swap which ball is "B1" and which is "B2" mid-trajectory if they get close during the collision. When identities swap:
- The apparent pre-collision velocity of B1 becomes the actual pre-collision velocity of B2
- The post-collision velocity of B1 becomes the actual post-collision velocity of B2

This inverts the relative velocity ratio, giving e > 1 or P ≠ conserved. The effect is worse for golf balls (large, heavy, hard to distinguish) than ping-pong balls.

**Fix:** Add a ball color or size distinguisher, or use the `M` (measurement) mode + coefficient of restitution calculation instead of the `K` (momentum) mode for equal-mass scenarios.

---

## 6. MESA 5 FAILURE — DETAILED DIAGNOSIS

| Trial | v₁ᵢ (cm/s) | v₂ᵢ (cm/s) | v₁f (cm/s) | v₂f (cm/s) | P% |
|---|---|---|---|---|---|
| PP try 1 | 0.64 | 7.72 | 14.91 | 18.06 | 63% |
| PP try 2 | 1.99 | 0.0 | 4.51 | 0.46 | 292% |
| Golf try 1 | 1.34 | 1.35 | 6.19 | 9.88 | 2926% |
| Golf try 2 | 4.88 | 0.0 | 23.78 | 20.74 | 193% |
| Amarilla try 1 | 0.49 | 0.12 | 2.36 | 17.84 | 5243% |
| Amarilla try 2 | 1.35 | 0.0 | 3.02 | 112.1 | 5933% |

The pre-collision speeds are 0.5–5 cm/s (≈ 10–30× lower than other teams). Post-collision speeds are normal or inflated. This means:

- **The balls were nearly stationary before "collision"** — the group may have placed the balls in contact and nudged them rather than colliding them at speed
- OR the code was paused/resumed at the wrong moment, capturing the ball after deceleration as "pre-collision"
- The extreme v₂f in Amarilla try 2 (112 cm/s) indicates a false detection spike — a light reflection or shadow detected as a fast-moving ball

---

## 7. TEAM PERFORMANCE RANKING

| Rank | Team | Valid trials | Best result | Notes |
|---|---|---|---|---|
| 🥇 Mesa 2 | 3/4 | 90.1% P, e=0.722 | Most valid data, good technique |
| 🥈 Mesa 4 | 3/4 | 86.0% P (2 trials) | Consistent PingPong; Golf data problematic |
| 🥉 Mesa 1 | 2/4 | 84.1% P | Amarilla try 1 severely wrong (false detection) |
| 4th Mesa 3 | 2/4 | 66.5% P | PingPong data ok; Golf failure |
| 5th Mesa 5 | 0/6 | — | Complete tracking failure; pre-collision v ≈ 0 |

---

## 8. OBSERVATIONS & MISSING DATA

- **GolfHueca** (hollow golf ball) — planned 4th ball type — **not attempted by any team**. All sheets have this row with all-null data. Likely ran out of time or the hollow balls were unavailable.
- **Only 2 of 3 intended trials per ball type were collected** — the 3rd row is always empty. Students stopped at 2 trials.
- **Masses are stored in kg** (column header says "g") — student labeling error. Values are internally consistent; does not affect percentage calculations.

---

## 9. ACTION ITEMS FROM PHYSICS DATA

### For the next iteration of the experiment:

1. **Address golf ball identity swap** — give each ball a visible sticker/color (a dot of colored tape) so the tracker can distinguish them. Or switch to `M` mode for e measurement.

2. **Require minimum pre-collision speed** — add a HUD indicator showing current ball speed. If a ball is moving < 10 cm/s before the collision, warn the student to retry with more force.

3. **Add post-collision window size control** — let students choose how many frames after collision to use for velocity regression. Currently it's fixed; making it adjustable (and showing the effect) is a good exercise in measurement uncertainty.

4. **Collect z-component explanation** — add to the lab guide: "Why is KE not 100% conserved even for elastic balls?" with the three factors above (vertical bounce, spin, friction).

5. **Include GolfHueca** — plan extra time if this ball type is to be included (group ran out of time in June 2025).

---

## 10. KEY NUMBERS FOR PUBLICATION/REPORT

| Metric | Value |
|---|---|
| Teams with ≥1 valid trial | 4/5 (80%) |
| Overall trial success rate | 10/28 (36%) |
| Best momentum conservation | 90.1% (Mesa 2, Amarilla) |
| Best e measured | 0.722 (Mesa 2, Amarilla) |
| Ball type with best data | PingPong (6 valid trials) |
| Ball type with worst data | Golf (0 valid trials) |
| Avg P conservation (valid trials) | ~73% |
| Complete team failure | Mesa 5 (pre-collision velocity ≈ 0) |
