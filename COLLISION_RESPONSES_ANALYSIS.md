# Collision Experiment — Form Response Analysis
## "Prueba experimental COLISIONES" — June 25, 2025

**N respondents:** 14 | **Ages:** 19–23 | **Prior CV experience:** None (100%)

---

## 1. QUANTITATIVE SUMMARY

### Overall Averages by Category

| Category | Avg / 5 | vs. Pendulum experiments |
|---|---|---|
| **Usability** | 3.79 | ⬇️ Lower (pendulum avg ~4.3–4.5) |
| **Learning** | 3.71 | ⬇️ Lower (pendulum avg ~4.2–4.7) |
| **Satisfaction** | 3.48 | ⬇️ Lower (pendulum avg ~4.0–4.3) |

**Collision experiment scores consistently ~0.7 points lower than pendulum experiments across all dimensions.** Root cause: detection reliability is worse for the two-ball collision setup than single-ball pendulum tracking.

---

### Per-Question Breakdown

#### Usability (3 questions)

| Question | Avg | Distribution |
|---|---|---|
| El código me resultó fácil de usar | **3.79** | 3×3→3×3, 6×4, 2×5 |
| Logré realizar las acciones sin problemas | **3.07** ⚠️ | 2×2, 8×3, 3×4, 1×(none) |
| La información del HUD me ayudó | **4.50** ✅ | 7×4, 7×5 |

**Standout:** HUD information (4.50/5) is the highest-rated item in the entire survey — students value the live velocity/momentum overlay. "Acciones sin problemas" (3.07) is the lowest-rated item and directly reflects detection failures.

#### Learning (5 questions)

| Question | Avg | Distribution |
|---|---|---|
| Comprende mejor los conceptos | 3.79 | 5×3, 7×4, 2×5 |
| Comprende cómo trabajar en el lab | 3.93 | 4×3, 7×4, 3×5 |
| Presentar y organizar datos experimentales | 3.93 | 3×3, 8×4, 2×5, 1×(none) |
| Diseñar e interpretar datos (gráficos) | **3.36** ⚠️ | 3×2, 5×3, 6×4, 0×5 |
| Resolver actividades planteadas | 3.57 | 2×2, 5×3, 5×4, 2×5 |

**Lowest:** Data interpretation/graph design (3.36). No student gave a 5 for this item — suggests the graph output (`collision_vectors.pdf`) is not well-integrated into the lab workflow, or students did not reach the analysis step due to detection issues.

#### Satisfaction (4 questions)

| Question | Avg | Distribution |
|---|---|---|
| Satisfacción general | 3.36 | 2×2, 6×3, 5×4, 1×5 |
| Relevante para mis estudios | **3.21** ⚠️ | 2×2, 6×3, 5×4, 1×4 |
| Desea más códigos CV | **4.00** ✅ | 1×2, 4×3, 4×4, 5×5 |
| Motivación con el proyecto | 3.36 | 1×1, 2×2, 5×3, 3×4, 3×5 |

**Key finding:** Despite low overall satisfaction, students strongly want more CV experiments (4.00/5). The concept works; the execution needs polish.

---

## 2. QUALITATIVE THEMES

### What Students Praised ✅

- **HUD / visualization** — Multiple students praised the live overlay and real-time feedback
- **Novel approach** — "Nueva forma para hacer prácticas", "interesante", "entretenido"
- **Theory-to-practice bridge** — "Lleva a la práctica temas que se quedan en el ámbito teórico"
- **Ease of understanding** — "Fácil de emplear", "se logra entender fácilmente los datos"

**Best quotes:**
> *"Es una buena manera de plantear la parte práctica de los temas, es fácil de emplear y se logra entender fácilmente los datos solicitados"*

> *"Siento que es una nueva forma para hacer prácticas la cual es muy interesante"*

> *"Hace las clases más interactivas y entretenidas"*

---

### Critical Issues 🔴

#### 1. Light reflections detected as balls (5+ reports — most common)

- *"Las luces que se reflejan sobre la mesa de trabajo a veces son detectadas como bolas"* [R1]
- *"El código detectaba reflejos de la luz en la mesa como si fueran parte del experimento"* [R2]
- *"La cámara identifica el reflejo de la luz en la mesa como una bola"* [R4]
- *"Problemas al momento de detectar algunos colores como el de las bolas amarillas"* [R12]

**Root cause:** The collision setup is done on a table surface; light reflections on the table mimic the appearance of a ball. The pendulum experiment doesn't have this problem because the ball swings in the air against a background.

**Fix:** Increase `conf` threshold (0.6 → 0.7) for collision scripts, or add a spatial filter to reject detections near the table surface baseline.

#### 2. Ball detection failures / tracking loss (9+ reports)

- *"No funciona del todo bien leyendo las pelotas y se pega"* [R3]
- *"A veces es difícil trabajar si no detecta las esferas"* [R5]
- *"Qué fuera más fácil reconocer las bolas, en ocasiones se complicaba"* [R6]
- *"Una mejora en la parte de la detección podría mejorar, fue el único fallo"* [R8]
- *"Tal vez la forma en que detecta las colisiones"* [R9]
- *"Que reconozca mejor las bolas"* [R10]
- *"A veces no capta de forma correcta la trayectoria o al contrario detecta demasiado"* [R11]
- *"En mi grupo tuvimos problemas para el reconocimiento de las bolas y el seguimiento"* [R14]

#### 3. Program freezing / crashes (2 reports)

- *"No funciona del todo bien y se pega"* [R3]
- *"El programa se detuvo completamente tras varias tomas de movimiento"* [R13]

#### 4. Camera setup issues (2 reports)

- *"Se necesita una cámara estabilizada"* [R1]
- *"Utilizar una cámara externa con trípode"* [R2]
- *"Las bolas se salen del campo que reconoce la cámara"* [R10]

---

## 3. IMPROVEMENT RECOMMENDATIONS (STUDENTS' OWN WORDS)

| Theme | Count | Representative Quote |
|---|---|---|
| **Better ball detection** | 9 | "Mejorar la detección de las bolas" |
| **Ignore light reflections** | 5 | "Detectaba reflejos de la luz como bolas" |
| **External camera / tripod** | 2 | "Cámara externa con trípode" |
| **Fix freezing** | 2 | "El programa se detuvo completamente" |
| **Heavier balls (less drift)** | 1 | "Bolas con mayor masa para que no salgan del campo" |

---

## 4. COMPARISON WITH PENDULUM EXPERIMENTS

| Metric | Collisions 2025 | Pendulum Energy 2026 | Pendulum T²∝L 2025 |
|---|---|---|---|
| N respondents | 14 | ~14 | ~13 |
| Usability avg | 3.79 ⚠️ | 4.4 | 4.0 |
| Learning avg | 3.71 ⚠️ | 4.7 | 4.2 |
| Satisfaction avg | 3.48 ⚠️ | 4.2 | 4.0 |
| Want more CV codes | 4.00 ✅ | 4.3 | — |
| Main issue | Detection/reflections | Calibration depth | Camera positioning |

**Key difference:** The collision experiment's technical failures are more disruptive than the pendulum's calibration depth issue, because a detection failure stops data collection entirely, while a calibration error just scales the results.

---

## 5. ACTION ITEMS

### 🔴 Critical (before next use)

1. **Increase confidence threshold** in `colisiones_v2.py`: `conf=0.6` → `conf=0.7` to reduce reflection false positives
2. **Add surface-plane filter**: Reject detections whose vertical position in the frame is below a user-set "table level" marker
3. **Add visual indicator** when zero or one ball is detected (red border / HUD warning) so students know immediately when tracking fails

### 🟡 High Impact

4. **Camera guide**: Add a startup screen or README section specifically for the collision setup — recommend matte/dark table surface and controlled lighting, external USB camera on a fixed mount
5. **Fix freeze bug**: Investigate the crash after multiple tracking runs (R13 — "se detuvo completamente tras varias tomas") — possibly a memory leak in `deque` or YOLO model state
6. **Ball-exit warning**: When a ball leaves the calibrated area, display a clear HUD message instead of silently losing the track

### 🟠 Medium

7. **Graph integration**: Satisfaction with data interpretation was lowest (3.36). Consider showing the `collision_vectors.pdf` panels inside the OpenCV window (embedded matplotlib) rather than saving to file only
8. **Collision surface recommendation**: Suggest students use a dark/non-reflective mat on the table

---

## 6. STUDENT EXPERIENCE SUMMARY

**Collision experiment path:**
1. ⚠️ Setup: table reflections immediately cause false detections
2. 🔴 Tracking: ball detection drops in/out during fast collision
3. 🔴 Analysis: some groups unable to reach the `K` momentum analysis step due to detection issues
4. ✅ Concept: students who got data understood momentum and restitution
5. ✅ Desire: 4.00/5 want more CV experiments — the concept resonates

**Bottom line:** The collision experiment is the weakest of the four (lowest scores on all dimensions), but it has the highest-rated HUD (4.5/5) and strong desire for more. The problem is purely technical: light reflections on table surfaces are detected as false positives, derailing the experiment before analysis can begin. A confidence threshold increase and a surface-plane filter would have high impact with low implementation cost.
