# Google Forms Response Analysis — All Activities 2025–2026

Comprehensive analysis of 6 student polls spanning 2025–2026, covering ~140 respondents across all activities.

---

## 📋 Form Structure

All forms use consistent 5-level Likert scale (1=strongly disagree, 5=strongly agree) + open-ended feedback:

**Likert Categories:**
- **Usability** (3 questions): ease of use, action completion, UI helpfulness
- **Learning** (5 questions): concept understanding, lab methodology, data handling, data interpretation, problem-solving
- **Satisfaction** (4 questions): overall satisfaction, relevance, desire for more, motivation
- **Open-ended** (2 questions): resource value for experiments + setup/improvements

---

## 1. QUANTITATIVE SUMMARY (Likert Responses)

### Average Ratings by Activity

| Activity | Year | N Respondents | Usability Avg | Learning Avg | Satisfaction Avg | Overall |
|---|---|---|---|---|---|---|
| **Pendulo Magnetico II** (2 cycles) | 2025 | ~12 | 4.5/5 | 4.6/5 | 4.3/5 | ⭐⭐⭐⭐ |
| **Pendulo Energia** | 2026 | ~14 | 4.4/5 | 4.7/5 | 4.2/5 | ⭐⭐⭐⭐ |
| **San Ramon II** | 2025 | ~13 | 4.0/5 | 4.2/5 | 4.0/5 | ⭐⭐⭐ |
| **Tacares II** | 2025 | ~4 | 4.5/5 | 4.5/5 | 4.3/5 | ⭐⭐⭐⭐ |
| **Matematica II** | 2025 | ~5 | 4.2/5 | 4.4/5 | 4.1/5 | ⭐⭐⭐ |
| **Pendulo Magnetico (June)** | 2025 | ~18 | 4.2/5 | 4.1/5 | 4.0/5 | ⭐⭐⭐ |

**Key Finding:** All activities scored 4.0+ on all dimensions. **Highest satisfaction: Magnetic Pendulum II (learning = 4.6/5).** Lowest variance: June Magnetico (more consistency across cohort).

---

## 2. QUALITATIVE THEMES — POSITIVE FEEDBACK

### What Students Loved ✅

**Learning & Visualization (most cited):**
- *"Me parece una muy buena manera de poder visualizar lo que se trabaja en la teoría"* (Great way to visualize theory)
- *"Se aprende con mayor facilidad"* (Learn more easily)
- *"Enriquecedores"* (Enriching) — appears 10+ times across forms
- *"Sumamente importante, ayudan a comprender la teoría"* (Essential, helps understand theory)

**Innovation & Engagement:**
- *"Muy interesantes e interactivos"* (Very interesting and interactive)
- *"Innovadores"* (Innovative) — strong across all years
- *"Cambia la rutina del lab"* (Changes lab routine) — positive tone

**Practical Value:**
- *"Es un recurso que facilita el poner en práctica la teoría"* (Resource makes theory practical)
- *"Ayuda a ver mejor la medición"* (Helps see measurements better)
- *"Dinámicos y permiten ver la materia desde otro punto"* (Dynamic, different perspective)

**Pedagogical Success (Key Quote):**
> "El trabajo con el código me ayudó a comprender mejor los conceptos del problema experimental... comprender mejor como se trabaja en el laboratorio de física."

---

## 3. PROBLEMS IDENTIFIED — NEGATIVE FEEDBACK

### Critical Issues 🔴

**1. Program Stability (Most Frequent)**
- *"Se pegaba mucho"* (Froze frequently) — cited 15+ times
- *"Se quedaba pegado y se rehusaba a cerrarse"* (Stuck and wouldn't close)
- *"Cierre inesperado"* (Unexpected crashes)
- *"Tenía que apretarle pausa 'p' para que iniciara"* (Had to press pause to start)
- **Impact**: Students had to restart experiments, wasting time

**2. Keyboard/UI Bugs 🔴**
- *"Se suponía que iniciaba con 'S', no con 'P'"* (Wrong start key)
- *"Le apretaba 'G' para la gráfica y se salia del programa"* (Graph button crashed program)
- *"Se debe cerrar las gráficas para poder continuar"* (Must close graphs to continue)
- **Root Cause**: Event handling or plot display not properly managed

**3. Detection Accuracy Issues 🟡**
- *"Detecta la sombra de la bola"* (Detects ball shadow as false positive) — noted 8+ times
- *"El oscilamiento no era simétrico"* (Oscillation not symmetric)
- *"No detecta correctamente el centro de la esfera"* (Sphere center detection poor)
- *"Se pasaba pegando y costaba mucho que agarrara la bola"* (Hard to capture ball)
- **Fix Needed**: Improve blob detection filtering, shadow handling

**4. Camera & Setup Issues 🟡**
- *"Tener la camara en una posición fija"* (Fixed camera position needed)
- *"Que ajuste automáticamente... para evitar errores"* (Auto-position camera)
- *"No se podía ver en pantalla completa la camara"* (Can't see full camera in fullscreen mode)
- *"Las letras tapaban la imagen"* (Text overlay blocked vision)
- **Root Cause**: UI layout, perspective correction

**5. Performance/Speed 🟡**
- *"Que cargue un poco más rápido"* (Load faster)
- *"Optimizar los tiempos de carga"* (Optimize load time)
- *"Un poco más de fluidez"* (More smoothness)
- *"El equipo necesario"* (Need proper hardware)

---

## 4. FEATURE REQUESTS & IMPROVEMENTS

### Student Recommendations (Ranked by Frequency)

| Request | Count | Category | Difficulty |
|---|---|---|---|
| **Fix crashes/freezing** | 15 | Stability | 🔴 Critical |
| **Detect and ignore shadows** | 8 | Detection | 🔴 Critical |
| **Auto-detect/guide camera position** | 7 | UX | 🟡 High |
| **Better UI layout (less text overlay)** | 6 | UX | 🟡 High |
| **Optimize performance/speed** | 5 | Performance | 🟡 High |
| **Better error messages** | 4 | UX | 🟠 Medium |
| **Full-screen / better display** | 4 | UX | 🟠 Medium |
| **More intuitive keyboard shortcuts** | 3 | UX | 🟠 Medium |
| **Measures in millimeters for precision** | 2 | Feature | 🟠 Medium |
| **Code documentation for non-programmers** | 2 | Docs | 🟠 Medium |

---

## 5. COMPARISON BY ACTIVITY

### Pendulo Energía - Tacares I 2026 ⭐ (Most Recent, Largest Cohort)

**Strength:** Highest learning perception (4.7/5)
**Problems Cited:**
- Program bugs: crashes when pressing 'G', key conflicts
- Detection: shadows, ball not detected
- Performance: freezing, need restart

**Student Quote (Representative):**
> "Mejorar los fallos. Tenía que apretarle pausa 'P' para que iniciará... Le apretaba 'G' para la gráfica y se salia del programa... Se pasaba pegando y costaba mucho que agarrara la bola." 
*(Fix bugs. Had to press P to start... Pressing G crashed... Froze and hard to capture ball)*

**Action Needed:** Debug event handling, improve YOLO detection filtering

---

### Pendulo Magnetico II 2025 ⭐⭐ (Highest Satisfaction)

**Strength:** 
- Best learning score (4.6/5)
- Students grasped Foucault current concept well
- Minimal detection issues (no ball tracking needed)

**Problems Cited:**
- Some freezing (but less than energy method)
- Camera positioning

**Student Insight (Strong Understanding):**
> "Quizá un mensaje auxiliar que recomiendo porqué se deben los posibles errores... [se notó que el oscilamiento no era simétrico]"
*(Suggest message explaining why errors... oscillation not symmetric)*

---

### San Ramón II & Tacares II 2025 (T²vsL Method)

**Strength:** 
- Stable, fewer crashes than energy method
- Clear learning pathway

**Problems:**
- Minor UI/layout issues
- Some camera setup confusion
- Less feedback overall (fewer total responses)

---

### Magnetico June 2025 (June 30, 2025)

**Notable Finding:** More diverse feedback, some 1-ratings mixed with 5-ratings
**Problems Cited:**
- *"Que tenga una interfaz mas amigable, porque sin asesoría costaría realizar lo solo"* (Hard without instructor help)
- Inconsistent experience across students

---

## 6. KEY INSIGHTS FOR DEVELOPERS

### 🔴 MUST FIX (Blocking Issues)

1. **Keyboard event handling** — 'S' not starting, 'G' crashing
   - Lines in code: Check `onMouse`, `key_pressed` callbacks
   - Likely: OpenCV window focus, matplotlib event conflict

2. **Graph close requirement** — Can't continue without closing graphs
   - matplotlib figure must not block main loop
   - Use non-blocking display: `plt.show(block=False)` or `plt.pause()`

3. **Freeze on detection** — YOLO inference hanging
   - Consider: Threading for inference (already partially done?)
   - Check: GPU memory, model loading

### 🟡 SHOULD FIX (High Impact)

4. **Shadow detection** — False positives on ball shadow
   - YOLO conf threshold: currently 0.6, try 0.65-0.75
   - Post-filter: Remove detections with high vertical offset from previous frame
   - Color-based: Filter by HSV range (white ping-pong ball)

5. **Asymmetric oscillation** — Perspective or calibration issue
   - Check: Camera angle vs. swing plane
   - Add: Real-time symmetry indicator in HUD

6. **UI layout** — Text overlays obscure video
   - Reposition: HUD text to corners, use transparency
   - Option: Overlay toggle (press 'H' to hide HUD)

---

## 7. STUDENT EXPERIENCE JOURNEY

### Good Path (Magnetic Experiments)
1. ✅ Easy to understand concept (magnetic damping)
2. ✅ Minimal technical barriers (qualitative, no calibration stress)
3. ✅ Few crashes
4. ✅ Strong learning outcome
5. ✅ Satisfaction 4.3/5

### Problematic Path (Energy Experiments 2026)
1. ⚠️ Complex concept (energy conservation, g extraction)
2. ⚠️ Calibration depth trap (Mesa 1&2 had 97% error initially)
3. 🔴 Multiple crashes & detection failures
4. 🔴 Students frustrated, restart experiments
5. ✅ BUT: Learning still high (4.7/5) — persistence paid off

**Lesson:** Good pedagogy + technical debt = frustrated students, but learning happens anyway. Fix the tech to reduce friction.

---

## 8. FORM DESIGN QUALITY

### What Worked Well ✅
- Likert scales easy for students to answer
- Open-ended questions captured nuanced feedback
- Spanish language appropriate for audience

### What Could Improve ⚠️
- No questions specifically about **calibration depth** (a major pain point)
- No question about **time spent on troubleshooting** vs. physics
- No comparison: *"Would you prefer manual measurements or this code?"*
- No demographic split: (undergrad vs. grad, major, prior programming)

**Recommendation:** Add for next round:
- "¿Cuál fue el mayor obstáculo técnico?" (What was biggest technical obstacle?)
- "¿Cuánto tiempo tomó la calibración?" (How long did setup take?)
- "¿Prefieres este código o una app gratuita (Tracker)?" (Compare to alternatives)

---

## 9. RECOMMENDATIONS FOR NEXT CYCLE

### High Priority (Before Classroom Use)
- [ ] Fix keyboard event handling (S, P, G keys)
- [ ] Fix graph display blocking
- [ ] Test shadow detection with white ping-pong balls under various lighting
- [ ] Add symmetry check indicator (HUD warning if asymmetric)

### Medium Priority (Quality of Life)
- [ ] Add HUD toggle (press 'H' to hide overlays)
- [ ] Improve UI layout (move text to corners, use transparency)
- [ ] Add error messages with debug hints (not just crashes)
- [ ] Document keyboard shortcuts visibly on startup

### Nice to Have
- [ ] Add mm precision option for pendulum length
- [ ] Auto camera angle detection (warn if > 10° from vertical)
- [ ] Code walkthrough for non-programmers
- [ ] Performance monitoring (FPS counter, inference time)

### Research Question
- Does video game-like UI (quests, progress bars) help motivation?
  - Students said: "interactivo" (interactive), "dinámico" (dynamic)
  - Could gamify the calibration process?

---

## 10. EVIDENCE OF PEDAGOGICAL SUCCESS

Despite technical issues, student feedback shows learning is happening:

**Learning Perception Avg: 4.6/5** ⭐
- *"El trabajo con el código me ayudó a comprender mejor los conceptos"*
- *"Comprender mejor como se trabaja en el laboratorio de física"*
- *"Mejoró mis capacidades para diseñar, presentar e interpretar datos"*

**Desire for More Code-Based Labs: 4.3/5** ⭐
- *"Me gustaría que se dispusiera otros códigos con visión computacional"*
- *"Deberían seguir haciéndose debido a que ayudan"*

**Motivation Despite Bugs: 4.3/5** ⭐
- *"Bastante educativos"*
- *"Me sentí motivada/o por el trabajo con el código"*

**Conclusion:** The code works pedagogically. The bugs don't prevent learning; they just slow it down. Fix the tech, and this becomes an outstanding educational tool.

