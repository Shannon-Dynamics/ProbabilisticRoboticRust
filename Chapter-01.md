# Chapter 1 — The Robot That Doubts — Why Probabilistic Robotics?

> Part I — Foundations: The Robot and Its Uncertainty · Estimated length: 5 web pages · Difficulty: Foundational

## 1. Purpose & Story Arc

This chapter is the book's thesis statement and voice calibration. It must convince a skeptical
engineer, in one sitting, that (a) a robot acting on a single best guess is quantifiably worse than
a robot acting on a belief, and (b) this book will prove every such claim three ways — with math
(F), with a widget they can poke (C), and with Rust code that *is* the widget (P). The "aha" the
reader leaves with: **uncertainty is not a nuisance to engineer away; it is state to be estimated,
and estimating it changes what the robot should do.** The voice set here rules the whole book:
hook-driven and playful ("the robot that doubts"), but every playful claim is immediately backed by
a number, an equation, or a runnable artifact — never hand-waving. Concrete before abstract, always.

Story line:

1. **Hook** — autoplay widget: Rusty is commanded to drive a square; dead reckoning says it did;
   ground truth says otherwise (w1.2). One sentence of narration: "Every robot you have ever seen
   is lying to itself about where it is."
2. **Problem** — the five sources of uncertainty (Thrun's list, modern examples), each pinned to a
   spot in the Apartment scene (w1.3).
3. **Intuition** — a single best guess vs. a belief: the two-corridor decision example with real
   numbers; the argmax fallacy.
4. **Formalism (light)** — first sighting of $x_t, u_t, z_t, bel(x_t)$; the hallway thought
   experiment run live (w1.1, flagship preview); "sensing sharpens, moving smears" spoken for the
   first time.
5. **Implications** — what the probabilistic paradigm buys (unstructured environments, weaker
   models, graceful degradation) and what it costs (computation, approximation) — honest on both.
6. **The contract** — tour of the book: the FCP method, the fixed chapter rhythm, the color code,
   the two running worlds (Hallway, Apartment), and Rusty, who is built in Ch. 4 and never leaves.
7. **Experiment** — reader setup: clone, `cargo run`, reproduce the chapter's numeric example on
   their own machine in under five minutes.

## 2. Prerequisites & Position

- **Builds on:** nothing — this is the entry point. Assumed: programming fluency in *some*
  language (Rust not required yet; Appendix A is the on-ramp), high-school probability, comfort
  with vectors/matrices at the "seen them before" level.
- **Feeds into:** Ch. 2 (formalizes the probability used informally here), Ch. 4 (the workspace
  cloned here becomes the lab), Ch. 5 (w1.1 grows into the full Hallway Belief Machine w5.1, and
  the informal sense/shift update is derived as the Bayes filter), Ch. 26 (the capstone closes the
  loop opened by the hook), every chapter (voice, color code, FCP rhythm).
- **Baseline sources:** Thrun et al. (1999–2000 draft) Ch. 1 §1.1 (uncertainty in robotics — the
  five sources), §1.2 (the probabilistic paradigm), §1.3 (implications), §1.4 (road map).
  Pedagogy baseline (research dossier): Labbe (intuition-first stance), Ciechanowski (autoplay,
  one-parameter widgets, "glanceable then explorable"), bzarg (color-coded equations), Victor
  (ladder of abstraction), kalmanfilter.net (fully numeric worked examples).

## 3. Foundation (F) — Mathematical Core

Deliberately light — this is the only chapter where C outweighs F — but nothing said here may be
retracted later. Chapter-scoped notation table (first appearance of the book-wide symbols):

| Symbol | Meaning | Formalized in |
|---|---|---|
| $x_t$ | state at time $t$ (for Rusty: pose, later pose+map) | Ch. 5 |
| $u_t$ | control asserted between $t-1$ and $t$ | Ch. 5, 9 |
| $z_t$ | measurement at time $t$ | Ch. 5, 10 |
| $bel(x_t) = p(x_t \mid z_{1:t}, u_{1:t})$ | belief: posterior over state given all data | Ch. 5 |
| $\eta$ | generic normalizer | Ch. 2 |

**Definitions introduced (named, one crisp sentence each):**

- **The five sources of uncertainty** (Thrun §1.1, retained verbatim as a taxonomy):
  *environments* (unpredictable, dynamic worlds), *sensors* (limits and noise), *actuation*
  (motors slip, gears have backlash), *models* (every model in this book is knowingly wrong),
  *computation* (real-time forces approximation). Each gets a modern example: dynamic obstacles in
  a warehouse; LiDAR dropout on black surfaces; wheel slip on tile; the point-mass map assumption;
  particle counts bounded by a WASM frame budget.
- **Point estimate vs. belief:** a point estimate is a single $\hat{x}_t$; a belief is a
  distribution $bel(x_t)$ — the book's central object.
- **The probabilistic paradigm** (statement, not yet theorem): represent information by probability
  distributions over state, update them by the laws of probability, and act on the *distribution*.

**Derivation 1 — "The best guess is not enough" (the argmax fallacy).**
*Statement:* the action optimal against the full belief, $a^\star = \arg\min_a \mathbb{E}_{p(x)}[L(a,x)]$,
can differ from — and strictly beat — the action optimal against the mode,
$\hat{a} = \arg\min_a L(a, \arg\max_x p(x))$.
*Sketch (4 steps):*
1. Rusty is at a T-junction; the charger is left or right: $p(\text{left}) = 0.55$, $p(\text{right}) = 0.45$.
2. Loss table: go-left costs $0$ if left, $10$ if right; go-right symmetric; sense-again costs $1$
   and then acts perfectly.
3. Expected losses: $\mathbb{E}[\text{go-left}] = 4.5$, $\mathbb{E}[\text{go-right}] = 5.5$,
   $\mathbb{E}[\text{sense}] = 1$.
4. The mode-follower goes left (its belief is a point, so sensing "gains nothing"); the
   belief-follower senses. Doubt has decision value.
*Collapsible:* the general statement with an arbitrary loss; pointer to Ch. 22 where this becomes
belief-space planning (the tiger problem is this example grown up).

**Derivation 2 — "Sense then move, by hand" (the hallway, numerically).**
*Statement:* two door-sensings separated by one motion localize Rusty in a 10-cell cyclic corridor
with doors at cells $\{1, 4, 5\}$, even though single sensings are ambiguous.
*Sketch (5 steps):*
1. Uniform prior: $bel(x) = 0.1$ for all 10 cells (blue).
2. Sense "door": multiply by likelihood $p(z{=}\text{door} \mid x) = 0.6$ at doors, $0.2$ elsewhere
   (green); normalize with $\eta$: doors $\to 0.1875$, others $\to 0.0625$ (purple).
3. Move right one cell (perfect motion, for now): the histogram shifts cyclically (orange).
4. Sense "door" again: only cell 5 was a door *and* is one cell right of a door; posterior at
   cell 5 $= 0.1125/0.325 = 9/26 \approx 0.3462$.
5. Moral spoken plainly: sensing multiplied, moving shifted (and in Ch. 5, will smear).
*Collapsible:* the full 10-cell table for all three steps, every number — this exact table is
reproduced by the chapter's code and its unit test (§5), seeding the book's convention that every
numeric worked example is executable.

**Named algorithms:** none formally introduced. Explicit forward pointer: the two informal
operations above are the two lines of `Bayes_filter` (Thrun Table 2.1), derived honestly in Ch. 5.
The informal signatures `sense(bel, z) -> bel'` and `shift(bel, u) -> bel'`, each $O(N)$ over $N$
cells, appear in the text so the reader meets the predict/correct shape on day one.

## 4. Conceptual (C) — Intuition & Visual Design

Widget discipline (set here, obeyed book-wide, per the pedagogy dossier): every widget autoplays a
sensible default; interaction is invitation, not requirement; one meaningful parameter each; every
widget ships a build-time static SVG fallback rendered from the same Rust code; the book color code
(prior **blue**, prediction **orange**, measurement **green**, posterior **purple**, ground truth
**gray dashed**) is introduced by w1.4 and never violated.

- **Widget w1.2: Dead Reckoner (the hook)** — type: wasm-sim, autoplay. Rusty drives a commanded
  square in the Apartment on a 10 s loop; the dead-reckoned pose (orange) diverges from ground
  truth (gray dashed); a small inset plots position error vs. time. Reader manipulates: one
  toggle — actuation noise on/off (plus the standard seed reroll). Observes: with noise off the
  traces coincide; with noise on, unbounded drift, different every seed. Misconception killed:
  *"if I know the commands, I know where the robot is."*
- **Widget w1.1: Hallway Belief Machine (preview)** — type: wasm-sim, autoplay animation — the
  flagship, the book's thesis in one loop. A 1D corridor with three indistinguishable doors;
  belief histogram below; the canned sequence sense → move → sense from Derivation 2 plays with
  color-coded phases (blue prior, green likelihood flash, purple posterior, orange smear on move).
  Reader manipulates: play/pause and single-step buttons only (the full parameter surface —
  noise sliders, Markov-breaking — is deliberately reserved for w5.1). Observes: ambiguous
  multi-modal belief collapsing to one peak; the numbers match Derivation 2 exactly. Misconception
  killed: *"uncertainty only ever grows"* and *"a robot needs one hypothesis."*
- **Widget w1.3: Five Uncertainties, One Apartment** — type: static-svg (annotated scene). The
  Apartment floorplan with five callouts pinning each uncertainty source to a concrete place
  (glass wall → sensor; rug → actuation; person walking → environment; "walls are line segments"
  → model; frame budget note → computation). No interaction; this is the chapter's shareable
  figure.
- **Widget w1.4: How to Read This Book** — type: static-svg. The FCP rhythm diagram
  (Hook → C → F → P → Lab → Exercises) plus the color-code legend rendered as a keyed strip; the
  same SVG is reused in the site header/appendix D. Establishes that equation terms, figure
  elements, and code comments share colors.

Layout sketch: w1.2 sits above the fold before any prose; w1.3 inline in the five-sources section;
w1.1 mid-chapter at the hallway narrative with the Derivation 2 table beside it (numbers in the
table highlight in sync with the animation step — the book's first taste of linked views); w1.4
closes the chapter next to the setup instructions.

## 5. Practical (P) — Rust Implementation

- **Crates used:** none — `demos/ch01-hello` is dependency-free std-only Rust by design, so
  the reader's *first build finishes in seconds* and cannot fail on native deps. (The workspace
  `Cargo.toml` already pins the full book stack from TOC.md — nalgebra 0.35, rand 0.9,
  parry2d 0.30, eframe 0.35, … — but nothing in this chapter touches it.) The w1.1/w1.2 demo
  crates use `eframe`/`egui` 0.35 + `widget-kit` (built in Ch. 4; Ch. 1's demos are authored
  against that framework even though the reader hasn't seen its internals yet).
- **Module plan:** `demos/ch01-hello/` (bin, the reader's first run) ·
  `demos/ch01-doubt/` (bins `w1-1-hallway-preview`, `w1-2-dead-reckoner`, wasm targets).
- **Reader setup path** (spelled out step-by-step in the chapter, tested in CI on Linux/macOS/
  Windows): install `rustup` → stable toolchain (any 2026 stable; edition 2024 requires
  ≥ 1.85) → `git clone <book repo> && cd prob-robotics-rust` → `cargo run -p ch01_hello` →
  `cargo test -p ch01_hello`. Optional (for widget hacking, deferred to Ch. 4):
  `rustup target add wasm32-unknown-unknown`, `cargo install trunk`; for building the book site:
  `mdbook` 0.5.x + `mdbook-katex` 0.10.

Key code (the entire `ch01-hello`, ~60 lines, printed in full in the chapter):

```rust
// demos/ch01-hello/src/main.rs — zero dependencies, edition 2024
const N: usize = 10;
const DOORS: [usize; 3] = [1, 4, 5];
const P_HIT: f64 = 0.6; // p(z = door | at a door)
const P_FALSE: f64 = 0.2; // p(z = door | not at a door)

fn normalize(bel: &mut [f64; N]) { let s: f64 = bel.iter().sum(); bel.iter_mut().for_each(|p| *p /= s); }

/// Measurement update: pointwise product with the likelihood, then normalize. (η in action.)
fn sense_door(bel: &mut [f64; N]) {
    for (i, p) in bel.iter_mut().enumerate() {
        *p *= if DOORS.contains(&i) { P_HIT } else { P_FALSE };
    }
    normalize(bel);
}

/// Motion update: perfect cyclic shift right. (Ch. 5 replaces this with a smearing convolution.)
fn move_right(bel: &[f64; N]) -> [f64; N] {
    std::array::from_fn(|i| bel[(i + N - 1) % N])
}

fn bar(p: f64) -> String { "█".repeat((p * 60.0).round() as usize) }

fn main() {
    let mut bel = [1.0 / N as f64; N];
    sense_door(&mut bel);
    bel = move_right(&bel);
    sense_door(&mut bel);
    for (i, p) in bel.iter().enumerate() { println!("cell {i}: {p:.4} {}", bar(*p)); }
}
```

- **Worked end-to-end example & expected output:** running it prints the Derivation 2 posterior —
  cell 5 at `0.3462` with the longest bar, cells 1/2/4/6 at `0.1154`, the rest at `0.0385`.
  A unit test `worked_example_ch01` asserts `bel[5] == 9.0/26.0` to 1e-12 — the first instance of
  the book-wide convention (formalized in Ch. 2) that every chapter's worked numbers are locked by
  a test.
- **Runnable artifact:** `cargo run -p ch01_hello` (terminal histogram above); the WASM artifacts
  are w1.1/w1.2 embedded in-page. The chapter ends by telling the reader: "the animation you
  watched and the numbers you just printed are the same computation — that is the whole book."

## 6. Simulations & Animations Manifest

| id | title | type | crate stack | interaction | teaches |
|---|---|---|---|---|---|
| w1.1 | Hallway Belief Machine (preview) | wasm-sim (autoplay animation) | eframe 0.35 + widget-kit + pr-core | play/pause, step, seed reroll | belief as histogram; sense sharpens, move shifts; multi-modality is fine |
| w1.2 | Dead Reckoner | wasm-sim (autoplay) | eframe 0.35 + widget-kit + sim | noise on/off toggle, seed reroll | commands ≠ position; drift is unbounded and seed-dependent |
| w1.3 | Five Uncertainties, One Apartment | static-svg | plotters (build-time) | none | Thrun's five uncertainty sources, concretely located |
| w1.4 | How to Read This Book | static-svg | plotters (build-time) | none | FCP rhythm; the book color code |

## 7. Exercises & Extensions

1. **(F)** Redo Derivation 1 with $p(\text{left}) = 0.9$. At what prior probability does
   sense-again stop being optimal? Derive the threshold as a function of the sensing cost.
2. **(F)** In Derivation 2, show that if the doors were at cells $\{1, 4, 7\}$ (evenly spaced in a
   9-cell corridor), no number of sense–move cycles with perfect motion ever yields a unimodal
   belief. What property of the door layout does localization depend on?
3. **(C — predict, then verify with w1.1)** Before pressing step: which cells hold the most
   probability after sense → move → move → sense? Verify, then explain why the answer differs
   from Derivation 2.
4. **(C — w1.2)** With noise on, watch five seeds. Is the *direction* of final drift predictable?
   Write down why not, in one sentence, using the word "distribution."
5. **(P)** Modify `ch01-hello`: add a `move_right_sloppy` that goes right two cells with
   probability 0.1, one cell with 0.8, and stays with 0.1. Re-run the sequence. This is the
   book's first convolution — keep your code; Ch. 5 derives what you just did.
6. **(P — setup)** Complete the setup path, run `cargo test -p ch01_hello`, and change `P_HIT` to
   0.9. Predict before running: does cell 5's posterior go up or down? (Both the test failing and
   *why it fails* are the lesson.)

## 8. Modernization Notes

- The baseline is the **1999–2000 16-chapter draft** of Thrun et al., whose Ch. 1 is conceptual
  and slightly dated: examples are museum tour-guide robots (Rhino/Minerva). We keep the five-source
  uncertainty taxonomy and the corridor thought experiment (both aged perfectly), and replace the
  motivating fleet with 2026-era examples (warehouse AMRs, sidewalk delivery, Rust's adoption in
  safety-critical autonomy) plus Rusty as the in-book protagonist.
- The draft's §1.3 "implications" claims are kept but made falsifiable in-book: "graceful
  degradation" is *demonstrated* in Ch. 12's kidnap recovery and Ch. 26's failure-mode tour rather
  than asserted.
- Added relative to any edition of the baseline: the decision-theoretic argmax-fallacy example
  (front-loading the Part VI payoff that beliefs exist *to be acted on*), the executable numeric
  worked example with its locking unit test, and the entire delivery mechanism (autoplay WASM
  widgets, color-coded equations, FCP contract) drawn from the pedagogy research (Labbe,
  Ciechanowski, bzarg, Victor).
- Dropped: the draft's §1.5 bibliographical-remarks format (the book uses per-chapter "further
  reading" boxes with modern citations instead); no attempt to survey robotics history.
