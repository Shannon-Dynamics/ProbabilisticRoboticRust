# Chapter 3 — The Geometry of Motion: Poses, Frames, and Lie Groups

> Part I — Foundations: The Robot and Its Uncertainty · Estimated length: 9 web pages · Difficulty: Intermediate

## 1. Purpose & Story Arc

Probabilistic robotics puts distributions *on poses*, and poses do not live in a vector space —
that single fact, ignored, produces half the bugs in student SLAM systems (angle wrap-around,
covariance ellipses on the wrong manifold, "just average the quaternions"). This chapter builds
the geometric substrate honestly: frames and rotation matrices the Craig way, homogeneous
transforms, SO(2)/SE(2) worked to the last decimal, SO(3)/SE(3) with quaternions and axis-angle,
exponential/log maps and screws the Lynch–Park way — and then the modern move the baselines don't
make: $\boxplus/\boxminus$ as the disciplined answer to "state + increment" on a manifold (Sola's
micro Lie theory). The "aha": **$\exp$ turns straight lines in tangent space into arcs in the
world, and $\boxplus$ lets every filter in this book pretend, locally and legitimately, that poses
are vectors.** The chapter also forges the book's most-reused artifact: the hand-rolled `SE2` type
that every filter, motion model, and SLAM system from here to Ch. 26 imports.

Story line:

1. **Hook** — widget w3.3: average two headings, 179° and −179°. The "mean" points backwards.
   One frame drag in w3.1 shows the same disease in 2D poses.
2. **Frames and transforms** — Craig's leading-superscript notation as a *type system on paper*;
   composition, inversion, the cancellation rule.
3. **Rotations** — SO(2) and SO(3) as matrix groups; parameterizations (angle, axis-angle,
   quaternion) and their traps (Spong/Craig's Euler-angle taxonomy compressed to one honest box).
4. **The exponential map** — from the ODE $\dot{R} = R\,[\omega]_\times$ to $\exp$; twists and
   screws; SE(2) closed forms derived fully; SE(3) stated with sophus as the executor.
5. **$\boxplus/\boxminus$** — why "pose + delta" needs a definition; the retraction axioms;
   local coordinates as the peace treaty between manifolds and filters.
6. **Uncertainty on manifolds (preview)** — a Gaussian in the tangent space pushed through
   $\exp$ becomes a banana in $(x, y)$ (w3.4); one paragraph, full treatment in Chs. 7 and 9.
7. **Experiment** — implement `SE2` (exp/log/compose/adjoint/⊞/⊟), property-test it, drive a
   square with it, and catch a frame bug at compile time with newtypes.

## 2. Prerequisites & Position

- **Builds on:** Ch. 1 (Rusty and the pose vector $(x, y, \theta)^\top$ seen informally), Ch. 2
  (Gaussians — needed for the tangent-space-uncertainty preview and the property-based tests'
  random inputs).
- **Feeds into:** Ch. 4 (diff-drive exact integration *is* `SE2::exp`), Ch. 7 (error-state EKF on
  the `Manifold` trait born here), Ch. 9 (on-manifold motion noise; the banana grown up), Ch. 11
  (EKF localization Jacobians), Chs. 15–16 (pose-graph optimization on SE(2)), Ch. 18 (SE(3) via
  sophus), Appendix C (the reference card is this chapter's tables, extracted).
- **Baseline sources:** Lynch & Park Ch. 3 §3.1 (planar rigid motions), §3.2 (rotations, angular
  velocity, exponential coordinates of rotation), §3.3 (homogeneous transforms, twists,
  exponential coordinates of rigid motion), Appendix B (quaternions, Euler angles); Craig Ch. 2
  §2.2–2.7 (frames, mappings vs. operators, compound transforms, transform equations), §2.8
  (orientation representations); Spong et al. Ch. 2 §2.2–2.6 (rotations, current-vs-fixed-frame
  composition rules, parameterizations incl. exponential coordinates, rigid motions), Appendix B
  (matrix exponential, Lie groups); Niku Ch. 3 (screw-based transforms — cross-reference only).
  **Modernization spine:** Sola, Deray, Atchuthan, *A micro Lie theory for state estimation in
  robotics* (arXiv:1812.01537); Hertzberg et al. (⊞/⊟ axioms).

## 3. Foundation (F) — Mathematical Core

Chapter notation table (extends TOC; one deliberate deviation, flagged):

| Symbol | Meaning |
|---|---|
| $\{A\}, \{B\}$ | coordinate frames (Craig braces) |
| ${}^{A}p$ | point $p$ expressed in $\{A\}$ |
| ${}^{A}_{B}T \in SE(2)/SE(3)$, ${}^{A}_{B}R$ | transform/rotation of $\{B\}$ relative to $\{A\}$; maps ${}^{B}p \mapsto {}^{A}p$ |
| $\boldsymbol{\tau} = (v_x, v_y, \omega)^\top$ | SE(2) tangent/twist coordinates — **translation-first** (Sola order; note: Lynch–Park write $(\omega, v)$; and we avoid the symbol $\xi$, reserved book-wide for the information vector) |
| $\tau^{\wedge}$, $(\cdot)^{\vee}$ | hat: coordinates → Lie-algebra matrix; vee: inverse |
| $\exp, \log$ | exponential/log maps $\mathfrak{se}(2) \leftrightarrow SE(2)$ etc. |
| $\operatorname{Ad}_T$ | adjoint of $T$ |
| $x \boxplus \delta,\ y \boxminus x$ | retraction $x \cdot \exp(\delta^{\wedge})$ and its local inverse $\log(x^{-1}y)^{\vee}$ — **right/local convention, fixed book-wide** |
| $q \in S^3$ | unit quaternion; double cover of SO(3) |

**Definitions:** frame; rotation matrix ($R^\top R = I$, $\det R = +1$); SO(2), SO(3), SE(2),
SE(3) as groups with $T = \begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix}$; Craig's mapping vs.
operator readings of the same matrix; skew operator $[\omega]_\times$; Lie algebra elements
$\tau^{\wedge}$; twist; screw axis and pitch (Chasles stated); geodesic; retraction (Hertzberg
axioms: $x \boxplus 0 = x$; $x \boxplus (y \boxminus x) = y$; smoothness); Rusty's pose is
${}^{W}_{B}T$ ("world from body"), and the book reads every transform name left-to-right as
"target-from-source" — a naming convention enforced later by the type system.

**Derivations:**

1. **Rotation matrices are stacked axes** (Craig §2.2). *Statement:* the columns of ${}^{A}_{B}R$
   are $\{B\}$'s unit axes expressed in $\{A\}$; orthonormality follows. *Sketch (3 steps):*
   expand ${}^{A}p = {}^{A}_{B}R\,{}^{B}p$ on basis vectors; read columns; dot products give
   $R^\top R = I$. *Collapsible:* why $\det = +1$ (orientation), and the mapping/operator duality.
2. **The cancellation rule.** *Statement:* ${}^{A}_{C}T = {}^{A}_{B}T\; {}^{B}_{C}T$ and
   $({}^{A}_{B}T)^{-1} = {}^{B}_{A}T$ with the closed-form inverse $(R^\top, -R^\top t)$.
   *Sketch (3 steps):* compose mappings; match sub/superscripts ("adjacent indices cancel");
   verify inverse by block multiplication. *Collapsible:* Craig's transform-equation technique
   (solving ${}^{A}_{B}T\,X = {}^{A}_{C}T$ for an unknown frame) — used again in Ch. 16's
   loop-closure constraints.
3. **$\exp$ for SO(2)/SO(3) from the rotation ODE.** *Statement:* $\dot{R} = R[\omega]_\times$
   with constant $\omega$ gives $R(t) = R(0)\exp([\omega]_\times t)$; for SO(3),
   $\exp([\hat\omega]_\times\theta) = I + \sin\theta\,[\hat\omega]_\times + (1-\cos\theta)[\hat\omega]_\times^2$
   (Rodrigues). *Sketch (4 steps):* differentiate $R^\top R = I$ to get skew velocity; solve the
   linear matrix ODE; use $[\hat\omega]_\times^3 = -[\hat\omega]_\times$ to collapse the series;
   read off Rodrigues. *Collapsible:* full series bookkeeping; the SO(2) specialization where
   everything commutes.
4. **SE(2) exp/log in closed form** — the chapter's centerpiece. *Statement:* for
   $\boldsymbol{\tau} = (\rho, \theta)$ with $\rho = (v_x, v_y)^\top$,
   $\exp(\tau^{\wedge}) = \big(R(\theta),\, V(\theta)\rho\big)$ where
   $$V(\theta) = \frac{1}{\theta}\begin{bmatrix} \sin\theta & -(1-\cos\theta) \\ 1-\cos\theta & \sin\theta \end{bmatrix},$$
   and $\log$ inverts it via $\theta = \operatorname{atan2}(R_{21}, R_{11})$, $\rho = V(\theta)^{-1} t$.
   *Sketch (5 steps):* integrate the constant-twist ODE for the translation column; the integral
   of $R(s\theta)$ over $s\in[0,1]$ yields $V$; check $V(\theta) \to I$ as $\theta \to 0$;
   invert $V$ in closed form; state the numerical guard (Taylor series for $|\theta| < 10^{-4}$).
   *Collapsible:* the full integration, $V^{-1}$'s closed form, and the Taylor coefficients used
   in code. *Payoff flagged for Ch. 4:* $\exp((\Delta s, 0, \Delta\theta))$ is exactly the
   constant-curvature arc — diff-drive odometry falls out for free.
5. **Quaternions in one honest page.** *Statement:* unit quaternions double-cover SO(3);
   $q = (\cos\frac{\theta}{2}, \hat\omega\sin\frac{\theta}{2})$; composition is quaternion
   product. *Sketch (3 steps):* verify the rotation action $q p q^{-1}$; norm preservation;
   $\pm q$ give the same $R$. *Collapsible:* conversion formulas both directions; why
   interpolation uses slerp; renormalization drift. (Implementation delegated to nalgebra's
   `UnitQuaternion` — we derive, we do not re-implement.)
6. **The adjoint moves twists between frames.** *Statement:* $T\exp(\tau^{\wedge})T^{-1} =
   \exp\big((\operatorname{Ad}_T \tau)^{\wedge}\big)$; for SE(2),
   $\operatorname{Ad}_T = \begin{bmatrix} R & -S\,t \\ 0 & 1 \end{bmatrix}$ with
   $S = \begin{bmatrix} 0 & -1 \\ 1 & 0\end{bmatrix}$. *Sketch (3 steps):* conjugate a one-parameter
   motion; differentiate at 0; collect blocks. *Collapsible:* full block algebra. **Design note in
   the margin:** sign conventions here are notorious; the book's stated authority is the property
   test `adjoint_conjugation` (below), not anyone's memory — a deliberate lesson in executable
   mathematics.
7. **⊞/⊟ make a legitimate local vector space.** *Statement:* with $x \boxplus \delta =
   x\exp(\delta^{\wedge})$ and $y \boxminus x = \log(x^{-1}y)^{\vee}$, the Hertzberg axioms hold,
   and for SE(2) $\boxminus$ is smooth wherever $|\theta_{x^{-1}y}| < \pi$. *Sketch (4 steps):*
   axiom 1 by $\exp(0) = I$; axiom 2 by $\exp\circ\log$; injectivity radius from Derivation 4's
   $\log$; remark that *right* ⊞ means increments live in the **body frame** — the physically
   meaningful choice for a robot (its odometry is body-frame), and the book's fixed convention.
   *Collapsible:* left-⊞ alternative and how to convert (via $\operatorname{Ad}$); pointer to
   Ch. 7 where the choice becomes the error-state EKF's error definition.

**Named algorithms** (no Thrun-table names exist for geometry; names are the book's Rust API):

- `se2_exp(tau: Tangent2) -> SE2` / `se2_log(T: SE2) -> Tangent2` — $O(1)$, Taylor-guarded at
  small $|\theta|$; round-trip error < 1e-12 over the test domain.
- `compose(a, b)`, `inverse(a)` — $O(1)$; group axioms property-tested.
- `adjoint(T) -> SMatrix<f64,3,3>` — $O(1)$.
- `boxplus(x, delta)`, `boxminus(y, x)` — $O(1)$; axioms property-tested.
- `so3_exp/so3_log` (axis-angle ↔ `UnitQuaternion`) — $O(1)$; delegated to nalgebra, edge cases
  ($\theta \approx 0$, $\theta \approx \pi$) tested against `sophus`.

## 4. Conceptual (C) — Intuition & Visual Design

Chapter metaphor: *tangent space is the flat map, the manifold is the globe; $\exp$ is the
projection that bends straight lines into arcs.* All widgets autoplay with one primary control
and ship static fallbacks.

- **Widget w3.1: Frame Composer** (flagship) — type: interactive sim. Three draggable frames
  $\{W\}, \{B\}, \{L\}$ (world, body, lidar) in the plane, each with axis glyphs; a live
  transform-chain panel renders ${}^{W}_{L}T = {}^{W}_{B}T\,{}^{B}_{L}T$ in Craig notation with
  the canceling indices highlighted on hover, and shows the numeric $3{\times}3$ homogeneous
  matrices. A test point $p$ (draggable) displays its coordinates in all three frames at once.
  Reader manipulates: drag/rotate any frame, drag $p$; one toggle: "compose about fixed vs.
  current frame" (pre- vs. post-multiplication, Spong §2.4) with the two results shown as ghost
  frames. Misconception killed: *transform composition is commutative / superscript notation is
  decoration* — the notation is a type system, and the toggle makes the non-commutativity
  physical.
- **Widget w3.2: Exp/Log Lens** (flagship) — type: interactive sim/animation. Left pane: the
  tangent space, a flat $(v_x, v_y, \omega)$ card with a draggable arrow $\boldsymbol{\tau}$
  (2D position of arrowhead = $v$, one slider = $\omega$). Right pane: the plane, where the screw
  motion $s \mapsto \exp(s\tau^{\wedge})$, $s \in [0,1]$ animates Rusty sliding along an arc
  around the (marked) instantaneous center of rotation. Autoplay sweeps $s$ continuously. Toggle:
  "naive interpolation" overlays the pose obtained by independently lerping $(x, y, \theta)$ —
  visibly leaving the arc, wheels side-slipping. Misconception killed: *pose interpolation =
  componentwise lerp.* A "3D" tab shows the same story for one fixed SE(3) screw (pitch slider)
  as a pre-rendered three-d animation — look, don't touch; sophus computes it.
- **Widget w3.3: The Wrap-Around Trap** — type: interactive sim (the chapter's hook). A compass
  rose with two draggable heading needles; the arithmetic mean of the two angles (red, wrong) vs.
  $\theta_1 \boxplus \frac{1}{2}(\theta_2 \boxminus \theta_1)$ (purple, right). Default: 179° and
  −179°, arithmetic mean pointing due backwards. Misconception killed: *angles are just numbers.*
  Caption forward-references w7.2, where the same disease breaks a filter.
- **Widget w3.4: Banana Preview** — type: interactive sim. Draw 500 seeded samples from a
  tangent-space Gaussian $\mathcal{N}(0, \operatorname{diag}(\sigma_v^2, \sigma_v^2, \sigma_\omega^2))$
  at a commanded arc $\boldsymbol{\tau}_0$, push each through
  $x_0 \boxplus (\boldsymbol{\tau}_0 + \delta)$, and scatter the resulting positions: a banana.
  Overlay (toggle): the naive vector-space Gaussian ellipse in $(x, y)$ — visibly the wrong
  shape. One slider: $\sigma_\omega$. Misconception killed: *pose uncertainty is an ellipse.*
  Caption: "Ch. 9 derives this banana; Ch. 7 teaches filters to respect it."

Layout: w3.3 opens the chapter (hook); w3.1 sits in the frames section with the Derivation 2
cancellation rule beside it, colors matched; w3.2 spans the exp/log section; w3.4 closes the
chapter as the bridge to Part II/III.

## 5. Practical (P) — Rust Implementation

- **Crates:** `nalgebra` 0.35 (`UnitComplex`, `UnitQuaternion`, `Isometry2/3` as the standard
  types we interoperate with; `SMatrix` for adjoints/Jacobians); `sophus` (pinned minor version —
  its API is self-declared unstable) for SO(3)/SE(3) exp/log and as the 3D cross-check oracle;
  `rand`/`rand_distr` (property-test inputs, banana sampling); `proptest` (dev-dependency:
  property-based tests); `plotters` (fallback figures).
- **Module plan:** `crates/pr-core/src/geom/{so2.rs, se2.rs, frames.rs, manifold.rs}` — the
  second deposit into `pr-core`; `demos/ch03-geometry/` (bins `w3-1-frame-composer`,
  `w3-2-exp-log-lens`, `w3-3-wrap-trap`, `w3-4-banana-preview`).

Key types & signatures:

```rust
// crates/pr-core/src/geom/se2.rs
use nalgebra::{Point2, SMatrix, SVector, UnitComplex, Vector2};

/// Tangent coordinates τ = (v_x, v_y, ω)ᵀ — translation-first (Sola order).
pub type Tangent2 = SVector<f64, 3>;

/// Hand-rolled SE(2): rotation stored as UnitComplex (no angle wrap bugs), translation as Vector2.
/// This type is used by every filter and SLAM system in the book.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SE2 {
    pub rot: UnitComplex<f64>,
    pub trans: Vector2<f64>,
}

impl SE2 {
    pub const IDENTITY: Self;
    pub fn new(x: f64, y: f64, theta: f64) -> Self;
    pub fn xytheta(&self) -> (f64, f64, f64); // for display only — never do math on the tuple
    pub fn exp(tau: &Tangent2) -> Self;       // Derivation 4; Taylor guard |θ| < 1e-4
    pub fn log(&self) -> Tangent2;
    pub fn inverse(&self) -> Self;            // (Rᵀ, −Rᵀ t)
    pub fn adjoint(&self) -> SMatrix<f64, 3, 3>; // Derivation 6
    pub fn act(&self, p: &Point2<f64>) -> Point2<f64>;
}

impl std::ops::Mul for SE2 { type Output = SE2; /* composition */ }

// crates/pr-core/src/geom/manifold.rs
/// The trait Ch. 7's EKF/UKF are generic over. D = tangent dimension.
pub trait Manifold<const D: usize>: Clone {
    fn boxplus(&self, delta: &SVector<f64, D>) -> Self;   // x ⊞ δ = x · exp(δ^)
    fn boxminus(&self, other: &Self) -> SVector<f64, D>;  // self ⊟ other = log(other⁻¹ · self)
}
impl Manifold<3> for SE2 { /* … */ }
impl<const N: usize> Manifold<N> for SVector<f64, N> { /* ⊞ = +, ⊟ = − : vectors are manifolds too */ }

// crates/pr-core/src/geom/frames.rs — compile-time frame safety via zero-sized markers
pub trait Frame {}
pub struct World; pub struct Body; pub struct LidarF; // impl Frame for each
/// Pose<A, B> ≙ ᴬ_BT — "A from B". Newtype over SE2, zero runtime cost.
pub struct Pose<A: Frame, B: Frame>(pub SE2, PhantomData<(A, B)>);
impl<A: Frame, B: Frame, C: Frame> Mul<Pose<B, C>> for Pose<A, B> {
    type Output = Pose<A, C>; // indices cancel — Craig's rule, enforced by rustc
}
impl<A: Frame, B: Frame> Pose<A, B> { pub fn inverse(self) -> Pose<B, A>; }
```

The chapter prints the deliberate compile error — `world_from_body * world_from_lidar` fails with
"expected `Pose<Body, _>`, found `Pose<World, LidarF>`" — and the prose says: *Craig invented this
type system in 1986 notation; rustc enforces it.* (`sim` and later chapters use bare `SE2` for
state and `Pose<_,_>` at subsystem boundaries — sensor extrinsics, world anchoring — a stated
convention.)

**Property-test suite** (in `pr-core/tests/geom.rs`, shown in the chapter as the executable form
of the derivations): `exp_log_roundtrip` (‖log(exp(τ)) − τ‖ < 1e-12 for ‖θ‖ < π);
`group_axioms` (associativity, identity, inverse); `adjoint_conjugation`
($T\exp(\tau)T^{-1} = \exp(\operatorname{Ad}_T\tau)$); `boxplus_axioms` (Hertzberg);
`so3_matches_sophus` (our nalgebra-based axis-angle vs. `sophus` `Isometry3`/`Rotation3` exp/log
at 1 000 random twists, incl. near-$\pi$ edge cases).

**Worked end-to-end example** (`cargo run -p pr-core --example square_dance`): (i) the quarter-arc
micro-example — $\mathrm{SE2::exp}((1, 0, \tfrac{\pi}{2})^\top)$ yields
$(x, y, \theta) = (\tfrac{2}{\pi}, \tfrac{2}{\pi}, \tfrac{\pi}{2}) \approx (0.63662, 0.63662, 1.57080)$
— printed and locked by `worked_example_ch03` to 1e-12 (this is the chapter's numeric worked
example per the Ch. 2 convention); (ii) Rusty traverses a unit square via four
`exp((1,0,0)) ∘ exp((0,0,π/2))` compositions; the closure error
$\| \text{final} \boxminus \mathrm{SE2::IDENTITY} \|$ prints as ≈ 1e-15; (iii) the same square
with naive $(x, y, \theta)$-tuple arithmetic and a wrap bug planted at $\theta = \pi$ — closure
error 0.04 rad — the before/after that justifies the whole chapter.

**Runnable artifact:** the example above; WASM widgets w3.1–w3.4. The demo w3.2 calls the very
`SE2::exp` the reader implemented; its arc *is* Derivation 4 executing.

## 6. Simulations & Animations Manifest

| id | title | type | crate stack | interaction | teaches |
|---|---|---|---|---|---|
| w3.1 | Frame Composer | wasm-sim | eframe 0.35 + pr-core + widget-kit | drag frames & point; fixed/current toggle | superscript notation as type system; non-commutativity; cancellation rule |
| w3.2 | Exp/Log Lens | wasm-sim + canned 3D animation | eframe + pr-core + widget-kit (2D); three-d + sophus (3D tab, pre-rendered) | drag τ arrow, ω slider, naive-lerp toggle | exp = twist → screw motion; geodesic vs. componentwise interpolation |
| w3.3 | The Wrap-Around Trap | wasm-sim | eframe + pr-core + widget-kit | drag two heading needles | angle averaging fails; ⊟ then ⊞ fixes it |
| w3.4 | Banana Preview | wasm-sim | eframe + egui_plot + pr-core + widget-kit | σ_ω slider, Gaussian-overlay toggle, seed reroll | tangent Gaussians bend into bananas; vector-space ellipses lie |
| — | per-widget fallback SVGs | static-svg | plotters, CI | none | static fallback discipline |

## 7. Exercises & Extensions

1. **(F)** Derive $V(\theta)$ of Derivation 4 by explicitly evaluating
   $\int_0^1 R(s\theta)\,ds$, and derive $V(\theta)^{-1}$ in closed form. Show both limits as
   $\theta \to 0$ recover $I$.
2. **(F)** Prove the SE(2) adjoint formula of Derivation 6 by block computation, and verify that
   $\operatorname{Ad}_{T_1 T_2} = \operatorname{Ad}_{T_1}\operatorname{Ad}_{T_2}$.
3. **(F)** Show that right-⊞ increments are body-frame quantities: if $x' = x \boxplus \delta$,
   the world-frame displacement is $R_x \delta_{v}$. Conclude why odometry naturally produces
   right increments.
4. **(C — predict, then verify with w3.2)** Two poses differ by a pure 180° rotation. Predict what
   the geodesic interpolation does at $s = 0.5$ and why $\log$ is ill-defined there; verify with
   the widget (it flags the injectivity boundary). Relate to Derivation 7's $|\theta| < \pi$
   condition.
5. **(C — w3.4)** Predict how the banana changes if $\sigma_\omega \to 0$ with $\sigma_v$ fixed.
   Verify, and explain in one sentence when the Gaussian overlay is *approximately* honest (this
   is the EKF's bet, named in Ch. 7).
6. **(P)** Implement `SO3` as a newtype over `UnitQuaternion` with `exp/log/boxplus/boxminus`
   under the `Manifold` trait; handle $\theta \approx \pi$ with the numerically stable branch;
   property-test against `sophus` and submit the near-$\pi$ round-trip error plot (plotters
   scaffold provided).

## 8. Modernization Notes

- **What the baselines lack, jointly:** all three geometry baselines are deterministic and
  manipulator-centric. Craig (2005 ed. of a 1986 design) has the best frame *pedagogy* but no
  exponential map at all; Spong has exponential coordinates but uses them lightly; Lynch–Park
  (2017) is the modern screw-theoretic treatment but never puts a distribution on a pose and has
  no ⊞/⊟. The chapter's reconciliation: **Craig's notation** (as the human-readable type system)
  + **Lynch–Park's exponential coordinates** (as the computational engine) + **Sola's micro Lie
  theory** (as the estimation-facing interface) — one coherent story none of the baselines tells.
- **Deliberate deviations, recorded:** tangent order is translation-first $(v_x, v_y, \omega)$
  per Sola, *not* Lynch–Park's $(\omega, v)$ — chosen to match the pose tuple $(x, y, \theta)$
  the reader already holds and factrs/sophus conventions; the symbol $\xi$ for twists (Lynch–Park)
  is avoided because the TOC reserves $\xi$ for the information vector; right-⊞ is fixed
  book-wide (body-frame increments) with the left alternative relegated to a collapsible.
- **Dropped from the baselines and why:** D–H parameters and manipulator kinematic chains (Rusty
  has no arm; Appendix pointer to Craig Ch. 3/Lynch–Park Ch. 4 and the `k`/`urdf-rs` crates);
  wrenches/statics (Lynch–Park §3.4 — no force control in this book); the 24 Euler-angle
  conventions (Craig App. B) compressed to one warning box; Cayley–Rodrigues parameters.
- **Added beyond all baselines:** uncertainty on manifolds (banana preview), the `Manifold` trait
  as the book's estimation interface, property-based testing as the arbiter of sign conventions,
  and compile-time frame safety — the Rust type system as the modern form of Craig's notation.
