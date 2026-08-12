# Chapter 19 — Modern Map Representations

> Part V — Mapping and SLAM · Estimated length: 12 web pages · Difficulty: Intermediate

## 1. Purpose & Story Arc

Part V has produced trajectories; this chapter asks what the *map* should be. Chapter 13's flat
occupancy grid answers one question ("is this cell occupied?") at one resolution, with memory that
scales like the floor area whether or not anything interesting is there. This chapter walks the
modern menagerie — octrees, TSDFs, ESDFs, meshes, and radiance fields — and lands one unifying
punchline: **every serious map representation is a recursive estimator wearing a costume.** The
octree runs Ch. 8's binary Bayes filter per node; TSDF fusion's weighted running average *is* a
per-voxel one-dimensional Kalman/information filter for a static state; even NeRF/3DGS mapping is
$\arg\max_m p(z \mid m, x)$ with a differentiable renderer as the measurement model. The reader
leaves knowing that choosing a representation means choosing which *query* — occupancy, distance,
surface, appearance — gets to be $O(1)$, and carries an ESDF forward as the cost substrate that
Ch. 20's planners and Ch. 23's MPPI will consume.

Story line:
1. Hook: the same Apartment, four maps, four memory/query price tags — the Representation Gallery teaser.
2. Beyond flat grids: hierarchy (octrees/OctoMap) — the Ch. 8/13 log-odds filter, now with pruning and clamping.
3. Distance, not occupancy: SDF → truncation → TSDF fusion derived as recursive least squares (the Bayes filter in disguise).
4. From TSDF to surfaces (marching squares/cubes) and to planning fields (ESDF, gradients).
5. The measured comparison: memory, update cost, query cost, on one log — no representation wins everywhere.
6. Frontier section, done principledly: differentiable rendering as a measurement model (NeRF/3DGS), with survey pointers and honest caveats.
7. Integration lab: RustSLAM-2D's output re-fused into all four representations; the ESDF handed to Part VI.

## 2. Prerequisites & Position

- Builds on: Ch. 8 §binary Bayes filter (log odds — the octree's engine), Ch. 10 (beam geometry,
  ray casting), Ch. 13 (`OccGrid`, inverse sensor models, the independence caveat — inherited
  per-voxel here), Ch. 15 (weighted least squares as MAP), Ch. 16 (RustSLAM-2D poses feed every
  fusion run), Ch. 18 (posed keyframes; rendering-as-measurement closes that loop).
- Feeds into: Ch. 20 (planning on costmaps/ESDFs), Ch. 23 (MPPI obstacle costs from this
  chapter's ESDF — explicit contract in the TOC), Ch. 24 (exploration frontiers over hierarchical
  maps), Ch. 25 (learned/differentiable map components), Ch. 26 (capstone).
- Baseline sources: Thrun et al. (1999–2000 draft) Ch. 9 §9.1–9.4 (occupancy mapping — the
  ancestor and the recursive-estimation template; `occupancy_grid_mapping`, Table 9.1) and draft
  Ch. 4 §4.1.4 (binary Bayes filter, Table 4.2). Everything else from the modernization set:
  OctoMap (Hornung et al. 2013), KinectFusion (Newcombe et al. 2011) for TSDF fusion, Voxblox
  (Oleynikova et al. 2017) for ESDF-from-TSDF and weighting variants, marching cubes (Lorensen &
  Cline), distance transforms (Felzenszwalb & Huttenlocher), and Tosi et al. (arXiv:2402.13255)
  as the NeRF/3DGS-SLAM survey anchor; NeRF (Mildenhall et al. 2020), 3DGS (Kerbl et al. 2023).

## 3. Foundation (F) — Mathematical Core

### Definitions & notation introduced

| Symbol | Meaning |
|---|---|
| $\ell_{t,i}$, $[\ell_{\min}, \ell_{\max}]$ | per-node log odds (Ch. 8) with OctoMap clamping bounds |
| $\phi(\mathbf{x})$ | signed distance to the nearest surface (negative inside obstacles) |
| $\tau$ | truncation distance; $\Phi(\mathbf{x}) = \mathrm{clamp}(\phi(\mathbf{x}), -\tau, \tau)$ |
| $D_t(v), W_t(v)$ | TSDF value and accumulated weight of voxel $v$ at time $t$ |
| $d_t(v), w_t(v)$ | per-scan projective distance observation of $v$ and its weight |
| $W_{\max}$ | weight clamp — the fading-memory knob |
| $\mathrm{esdf}(v)$ | Euclidean distance from $v$ to the extracted surface/occupied set |
| $F_\theta(\mathbf{x}, \mathbf{d}) \to (\mathbf{c}, \sigma)$ | radiance field: color + density queried at point $\mathbf{x}$, view direction $\mathbf{d}$ |

### Derivations

1. **The octree is the Ch. 8 filter plus a compression theorem.**
   *Statement:* per-leaf updates $\ell \leftarrow \mathrm{clamp}(\ell + \mathrm{inv\_sensor}(z_t, x_t) - \ell_0,\; \ell_{\min}, \ell_{\max})$
   (draft Tables 4.2/9.1, unchanged); a parent whose children all sit at a clamp bound may prune
   them losslessly *with respect to the clamped filter*; expected memory for an indoor world is
   $O(\text{surface})$ not $O(\text{volume})$.
   *Sketch:* clamping bounds the filter's state space → equal saturated children carry no extra
   information → prune; free/unknown space dominates volume and collapses to shallow nodes;
   queries descend $O(\text{depth})$; multi-resolution queries read inner nodes as max/mean of leaves.
   *Collapsible:* why clamping also fixes the Ch. 13 "stubborn map" problem (bounded evidence ⇒
   bounded forgetting time for dynamic changes), with the update-count arithmetic; octree
   arithmetic (Morton codes) as an implementation note.
2. **TSDF fusion = per-voxel recursive least squares = a Bayes filter in disguise.** *(The chapter's centerpiece.)*
   *Statement:* model each scan's projective distance as $d_t(v) = D^\ast(v) + \epsilon_t$,
   $\epsilon_t \sim \mathcal{N}(0, \sigma^2 / w_t(v))$. The MAP estimate after $t$ scans is the weighted mean, and it
   admits the recursion
   $$D_t = \frac{W_{t-1} D_{t-1} + w_t\, d_t}{W_{t-1} + w_t}, \qquad W_t = \min(W_{t-1} + w_t,\; W_{\max}),$$
   which is exactly a scalar information filter for a static state: $W$ is accumulated information
   (inverse variance, Ch. 6's $\Omega$), the update is precision-weighted averaging (Ch. 6's Kalman
   gain $K = w_t / (W_{t-1} + w_t)$), and the clamp turns it into a fading-memory filter.
   *Sketch:* (i) Gaussian likelihoods multiply → weighted least squares; (ii) rewrite the batch
   weighted mean recursively; (iii) identify the gain and match it, symbol for symbol, to the 1D
   Kalman update of Ch. 6; (iv) show clamping ≡ exponential forgetting with rate $w_t / W_{\max}$;
   (v) note what the Gaussian model hides — projective distance is a *biased* estimate of true
   signed distance away from the surface, which is exactly why $\tau$ truncates the domain to where
   the approximation holds.
   *Collapsible:* the bias analysis (grazing-angle geometry, projective vs true distance), Voxblox's
   weight choices ($w \propto 1/z^2$, angle terms) as variance modeling, and the honest caveat that
   per-voxel independence is Ch. 13's independence assumption all over again — stated, not hidden.
3. **Surfaces and distance fields from a TSDF.**
   *Statement:* the estimated surface is the zero level set $\{\mathbf{x} : D(\mathbf{x}) = 0\}$; marching
   squares/cubes extracts it cell-by-cell with linear interpolation along edges, $O(\text{cells})$;
   the ESDF replaces truncated projective distance with true Euclidean distance via a two-pass
   distance transform, $O(n)$ total (Felzenszwalb), and $\nabla \mathrm{esdf}$ gives the collision-avoidance
   gradients planners want.
   *Sketch:* sign changes on cell edges ⇒ crossing; interpolation is exact for the piecewise-linear
   model; case tables (16 in 2D); for the ESDF: lower-envelope-of-parabolas argument in 1D, then
   row/column separability in 2D.
   *Collapsible:* the 2D case table drawn out; ambiguity cases; proof of the distance transform's
   correctness and its exact $O(n)$ bound; why ESDF ≠ TSDF (truncation, projective bias, and
   free-space coverage) with a side-by-side figure spec.
4. **Representation economics (a theorem-flavored comparison, measured in w19.2).**
   *Statement:* for world diameter $L$, resolution $r$, surface measure $S$: flat grid memory
   $\Theta((L/r)^d)$, octree $O((S/r^{d-1}) \log(L/r))$; occupancy query grid $O(1)$ vs octree
   $O(\log(L/r))$; distance query: ESDF $O(1)$ vs mesh $O(\log F)$ (BVH) vs grid/octree "not
   answerable without search." No representation dominates — the map is chosen by its query workload.
   *Sketch:* count nodes along the surface per level; geometric series; per-query data-structure
   walkthroughs. Each claim is then *measured live* in w19.2 on the same Apartment log.
5. **Differentiable rendering as a measurement model (the one principled frontier section).**
   *Statement:* volume rendering $\hat{C}(\mathbf{r}) = \int_0^\infty T(s)\, \sigma(\mathbf{r}(s))\, \mathbf{c}(\mathbf{r}(s), \mathbf{d})\, ds$ with
   $T(s) = \exp(-\int_0^s \sigma)$ defines a *generative model of images*; with Gaussian photometric
   noise, $-\log p(I_t \mid \theta, T_{cw,t}) \propto \| I_t - \hat{I}(\theta, T_{cw,t}) \|^2$, so NeRF-style mapping is MAP
   estimation of map parameters $\theta$ (and poses) under a rendering measurement model — the same
   $\eta\, p(z \mid m, x)$ structure as Ch. 13, optimized by gradients instead of closed-form updates.
   3DGS swaps the field for explicit Gaussian primitives and rasterization for the integral —
   trading the model class for two orders of magnitude in speed.
   *Sketch:* (i) rendering equation discretized (alpha compositing); (ii) photometric residual as
   log-likelihood; (iii) gradient flows through the renderer to $\theta$ — "the inverse sensor model
   is now computed by autodiff"; (iv) pose tracking = the same residual optimized over $T_{cw}$
   (Ch. 16's scan-to-map, with images); (v) caveats: geometry-appearance ambiguity, no closed-form
   uncertainty, compute; survey pointer (Tosi et al.) instead of a system zoo.
   *Collapsible:* discrete alpha-compositing derivation from piecewise-constant density, and one
   worked 1D example (the exact toy w19.4 animates).

### Named algorithms

| Algorithm | Signature | Complexity |
|---|---|---|
| `octree_insert_scan` | $(\text{tree}, x_t, z_t) \to \text{tree}$ | $O(\text{beams} \cdot \frac{\text{range}}{r} \cdot \log\frac{L}{r})$; lossless prune pass amortized |
| `tsdf_integrate` | $(\text{grid}, T_{wc}, \mathcal{P}_t) \to \text{grid}$ | $O(\text{beams} \cdot \tau / r)$ — only the truncation band updates |
| `marching_squares` | $(\text{tsdf}) \to \{\text{segments}\}$ | $O(\text{cells})$, embarrassingly parallel |
| `esdf_from_tsdf` | $(\text{tsdf}) \to \text{field}$ | $O(n)$ two-pass (Felzenszwalb), $n$ = cells |
| `esdf_query` / `esdf_gradient` | $(\text{field}, \mathbf{x}) \to (d, \nabla d)$ | $O(1)$ bilinear — the Ch. 20/23 contract |
| ancestor: `occupancy_grid_mapping` | draft Table 9.1 | the recursion every algorithm above secretly reruns |

## 4. Conceptual (C) — Intuition & Visual Design

- **Widget w19.1: TSDF Sculptor** *(flagship; interactive sim)* — Rusty's depth scans carve a 2D
  TSDF of the Apartment, rendered as a diverging heatmap (**orange** negative/inside through white
  zero to **blue** positive/free, the one place the chapter borrows a diverging ramp), with the
  extracted marching-squares contour drawn in **purple** and ground-truth walls **gray dashed**.
  Autoplays a lap. One meaningful parameter: the $W_{\max}$ slider. A "voxel inspector" hover shows
  the tapped cell's running $(D, W)$ as a tiny 1D filter strip — prior in **blue**, incoming scan
  sample in **green**, fused posterior in **purple** — making Derivation 2 literal. Mid-run, a
  chair is teleported (button): with $W_{\max}$ large its ghost persists; small, it fades fast but
  the map gets noisy. *Misconception killed:* "TSDF fusion is a graphics hack" — the inspector
  shows a Bayes filter running in every voxel, and $W_{\max}$ is its memory, trading stability
  against responsiveness exactly like Ch. 6's noise ratio. Static fallback: carve-sequence
  filmstrip + one inspector inset.
- **Widget w19.2: Representation Gallery** *(flagship; interactive sim, chapter dashboard)* —
  Four synchronized panes of the same Apartment log: flat `OccGrid`, quadtree (2D octree),
  TSDF+contour, extracted mesh/segments. Beneath each, live meters: memory (bytes, measured),
  update time per scan, and the running cost of a stream of "distance to nearest obstacle?"
  queries at random probe points (probes drawn as pins). Controls: resolution stepper
  (2.5/5/10 cm) and a query-workload switch (occupancy probes vs distance probes vs surface
  extraction) that visibly reshuffles which pane is "winning" (its meter turns **green**).
  Autoplays with the workload cycling. *Misconception killed:* "there is a best map" — the ranking
  inverts before the reader's eyes as the query workload changes; representation choice is
  workload choice. Static fallback: the measured comparison table + four thumbnails.
- **Widget w19.3: Wavefront** *(supporting; animation)* — The ESDF distance transform sweeping
  outward from the w19.1 contour as an expanding isoline animation; then $-\nabla \mathrm{esdf}$ quiver
  arrows fade in, and a test particle dropped anywhere slides downhill *away* from obstacles.
  One toggle: ESDF vs raw TSDF values beyond $\tau$ (the TSDF flatlines — visibly useless for
  far-field planning). *Misconception killed:* "TSDF and ESDF are the same thing" — truncation vs
  full Euclidean distance, projective vs true, shown side by side; also previews why Ch. 23's MPPI
  wants this field's gradients.
- **Widget w19.4: Photometric Descent** *(supporting; animation, one slider)* — The 1D toy from
  Derivation 5's collapsible: a 1D density field $\sigma(x)$ (unknown, **orange** estimate vs
  **gray dashed** truth) is optimized so its volume-rendered depths match observations from a few
  1D "cameras" (**green** measurements); the loss curve falls; the estimated density sharpens into
  walls. Slider: observation noise. *Misconception killed:* "NeRF mapping is magic" — it is
  gradient descent on a photometric measurement model, watchable at 1D scale; also honestly shows
  a local-minimum run (seed button) where density parks in the wrong place.

## 5. Practical (P) — Rust Implementation

Crates: `nalgebra` 0.35 (poses, probe math), `ndarray` 0.17 (2D scalar fields for TSDF/ESDF —
the one place the book prefers array semantics over matrix semantics), `parry2d` 0.30 (simulated
depth scans; mesh-distance BVH cross-check in the gallery), `rand`/`rand_distr` 0.9/0.6 (seeded
probes), `egui`/`eframe` 0.35 + `egui_plot` 0.34 (widgets, meters), `three-d` 0.19 (the one 3D
showpiece: the Apartment's TSDF extruded and mesh-extracted, orbitable), `plotters` (static
comparison figures). Reuses `sim` (Ch. 4), `OccGrid` (Ch. 13), RustSLAM-2D logs (Ch. 16).

Module plan: `crates/ch19_maps/` — `quadtree.rs`, `tsdf.rs`, `esdf.rs`, `marching.rs`,
`gallery.rs` (benchmark harness with a `MapRepr` trait), `toy_nerf.rs` (the 1D photometric toy);
demos `demos/ch19-tsdf-sculptor/`, `ch19-gallery/`, `ch19-wavefront/`, `ch19-showpiece3d/`.

```rust
use nalgebra::{Point2, Vector2};

/// One trait, four impls — the gallery's measurement harness and the chapter's thesis in code.
pub trait MapRepr {
    fn integrate_scan(&mut self, pose: &SE2, scan: &Scan);
    fn occupancy(&self, p: Point2<f64>) -> Option<f64>;                 // None = unknown
    fn distance(&self, p: Point2<f64>) -> Option<f64>;                  // None = not supported O(1)
    fn memory_bytes(&self) -> usize;
}

pub struct QuadTree { root: Node, resolution: f64, l_clamp: (f64, f64), l0: f64 }
impl QuadTree {
    pub fn insert_scan(&mut self, pose: &SE2, scan: &Scan);             // Derivation 1: Ch. 8 filter/leaf
    pub fn prune(&mut self) -> usize;                                   // returns nodes freed
    pub fn occupancy_at(&self, p: Point2<f64>, max_depth: Option<u8>) -> f64;
}

pub struct Tsdf2 {
    trunc: f64, resolution: f64,
    d: ndarray::Array2<f32>, w: ndarray::Array2<f32>, w_max: f32,
}
impl Tsdf2 {
    pub fn integrate(&mut self, pose: &SE2, scan: &Scan);               // Derivation 2, verbatim
    pub fn voxel(&self, ij: (usize, usize)) -> (f32, f32);              // (D, W) — powers the inspector
    pub fn surface(&self) -> Vec<[Point2<f64>; 2]>;                     // marching squares segments
}

pub struct Esdf2 { d: ndarray::Array2<f32>, resolution: f64 }
pub fn esdf_from_tsdf(tsdf: &Tsdf2) -> Esdf2;                           // Felzenszwalb two-pass, O(n)
impl Esdf2 {
    pub fn distance(&self, p: Point2<f64>) -> f64;                      // O(1) bilinear — Ch. 20/23 contract
    pub fn gradient(&self, p: Point2<f64>) -> Vector2<f64>;
}
```

Worked end-to-end example (`cargo run --example representation_gallery`): replay the Ch. 16
RustSLAM-2D Apartment log (seed `0xC0FFEE`, poses from the optimized graph) into all four
`MapRepr` impls at 5 cm. Expected (unit-tested) output table — flat grid ≈ 43 kB / occupancy query
≈ 20 ns / no $O(1)$ distance; quadtree ≈ 8 kB after pruning / ≈ 85 ns; TSDF ≈ 340 kB (f32 $D{+}W$)
/ contour of ≈ 1.9 k segments; ESDF adds ≈ 170 kB and answers distance in $O(1)$, matching
`parry2d`'s BVH mesh-distance to < 1 cell everywhere. The example emits the chapter's comparison
table (markdown) and figures (SVG), and hands `esdf.bin` to Part VI — the artifact Ch. 23 loads.
The 3D showpiece (`ch19-showpiece3d`, `three-d`) extrudes the TSDF to 2.5D and mesh-extracts it,
orbitable in-browser. WASM demos: w19.1–w19.4.

## 6. Simulations & Animations Manifest

| id | title | type | crate stack | interaction | teaches |
|---|---|---|---|---|---|
| w19.1 | TSDF Sculptor | wasm-sim | eframe, egui_plot, ch19_maps, sim | $W_{\max}$ slider; voxel inspector hover; teleport-chair button | TSDF fusion as a per-voxel Bayes filter; memory vs responsiveness |
| w19.2 | Representation Gallery | wasm-sim (dashboard) | eframe, egui_plot, ch19_maps, parry2d | resolution stepper; query-workload switch | representation economics; the map is chosen by its queries |
| w19.3 | Wavefront | animation + toggle | eframe, ch19_maps | ESDF/TSDF toggle; drop test particle | distance transforms; ESDF ≠ TSDF; gradients for planning |
| w19.4 | Photometric Descent | animation + slider | eframe, egui_plot, ch19_maps::toy_nerf | noise slider; reseed button | differentiable rendering as a measurement model, demystified in 1D |
| f19.1 | 3D showpiece: meshed Apartment | wasm-sim (3D) | three-d, ch19_maps | orbit camera | the same estimation, one dimension up |

## 7. Exercises & Extensions

1. **[F]** Prove Derivation 2's equivalence in full: show the TSDF recursion is the scalar information-filter update of Ch. 6 with $\Omega_t = W_t$, and derive the effective forgetting factor $\lambda = 1 - w_t/W_{\max}$ once the clamp is active. What is the steady-state variance under constant re-observation?
2. **[F]** Derive the octree memory bound of Derivation 4 for a rectangular room of perimeter $S$ and diameter $L$, and predict the quadtree/grid memory ratio for the Apartment at 5 cm; check against w19.2's meters.
3. **[C]** In w19.1, predict how many re-observations the teleported chair's ghost survives at $W_{\max} \in \{16, 128\}$ using your exercise-1 forgetting factor; verify with the inspector.
4. **[C]** In w19.2, find a query workload ordering under which each of the four representations is "winning" at some point (green meter). Write the one-sentence workload description for each.
5. **[P]** Implement `MapRepr` for a fifth representation — a point cloud with a k-d tree — and add it to the gallery. Where does it win, and why do production systems still not use it as the primary map?
6. **[P]** Extend `toy_nerf.rs` from depth-only to a two-channel (depth + reflectance) 1D renderer and show that appearance disambiguates a geometry local minimum from exercise seed 7 — the chapter's differentiable-rendering claim, verified at 1D scale.

## 8. Modernization Notes

- The 2005 baseline (and the 1999–2000 draft, Ch. 9) ends at flat occupancy grids; octrees, TSDF,
  ESDF, meshes, and radiance fields are all post-baseline and sourced from the modernization set
  (OctoMap 2013; KinectFusion 2011; Voxblox 2017; Tosi et al. survey for NeRF/3DGS SLAM). The
  chapter's *framing* — every representation as recursive estimation — is this book's contribution
  and is what keeps the new material continuous with the Bayes-filter spine rather than a zoo tour.
- Per the modernization guidance, NeRF/3DGS gets **one principled section plus survey pointers,
  not system worship**: the field churned through dozens of systems in 2024–2026 and canonizing
  any one of them in print would date the book on arrival. The 1D toy (w19.4) teaches the
  mechanism; Tosi et al. carries the reader to the frontier.
- Dropped or compressed: surfel maps (one paragraph; superseded for the book's purposes by TSDF +
  mesh), GPU fusion pipelines (nvblox named as the production path, not taught), 3D octree
  implementation detail (Morton codes an implementation note; the 2D quadtree carries the
  pedagogy), and Ch. 13's MAP-with-forward-models thread is referenced but not re-run here —
  its modern descendant *is* the differentiable-rendering section.
- Honesty note: per-voxel independence — flagged in Ch. 13 as occupancy mapping's structural lie —
  is inherited by OctoMap and TSDF fusion alike, and the text says so where each update rule is
  derived; the correlated-field alternative (GP implicit surfaces) gets a pointer box.
