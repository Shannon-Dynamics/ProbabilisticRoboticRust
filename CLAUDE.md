# CLAUDE.md — Probabilistic Robotics via Rust

Guidance for Claude Code (and human contributors) when developing this project.

## What this project is

An **interactive web book**, *Probabilistic Robotics via Rust*, taught the **FCP way** — every
chapter delivers **F**oundation (full mathematical formalism and derivations), **C**onceptual
(interactive visual dashboards, simulations, and animations in the page), and **P**ractical
(idiomatic Rust with the best current crates). The book modernizes Thrun/Burgard/Fox's
*Probabilistic Robotics* using the reference texts in `Resource/` as baseline, extended with
factor graphs, Lie-group estimation, scan matching, VIO, TSDF mapping, MPPI, modern POMDP
solvers, active SLAM, and learning-in-the-loop.

**The core promise: the Rust code printed in the text is the same code that compiles to WASM and
powers the in-page simulations.** Never break this — no demo may fork away from the library code
it claims to demonstrate.

## Current state & source-of-truth documents

| File | Role |
|---|---|
| `TOC.md` | **The contract.** Vision, part/chapter structure (26 chapters, 7 parts, 4 appendices), book-wide notation table, color code, verified crate stack, publishing architecture. Nothing may contradict it; change TOC.md *first*, then dependents. |
| `Chapter-01.md` … `Chapter-26.md` | Per-chapter design docs: storyline, F/C/P plan, widget manifests (ids `w<chapter>.<k>`), Rust module plans, exercises, modernization notes. These are the specs to implement. |
| `Resource/Fundamental Robotics via Rust/*.pdf` | Baseline reference books. **Note:** the Probabilistic Robotics PDF is the 1999–2000 16-chapter draft, not the 2005 edition — FastSLAM, GraphSLAM (published form), UKF, KLD-sampling, and exploration are *not* in it; the chapter designs already account for this. |
| `CLAUDE.md` | This file: how to build the book. |

The running lab: robot **Rusty** (differential drive), worlds **Hallway** (1D) and **Apartment**
(2D floorplan, ray-cast LiDAR). Use these names and no others.

## Repository layout

The book is a **React / Next.js static site** in `web/`. (An earlier plan used mdBook + Rust→WASM;
the web app supersedes it. The Rust remains the canonical implementation that the book teaches.)

```
web/
  app/                    # Next.js App Router: (home)/, chapters/[[...slug]]/, notation/, global.css
  content/chapters/       # chNN-slug.mdx — one file per chapter + meta.json (sidebar order)
  components/
    book/                 # prose components: Overview, Epigraph, Derivation, Algorithm, Exercises, References…
    sim/                  # WidgetFrame, SimCanvas, Transport/Slider/ControlPanel
    viz/                  # Nivo charts + Dashboard/StatTile
    ch/chNN/              # that chapter's interactive widgets (client components)
    home/                 # landing-page hero
  lib/
    prob/ geom/ sim/ models/ filters/ mapping/   # the TypeScript algorithm port powering the sims
    __checks__.ts         # numerical self-checks for that port
    katex-macros.ts       # the single global KaTeX macro table
    book-structure.ts     # canonical chapter numbers, slugs, titles
  AUTHORING.md            # how to write a chapter — read before authoring
Chapter-01.md … Chapter-26.md, TOC.md    # design docs (repo root) — the specs
Resource/                 # baseline reference PDFs
```

**Two implementations, on purpose.** The Rust in the prose is canonical and teachable. `web/lib/`
is a faithful TypeScript port that runs the in-page simulations, so the reader can compare them
side by side. When you change one, change the other, and make sure the chapter's worked numeric
example still holds in both.

## Toolchain & commands

- **Node ≥ 20.9** (Next.js 16 requires it). This machine's system Node is 18, so a local
  Node 24 LTS lives at `~/.local/node` — prefix commands with
  `export PATH="$HOME/.local/node/bin:$PATH"`.
- Web stack versions are pinned in `web/package.json` and documented in `TOC.md §2`.
- Rust versions for the printed listings: see the crate table in `TOC.md §2`.

```sh
cd web
npm run dev         # local preview
npm run build       # typecheck + static export (must be clean before shipping a chapter)
npm run typecheck   # tsc --noEmit
npm run check       # numerical self-checks of the TypeScript algorithm port
npx fumadocs-mdx    # regenerate the .source content index after adding a chapter file
```

If `next build` reports type errors in files that look already-fixed, clear the stale cache:
`rm -rf .next tsconfig.tsbuildinfo`.

## Non-negotiable conventions

1. **Notation** comes from `TOC.md §2` (Thrun-compatible + $\boxplus/\boxminus$ manifold
   extensions). Each chapter may only *add* symbols via its own notation table.
2. **Color code** everywhere (figures, KaTeX terms, code comments, widget UI):
   prior **blue** · prediction **orange** · measurement **green** · posterior **purple** ·
   ground truth **gray dashed**. On the web these are the CSS custom properties `--pr-prior`,
   `--pr-prediction`, `--pr-measurement`, `--pr-posterior`, `--pr-truth`, defined once in
   `web/app/global.css` and redefined for dark mode. Never hardcode a hex anywhere else, and never
   use a data color for chrome — the chrome accent is teal, deliberately outside the data hues.
   In equations, tint terms with `\htmlClass{term-prior}{…}` and friends.
3. **Determinism:** every stochastic demo/test is seeded, with a visible seed control ("re-roll"
   shows the seed). Rust: `SmallRng`/`Pcg64`, never `thread_rng()`. Web: `lib/prob/rng.ts`,
   never `Math.random()`.
4. **Widgets:** autoplay a sensible default; foreground one meaningful parameter; `teaches` names
   the misconception the widget kills; every widget is wrapped in `WidgetFrame` with a caption
   saying what to notice and what to try; ids `w<chapter>.<k>` must match the design doc's
   manifest table.
5. **Filters are hand-rolled** on nalgebra — that is the book's point. Crates like `adskalman`
   appear only as cross-validation in tests. Conversely, don't hand-roll what the stack
   assigns to a crate (sparse solves → `faer`, production factor graphs → `factrs`, collision/
   ray-cast → `parry2d`).
6. **Ownership of shared artifacts** (introduce once, reuse after): `SE2` type → Ch. 3; `sim` +
   widget framework → Ch. 4; `trait BayesFilter` → Ch. 5; particle machinery → Ch. 8; GN/LM
   optimizer → Ch. 15; RustSLAM-2D → Ch. 16. A later chapter must not re-introduce these. On the
   web side the same rule applies to `web/lib/` — a chapter widget imports from it, never
   reimplements it, and only adds genuinely new algorithms as new files.
7. **Every worked numeric example in the text is reproduced by a test** (Labbe pattern) — a Rust
   unit test in the prose, and an invariant in `web/lib/__checks__.ts` where the port covers it.
   Property-test samplers against closed forms (e.g. KS tests, motion models).
8. **Algorithm names** follow Thrun's table style (`sample_motion_model_velocity`, `MCL`,
   `EKF_SLAM`, …) in both pseudocode and Rust function names where reasonable.
9. Derivations: statement-first, 3–8 named steps in the text, full algebra in a collapsible
   block. Concrete before abstract; theorem-proof ordering only inside Foundation sections.

## Workflow: implementing a chapter

Read `web/AUTHORING.md` first — it is the detailed contract. In short:

1. Read `Chapter-NN.md` (the design doc: storyline, math, widget manifest, Rust plan) and study
   `web/content/chapters/ch05-bayes-filter.mdx` plus its widgets in `web/components/ch/ch05/`,
   which set the quality bar.
2. Library: if the chapter needs an algorithm `web/lib/` lacks, add it as a new file under the
   matching subdirectory, and add an invariant to `lib/__checks__.ts`.
3. Widgets: build the manifest entries as client components in `web/components/ch/chNN/`,
   importing the real algorithms from `web/lib/`.
4. Prose: write `web/content/chapters/chNN-slug.mdx` following the FCP rhythm (Hook → Conceptual
   → Foundation → Practical → integration → Exercises → References), with the epigraph, the
   Rust listings, a hand-checkable worked example, and verified citations.
5. Gates: `npm run build` clean (it type-checks and statically exports), `npm run check` passing,
   widget ids and colors matching the manifest, notation matching `TOC.md`.

**Definition of done** = step 5 plus: cross-chapter links resolve to real slugs, exercises span
F/C/P, **every citation and the epigraph verified to exist**, and the design doc's §8
modernization claims actually reflected in the text.

## Editing the design docs themselves

- Keep the 8-section skeleton used by all `Chapter-NN.md` files (Purpose/Story Arc; Prerequisites
  & Position; Foundation; Conceptual; Practical; Manifest; Exercises; Modernization Notes).
- Never renumber chapters or reuse widget ids; TOC.md and all cross-references must move in the
  same change.
- New crates or version bumps: update `TOC.md §2` first, then affected chapter docs, then this
  file if commands change. Verify WASM compatibility before adopting any crate (factrs/sophus
  WASM builds are CI-tested early for exactly this reason).

## Style

- Voice: rigorous but narrative-first; honest about approximations and failure modes; playful is
  fine, imprecise is not. British/American spelling: American.
- Rust: idiomatic, `clippy`-clean, const generics for dimensions where natural, newtypes for
  frames/units; a deliberate compile-error example is a feature of Practical sections, not a bug.
- No placeholder content in the published book — every widget listed in a manifest exists or the
  chapter doesn't ship.
