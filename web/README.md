# Probabilistic Robotics via Rust — the web book

The interactive book itself: a statically-exported Next.js site where every chapter carries live
simulations, and the algorithms in those simulations are the ones the chapter teaches.

## Running it

Requires **Node ≥ 20.9** (Next.js 16). If your system Node is older, a local install works fine:

```sh
curl -sSL https://nodejs.org/dist/v24.19.0/node-v24.19.0-linux-x64.tar.xz \
  | tar -xJ -C ~/.local/node --strip-components=1
export PATH="$HOME/.local/node/bin:$PATH"
```

Then:

```sh
npm install
npm run dev        # http://localhost:3000
npm run build      # typecheck + static export to out/
npm run check      # numerical self-checks of the algorithm library
npm run typecheck
```

After adding or renaming a chapter file, regenerate the content index with `npx fumadocs-mdx`
(the `postinstall` hook also does this).

## How it fits together

```
app/                    routes: landing page, /chapters/[[...slug]], /notation
content/chapters/       one MDX file per chapter + meta.json for sidebar order
components/
  book/                 prose components (Overview, Derivation, Algorithm, Exercises, References…)
  sim/                  WidgetFrame, SimCanvas, transport + parameter controls
  viz/                  Nivo charts, dashboard shell, stat tiles
  ch/chNN/              each chapter's interactive widgets
lib/
  prob/ geom/ models/   the algorithm library the simulations run
  filters/ mapping/ sim/
  __checks__.ts         numerical invariants for all of the above
```

**The simulations are not mock-ups.** `lib/` is a faithful TypeScript port of the Rust the book
prints, and `lib/__checks__.ts` pins it with invariants the mathematics guarantees — exp/log
round-trips on SE(2), a hand-computed Kalman update, beam-model normalization, the entropy
behaviour of prediction versus correction. `npm run check` runs them; CI fails if any breaks.

## Design constraints worth knowing before editing

- **The four estimation colors are reserved for data.** Blue is prior, orange is prediction, green
  is measurement, purple is posterior, gray is ground truth — in prose, in equations
  (`\htmlClass{term-prior}{…}`), in figures, and in widget UI. Chrome uses teal, deliberately
  outside those hues. Always reference them as `var(--pr-*)`, never a literal hex, so both themes
  work.
- **Everything stochastic is seeded.** `lib/prob/rng.ts`, never `Math.random()`, and the seed is
  shown in the widget's control bar so a reader can reproduce or re-roll a run.
- **Widgets autoplay.** Interaction is an invitation, not a requirement: a reader who touches
  nothing should still learn the point.
- **KaTeX is pre-rendered** at build time with a frozen global macro table (`lib/katex-macros.ts`),
  and `katex` is pinned via `overrides` — a version skew between the renderer and the stylesheet
  silently breaks every equation on the site.

Writing a chapter: see [AUTHORING.md](./AUTHORING.md). The design specs live in `../Chapter-NN.md`,
and `../TOC.md` is the book's contract.
