# Authoring a chapter

How to turn a `Chapter-NN.md` design doc (in the repo root, one level up) into a chapter of the
web book. Read this in full before writing MDX.

## Where things live

| Path | What |
|---|---|
| `content/chapters/chNN-slug.mdx` | The chapter itself. One file. |
| `components/ch/chNN/*.tsx` | That chapter's interactive widgets. Client components. |
| `lib/` | Shared algorithm library — the simulations run **real implementations**, not fakes. |
| `components/book/*` | Prose components (Overview, Derivation, Algorithm, Exercises, References…). |
| `components/viz/*` | Nivo charts + dashboard shell. |
| `components/sim/*` | `WidgetFrame`, `SimCanvas`, `Transport`, `Slider`, `ControlPanel`. |
| `lib/book-structure.ts` | Canonical chapter numbers, slugs, titles. Never invent a slug. |

## Frontmatter (required)

```yaml
---
title: The Bayes Filter
description: One sentence, shown in search results and on the chapter card.
chapter: 5
part: PART II
partTitle: The Bayes Filter Family
difficulty: Foundational      # Foundational | Intermediate | Advanced
readingTime: 45 min
quote: The robot's belief is not a location. It is a distribution over locations.
quoteAuthor: Sebastian Thrun
quoteSource: Probabilistic Robotics (2005)
---
```

The quote is **required** and must be a real, verifiable statement by a genuine robotics
researcher, with the real source. Paraphrase only if you mark it as a paraphrase. Never invent a
quotation and never attribute an invented line to a real person.

## Chapter skeleton

Follow this order. It is the FCP rhythm: hook → conceptual → foundation → practical → exercises.

```mdx
<Overview goals={[...]} prerequisites={[...]}>
Two or three paragraphs: what this chapter is for, and why the reader should care.
</Overview>

## The problem                      ← the hook, with the first widget
## Building intuition               ← Conceptual: widgets first, math named but not yet derived
## The mathematics                  ← Foundation: definitions, theorems, derivations
## The algorithm                    ← <Algorithm> box
## Implementation in Rust           ← Practical
## Putting it together              ← the integration lab / dashboard
## Exercises
## References
```

## Math

KaTeX with `$…$` and `$$…$$`. Global macros are defined in `lib/katex-macros.ts` — use them:
`\bel`, `\belbar`, `\Normal`, `\E`, `\SEtwo`, `\bplus`, `\bminus`, `\T`, `\norm{}`, `\mat{}`.

Color-code equation terms so they match the figures:

```latex
$$
\belbar(x_t) = \int \htmlClass{term-prediction}{p(x_t \mid u_t, x_{t-1})}\,
               \htmlClass{term-prior}{\bel(x_{t-1})}\, dx_{t-1}
$$
```

Available classes: `term-prior`, `term-prediction`, `term-measurement`, `term-posterior`,
`term-truth`.

Note: JSX *attribute strings* never pass through remark-math, so `title="$x$"` renders literally.
Props that may need math (`Derivation`'s `result`, `Exercise`'s `hint` and `solution`) accept a
node — but the simplest approach is to keep math in the component's children, where MDX processes
it normally.

Long algebra goes in a `<Derivation>`:

```mdx
<Derivation title="Deriving the Kalman gain" result="K_t = \Sigma_t H_t^\mathsf{T} S_t^{-1}">
Step-by-step algebra here. The reader can skip this entirely on a first pass.
</Derivation>
```

## Code

Rust only, in fenced blocks with a title and (optionally) highlighted lines:

````mdx
```rust title="crates/pr-core/src/filters/bayes.rs" {4-7}
pub trait BayesFilter {
    type Control;
    type Measurement;
    fn predict(&mut self, u: &Self::Control);
    fn correct(&mut self, z: &Self::Measurement);
}
```
````

Rules for code:
- It must be real, compilable-in-spirit Rust using the book's crate stack (`nalgebra` 0.35,
  `rand` 0.9, `faer` 0.24, `parry2d` 0.30, `factrs` 0.3, `petgraph` 0.8).
- Show types. `SVector<f64, 3>`, not `Vec<f64>`, when the dimension is fixed.
- Comment the *why*, never the *what*.
- Every chapter has at least three substantial listings: the core type, the algorithm, and a
  worked example with its expected output.

## Widgets

Every widget is a client component in `components/ch/chNN/`, wrapped in `WidgetFrame`, using
`SimCanvas` + `useSimulation` for animation and the real algorithms from `lib/`.

```tsx
'use client';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { SimCanvas } from '@/components/sim/sim-canvas';
import { Transport, Slider, ControlPanel } from '@/components/sim/controls';
import { useSimulation } from '@/lib/sim/use-simulation';

export function HallwayBeliefMachine() {
  const sim = useSimulation<State>({ init, step, fps: 12 });
  return (
    <WidgetFrame
      id="w5.1"
      title="Hallway Belief Machine"
      teaches="Sensing sharpens the belief; moving smears it."
      colorKey={['prior', 'prediction', 'measurement', 'posterior', 'truth']}
      caption={<>What to notice, and what to try changing.</>}
    >
      <SimCanvas world={...} draw={...} deps={[sim.tick]} ariaLabel="..." />
      <ControlPanel>
        <Slider label="Sensor noise σ" role="measurement" value={...} min={0.01} max={1} onChange={...} />
      </ControlPanel>
      <Transport {...sim} onToggle={sim.toggle} onStep={sim.stepOnce} onReset={sim.reset} onReseed={sim.reseed} />
    </WidgetFrame>
  );
}
```

Then import it at the top of the MDX file and place it in the prose.

Widget rules, in priority order:

1. **Autoplay a sensible default.** The reader must learn something without touching anything.
2. **One idea per widget.** If it needs a paragraph to explain the controls, split it.
3. **Foreground one parameter.** Others can exist, but one slider is the point of the widget.
4. **Use the book color code**, always via `var(--pr-*)` — never a literal hex, never a color that
   only reads in one theme.
5. **Name the misconception it kills** in `teaches`.
6. IDs come from the chapter design doc (`w5.1`, `w5.2`, …) and must match the design's manifest.

## Dashboards

For chapters where the point is *monitoring* an algorithm rather than watching a scene, use the
dashboard components: `Dashboard`, `DashboardPanel`, `StatTile`, and the Nivo charts
(`LineChart`, `BarChart`, `HeatMap`, `ScatterChart`, `NetworkGraph`). Series accept a
`role` prop that applies the book color automatically.

Good dashboard candidates: filter innovation/NEES over time (Ch. 6, 11, 14), particle effective
sample size (Ch. 8, 12), map entropy during exploration (Ch. 13, 24), optimizer cost per
iteration (Ch. 15, 16).

## Exercises

Three to six per chapter, spanning the three passes, using the `level` prop:

```mdx
<Exercises>
  <Exercise level="F" difficulty={2} title="Marginalize the joint">
    Body. Math is fine here.
    <details>…</details>
  </Exercise>
</Exercises>
```

`F` = derive something. `C` = predict what a widget will do, then check. `P` = write Rust.

## References

Every chapter ends with a real bibliography — a mix of the foundational papers and **recent**
work (2020–2026). Verify each one exists before citing it: authors, year, title, venue, and a URL
or DOI. Add a `note` saying why the reference matters.

```mdx
<References>
  <Reference
    authors="Dellaert, F. and Kaess, M."
    year={2017}
    title="Factor Graphs for Robot Perception"
    venue="Foundations and Trends in Robotics 6(1–2)"
    doi="10.1561/2300000043"
    note="The standard modern treatment; Chapter 15 follows its formulation."
  />
</References>
```

**Never cite a paper you have not verified exists.** Fabricated citations are the one failure
mode that would discredit the whole book.

## Voice

Rigorous but narrative-first. Concrete before abstract. Say what breaks and when. Prefer the
active voice and a real example over a general claim. The reader is smart and busy: never pad,
never hedge, and never write "as we saw earlier" without a link.
