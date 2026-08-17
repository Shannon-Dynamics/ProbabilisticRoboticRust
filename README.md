# Probabilistic Robotics via Rust

An interactive book on probabilistic robotics, taught the **FCP way** — rigorous
mathematical **F**oundations, **C**onceptual understanding through simulations you
can manipulate, and **P**ractical implementations in Rust.

Twenty-six chapters take a differential-drive rover called **Rusty** from its
first noisy encoder tick to autonomously exploring and mapping a floorplan it has
never seen: Bayes filters, Kalman and particle filters, motion and sensor models,
localization, occupancy grids, SLAM as sparse least squares, scan matching,
visual-inertial odometry, planning under uncertainty, MPPI, POMDPs, and active
SLAM.

## Two editions, one source

| | |
|---|---|
| **Web** — [`web/`](web/) | Next.js static site. 26 chapters, 99 interactive simulations. |
| **Print** — [`pr-latex/`](pr-latex/) | 749-page PDF, typeset in the Shannon Robotics design language. |

The print edition is *generated* from the web edition, so the two cannot drift
apart. Even the printed descriptions of the interactive figures are extracted
from the React components themselves.

The web edition is published to
**<https://shannon-dynamics.github.io/ProbabilisticRoboticRust/>** from `main` on
every push — see [`web/README.md`](web/README.md#deployment) for how the build is
mounted under that sub-path.

```sh
cd web && npm install && npm run dev     # read it locally
cd web && npm run verify                 # checks + static export
cd pr-latex && make                      # regenerate main.pdf
```

`web` requires Node ≥ 20.9; `pr-latex` requires a TeX distribution with
`newpx`, `tcolorbox`, `fontawesome5` and `listings`.

## The simulations are the algorithms

[`web/lib/`](web/lib/) is a TypeScript port of the Rust the book teaches, and it
is what the in-page simulations actually run — not mock-ups. It is pinned by
numerical invariants the mathematics guarantees: SE(2) exp/log round-trips, a
hand-computed Kalman update, beam-model normalization, ray casting against
closed-form distances, and the theorem that prediction never sharpens a belief.

```sh
cd web
npm run check        # 32 numerical and math-rendering invariants
npm run check:book   # chapters, widget ids, cross-links, citations
```

## Repository layout

```
TOC.md                the book's contract: structure, notation, colour code, crate stack
Chapter-01.md … 26    per-chapter design docs (storyline, maths, widget manifest, Rust plan)
CLAUDE.md             how to develop this project
web/                  the interactive edition
pr-latex/             the print edition and its MDX→LaTeX converter
```

## Reference material

The book is built on six baseline texts — Thrun, Burgard & Fox's *Probabilistic
Robotics*; Lynch & Park's *Modern Robotics*; Craig; Niku; Choset et al.'s
*Principles of Robot Motion*; and Spong, Hutchinson & Vidyasagar — all cited
throughout.

Those PDFs are **not committed**: they are third-party copyrighted works and
about 120 MB together. If you want the tooling that reads them (the converter
verifies epigraphs against the sources), place them in
`Resource/Fundamental Robotics via Rust/`. Nothing in the build requires them.
