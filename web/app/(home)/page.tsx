import Link from 'next/link';
import { HeroLocalization } from '@/components/home/hero-localization';
import { PARTS } from '@/lib/book-structure';

export default function HomePage() {
  return (
    <main className="mx-auto w-full max-w-6xl px-4 py-12 sm:px-6 lg:py-16">
      {/* ---------------------------------------------------------------- */}
      {/* Hero: the thesis, stated and then shown running.                  */}
      {/* ---------------------------------------------------------------- */}
      <section className="grid items-center gap-10 lg:grid-cols-[1.05fr_1fr] lg:gap-14">
        <div>
          <p className="eyebrow mb-4">An interactive book · Foundation · Conceptual · Practical</p>
          <h1 className="font-display text-4xl leading-[1.08] font-semibold tracking-tight sm:text-5xl">
            Probabilistic Robotics
            <span className="block text-fd-muted-foreground">via Rust</span>
          </h1>
          <p className="mt-5 max-w-xl font-prose text-lg leading-relaxed text-fd-foreground/85">
            A robot never knows exactly where it is. This book takes that fact seriously: it
            derives the mathematics of reasoning under uncertainty in full, makes every hard idea
            something you can <em>play with</em> in the page, and implements all of it in Rust.
          </p>
          <p className="mt-4 max-w-xl font-prose text-base leading-relaxed text-fd-muted-foreground">
            Twenty-six chapters take a differential-drive rover called Rusty from its first noisy
            encoder tick to autonomously exploring and mapping a floorplan it has never seen —
            through Bayes filters, Kalman and particle filters, SLAM as least squares, planning
            under uncertainty, and modern factor-graph estimation.
          </p>

          <div className="mt-7 flex flex-wrap items-center gap-3">
            <Link
              href="/chapters"
              className="rounded-md bg-fd-primary px-4 py-2 font-ui text-sm font-medium text-fd-primary-foreground transition-opacity hover:opacity-90"
            >
              Start reading
            </Link>
            <Link
              href="/chapters/ch05-bayes-filter"
              className="rounded-md border border-fd-border px-4 py-2 font-ui text-sm font-medium transition-colors hover:bg-fd-accent"
            >
              Jump to the Bayes filter
            </Link>
          </div>
        </div>

        <div>
          <HeroLocalization />
          <p className="mt-2 font-ui text-xs leading-relaxed text-fd-muted-foreground">
            <strong className="font-semibold text-fd-foreground">Monte Carlo localization,
            running live.</strong>{' '}
            Nine hundred hypotheses start spread across the whole map. Each one moves with the
            robot and is weighted by how well its predicted range reading matches the real one.
            Within a few seconds, the cloud has found the robot — that collapse is what this book
            is about. Chapter 12 builds it properly.
          </p>
        </div>
      </section>

      {/* ---------------------------------------------------------------- */}
      {/* The method                                                        */}
      {/* ---------------------------------------------------------------- */}
      <section className="mt-20 border-t border-fd-border pt-10">
        <p className="eyebrow">The method</p>
        <h2 className="mt-1 font-display text-2xl font-semibold">Three passes over every idea</h2>
        <div className="mt-6 grid gap-px overflow-hidden rounded-lg border border-fd-border bg-fd-border sm:grid-cols-3">
          {[
            {
              k: 'F',
              t: 'Foundation',
              d: 'The full mathematics: definitions, assumptions stated out loud, and derivations carried through — not gestured at. Long algebra folds away for readers who want the result first.',
            },
            {
              k: 'C',
              t: 'Conceptual',
              d: 'Every hard idea becomes something you can manipulate. Drag the robot, turn up the noise, scrub through time, re-roll the seed, and watch what the equations were trying to tell you.',
            },
            {
              k: 'P',
              t: 'Practical',
              d: 'Then you build it in Rust, with the crates the field actually uses — nalgebra, faer, factrs, parry — and code you could lift into a real robot.',
            },
          ].map((c) => (
            <div key={c.k} className="bg-fd-card p-5">
              <div className="flex items-baseline gap-2">
                <span className="font-mono text-sm font-semibold text-fd-primary">{c.k}</span>
                <h3 className="font-display text-base font-semibold">{c.t}</h3>
              </div>
              <p className="mt-2 font-prose text-sm leading-relaxed text-fd-muted-foreground">
                {c.d}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* ---------------------------------------------------------------- */}
      {/* Contents                                                          */}
      {/* ---------------------------------------------------------------- */}
      <section className="mt-16">
        <p className="eyebrow">Contents</p>
        <h2 className="mt-1 font-display text-2xl font-semibold">Seven parts, twenty-six chapters</h2>
        <div className="mt-6 space-y-8">
          {PARTS.map((part) => (
            <div key={part.id}>
              <div className="flex items-baseline gap-3 border-b border-fd-border pb-1.5">
                <span className="font-mono text-[0.7rem] tracking-widest text-fd-primary">
                  {part.id}
                </span>
                <h3 className="font-display text-base font-semibold">{part.title}</h3>
              </div>
              <ul className="mt-2">
                {part.chapters.map((ch) => (
                  <li key={ch.n} className="border-b border-fd-border/60 last:border-0">
                    <Link
                      href={`/chapters/${ch.slug}`}
                      className="group grid grid-cols-[2rem_1fr] gap-3 py-2 transition-colors hover:bg-fd-accent/40"
                    >
                      <span className="pt-0.5 font-mono text-xs text-fd-muted-foreground tabular-nums">
                        {String(ch.n).padStart(2, '0')}
                      </span>
                      <span>
                        <span className="font-ui text-sm font-medium group-hover:text-fd-primary">
                          {ch.title}
                        </span>
                        <span className="block font-prose text-[0.82rem] leading-snug text-fd-muted-foreground">
                          {ch.blurb}
                        </span>
                      </span>
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </section>

      <footer className="mt-16 border-t border-fd-border pt-6 font-ui text-xs text-fd-muted-foreground">
        Built on the classic references — Thrun, Burgard &amp; Fox; Lynch &amp; Park; Craig; Niku;
        Choset et al.; Spong et al. — and brought up to date with factor graphs, estimation on Lie
        groups, and modern planning under uncertainty.
      </footer>
    </main>
  );
}
