// source.config.ts
import { defineDocs, defineConfig, frontmatterSchema } from "fumadocs-mdx/config";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import * as z from "zod";

// lib/katex-macros.ts
var MACROS = Object.freeze({
  // Sets
  "\\R": "\\mathbb{R}",
  "\\N": "\\mathbb{N}",
  "\\Z": "\\mathbb{Z}",
  // Probability
  "\\E": "\\mathbb{E}",
  "\\Var": "\\operatorname{Var}",
  "\\Cov": "\\operatorname{Cov}",
  "\\Prob": "\\operatorname{p}",
  "\\given": "\\mid",
  "\\Normal": "\\mathcal{N}",
  "\\KL": "\\operatorname{KL}",
  // The Bayes-filter vocabulary
  "\\bel": "\\operatorname{bel}",
  "\\belbar": "\\overline{\\operatorname{bel}}",
  // Lie groups and manifolds
  "\\SOtwo": "\\mathrm{SO}(2)",
  "\\SOthree": "\\mathrm{SO}(3)",
  "\\SEtwo": "\\mathrm{SE}(2)",
  "\\SEthree": "\\mathrm{SE}(3)",
  "\\sotwo": "\\mathfrak{so}(2)",
  "\\sethree": "\\mathfrak{se}(3)",
  "\\bplus": "\\boxplus",
  "\\bminus": "\\boxminus",
  "\\Ad": "\\operatorname{Ad}",
  // Matrix helpers
  "\\tr": "\\operatorname{tr}",
  "\\diag": "\\operatorname{diag}",
  "\\rank": "\\operatorname{rank}",
  "\\T": "^{\\mathsf{T}}",
  // Parameterized
  "\\norm": "\\left\\lVert #1 \\right\\rVert",
  "\\abs": "\\left\\lvert #1 \\right\\rvert",
  "\\vv": "\\mathbf{#1}",
  "\\mat": "\\mathbf{#1}"
});
var katexMacros = () => ({ ...MACROS });
var katexOptions = () => ({
  macros: katexMacros(),
  // Turbopack cannot serialize functions into plugin options, so every value
  // here is a plain one.
  strict: "ignore",
  trust: true,
  minRuleThickness: 0.06,
  maxSize: 20
});

// source.config.ts
var docs = defineDocs({
  dir: "content/chapters",
  docs: {
    schema: frontmatterSchema.extend({
      chapter: z.number().optional(),
      part: z.string().optional(),
      partTitle: z.string().optional(),
      difficulty: z.enum(["Foundational", "Intermediate", "Advanced"]).optional(),
      readingTime: z.string().optional(),
      quote: z.string().optional(),
      quoteAuthor: z.string().optional(),
      quoteSource: z.string().optional()
    })
  }
});
var source_config_default = defineConfig({
  mdxOptions: {
    remarkPlugins: [remarkMath],
    rehypePlugins: (v) => [
      // The macro table must be a mutable copy: KaTeX writes \cr into it for
      // every matrix/cases/aligned environment. See lib/katex-macros.ts.
      [rehypeKatex, katexOptions()],
      ...v
    ]
  }
});
export {
  source_config_default as default,
  docs
};
