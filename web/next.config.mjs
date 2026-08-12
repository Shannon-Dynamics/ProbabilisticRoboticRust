import { createMDX } from 'fumadocs-mdx/next';

const withMDX = createMDX();

/** @type {import('next').NextConfig} */
const config = {
  reactStrictMode: true,
  // The book is a static site: every chapter renders at build time, so readers
  // download HTML + the simulation bundles and nothing else.
  output: 'export',
  images: { unoptimized: true },
  typescript: { ignoreBuildErrors: false },
};

export default withMDX(config);
