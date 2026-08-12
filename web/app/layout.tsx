import './global.css';
import type { Metadata } from 'next';
import { Fraunces, Source_Serif_4, IBM_Plex_Sans, IBM_Plex_Mono } from 'next/font/google';
import { RootProvider } from 'fumadocs-ui/provider/next';

const fraunces = Fraunces({
  subsets: ['latin'],
  display: 'swap',
  axes: ['SOFT', 'WONK'],
  variable: '--font-fraunces',
});

const sourceSerif = Source_Serif_4({
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-source-serif',
});

const plexSans = IBM_Plex_Sans({
  subsets: ['latin'],
  weight: ['400', '500', '600'],
  display: 'swap',
  variable: '--font-plex-sans',
});

const plexMono = IBM_Plex_Mono({
  subsets: ['latin'],
  weight: ['400', '500'],
  display: 'swap',
  variable: '--font-plex-mono',
});

export const metadata: Metadata = {
  title: {
    default: 'Probabilistic Robotics via Rust',
    template: '%s · Probabilistic Robotics via Rust',
  },
  description:
    'An interactive web book on probabilistic robotics: rigorous mathematical foundations, ' +
    'interactive simulations for every hard idea, and implementations in Rust.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html
      lang="en"
      className={`${fraunces.variable} ${sourceSerif.variable} ${plexSans.variable} ${plexMono.variable}`}
      suppressHydrationWarning
    >
      <body className="flex min-h-screen flex-col">
        <RootProvider>{children}</RootProvider>
      </body>
    </html>
  );
}
