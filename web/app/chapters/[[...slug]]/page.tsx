import { source } from '@/lib/source';
import { DocsPage, DocsBody, DocsDescription, DocsTitle } from 'fumadocs-ui/layouts/docs/page';
import { notFound } from 'next/navigation';
import { getMDXComponents } from '@/mdx-components';
import { ChapterHeader } from '@/components/book/chapter-header';
import { Epigraph } from '@/components/book/epigraph';
import { ChapterNav } from '@/components/book/chapter-nav';

export default async function Page(props: { params: Promise<{ slug?: string[] }> }) {
  const params = await props.params;
  const page = source.getPage(params.slug);
  if (!page) notFound();

  const MDX = page.data.body;
  const fm = page.data as unknown as {
    chapter?: number;
    part?: string;
    partTitle?: string;
    difficulty?: string;
    readingTime?: string;
    quote?: string;
    quoteAuthor?: string;
    quoteSource?: string;
  };

  return (
    <DocsPage toc={page.data.toc} full={page.data.full}>
      <ChapterHeader
        chapter={fm.chapter}
        part={fm.part}
        partTitle={fm.partTitle}
        difficulty={fm.difficulty}
        readingTime={fm.readingTime}
      />
      <DocsTitle>{page.data.title}</DocsTitle>
      <DocsDescription>{page.data.description}</DocsDescription>
      {fm.quote ? (
        <Epigraph author={fm.quoteAuthor} source={fm.quoteSource}>
          {fm.quote}
        </Epigraph>
      ) : null}
      <DocsBody>
        <MDX components={getMDXComponents()} />
        <ChapterNav chapter={fm.chapter} />
      </DocsBody>
    </DocsPage>
  );
}

export function generateStaticParams() {
  return source.generateParams();
}

export async function generateMetadata(props: { params: Promise<{ slug?: string[] }> }) {
  const params = await props.params;
  const page = source.getPage(params.slug);
  if (!page) notFound();
  return {
    title: page.data.title,
    description: page.data.description,
  };
}
