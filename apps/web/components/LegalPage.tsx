import type { ReactNode } from "react";

export function LegalPage({ title, updated, children }: { title: string; updated: string; children: ReactNode }) {
  return (
    <article className="max-w-3xl">
      <h1 className="font-serif text-3xl">{title}</h1>
      <p className="mt-2 text-xs text-stone">Stand: {updated}</p>
      <div className="mt-8 space-y-8 text-[15px] leading-relaxed">{children}</div>
    </article>
  );
}

export function LegalSection({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="space-y-3">
      <h2 className="font-serif text-xl">{title}</h2>
      {children}
    </section>
  );
}
