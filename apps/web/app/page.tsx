import Link from "next/link";
import { Card, CardTitle } from "@portfolio/ui";
import { PROJECTS } from "@/lib/site";

const PROJECT_LINK_CLASS = "mt-4 inline-block text-sm text-moss underline underline-offset-4 hover:text-gold-deep";

export default function HomePage() {
  return (
    <div className="space-y-10">
      <section className="max-w-3xl">
        <p className="text-xs uppercase tracking-widest text-gold-deep">Portfolio</p>
        <h1 className="mt-2 font-serif text-4xl">Projekte</h1>
        <p className="mt-3 text-stone">
          Drei Projekte zu Finanzdaten, Textklassifikation und Netzwerkanalyse. DepotDoktor und KontoKlar laufen unter
          eigenen Adressen, NetzRadar auf dieser Seite.
        </p>
      </section>
      <ul className="grid gap-6 md:grid-cols-3">
        {PROJECTS.map((project) => (
          <li key={project.slug} className="flex" data-testid={`project-${project.slug}`}>
            <Card className="flex w-full flex-col">
              <p className="text-xs uppercase tracking-widest text-gold-deep">Projekt {project.code}</p>
              <CardTitle className="mt-1">{project.title}</CardTitle>
              <p className="flex-1 text-sm text-stone">{project.description}</p>
              {project.external ? (
                <a href={project.href} rel="noopener noreferrer" className={PROJECT_LINK_CLASS}>
                  Zum Projekt <span className="text-stone">(externe Seite)</span>
                  <span className="sr-only"> {project.title}</span>
                </a>
              ) : (
                <Link href={project.href} className={PROJECT_LINK_CLASS}>
                  Zum Projekt
                  <span className="sr-only"> {project.title}</span>
                </Link>
              )}
            </Card>
          </li>
        ))}
      </ul>
    </div>
  );
}
