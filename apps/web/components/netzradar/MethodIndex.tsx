"use client";

export interface MethodIndexItem {
  id: string;
  label: string;
}

function openTarget(id: string): void {
  const target = document.getElementById(id);
  const details = target instanceof HTMLDetailsElement ? target : target?.closest("details");
  if (details && !details.open) details.open = true;
}

export function MethodIndex({ items, className }: { items: readonly MethodIndexItem[]; className?: string }) {
  return (
    <nav aria-labelledby="method-index-title" className={className} data-testid="method-index">
      <p id="method-index-title" className="eyebrow">
        Inhalt der Methodik
      </p>
      <ol className="mt-4 border-l border-line">
        {items.map((item, index) => (
          <li key={item.id}>
            <a
              href={`#${item.id}`}
              onClick={() => openTarget(item.id)}
              className="-ml-px flex items-baseline gap-3 border-l border-transparent py-1.5 pl-4 text-sm leading-snug text-slate transition-colors duration-150 hover:border-gold-deep hover:text-gold-deep"
            >
              <span aria-hidden="true" className="display num w-5 shrink-0 text-[0.9375rem] text-gold-deep">
                {String(index + 1).padStart(2, "0")}
              </span>
              <span>{item.label}</span>
            </a>
          </li>
        ))}
      </ol>
    </nav>
  );
}
