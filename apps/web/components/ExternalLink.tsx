import type { ReactNode } from "react";

export const PROSE_LINK_CLASS = "text-moss underline underline-offset-2 hover:text-gold-deep";

export function ExternalLink({
  href,
  children,
  className = PROSE_LINK_CLASS,
  testId,
}: {
  href: string;
  children: ReactNode;
  className?: string;
  testId?: string;
}) {
  return (
    <a href={href} rel="noopener noreferrer" className={className} data-testid={testId}>
      {children}
      <span className="sr-only"> (externe Seite)</span>
    </a>
  );
}
