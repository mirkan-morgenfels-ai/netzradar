import type { ReactNode } from "react";
import { cx } from "./cx";

export function Card({ children, className, testId }: { children: ReactNode; className?: string; testId?: string }) {
  return (
    <section className={cx("rounded-2xl border border-line bg-surface p-6 shadow-card sm:p-8", className)} data-testid={testId}>
      {children}
    </section>
  );
}

export function CardTitle({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <h2 className={cx("font-display text-[1.625rem] leading-tight font-medium tracking-[-0.01em] text-ink sm:text-[1.75rem]", className)}>
      {children}
    </h2>
  );
}

export function CardEyebrow({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <p className={cx("text-[0.6875rem] font-medium tracking-[0.16em] text-gold-deep uppercase", className)}>{children}</p>
  );
}
