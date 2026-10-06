import type { ReactNode } from "react";
import { cx } from "./cx";

export function Card({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <section className={cx("rounded-lg border border-line bg-surface p-6 shadow-sm", className)}>
      {children}
    </section>
  );
}

export function CardTitle({ children, className }: { children: ReactNode; className?: string }) {
  return <h2 className={cx("mb-4 font-serif text-xl text-ink", className)}>{children}</h2>;
}
