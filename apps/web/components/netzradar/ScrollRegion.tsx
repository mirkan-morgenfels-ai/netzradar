import type { ReactNode } from "react";
import { cx } from "@portfolio/ui";

export type ScrollRegionName = { labelledBy: string; label?: never } | { label: string; labelledBy?: never };

export type ScrollRegionProps = ScrollRegionName & {
  className?: string;
  testId?: string;
  children: ReactNode;
};

export function ScrollRegion({ labelledBy, label, className, testId, children }: ScrollRegionProps) {
  return (
    <div
      role="region"
      tabIndex={0}
      aria-labelledby={labelledBy}
      aria-label={label}
      data-testid={testId}
      className={cx("overflow-x-auto", className)}
    >
      {children}
    </div>
  );
}
