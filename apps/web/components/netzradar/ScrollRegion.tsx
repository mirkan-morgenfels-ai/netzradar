"use client";

import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { cx } from "@portfolio/ui";

export type ScrollRegionName = { labelledBy: string; label?: never } | { label: string; labelledBy?: never };

export type ScrollRegionProps = ScrollRegionName & {
  className?: string;
  testId?: string;
  hint?: string;
  children: ReactNode;
};

export function ScrollRegion({ labelledBy, label, className, testId, hint, children }: ScrollRegionProps) {
  const ref = useRef<HTMLDivElement>(null);
  const [overflowing, setOverflowing] = useState(false);
  const [moreRight, setMoreRight] = useState(false);

  const measure = useCallback(() => {
    const element = ref.current;
    if (!element) return;
    const hasOverflow = element.scrollWidth - element.clientWidth > 1;
    setOverflowing(hasOverflow);
    setMoreRight(hasOverflow && element.scrollLeft + element.clientWidth < element.scrollWidth - 1);
  }, []);

  useEffect(() => {
    measure();
    const element = ref.current;
    if (!element || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    if (element.firstElementChild) observer.observe(element.firstElementChild);
    return () => observer.disconnect();
  }, [measure]);

  return (
    <div>
      <div className="relative">
        <div
          ref={ref}
          role="region"
          tabIndex={0}
          aria-labelledby={labelledBy}
          aria-label={label}
          data-testid={testId}
          data-overflowing={overflowing ? "true" : "false"}
          onScroll={measure}
          className={cx("overflow-x-auto", className)}
        >
          {children}
        </div>
        {moreRight ? (
          <div
            aria-hidden="true"
            className="pointer-events-none absolute inset-y-px right-px w-10 rounded-r-2xl bg-linear-to-l from-surface to-transparent"
          />
        ) : null}
      </div>
      {overflowing && hint ? <p className="mt-2 text-xs text-slate sm:hidden">{hint}</p> : null}
    </div>
  );
}
