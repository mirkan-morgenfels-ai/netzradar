"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cx } from "@portfolio/ui";
import { isActivePath, type NavLink } from "@/lib/site";

const LINK_CLASS = "hover:text-gold-deep";
const ACTIVE_CLASS = "underline decoration-gold-deep underline-offset-4";

export function NavLinks({ links }: { links: readonly NavLink[] }) {
  const pathname = usePathname();
  return (
    <ul className="flex flex-wrap gap-x-5 gap-y-2 text-sm">
      {links.map((link) => {
        if (link.external) {
          return (
            <li key={link.href}>
              <a href={link.href} rel="noopener noreferrer" className={LINK_CLASS}>
                {link.label}
                <span className="sr-only"> (externe Seite)</span>
              </a>
            </li>
          );
        }
        const active = isActivePath(pathname, link.href);
        return (
          <li key={link.href}>
            <Link
              href={link.href}
              aria-current={active ? "page" : undefined}
              className={cx(LINK_CLASS, active ? ACTIVE_CLASS : undefined)}
            >
              {link.label}
            </Link>
          </li>
        );
      })}
    </ul>
  );
}
