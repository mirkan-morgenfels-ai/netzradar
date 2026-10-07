import type { ButtonHTMLAttributes } from "react";
import { cx } from "./cx";

export type ButtonVariant = "primary" | "secondary" | "gold" | "outline-light";

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: "sm" | "md";
}

const LOOK: Record<ButtonVariant, string> = {
  primary: "bg-navy-950 text-ivory hover:bg-navy-800",
  secondary: "border border-line-strong bg-surface text-ink hover:border-ink",
  gold: "bg-gold text-navy-950 hover:bg-gold-light",
  "outline-light": "border border-gold/70 text-ivory hover:border-gold-light hover:bg-navy-800",
};

export function Button({ variant = "primary", size = "md", className, ...rest }: ButtonProps) {
  const base =
    "inline-flex items-center justify-center gap-2 rounded-full font-medium whitespace-nowrap transition-colors duration-150 disabled:cursor-not-allowed disabled:opacity-50";
  const dimensions = size === "sm" ? "px-4 py-2 text-[0.8125rem]" : "px-5 py-2.5 text-sm";
  return <button className={cx(base, dimensions, LOOK[variant], className)} {...rest} />;
}
