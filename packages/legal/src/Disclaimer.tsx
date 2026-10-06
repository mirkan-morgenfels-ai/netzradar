import { DISCLAIMER_LONG, DISCLAIMER_SHORT } from "./texts";

export interface DisclaimerProps {
  variant?: "short" | "long";
  className?: string;
}

export function Disclaimer({ variant = "short", className }: DisclaimerProps) {
  return (
    <p role="note" className={className}>
      {variant === "long" ? DISCLAIMER_LONG : DISCLAIMER_SHORT}
    </p>
  );
}
