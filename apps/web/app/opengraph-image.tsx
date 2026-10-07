import { ImageResponse } from "next/og";
import { BrandMark } from "@/components/BrandMark";
import { NET_EDGES, NET_FAN_IN, NET_FLAGGED, NET_NODES, NET_RINGS, NET_SWEEP, netPoint, sweepWedge } from "@/components/site/netMotif";
import { OG_IMAGE_ALT, OG_IMAGE_SIZE } from "@/lib/metadata";
import { OG_GLYPHS } from "@/lib/og-glyphs";
import { PROJECTS } from "@/lib/site";

export const alt = OG_IMAGE_ALT;
export const size = OG_IMAGE_SIZE;
export const contentType = "image/png";

const NAVY = "#0b1626";
const IVORY = "#f7f3ea";
const GOLD = "#c9a548";
const GOLD_LIGHT = "#d8bd72";
const NAVY_300 = "#8f9bb0";

const PROJECT = PROJECTS.find((project) => project.slug === "netzradar");
const FAN_IN = new Set(NET_FAN_IN);

function Glyphs({
  id,
  height,
  top,
  bottom,
  colors,
}: {
  id: keyof typeof OG_GLYPHS;
  height: number;
  top: number;
  bottom: number;
  colors: string[];
}) {
  const glyphs = OG_GLYPHS[id];
  const span = bottom - top;
  const width = Math.ceil((glyphs.width * height) / span);
  return (
    <svg width={width} height={height} viewBox={`0 ${top} ${glyphs.width} ${span}`} xmlns="http://www.w3.org/2000/svg">
      {glyphs.runs.map((d, index) => (
        <path key={index} d={d} fill={colors[index] ?? colors[0]} />
      ))}
    </svg>
  );
}

function NetMotif() {
  const [fx, fy] = NET_FLAGGED;
  return (
    <svg
      width="500"
      height="281"
      viewBox="0 0 640 360"
      xmlns="http://www.w3.org/2000/svg"
      style={{ position: "absolute", right: 32, top: 36 }}
    >
      <defs>
        <linearGradient id="og-fade" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0.15" stopColor="#fff" stopOpacity="0" />
          <stop offset="0.5" stopColor="#fff" stopOpacity="1" />
        </linearGradient>
        <radialGradient id="og-sweep" cx={fx} cy={fy} r={NET_SWEEP.radius} gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor={GOLD} stopOpacity="0.24" />
          <stop offset="1" stopColor={GOLD} stopOpacity="0" />
        </radialGradient>
        <mask id="og-mask">
          <rect width="640" height="360" fill="url(#og-fade)" />
        </mask>
      </defs>
      <g mask="url(#og-mask)">
        {NET_RINGS.map((radius, index) => (
          <circle
            key={radius}
            cx={fx}
            cy={fy}
            r={radius}
            fill="none"
            stroke={GOLD}
            strokeOpacity={0.26 - index * 0.04}
            strokeWidth="1"
            strokeDasharray="2 7"
          />
        ))}
        <path d={sweepWedge(NET_SWEEP.radius, NET_SWEEP.from, NET_SWEEP.to)} fill="url(#og-sweep)" />
        {NET_EDGES.map(([a, b]) => {
          const [x1, y1] = netPoint(a);
          const [x2, y2] = netPoint(b);
          return (
            <line key={`${a}-${b}`} x1={x1} y1={y1} x2={x2} y2={y2} stroke={GOLD} strokeOpacity="0.32" strokeWidth="1" />
          );
        })}
        {NET_FAN_IN.map((index) => {
          const [x, y] = netPoint(index);
          return <line key={index} x1={x} y1={y} x2={fx} y2={fy} stroke={GOLD_LIGHT} strokeWidth="2" strokeLinecap="round" />;
        })}
        {NET_NODES.map(([x, y], index) => (
          <circle
            key={`${x}-${y}`}
            cx={x}
            cy={y}
            r={FAN_IN.has(index) ? 4 : 3.4}
            fill={NAVY}
            stroke={FAN_IN.has(index) ? GOLD_LIGHT : GOLD}
            strokeOpacity={FAN_IN.has(index) ? 1 : 0.6}
            strokeWidth="1.4"
          />
        ))}
      </g>
      <circle cx={fx} cy={fy} r="24" fill="none" stroke={GOLD_LIGHT} strokeOpacity="0.2" strokeWidth="1" />
      <circle cx={fx} cy={fy} r="13" fill={NAVY} stroke={GOLD_LIGHT} strokeOpacity="0.5" strokeWidth="1" />
      <circle cx={fx} cy={fy} r="5.5" fill={GOLD_LIGHT} />
    </svg>
  );
}

export default function OpengraphImage() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          position: "relative",
          backgroundColor: NAVY,
          backgroundImage: "radial-gradient(circle at 92% -10%, rgba(62,106,158,0.38) 0%, rgba(62,106,158,0) 55%)",
          color: IVORY,
        }}
      >
        <NetMotif />
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            justifyContent: "space-between",
            width: "100%",
            padding: "60px 76px 56px",
          }}
        >
          <div style={{ display: "flex", alignItems: "center" }}>
            <BrandMark size={44} diamond={2.4} />
            <div style={{ display: "flex", flexDirection: "column", marginLeft: 14 }}>
              <div style={{ fontSize: 16, letterSpacing: 3.2, textTransform: "uppercase", color: IVORY }}>
                Mirkan Deniz Günkaya
              </div>
              <div style={{ fontSize: 11, letterSpacing: 3, textTransform: "uppercase", color: NAVY_300, marginTop: 6 }}>
                Portfolio · Daten, KI, Finanzen
              </div>
            </div>
          </div>

          <div style={{ display: "flex", flexDirection: "column" }}>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                fontSize: 15,
                letterSpacing: 3.6,
                textTransform: "uppercase",
                color: GOLD_LIGHT,
              }}
            >
              <div style={{ width: 40, height: 1, backgroundColor: GOLD, marginRight: 16 }} />
              {`${PROJECT?.kicker ?? "Projekt 03"} · ${PROJECT?.topic ?? "Graph-ML"}`}
            </div>
            <div style={{ display: "flex", marginTop: 26 }}>
              <Glyphs id="title" height={150} top={-760} bottom={250} colors={[IVORY, GOLD_LIGHT]} />
            </div>
            <div style={{ display: "flex", marginTop: 14 }}>
              <Glyphs id="tagline" height={44} top={-760} bottom={250} colors={[GOLD_LIGHT]} />
            </div>
          </div>

          <div style={{ display: "flex", flexDirection: "column", width: 620, fontSize: 19, lineHeight: 1.5, color: NAVY_300 }}>
            <div style={{ width: 56, height: 1, backgroundColor: GOLD, marginBottom: 18 }} />
            <div style={{ display: "flex" }}>GCN und GraphSAGE gegen Baselines und eine MLP-Kontrolle,</div>
            <div style={{ display: "flex" }}>zeitlicher Split, PR-AUC, alle Ergebnisse vorab berechnet.</div>
          </div>
        </div>
      </div>
    ),
    size,
  );
}
