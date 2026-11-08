import { readFileSync, writeFileSync } from "node:fs";

const METRICS = new URL("../../../data/k3/metrics.json", import.meta.url);
const OUTPUT = new URL("../../../docs/ergebnisse.svg", import.meta.url);

const NAVY = "#0b1626";
const NAVY_700 = "#26354d";
const IVORY = "#f7f3ea";
const GOLD = "#c9a548";
const GOLD_LIGHT = "#d8bd72";
const MUTED = "#8f9bb0";
const SANS = "Inter, 'Segoe UI', 'Helvetica Neue', Helvetica, Arial, sans-serif";

const STYLES = {
  zscore: { name: "Robuste Z-Scores", kind: "Baseline, unüberwacht", fill: MUTED, opacity: 0.55, value: IVORY },
  iforest: { name: "Isolation Forest", kind: "Baseline, unüberwacht", fill: MUTED, opacity: 0.55, value: IVORY },
  hgb: { name: "Gradient Boosting", kind: "überwacht, ohne Kanten", fill: IVORY, opacity: 0.7, value: IVORY },
  mlp: { name: "MLP", kind: "überwacht, ohne Kanten", fill: IVORY, opacity: 0.45, value: IVORY },
  gcn: { name: "GCN", kind: "Graph Neural Network", fill: GOLD, opacity: 0.85, value: GOLD_LIGHT },
  graphsage: { name: "GraphSAGE", kind: "Graph Neural Network", fill: GOLD, opacity: 1, value: GOLD_LIGHT },
};

const metrics = JSON.parse(readFileSync(METRICS, "utf8"));
const runs = [...metrics.runs].sort((a, b) => b.prAuc - a.prAuc);
for (const run of runs) {
  if (!(run.method in STYLES)) throw new Error(`Kein Stil für das Verfahren ${run.method}`);
}
const evaluation = metrics.evaluation;
const random = evaluation.randomPrAucExpected;
const format = (value) => value.toFixed(4).replace(".", ",");

const width = 1100;
const left = 270;
const right = 1000;
const top = 126;
const row = 56;
const bottom = top + runs.length * row;
const height = bottom + 76;
const x = (value) => left + (right - left) * value;

const ticks = [0, 0.25, 0.5, 0.75, 1]
  .map(
    (tick) =>
      `<path d="M${x(tick)} ${top - 4}V${bottom + 6}" stroke="${NAVY_700}" stroke-width="1"/>` +
      `<text x="${x(tick)}" y="${bottom + 32}" text-anchor="middle" fill="${MUTED}" font-family="${SANS}" font-size="15">${tick.toFixed(2).replace(".", ",")}</text>`,
  )
  .join("");

const bars = runs
  .map((run, index) => {
    const style = STYLES[run.method];
    const y = top + index * row;
    return [
      `<text x="${left - 20}" y="${y + 20}" text-anchor="end" fill="${IVORY}" font-family="${SANS}" font-size="19" font-weight="500">${style.name}</text>`,
      `<text x="${left - 20}" y="${y + 41}" text-anchor="end" fill="${MUTED}" font-family="${SANS}" font-size="15">${style.kind}</text>`,
      `<rect x="${left}" y="${y + 8}" width="${(x(run.prAuc) - left).toFixed(1)}" height="26" rx="4" fill="${style.fill}" fill-opacity="${style.opacity}"/>`,
      `<text x="${(x(run.prAuc) + 10).toFixed(1)}" y="${y + 28}" fill="${style.value}" font-family="${SANS}" font-size="18" font-weight="600">${format(run.prAuc)}</text>`,
    ].join("\n");
  })
  .join("\n");

const summary = runs.map((run) => `${STYLES[run.method].name} ${format(run.prAuc)}`).join(", ");
const labelled = evaluation.testPositives + evaluation.testNegatives;

const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" role="img" aria-label="PR-AUC auf dem Testteil des synthetischen Netzes, Seed ${metrics.seed}: ${summary}; zufällige Rangfolge ${format(random)}">
<title>PR-AUC je Verfahren auf dem Testteil (Seed ${metrics.seed})</title>
<rect width="${width}" height="${height}" rx="18" fill="${NAVY}"/>
<rect x=".5" y=".5" width="${width - 1}" height="${height - 1}" rx="17.5" fill="none" stroke="${GOLD}" stroke-opacity=".22"/>
<path d="M64 52H104" stroke="${GOLD}" stroke-width="1.5"/>
<text x="120" y="57" fill="${GOLD_LIGHT}" font-family="${SANS}" font-size="15" font-weight="500" letter-spacing="3">ERGEBNIS · PR-AUC AUF DEM TESTTEIL (SEED ${metrics.seed})</text>
<text x="64" y="90" fill="${MUTED}" font-family="${SANS}" font-size="16">Synthetisches Netz, zeitlicher Split, ${labelled} gelabelte Testknoten, davon ${evaluation.testPositives} mit Label illegal</text>
${ticks}
${bars}
<path d="M${x(random).toFixed(1)} ${top - 4}V${bottom + 6}" stroke="${IVORY}" stroke-opacity=".75" stroke-width="2" stroke-dasharray="5 5"/>
<text x="${(x(random) + 8).toFixed(1)}" y="${top - 10}" fill="${IVORY}" font-family="${SANS}" font-size="15">zufällige Rangfolge ${format(random)}</text>
</svg>
`;

writeFileSync(OUTPUT, svg);
