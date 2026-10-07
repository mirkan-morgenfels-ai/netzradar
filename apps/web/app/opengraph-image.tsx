import { ImageResponse } from "next/og";
import { CHART_COLORS } from "@portfolio/charts/theme";
import { BrandMark } from "@/components/BrandMark";
import { OG_IMAGE_ALT, OG_IMAGE_SIZE, OG_SUBTITLE } from "@/lib/metadata";
import { SITE_NAME } from "@/lib/site";

export const alt = OG_IMAGE_ALT;
export const size = OG_IMAGE_SIZE;
export const contentType = "image/png";

export default function OpengraphImage() {
  return new ImageResponse(
    <div
      style={{
        width: "100%",
        height: "100%",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "80px",
        background: CHART_COLORS.ink,
        color: CHART_COLORS.paper,
      }}
    >
      <div style={{ display: "flex", flexDirection: "column", maxWidth: "800px" }}>
        <div style={{ width: "120px", height: "6px", background: CHART_COLORS.gold }} />
        <div style={{ marginTop: "40px", fontSize: "104px", lineHeight: 1 }}>{SITE_NAME}</div>
        <div style={{ marginTop: "28px", fontSize: "42px", lineHeight: 1.25, color: CHART_COLORS.goldSoft }}>
          {OG_SUBTITLE}
        </div>
        <div style={{ marginTop: "48px", fontSize: "27px", lineHeight: 1.4, color: CHART_COLORS.line }}>
          GCN und GraphSAGE gegen Baselines · zeitlicher Split · PR-AUC
        </div>
      </div>
      <BrandMark size={240} />
    </div>,
    size,
  );
}
