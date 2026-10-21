import { describe, expect, it } from "vitest";
import {
  EDGE_BODY_LAYER_GLSL,
  EDGE_FEATHER_PIXELS,
  GL_FLOAT,
  GL_UNSIGNED_BYTE,
  RING_LAYER_GLSL,
  edgeBodyLayer,
  ringLayer,
} from "../layers";

describe("ring layer", () => {
  it("reads fill, ring colour and ring share from the node attributes", () => {
    const layer = ringLayer();
    expect(layer.name).toBe("ring");
    expect(layer.uniforms).toEqual([]);
    expect(layer.attributes).toEqual([
      { name: "ringFill", size: 4, type: GL_UNSIGNED_BYTE, normalized: true, source: "color" },
      { name: "ringColor", size: 4, type: GL_UNSIGNED_BYTE, normalized: true, source: "borderColor" },
      { name: "ringSize", size: 1, type: GL_FLOAT, source: "borderSize", defaultValue: 0 },
    ]);
    expect(GL_UNSIGNED_BYTE).toBe(5121);
    expect(GL_FLOAT).toBe(5126);
  });

  it("takes its parameters in the order of the attributes", () => {
    expect(RING_LAYER_GLSL).toContain("vec4 layer_ring(vec4 v_ringFill, vec4 v_ringColor, float v_ringSize)");
  });

  it("measures the ring as a share of the radius with a transition of two antialiasing widths", () => {
    expect(RING_LAYER_GLSL).toContain("float aa = 2.0 * context.aaWidth;");
    expect(RING_LAYER_GLSL).toContain("float edge = 1.0 - v_ringSize;");
    expect(RING_LAYER_GLSL).toContain("float toRing = clamp((dist - edge + aa) / aa, 0.0, 1.0);");
  });

  it("drops a ring that is not wider than one antialiasing width and keeps the fill", () => {
    expect(RING_LAYER_GLSL).toContain("if (v_ringSize > context.aaWidth) {");
    expect(RING_LAYER_GLSL).toContain("return vec4(v_ringFill.rgb, v_ringFill.a * inside * (1.0 - toRing));");
  });
});

describe("edge body layer", () => {
  it("fades the body over one device pixel inside its width and leaves the arrow head alone", () => {
    expect(EDGE_FEATHER_PIXELS).toBe(1);
    expect(EDGE_BODY_LAYER_GLSL).toContain("uniform float u_pixelRatio;");
    expect(EDGE_BODY_LAYER_GLSL).toContain("if (v_zone < 0.5 || v_zone > 1.5) return v_color;");
    expect(EDGE_BODY_LAYER_GLSL).toContain("float feather = 2.0 * u_correctionRatio / u_pixelRatio;");
    expect(EDGE_BODY_LAYER_GLSL).toContain("smoothstep(-feather, 0.0, ctx.sdf)");
    expect(edgeBodyLayer()).toEqual({ name: "body", uniforms: [], attributes: [], glsl: EDGE_BODY_LAYER_GLSL });
  });
});
