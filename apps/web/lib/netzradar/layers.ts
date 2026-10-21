import type { EdgeLayer, FragmentLayer } from "sigma/rendering";

export const GL_UNSIGNED_BYTE = 0x1401;
export const GL_FLOAT = 0x1406;

export const RING_LAYER_GLSL = `
vec4 layer_ring(vec4 v_ringFill, vec4 v_ringColor, float v_ringSize) {
  float aa = 2.0 * context.aaWidth;
  float dist = length(context.uv);
  float edge = 1.0 - v_ringSize;
  float toRing = clamp((dist - edge + aa) / aa, 0.0, 1.0);
  if (v_ringSize > context.aaWidth) {
    return dist > edge ? v_ringColor : mix(v_ringFill, v_ringColor, toRing);
  }
  float inside = 1.0 - smoothstep(1.0 - aa - context.aaWidth, 1.0 - aa + context.aaWidth, dist);
  return vec4(v_ringFill.rgb, v_ringFill.a * inside * (1.0 - toRing));
}
`;

export function ringLayer(): FragmentLayer {
  return {
    name: "ring",
    uniforms: [],
    attributes: [
      { name: "ringFill", size: 4, type: GL_UNSIGNED_BYTE, normalized: true, source: "color" },
      { name: "ringColor", size: 4, type: GL_UNSIGNED_BYTE, normalized: true, source: "borderColor" },
      { name: "ringSize", size: 1, type: GL_FLOAT, source: "borderSize", defaultValue: 0 },
    ],
    glsl: RING_LAYER_GLSL,
  };
}

export const EDGE_FEATHER_PIXELS = 1;

export const EDGE_BODY_LAYER_GLSL = `
uniform float u_pixelRatio;

vec4 layer_body(EdgeContext ctx) {
  if (v_zone < 0.5 || v_zone > 1.5) return v_color;
  float feather = ${2 * EDGE_FEATHER_PIXELS}.0 * u_correctionRatio / u_pixelRatio;
  return vec4(v_color.rgb, v_color.a * (1.0 - smoothstep(-feather, 0.0, ctx.sdf)));
}
`;

export function edgeBodyLayer(): EdgeLayer {
  return { name: "body", uniforms: [], attributes: [], glsl: EDGE_BODY_LAYER_GLSL };
}
