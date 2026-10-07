import type { MetadataRoute } from "next";
import { SITEMAP_PATHS, siteUrl } from "@/lib/site";

export default function sitemap(): MetadataRoute.Sitemap {
  const base = siteUrl();
  return SITEMAP_PATHS.map((path) => ({ url: new URL(path, base).href }));
}
