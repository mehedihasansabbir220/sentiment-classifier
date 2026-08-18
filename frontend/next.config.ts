import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Emit .next/standalone so the Docker runtime stage ships only the files the
  // production server actually needs, instead of all of node_modules.
  output: "standalone",
};

export default nextConfig;
