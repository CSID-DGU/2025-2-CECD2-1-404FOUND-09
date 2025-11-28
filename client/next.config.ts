import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  compiler: {
    styledComponents: true,
  },
  eslint: {
    ignoreDuringBuilds: true,
  },
  output: 'standalone',
  env: {
    PORT: '8081',
  },
};

export default nextConfig;
