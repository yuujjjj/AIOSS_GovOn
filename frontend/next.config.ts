import type { NextConfig } from "next";

const isProd = process.env.NODE_ENV === 'production';
const previewPath = process.env.NEXT_PUBLIC_BASE_PATH || '';
const basePath = isProd ? (previewPath || '/GovOn') : '';

const nextConfig: NextConfig = {
  output: 'export',
  basePath: basePath,
  assetPrefix: basePath ? `${basePath}/` : '',
  // Images optimization is not supported with static export, so we need to disable it
  images: {
    unoptimized: true,
  },
};

export default nextConfig;
