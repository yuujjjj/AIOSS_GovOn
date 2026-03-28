import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright E2E 테스트 설정
 *
 * 현재: FastAPI 백엔드 API 엔드포인트 + Swagger UI 테스트
 * 향후: 프론트엔드 개발 시 브라우저 기반 UI 테스트로 확장
 */
export default defineConfig({
  // 테스트 디렉토리
  testDir: './e2e',

  // 전체 테스트 타임아웃 (밀리초)
  timeout: 30000,

  // expect() 타임아웃
  expect: {
    timeout: 5000,
  },

  // 테스트 결과물 저장 디렉토리
  outputDir: 'test-results/',

  // 병렬 실행 설정
  fullyParallel: true,
  workers: process.env.CI ? 1 : undefined,

  // CI 환경에서 재시도 2회
  retries: process.env.CI ? 2 : 0,

  // 리포터 설정
  reporter: process.env.CI ? 'github' : 'html',

  // 공통 설정
  use: {
    // FastAPI 서버 기본 URL
    baseURL: 'http://localhost:8000',

    // 실패 시에만 스크린샷 저장
    screenshot: 'only-on-failure',

    // 실패 시에만 비디오 저장
    video: 'retain-on-failure',

    // 트레이스: 첫 재시도 시 수집
    trace: 'on-first-retry',
  },

  // 브라우저 프로젝트: CI 경량화를 위해 chromium만 사용
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
    // 향후 프론트엔드 개발 시 추가 브라우저 테스트 활성화
    // {
    //   name: 'firefox',
    //   use: { ...devices['Desktop Firefox'] },
    // },
    // {
    //   name: 'mobile-chrome',
    //   use: { ...devices['Pixel 5'] },
    // },
  ],

  // 로컬 개발 시 FastAPI 서버 자동 실행
  // CI에서는 워크플로우에서 별도로 서버를 시작하므로 비활성화
  // webServer: {
  //   command: 'SKIP_MODEL_LOAD=true uvicorn src.inference.api_server:app --host 0.0.0.0 --port 8000',
  //   port: 8000,
  //   reuseExistingServer: !process.env.CI,
  //   timeout: 60000,
  // },
});
