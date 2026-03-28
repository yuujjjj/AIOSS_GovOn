import { test, expect } from '@playwright/test';

/**
 * FastAPI 백엔드 헬스체크 및 API 문서 E2E 테스트
 *
 * 검증 대상:
 * - /health 엔드포인트 정상 응답
 * - /docs (Swagger UI) 페이지 접근
 * - /openapi.json 스키마 제공
 *
 * 향후 확장:
 * - 프론트엔드 개발 후 메인 페이지 렌더링 테스트 추가
 * - 인증 플로우 E2E 테스트 추가
 */

test.describe('API 헬스체크', () => {
  test('/health 엔드포인트가 정상 응답을 반환한다', async ({ request }) => {
    const response = await request.get('/health');

    // 상태 코드 200 확인
    expect(response.status()).toBe(200);

    // 응답 본문 검증
    const body = await response.json();
    expect(body).toHaveProperty('status', 'healthy');

    // RAG 관련 필드 존재 확인 (값은 환경에 따라 다를 수 있음)
    expect(body).toHaveProperty('rag_enabled');
    expect(body).toHaveProperty('hybrid_search_enabled');
    expect(body).toHaveProperty('pii_masking_enabled');
  });

  test('/health 응답에 feature_flags가 포함된다', async ({ request }) => {
    const response = await request.get('/health');
    const body = await response.json();

    expect(body).toHaveProperty('feature_flags');
    expect(body.feature_flags).toHaveProperty('use_rag_pipeline');
    expect(body.feature_flags).toHaveProperty('model_version');
  });
});

test.describe('API 문서 (Swagger UI)', () => {
  test('/docs 페이지에 접근할 수 있다', async ({ page }) => {
    const response = await page.goto('/docs');

    // 페이지 로드 성공 확인
    expect(response?.status()).toBe(200);

    // Swagger UI 렌더링 대기
    await page.waitForSelector('#swagger-ui', { timeout: 10000 });

    // Swagger UI 컨테이너 존재 확인
    const swaggerUI = page.locator('#swagger-ui');
    await expect(swaggerUI).toBeVisible();
  });

  test('/openapi.json 스키마가 유효한 JSON을 반환한다', async ({ request }) => {
    const response = await request.get('/openapi.json');

    expect(response.status()).toBe(200);

    const schema = await response.json();
    expect(schema).toHaveProperty('openapi');
    expect(schema).toHaveProperty('info');
    expect(schema).toHaveProperty('paths');

    // /health 경로가 스키마에 존재하는지 확인
    expect(schema.paths).toHaveProperty('/health');
  });

  // 향후 프론트엔드 개발 시 활성화할 테스트
  // test('메인 페이지가 정상 렌더링된다', async ({ page }) => {
  //   await page.goto('/');
  //   // TODO: 프론트엔드 메인 페이지 렌더링 검증
  //   // await expect(page.locator('h1')).toContainText('GovOn');
  // });
});

test.describe('API 엔드포인트 기본 검증', () => {
  test('존재하지 않는 경로에 404를 반환한다', async ({ request }) => {
    const response = await request.get('/nonexistent-path', {
      failOnStatusCode: false,
    });

    expect(response.status()).toBe(404);
  });

  // 향후 확장 포인트: 인증이 필요한 엔드포인트 테스트
  // test('인증 없이 보호된 엔드포인트에 접근하면 401/403을 반환한다', async ({ request }) => {
  //   const response = await request.post('/api/analyze', {
  //     failOnStatusCode: false,
  //   });
  //   expect([401, 403]).toContain(response.status());
  // });
});
