import { test, expect } from '@playwright/test';

/**
 * 민원 제출 프론트엔드 E2E 테스트 (스텁)
 *
 * 프론트엔드 개발 전까지 test.skip()으로 비활성화.
 * 프론트엔드 개발 완료 후:
 * 1. test.skip() 제거
 * 2. 각 테스트의 TODO 주석을 실제 구현으로 교체
 * 3. 필요한 선택자(selector)를 실제 UI에 맞게 수정
 */

test.describe('민원 제출 페이지', () => {
  // 프론트엔드 미구현 - 개발 완료 시 이 줄 제거
  test.skip();

  test('민원 제출 페이지에 접근할 수 있다', async ({ page }) => {
    // TODO: 실제 민원 제출 페이지 경로로 변경
    await page.goto('/complaints/new');

    // TODO: 페이지 타이틀 또는 헤더 검증
    // await expect(page.locator('h1')).toContainText('민원 접수');

    // TODO: 제출 폼 요소 존재 확인
    // await expect(page.locator('form#complaint-form')).toBeVisible();
    // await expect(page.locator('textarea[name="content"]')).toBeVisible();
    // await expect(page.locator('button[type="submit"]')).toBeVisible();
  });

  test('민원을 작성하고 제출할 수 있다', async ({ page }) => {
    await page.goto('/complaints/new');

    // TODO: 민원 내용 입력
    // await page.fill('textarea[name="content"]', '도로 파손 민원입니다. 인도 블록이 깨져 있어 보행에 위험합니다.');

    // TODO: 카테고리 선택 (있는 경우)
    // await page.selectOption('select[name="category"]', 'infrastructure');

    // TODO: 제출 버튼 클릭
    // await page.click('button[type="submit"]');

    // TODO: 제출 완료 확인
    // await expect(page.locator('.success-message')).toContainText('민원이 접수되었습니다');
  });

  test('필수 필드 미입력 시 유효성 검사가 동작한다', async ({ page }) => {
    await page.goto('/complaints/new');

    // TODO: 빈 폼 제출 시도
    // await page.click('button[type="submit"]');

    // TODO: 유효성 검사 에러 메시지 확인
    // await expect(page.locator('.error-message')).toBeVisible();
  });
});

test.describe('민원 분류 결과 표시', () => {
  // 프론트엔드 미구현 - 개발 완료 시 이 줄 제거
  test.skip();

  test('AI 분류 결과가 올바르게 표시된다', async ({ page }) => {
    // TODO: 민원 상세 페이지 또는 결과 페이지 접근
    // await page.goto('/complaints/1/result');

    // TODO: 분류 결과 카테고리 표시 확인
    // await expect(page.locator('.classification-result')).toBeVisible();
    // await expect(page.locator('.category-badge')).toBeVisible();

    // TODO: 신뢰도 점수 표시 확인
    // await expect(page.locator('.confidence-score')).toBeVisible();
  });

  test('유사 민원 검색 결과가 표시된다', async ({ page }) => {
    // TODO: 민원 상세 페이지에서 유사 민원 섹션 확인
    // await page.goto('/complaints/1/result');

    // TODO: 유사 민원 목록 존재 확인
    // await expect(page.locator('.similar-complaints')).toBeVisible();
    // const similarItems = page.locator('.similar-complaint-item');
    // await expect(similarItems).toHaveCount(/* 기대하는 개수 */);
  });
});

test.describe('검색 기능', () => {
  // 프론트엔드 미구현 - 개발 완료 시 이 줄 제거
  test.skip();

  test('민원 검색 페이지에 접근할 수 있다', async ({ page }) => {
    // TODO: 검색 페이지 경로로 변경
    // await page.goto('/search');

    // TODO: 검색 입력 필드 존재 확인
    // await expect(page.locator('input[name="query"]')).toBeVisible();
  });

  test('검색어 입력 시 결과가 표시된다', async ({ page }) => {
    // TODO: 검색 페이지 접근
    // await page.goto('/search');

    // TODO: 검색어 입력 및 검색 실행
    // await page.fill('input[name="query"]', '도로 보수');
    // await page.click('button.search-btn');

    // TODO: 검색 결과 목록 확인
    // await expect(page.locator('.search-results')).toBeVisible();
    // const resultItems = page.locator('.search-result-item');
    // expect(await resultItems.count()).toBeGreaterThan(0);
  });

  test('검색 결과가 없을 때 적절한 메시지가 표시된다', async ({ page }) => {
    // TODO: 검색 페이지 접근
    // await page.goto('/search');

    // TODO: 결과가 없을 검색어 입력
    // await page.fill('input[name="query"]', 'xxxxxxxxxxxxxxxxx');
    // await page.click('button.search-btn');

    // TODO: '결과 없음' 메시지 확인
    // await expect(page.locator('.no-results')).toContainText('검색 결과가 없습니다');
  });
});
