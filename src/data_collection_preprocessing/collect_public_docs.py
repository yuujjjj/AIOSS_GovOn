"""
행안부 공공문서 AI 학습데이터 수집 모듈

공공데이터포털(data.go.kr) 행안부 공공문서 API를 호출하여
보도자료/연설문/간행물/보고서/계획서 텍스트와 task 학습 데이터를 수집한다.

API: http://apis.data.go.kr/1741000/publicDoc
Issue: #389
"""

import argparse
import asyncio
import json
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from .config import PublicDocumentConfig, get_config

try:
    import httpx

    _HTTPX_AVAILABLE = True
except ImportError:
    httpx = None  # type: ignore
    _HTTPX_AVAILABLE = False


# ---------------------------------------------------------------------------
# 결과 데이터클래스
# ---------------------------------------------------------------------------


@dataclass
class CollectionResult:
    """수집 결과 요약"""

    success: bool
    total_documents: int
    category_counts: Dict[str, int] = field(default_factory=dict)
    failed_pages: Dict[str, List[int]] = field(default_factory=dict)
    output_path: Optional[str] = None
    duration_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)

    @property
    def meets_minimum(self) -> bool:
        """최소 수집 기준(기본 1000건) 충족 여부"""
        return self.total_documents >= 1000


# ---------------------------------------------------------------------------
# 수집기
# ---------------------------------------------------------------------------


class PublicDocumentCollector:
    """
    행안부 공공문서 API 수집기

    5개 카테고리(press/speech/publication/report/plan)를 순차 수집하고
    min_docs 미달 시 getDocAll로 보충한다.
    """

    def __init__(self, config: Optional[PublicDocumentConfig] = None) -> None:
        self.config = config or get_config().public_doc
        self._last_request_time: float = 0.0
        self._min_interval: float = 1.0 / max(self.config.requests_per_second, 0.1)

    # ------------------------------------------------------------------
    # 공개 메서드
    # ------------------------------------------------------------------

    async def collect_all(self, min_docs: int = 1000) -> CollectionResult:
        """
        전체 카테고리에서 문서를 수집한다.

        5개 카테고리(getDocAll 제외)를 먼저 수집한 뒤,
        합산 문서 수가 min_docs에 미달하면 getDocAll로 보충한다.

        Parameters
        ----------
        min_docs : int
            최소 수집 목표 건수. 기본값 1000.

        Returns
        -------
        CollectionResult
            수집 결과 요약.
        """
        if not _HTTPX_AVAILABLE:
            msg = "httpx 패키지가 설치되지 않았습니다. pip install httpx>=0.27.0"
            logger.error(msg)
            return CollectionResult(success=False, total_documents=0, errors=[msg])

        if not self.config.api_key:
            msg = "DATA_GO_KR_API_KEY 환경변수가 설정되지 않았습니다."
            logger.error(msg)
            return CollectionResult(success=False, total_documents=0, errors=[msg])

        start_time = time.monotonic()
        all_documents: List[Dict[str, Any]] = []
        category_counts: Dict[str, int] = {}
        failed_pages: Dict[str, List[int]] = {}

        # getDocAll 제외한 5개 카테고리
        primary_categories = {k: v for k, v in self.config.categories.items() if k != "all"}

        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            for cat_key in primary_categories:
                logger.info(f"[public_doc] 카테고리 수집 시작: {cat_key}")
                docs = await self.collect_category(cat_key, client)
                all_documents.extend(docs)
                category_counts[cat_key] = len(docs)
                logger.info(f"[public_doc] {cat_key}: {len(docs)}건 수집 완료")

            # min_docs 미달 시 getDocAll 보충
            if len(all_documents) < min_docs and "all" in self.config.categories:
                needed = min_docs - len(all_documents)
                logger.info(
                    f"[public_doc] 현재 {len(all_documents)}건 < 목표 {min_docs}건, "
                    f"getDocAll 보충 시작 (필요: {needed}건 이상)"
                )
                supplemental = await self.collect_category("all", client)
                before = len(all_documents)
                all_documents.extend(supplemental)
                category_counts["all"] = len(supplemental)
                logger.info(
                    f"[public_doc] getDocAll 보충: {len(supplemental)}건 "
                    f"(누적 {before} → {len(all_documents)}건)"
                )

        # doc_id 기준 중복 제거
        seen_ids: set = set()
        deduped: List[Dict[str, Any]] = []
        for doc in all_documents:
            doc_id = doc.get("doc_id", "")
            if doc_id and doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            deduped.append(doc)

        removed = len(all_documents) - len(deduped)
        if removed > 0:
            logger.info(f"[public_doc] 중복 제거: {removed}건 제거, 최종 {len(deduped)}건")

        # 저장
        output_path: Optional[str] = None
        if deduped:
            saved = self.save_results(deduped)
            output_path = str(saved)

        duration = time.monotonic() - start_time
        result = CollectionResult(
            success=len(deduped) > 0,
            total_documents=len(deduped),
            category_counts=category_counts,
            failed_pages=failed_pages,
            output_path=output_path,
            duration_seconds=duration,
        )

        logger.info(
            f"[public_doc] 수집 완료: {result.total_documents}건, "
            f"{duration:.1f}초, 저장={output_path}"
        )
        return result

    async def collect_category(
        self,
        category_key: str,
        client: "httpx.AsyncClient",
    ) -> List[Dict[str, Any]]:
        """
        단일 카테고리를 페이지네이션으로 수집한다.

        첫 페이지에서 totalCount를 확인한 뒤 전체 페이지를 순차 수집한다.

        Parameters
        ----------
        category_key : str
            카테고리 키 (예: "press", "speech").
        client : httpx.AsyncClient
            공유 HTTP 클라이언트.

        Returns
        -------
        List[Dict[str, Any]]
            정규화된 문서 목록.
        """
        endpoint = self.config.categories.get(category_key)
        if not endpoint:
            logger.warning(f"[public_doc] 알 수 없는 카테고리: {category_key}")
            return []

        documents: List[Dict[str, Any]] = []

        # 첫 페이지로 totalCount 확인
        first_page = await self._fetch_page(endpoint, 1, client)
        if first_page is None:
            logger.warning(f"[public_doc] {category_key} 첫 페이지 수집 실패")
            return []

        items, total_count = self._parse_response(first_page)
        for item in items:
            documents.append(self._normalize_document(item, category_key))

        if total_count <= 0:
            return documents

        total_pages = min(
            (total_count + self.config.num_of_rows - 1) // self.config.num_of_rows,
            self.config.max_pages_per_category,
        )

        logger.debug(
            f"[public_doc] {category_key}: totalCount={total_count}, "
            f"수집 예정 페이지={total_pages}"
        )

        # 2페이지부터 순차 수집
        for page_no in range(2, total_pages + 1):
            page_data = await self._fetch_page(endpoint, page_no, client)
            if page_data is None:
                logger.warning(f"[public_doc] {category_key} p{page_no} 수집 실패, 건너뜀")
                continue
            items, _ = self._parse_response(page_data)
            for item in items:
                documents.append(self._normalize_document(item, category_key))

        return documents

    def save_results(
        self,
        documents: List[Dict[str, Any]],
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        문서 목록을 JSONL 파일로 저장한다.

        Parameters
        ----------
        documents : List[Dict[str, Any]]
            저장할 문서 목록.
        output_path : Optional[Path]
            저장 경로. None이면 config.output_dir 기준 타임스탬프 파일명 사용.

        Returns
        -------
        Path
            실제 저장된 파일 경로.
        """
        if output_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = Path(self.config.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            output_path = out_dir / f"public_docs_{ts}.jsonl"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for doc in documents:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")

        logger.info(f"[public_doc] {len(documents)}건 저장 완료: {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # 내부 HTTP 메서드
    # ------------------------------------------------------------------

    async def _fetch_page(
        self,
        endpoint: str,
        page_no: int,
        client: "httpx.AsyncClient",
    ) -> Optional[Dict[str, Any]]:
        """
        단일 페이지를 가져온다.

        Parameters
        ----------
        endpoint : str
            API 엔드포인트 이름 (예: "getDocPress").
        page_no : int
            페이지 번호 (1-based).
        client : httpx.AsyncClient
            공유 HTTP 클라이언트.

        Returns
        -------
        Optional[Dict[str, Any]]
            파싱된 응답 데이터. 실패 시 None.
        """
        url = f"{self.config.base_url}/{endpoint}"
        params: Dict[str, Any] = {
            "serviceKey": self.config.api_key,
            "numOfRows": self.config.num_of_rows,
            "pageNo": page_no,
            "format": self.config.response_format,
        }
        return await self._fetch_with_retry(url, params, client)

    async def _fetch_with_retry(
        self,
        url: str,
        params: Dict[str, Any],
        client: "httpx.AsyncClient",
    ) -> Optional[Dict[str, Any]]:
        """
        재시도 로직을 포함한 HTTP GET 요청.

        timeout / 429 / 5xx 에러만 재시도한다.
        4xx(429 제외)는 즉시 실패 반환.

        Parameters
        ----------
        url : str
            요청 URL.
        params : Dict[str, Any]
            쿼리 파라미터.
        client : httpx.AsyncClient
            공유 HTTP 클라이언트.

        Returns
        -------
        Optional[Dict[str, Any]]
            파싱된 응답. 실패 시 None.
        """
        for attempt in range(1, self.config.max_retries + 1):
            await self._rate_limit()
            try:
                response = await client.get(url, params=params)

                # 재시도 불필요한 4xx
                if 400 <= response.status_code < 500 and response.status_code != 429:
                    logger.warning(
                        f"[public_doc] HTTP {response.status_code} — 즉시 실패 반환 "
                        f"(url={url}, page={params.get('pageNo')})"
                    )
                    return None

                # 429 / 5xx: 재시도
                if response.status_code == 429 or response.status_code >= 500:
                    wait = self.config.retry_backoff_base**attempt
                    logger.warning(
                        f"[public_doc] HTTP {response.status_code} — "
                        f"{wait:.1f}초 후 재시도 (attempt {attempt}/{self.config.max_retries})"
                    )
                    await asyncio.sleep(wait)
                    continue

                response.raise_for_status()
                content_type = response.headers.get("content-type", "")

                # JSON 우선, XML 폴백
                if self.config.response_format == "json" and "json" in content_type:
                    return response.json()
                elif "xml" in content_type or response.text.lstrip().startswith("<"):
                    parsed_items, total = self._parse_xml_response(response.text)
                    return {"_xml_items": parsed_items, "_total_count": total}
                else:
                    # JSON 파싱 시도
                    try:
                        return response.json()
                    except Exception:
                        parsed_items, total = self._parse_xml_response(response.text)
                        return {"_xml_items": parsed_items, "_total_count": total}

            except httpx.TimeoutException as exc:  # type: ignore[union-attr]
                wait = self.config.retry_backoff_base**attempt
                logger.warning(
                    f"[public_doc] 타임아웃 — {wait:.1f}초 후 재시도 "
                    f"(attempt {attempt}/{self.config.max_retries}): {exc}"
                )
                if attempt < self.config.max_retries:
                    await asyncio.sleep(wait)
                else:
                    logger.error(f"[public_doc] 최대 재시도 초과 (타임아웃): {url}")
                    return None

            except Exception as exc:
                logger.error(f"[public_doc] 요청 오류 (attempt {attempt}): {exc}", exc_info=True)
                return None

        return None

    # ------------------------------------------------------------------
    # 파싱 메서드
    # ------------------------------------------------------------------

    def _parse_response(self, raw: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], int]:
        """
        API 응답을 파싱하여 (아이템 목록, totalCount) 를 반환한다.

        _fetch_with_retry 에서 XML을 미리 파싱한 경우(_xml_items 키)와
        JSON 응답 두 가지를 모두 처리한다.

        Parameters
        ----------
        raw : Dict[str, Any]
            _fetch_with_retry 반환 값.

        Returns
        -------
        Tuple[List[Dict[str, Any]], int]
            (아이템 목록, totalCount).
        """
        if "_xml_items" in raw:
            return raw["_xml_items"], raw.get("_total_count", 0)
        return self._parse_json_response(raw)

    def _parse_xml_response(self, content: str) -> Tuple[List[Dict[str, Any]], int]:
        """
        XML 응답을 파싱한다.

        Parameters
        ----------
        content : str
            XML 문자열.

        Returns
        -------
        Tuple[List[Dict[str, Any]], int]
            (아이템 목록, totalCount).
        """
        try:
            root = ET.fromstring(content)
        except ET.ParseError as exc:
            logger.warning(f"[public_doc] XML 파싱 실패: {exc}")
            return [], 0

        # resultCode 확인
        result_code = root.findtext("header/resultCode") or root.findtext(".//resultCode") or "00"
        if result_code not in ("00", "0", "200"):
            result_msg = root.findtext("header/resultMsg") or root.findtext(".//resultMsg") or ""
            logger.warning(f"[public_doc] API 에러 코드={result_code}: {result_msg}")
            return [], 0

        total_count_text = root.findtext(".//totalCount") or "0"
        try:
            total_count = int(total_count_text)
        except ValueError:
            total_count = 0

        items: List[Dict[str, Any]] = []
        for item_el in root.findall(".//resultList"):
            item_dict: Dict[str, Any] = {}
            for child in item_el:
                item_dict[child.tag] = child.text or ""
            items.append(item_dict)

        return items, total_count

    def _parse_json_response(self, body: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], int]:
        """
        JSON 응답을 파싱한다.

        응답 구조: response > header / body > resultList[]

        Parameters
        ----------
        body : Dict[str, Any]
            API JSON 응답 전체.

        Returns
        -------
        Tuple[List[Dict[str, Any]], int]
            (아이템 목록, totalCount).
        """
        # response 래퍼 벗기기
        response = body.get("response", body)
        if not isinstance(response, dict):
            response = body

        header = response.get("header", {})
        result_code = str(header.get("resultCode", "00"))
        if result_code not in ("00", "0", "200"):
            result_msg = header.get("resultMsg", "")
            logger.warning(f"[public_doc] API 에러 코드={result_code}: {result_msg}")
            return [], 0

        api_body = response.get("body", {})
        if not isinstance(api_body, dict):
            logger.warning("[public_doc] body 필드가 dict가 아님")
            return [], 0

        total_count = int(api_body.get("totalCount", 0))
        result_list = api_body.get("resultList", [])

        # 단일 dict 래핑 처리
        if isinstance(result_list, dict):
            result_list = [result_list]
        elif not isinstance(result_list, list):
            result_list = []

        return result_list, total_count

    # ------------------------------------------------------------------
    # 정규화
    # ------------------------------------------------------------------

    def _normalize_document(self, raw_item: Dict[str, Any], category: str) -> Dict[str, Any]:
        """
        API 원시 아이템을 출력 스키마로 정규화한다.

        카테고리별 meta 필드 차이를 처리하며, 없는 필드는 빈 문자열로 채운다.
        HTML 본문과 task 데이터는 원형 그대로 보존한다.

        Parameters
        ----------
        raw_item : Dict[str, Any]
            API 반환 원시 아이템.
        category : str
            카테고리 키 (예: "press").

        Returns
        -------
        Dict[str, Any]
            정규화된 문서 딕셔너리.
        """
        endpoint = self.config.categories.get(category, "")

        meta = raw_item.get("meta", {}) if isinstance(raw_item.get("meta"), dict) else {}
        data = raw_item.get("data", {}) if isinstance(raw_item.get("data"), dict) else {}

        # meta 필드가 raw_item 최상위에 플랫하게 있는 경우도 처리
        if not meta:
            meta = raw_item

        def _get(key: str, fallback: str = "") -> str:
            val = meta.get(key) or raw_item.get(key) or fallback
            return str(val) if val is not None else fallback

        html_text = str(data.get("text") or raw_item.get("text") or "")
        task_raw = data.get("task") or raw_item.get("task") or []
        tasks = self._normalize_tasks(task_raw)
        image_paths = self._extract_image_paths(html_text)

        return {
            "doc_id": _get("docId") or _get("doc_id"),
            "category": category,
            "category_endpoint": endpoint,
            "doc_type": _get("docType") or _get("doc_type"),
            "title": _get("title"),
            "file_name": _get("fileName") or _get("file_name"),
            "date": _get("date"),
            "time": _get("time"),
            "ministry": _get("ministry"),
            "department": _get("department"),
            "manager": _get("manager"),
            "relevant_departments": _get("relevantdepartments") or _get("relevant_departments"),
            "place": _get("place"),
            "person": _get("person"),
            "text": html_text,
            "image_paths": image_paths,
            "tasks": tasks,
            "raw_meta": meta,
            "collected_at": datetime.now().isoformat(),
            "_source": "public_doc",
        }

    @staticmethod
    def _normalize_tasks(task_raw: Any) -> List[Dict[str, Any]]:
        """
        task 필드를 정규화된 리스트로 변환한다.

        Parameters
        ----------
        task_raw : Any
            API에서 반환된 task 필드 원시값 (list / dict / None).

        Returns
        -------
        List[Dict[str, Any]]
            task_class, task_type, instruction, input, output 키를 포함한 목록.
        """
        if not task_raw:
            return []

        if isinstance(task_raw, dict):
            task_raw = [task_raw]
        elif not isinstance(task_raw, list):
            return []

        normalized = []
        for t in task_raw:
            if not isinstance(t, dict):
                continue
            normalized.append(
                {
                    "task_class": str(t.get("task_class") or ""),
                    "task_type": str(t.get("task_type") or ""),
                    "instruction": str(t.get("instruction") or ""),
                    "input": str(t.get("input") or ""),
                    "output": str(t.get("output") or ""),
                }
            )
        return normalized

    @staticmethod
    def _extract_image_paths(html_content: str) -> List[str]:
        """
        HTML 본문에서 이미지 경로를 추출한다.

        <img src="..."> 패턴을 단순 문자열 파싱으로 추출한다.

        Parameters
        ----------
        html_content : str
            원문 HTML 텍스트.

        Returns
        -------
        List[str]
            이미지 URL/경로 목록.
        """
        if not html_content:
            return []

        paths: List[str] = []
        search = html_content
        while True:
            idx = search.lower().find("<img")
            if idx == -1:
                break
            search = search[idx:]
            src_idx = search.lower().find("src=")
            if src_idx == -1 or src_idx > 200:
                search = search[4:]
                continue
            src_part = search[src_idx + 4 :]
            if src_part.startswith('"'):
                end = src_part.find('"', 1)
                if end != -1:
                    paths.append(src_part[1:end])
            elif src_part.startswith("'"):
                end = src_part.find("'", 1)
                if end != -1:
                    paths.append(src_part[1:end])
            search = search[4:]

        return paths

    # ------------------------------------------------------------------
    # Rate limit
    # ------------------------------------------------------------------

    async def _rate_limit(self) -> None:
        """
        requests_per_second 설정에 따라 요청 간격을 제어한다.

        timestamp 기반으로 마지막 요청과의 최소 간격을 보장한다.
        """
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            await asyncio.sleep(self._min_interval - elapsed)
        self._last_request_time = time.monotonic()


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="행안부 공공문서 학습데이터 수집 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python -m src.data_collection_preprocessing.collect_public_docs --min-docs 1000
  python -m src.data_collection_preprocessing.collect_public_docs --dry-run
  python -m src.data_collection_preprocessing.collect_public_docs --categories press speech
        """,
    )
    parser.add_argument(
        "--min-docs",
        type=int,
        default=1000,
        help="최소 수집 목표 건수 (기본값: 1000)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="실제 API 호출 없이 설정만 확인",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=["press", "speech", "publication", "report", "plan", "all"],
        default=None,
        help="수집할 카테고리 선택 (기본값: 전체)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="저장 경로 (기본값: data/raw/public_docs/public_docs_YYYYMMDD_HHMMSS.jsonl)",
    )
    return parser


async def _run_async(args: argparse.Namespace) -> int:
    config = get_config().public_doc

    if args.dry_run:
        logger.info("[public_doc] --dry-run 모드")
        logger.info(f"  base_url        = {config.base_url}")
        logger.info(f"  api_key 설정됨  = {bool(config.api_key)}")
        logger.info(f"  num_of_rows     = {config.num_of_rows}")
        logger.info(f"  max_pages       = {config.max_pages_per_category}")
        logger.info(f"  rate_limit      = {config.requests_per_second} tps")
        logger.info(f"  categories      = {list(config.categories.keys())}")
        return 0

    if args.categories:
        config.categories = {k: v for k, v in config.categories.items() if k in args.categories}

    collector = PublicDocumentCollector(config)
    result = await collector.collect_all(min_docs=args.min_docs)

    if args.output and result.output_path:
        # 지정된 경로로 이동
        import shutil

        shutil.move(result.output_path, args.output)
        logger.info(f"[public_doc] 결과 이동: {result.output_path} → {args.output}")

    logger.info(f"[public_doc] 최종 결과: {result.total_documents}건, 성공={result.success}")
    for cat, cnt in result.category_counts.items():
        logger.info(f"  {cat}: {cnt}건")

    return 0 if result.success else 1


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    exit_code = asyncio.run(_run_async(args))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
