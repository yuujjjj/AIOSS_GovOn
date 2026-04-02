"""
행안부 공공문서 수집 모듈 단위 테스트

테스트 구조:
  TestResponseParsing       — JSON/XML 파싱, 빈 응답, 에러 코드, task 보존
  TestDocumentNormalization — HTML 보존, 이미지 경로, 누락 필드, 엔드포인트별 차이
  TestPagination            — totalCount→페이지, max_pages cap, 다중 페이지
  TestRetry                 — timeout/429/500 재시도, 400 즉시 실패
  TestCollectAll            — 5개 카테고리, min_docs, 부분 실패, doc_id dedup
  TestSaveResults           — JSONL 형식, task 보존
"""

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.data_collection_preprocessing.collect_public_docs import (
    CollectionResult,
    PublicDocumentCollector,
)
from src.data_collection_preprocessing.config import PublicDocumentConfig

# ---------------------------------------------------------------------------
# 픽스처 & 헬퍼
# ---------------------------------------------------------------------------


def _make_config(**overrides) -> PublicDocumentConfig:
    """테스트용 최소 설정 반환"""
    defaults = {
        "api_key": "test_key_dummy",
        "num_of_rows": 10,
        "max_pages_per_category": 3,
        "requests_per_second": 100.0,  # 테스트에서 rate limit 없애기
        "max_retries": 2,
        "retry_backoff_base": 0.01,
        "timeout": 5.0,
        "response_format": "json",
    }
    defaults.update(overrides)
    return PublicDocumentConfig(**defaults)


def _json_response(
    items: List[Dict],
    total_count: int,
    result_code: str = "00",
) -> Dict[str, Any]:
    """정상 JSON 응답 픽스처"""
    return {
        "response": {
            "header": {"resultCode": result_code, "resultMsg": "OK"},
            "body": {
                "totalCount": total_count,
                "resultList": items,
            },
        }
    }


def _xml_response(items: List[Dict], total_count: int) -> str:
    """단순 XML 응답 픽스처"""
    item_xml = ""
    for item in items:
        fields = "".join(f"<{k}>{v}</{k}>" for k, v in item.items())
        item_xml += f"<resultList>{fields}</resultList>"
    return (
        f"<response>"
        f"<header><resultCode>00</resultCode><resultMsg>OK</resultMsg></header>"
        f"<body><totalCount>{total_count}</totalCount>{item_xml}</body>"
        f"</response>"
    )


def _make_item(doc_id: str = "01_01_00000001", category: str = "press") -> Dict[str, Any]:
    """문서 원시 아이템 픽스처"""
    return {
        "meta": {
            "docId": doc_id,
            "docType": "보도자료",
            "title": f"제목_{doc_id}",
            "fileName": "test.hwp",
            "date": "2024-01-01",
            "time": "10:00",
            "ministry": "행정안전부",
            "department": "창조정부기획과",
            "manager": "과장 홍길동",
            "relevantdepartments": "",
        },
        "data": {
            "text": "<p>본문 HTML</p>",
            "task": [
                {
                    "task_class": "요약",
                    "task_type": "abstractive",
                    "instruction": "요약하시오",
                    "input": "입력",
                    "output": "출력",
                }
            ],
        },
    }


# ---------------------------------------------------------------------------
# TestResponseParsing
# ---------------------------------------------------------------------------


class TestResponseParsing:
    """JSON / XML 파싱, 에러 코드, task 보존"""

    def setup_method(self):
        self.collector = PublicDocumentCollector(_make_config())

    # --- JSON 파싱 ---

    def test_parse_json_normal(self):
        items = [_make_item("DOC001"), _make_item("DOC002")]
        resp = _json_response(items, total_count=2)
        parsed_items, total = self.collector._parse_json_response(resp)
        assert total == 2
        assert len(parsed_items) == 2

    def test_parse_json_empty_list(self):
        resp = _json_response([], total_count=0)
        parsed_items, total = self.collector._parse_json_response(resp)
        assert total == 0
        assert parsed_items == []

    def test_parse_json_error_code(self):
        resp = _json_response([], total_count=0, result_code="99")
        parsed_items, total = self.collector._parse_json_response(resp)
        assert total == 0
        assert parsed_items == []

    def test_parse_json_single_dict_result_list(self):
        """resultList가 list 대신 dict 단일 항목인 경우"""
        resp = {
            "response": {
                "header": {"resultCode": "00", "resultMsg": "OK"},
                "body": {
                    "totalCount": 1,
                    "resultList": _make_item("DOC001"),
                },
            }
        }
        parsed_items, total = self.collector._parse_json_response(resp)
        assert total == 1
        assert len(parsed_items) == 1

    def test_parse_json_task_preserved(self):
        """task 데이터가 파싱 결과에 보존되는지 확인"""
        item = _make_item("DOC001")
        resp = _json_response([item], total_count=1)
        parsed_items, _ = self.collector._parse_json_response(resp)
        assert parsed_items[0]["data"]["task"][0]["task_class"] == "요약"

    # --- XML 파싱 ---

    def test_parse_xml_normal(self):
        items = [{"docId": "001", "title": "제목A"}, {"docId": "002", "title": "제목B"}]
        xml_str = _xml_response(items, total_count=2)
        parsed_items, total = self.collector._parse_xml_response(xml_str)
        assert total == 2
        assert len(parsed_items) == 2

    def test_parse_xml_empty(self):
        xml_str = _xml_response([], total_count=0)
        parsed_items, total = self.collector._parse_xml_response(xml_str)
        assert total == 0
        assert parsed_items == []

    def test_parse_xml_error_code(self):
        xml_str = (
            "<response>"
            "<header><resultCode>99</resultCode><resultMsg>ERROR</resultMsg></header>"
            "<body><totalCount>0</totalCount></body>"
            "</response>"
        )
        parsed_items, total = self.collector._parse_xml_response(xml_str)
        assert parsed_items == []

    def test_parse_xml_malformed(self):
        parsed_items, total = self.collector._parse_xml_response("<bad><xml")
        assert parsed_items == []
        assert total == 0

    # --- _parse_response 분기 ---

    def test_parse_response_xml_path(self):
        """_xml_items 키가 있으면 XML 파싱 결과를 직접 반환"""
        raw = {"_xml_items": [{"docId": "X"}], "_total_count": 1}
        items, total = self.collector._parse_response(raw)
        assert total == 1
        assert items[0]["docId"] == "X"

    def test_parse_response_json_path(self):
        resp = _json_response([_make_item("J001")], total_count=1)
        items, total = self.collector._parse_response(resp)
        assert total == 1


# ---------------------------------------------------------------------------
# TestDocumentNormalization
# ---------------------------------------------------------------------------


class TestDocumentNormalization:
    """HTML 보존, 이미지 경로, 누락 필드, 엔드포인트별 차이"""

    def setup_method(self):
        self.collector = PublicDocumentCollector(_make_config())

    def test_html_preserved(self):
        item = _make_item("DOC001")
        item["data"]["text"] = "<p>HTML 본문 <b>강조</b></p>"
        doc = self.collector._normalize_document(item, "press")
        assert "<p>HTML 본문 <b>강조</b></p>" == doc["text"]

    def test_image_paths_extracted(self):
        item = _make_item("DOC001")
        item["data"]["text"] = '<img src="http://example.com/a.png"><img src="http://b.com/b.jpg">'
        doc = self.collector._normalize_document(item, "press")
        assert "http://example.com/a.png" in doc["image_paths"]
        assert "http://b.com/b.jpg" in doc["image_paths"]

    def test_image_paths_empty_when_no_img(self):
        item = _make_item("DOC001")
        item["data"]["text"] = "<p>텍스트만</p>"
        doc = self.collector._normalize_document(item, "press")
        assert doc["image_paths"] == []

    def test_tasks_normalized(self):
        item = _make_item("DOC001")
        doc = self.collector._normalize_document(item, "press")
        assert len(doc["tasks"]) == 1
        task = doc["tasks"][0]
        assert task["task_class"] == "요약"
        assert task["task_type"] == "abstractive"
        assert "instruction" in task
        assert "input" in task
        assert "output" in task

    def test_missing_fields_become_empty_string(self):
        """speech 카테고리에 ministry/department 없어도 빈 문자열 반환"""
        item = {
            "meta": {
                "docId": "S001",
                "title": "연설문",
                "place": "청와대",
                "person": "대통령",
            },
            "data": {"text": "연설 내용", "task": []},
        }
        doc = self.collector._normalize_document(item, "speech")
        assert doc["place"] == "청와대"
        assert doc["person"] == "대통령"
        assert doc["ministry"] == ""
        assert doc["department"] == ""

    def test_press_specific_fields(self):
        item = _make_item("P001")
        item["meta"]["relevantdepartments"] = "기획조정실"
        doc = self.collector._normalize_document(item, "press")
        assert doc["relevant_departments"] == "기획조정실"
        assert doc["ministry"] == "행정안전부"

    def test_source_field_set(self):
        doc = self.collector._normalize_document(_make_item("DOC001"), "press")
        assert doc["_source"] == "public_doc"

    def test_collected_at_present(self):
        doc = self.collector._normalize_document(_make_item("DOC001"), "press")
        assert "collected_at" in doc
        assert doc["collected_at"]  # ISO datetime 문자열

    def test_category_endpoint_set(self):
        doc = self.collector._normalize_document(_make_item("DOC001"), "press")
        assert doc["category"] == "press"
        assert doc["category_endpoint"] == "getDocPress"

    def test_flat_meta_item(self):
        """meta/data 래퍼 없이 플랫 구조인 경우도 정규화"""
        item = {
            "docId": "FLAT001",
            "title": "플랫 제목",
            "text": "본문",
            "task": [],
        }
        doc = self.collector._normalize_document(item, "report")
        assert doc["doc_id"] == "FLAT001"
        assert doc["title"] == "플랫 제목"


# ---------------------------------------------------------------------------
# TestPagination
# ---------------------------------------------------------------------------


class TestPagination:
    """totalCount→페이지 계산, max_pages cap, 다중 페이지 수집"""

    def setup_method(self):
        self.config = _make_config(num_of_rows=2, max_pages_per_category=3)
        self.collector = PublicDocumentCollector(self.config)

    @pytest.mark.asyncio
    async def test_total_count_to_pages(self):
        """totalCount=5, num_of_rows=2 → 3 페이지"""
        call_count = 0

        async def fake_fetch(endpoint, page_no, client):
            nonlocal call_count
            call_count += 1
            items = [_make_item(f"D{page_no:02d}_{i}") for i in range(2)]
            return _json_response(items, total_count=5)

        self.collector._fetch_page = fake_fetch
        docs = await self.collector.collect_category("press", client=None)
        # 페이지 1,2,3 호출 (3페이지째는 남은 1건이지만 2건 반환되도 OK)
        assert call_count == 3
        assert len(docs) == 6  # 2건 × 3페이지

    @pytest.mark.asyncio
    async def test_max_pages_cap(self):
        """totalCount가 매우 커도 max_pages_per_category(=3)까지만 수집"""
        call_count = 0

        async def fake_fetch(endpoint, page_no, client):
            nonlocal call_count
            call_count += 1
            items = [_make_item(f"BIG{page_no:02d}_{i}") for i in range(2)]
            return _json_response(items, total_count=10000)

        self.collector._fetch_page = fake_fetch
        docs = await self.collector.collect_category("press", client=None)
        assert call_count == 3  # max_pages_per_category=3
        assert len(docs) == 6

    @pytest.mark.asyncio
    async def test_single_page_total_count(self):
        """totalCount <= num_of_rows 이면 1페이지만"""
        call_count = 0

        async def fake_fetch(endpoint, page_no, client):
            nonlocal call_count
            call_count += 1
            items = [_make_item(f"S{i}") for i in range(1)]
            return _json_response(items, total_count=1)

        self.collector._fetch_page = fake_fetch
        docs = await self.collector.collect_category("press", client=None)
        assert call_count == 1
        assert len(docs) == 1

    @pytest.mark.asyncio
    async def test_first_page_failure_returns_empty(self):
        """첫 페이지 실패 시 빈 목록 반환"""

        async def fake_fetch(endpoint, page_no, client):
            return None

        self.collector._fetch_page = fake_fetch
        docs = await self.collector.collect_category("press", client=None)
        assert docs == []


# ---------------------------------------------------------------------------
# TestRetry
# ---------------------------------------------------------------------------


class TestRetry:
    """timeout/429/500 재시도, 400 즉시 실패"""

    def setup_method(self):
        self.config = _make_config(max_retries=2, retry_backoff_base=0.01)
        self.collector = PublicDocumentCollector(self.config)

    @pytest.mark.asyncio
    async def test_timeout_retries(self):
        """TimeoutException 발생 시 max_retries만큼 재시도"""
        import httpx

        attempt_count = 0

        async def mock_get(*args, **kwargs):
            nonlocal attempt_count
            attempt_count += 1
            raise httpx.TimeoutException("timeout")

        mock_client = AsyncMock()
        mock_client.get = mock_get

        result = await self.collector._fetch_with_retry("http://x", {}, mock_client)
        assert result is None
        assert attempt_count == self.config.max_retries

    @pytest.mark.asyncio
    async def test_429_retries(self):
        """429 응답 시 재시도"""
        import httpx

        attempt_count = 0

        async def mock_get(*args, **kwargs):
            nonlocal attempt_count
            attempt_count += 1
            resp = MagicMock()
            resp.status_code = 429
            resp.raise_for_status = MagicMock()
            return resp

        mock_client = AsyncMock()
        mock_client.get = mock_get

        result = await self.collector._fetch_with_retry("http://x", {}, mock_client)
        assert result is None
        assert attempt_count == self.config.max_retries

    @pytest.mark.asyncio
    async def test_500_retries(self):
        """500 응답 시 재시도"""
        import httpx

        attempt_count = 0

        async def mock_get(*args, **kwargs):
            nonlocal attempt_count
            attempt_count += 1
            resp = MagicMock()
            resp.status_code = 500
            resp.raise_for_status = MagicMock()
            return resp

        mock_client = AsyncMock()
        mock_client.get = mock_get

        result = await self.collector._fetch_with_retry("http://x", {}, mock_client)
        assert result is None
        assert attempt_count == self.config.max_retries

    @pytest.mark.asyncio
    async def test_400_immediate_failure(self):
        """400 응답 시 즉시 실패 (재시도 없음)"""
        attempt_count = 0

        async def mock_get(*args, **kwargs):
            nonlocal attempt_count
            attempt_count += 1
            resp = MagicMock()
            resp.status_code = 400
            resp.raise_for_status = MagicMock()
            return resp

        mock_client = AsyncMock()
        mock_client.get = mock_get

        result = await self.collector._fetch_with_retry("http://x", {}, mock_client)
        assert result is None
        assert attempt_count == 1  # 재시도 없음

    @pytest.mark.asyncio
    async def test_success_on_second_attempt(self):
        """첫 시도 500, 두 번째 시도 200 성공"""
        import httpx

        attempt_count = 0
        good_response = _json_response([_make_item("OK1")], total_count=1)

        async def mock_get(*args, **kwargs):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count == 1:
                resp = MagicMock()
                resp.status_code = 500
                resp.raise_for_status = MagicMock()
                return resp
            # 두 번째 성공
            resp = MagicMock()
            resp.status_code = 200
            resp.headers = {"content-type": "application/json"}
            resp.raise_for_status = MagicMock()
            resp.json = MagicMock(return_value=good_response)
            resp.text = json.dumps(good_response)
            return resp

        mock_client = AsyncMock()
        mock_client.get = mock_get

        result = await self.collector._fetch_with_retry("http://x", {}, mock_client)
        assert result is not None
        assert attempt_count == 2


# ---------------------------------------------------------------------------
# TestCollectAll
# ---------------------------------------------------------------------------


class TestCollectAll:
    """5개 카테고리, min_docs, 부분 실패, doc_id dedup"""

    def setup_method(self):
        self.config = _make_config(
            num_of_rows=5,
            max_pages_per_category=2,
            requests_per_second=1000.0,
        )

    @pytest.mark.asyncio
    async def test_five_primary_categories(self):
        """getDocAll 제외한 5개 카테고리가 모두 수집된다"""
        collected_endpoints = []

        async def fake_collect_category(cat_key, client):
            collected_endpoints.append(cat_key)
            return [_make_item(f"{cat_key}_001")]

        collector = PublicDocumentCollector(self.config)
        collector.collect_category = fake_collect_category

        with patch("src.data_collection_preprocessing.collect_public_docs._HTTPX_AVAILABLE", True):
            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
                mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
                result = await collector.collect_all(min_docs=1)

        primary = {"press", "speech", "publication", "report", "plan"}
        assert primary.issubset(set(collected_endpoints))
        assert "all" not in collected_endpoints

    @pytest.mark.asyncio
    async def test_supplemental_when_below_min_docs(self):
        """min_docs 미달 시 getDocAll 보충 수집"""
        collected = []

        async def fake_collect(cat_key, client):
            collected.append(cat_key)
            # 각 카테고리에서 1건만 반환 → 5건 < min_docs=100
            return [_make_item(f"{cat_key}_001")]

        collector = PublicDocumentCollector(self.config)
        collector.collect_category = fake_collect

        with patch("src.data_collection_preprocessing.collect_public_docs._HTTPX_AVAILABLE", True):
            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
                mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
                result = await collector.collect_all(min_docs=100)

        assert "all" in collected

    @pytest.mark.asyncio
    async def test_doc_id_dedup(self):
        """동일 doc_id는 중복 제거된다"""
        # collect_category는 _normalize_document를 거친 dict(doc_id 키 포함)를 반환한다.
        # 테스트에서도 정규화된 형태로 반환해야 dedup 로직이 동작한다.
        collector = PublicDocumentCollector(self.config)

        async def fake_collect(cat_key, client):
            return [collector._normalize_document(_make_item("SAME_ID"), cat_key)]

        collector.collect_category = fake_collect

        with patch("src.data_collection_preprocessing.collect_public_docs._HTTPX_AVAILABLE", True):
            with patch("httpx.AsyncClient") as mock_client_cls:
                mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
                mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
                result = await collector.collect_all(min_docs=1)

        # 5개 카테고리에서 같은 SAME_ID → 중복 제거 후 1건
        assert result.total_documents == 1

    @pytest.mark.asyncio
    async def test_no_api_key_returns_failure(self):
        config = _make_config(api_key="")
        collector = PublicDocumentCollector(config)
        result = await collector.collect_all(min_docs=1)
        assert not result.success
        assert result.total_documents == 0

    @pytest.mark.asyncio
    async def test_httpx_not_available_returns_failure(self):
        collector = PublicDocumentCollector(self.config)
        with patch("src.data_collection_preprocessing.collect_public_docs._HTTPX_AVAILABLE", False):
            result = await collector.collect_all(min_docs=1)
        assert not result.success


# ---------------------------------------------------------------------------
# TestSaveResults
# ---------------------------------------------------------------------------


class TestSaveResults:
    """JSONL 형식 저장, task 보존"""

    def setup_method(self):
        self.config = _make_config()
        self.collector = PublicDocumentCollector(self.config)

    def test_save_creates_jsonl(self):
        docs = [
            self.collector._normalize_document(_make_item(f"DOC{i:03d}"), "press") for i in range(5)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "test_output.jsonl"
            saved = self.collector.save_results(docs, output_path=out_path)
            assert saved.exists()
            lines = saved.read_text(encoding="utf-8").strip().splitlines()
            assert len(lines) == 5

    def test_each_line_valid_json(self):
        docs = [self.collector._normalize_document(_make_item("D001"), "press")]
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "out.jsonl"
            saved = self.collector.save_results(docs, output_path=out_path)
            lines = saved.read_text(encoding="utf-8").strip().splitlines()
            for line in lines:
                parsed = json.loads(line)
                assert "_source" in parsed
                assert parsed["_source"] == "public_doc"

    def test_tasks_preserved_in_jsonl(self):
        doc = self.collector._normalize_document(_make_item("T001"), "press")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "task_test.jsonl"
            saved = self.collector.save_results([doc], output_path=out_path)
            line = json.loads(saved.read_text(encoding="utf-8").strip())
            assert "tasks" in line
            assert len(line["tasks"]) == 1
            assert line["tasks"][0]["task_class"] == "요약"

    def test_save_auto_path_created(self):
        """output_path=None이면 config.output_dir 아래에 자동 생성"""
        docs = [self.collector._normalize_document(_make_item("AUTO001"), "press")]
        with tempfile.TemporaryDirectory() as tmpdir:
            self.collector.config.output_dir = str(Path(tmpdir) / "auto_out")
            saved = self.collector.save_results(docs)
            assert saved.exists()
            assert saved.suffix == ".jsonl"

    def test_empty_documents_creates_empty_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "empty.jsonl"
            saved = self.collector.save_results([], output_path=out_path)
            assert saved.exists()
            assert saved.read_text() == ""

    def test_html_preserved_in_jsonl(self):
        item = _make_item("HTML001")
        item["data"]["text"] = "<div><p>HTML <b>테스트</b></p></div>"
        doc = self.collector._normalize_document(item, "press")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "html_test.jsonl"
            saved = self.collector.save_results([doc], output_path=out_path)
            loaded = json.loads(saved.read_text(encoding="utf-8").strip())
            assert loaded["text"] == "<div><p>HTML <b>테스트</b></p></div>"


# ---------------------------------------------------------------------------
# TestExtractImagePaths
# ---------------------------------------------------------------------------


class TestExtractImagePaths:
    """이미지 경로 추출 유닛 테스트"""

    def test_double_quote_src(self):
        html = '<img src="http://example.com/a.png">'
        result = PublicDocumentCollector._extract_image_paths(html)
        assert result == ["http://example.com/a.png"]

    def test_single_quote_src(self):
        html = "<img src='http://example.com/b.jpg'>"
        result = PublicDocumentCollector._extract_image_paths(html)
        assert result == ["http://example.com/b.jpg"]

    def test_multiple_images(self):
        html = '<img src="a.png"><img src="b.png"><img src="c.png">'
        result = PublicDocumentCollector._extract_image_paths(html)
        assert len(result) == 3

    def test_no_images(self):
        html = "<p>이미지 없음</p>"
        result = PublicDocumentCollector._extract_image_paths(html)
        assert result == []

    def test_empty_string(self):
        assert PublicDocumentCollector._extract_image_paths("") == []


# ---------------------------------------------------------------------------
# TestCollectionResultProperty
# ---------------------------------------------------------------------------


class TestCollectionResultProperty:
    """CollectionResult.meets_minimum 속성 테스트"""

    def test_meets_minimum_true(self):
        r = CollectionResult(success=True, total_documents=1000)
        assert r.meets_minimum is True

    def test_meets_minimum_false(self):
        r = CollectionResult(success=True, total_documents=999)
        assert r.meets_minimum is False

    def test_meets_minimum_exact(self):
        r = CollectionResult(success=True, total_documents=1000)
        assert r.meets_minimum is True
