"""
행정안전부 정부 공문서 AI 학습데이터 최종 수집 스크립트.

이슈 #157: 고품질 행정 공문서(보도자료, 보고서 등) 수집 및 인덱싱.
Base URL: apis.data.go.kr/1741000/publicDoc
상세기능: /getDocAll (5종 전체 조회)
"""

import argparse
import json
import os
import zipfile
import io
from pathlib import Path
from typing import Dict, List

import requests
from loguru import logger

BASE_URL = "https://apis.data.go.kr/1741000/publicDoc/getDocAll"

class PublicDocCollector:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch_all_docs(self, limit: int = 50) -> List[Dict]:
        """모든 유형의 공문서 AI 데이터를 일괄 수집한다. ZIP 다운로드 방식을 지원함."""
        params = {
            "serviceKey": self.api_key,
            "pageNo": 1,
            "numOfRows": limit,
            "type": "json"
        }
        
        try:
            logger.info(f"정부 공문서 AI 데이터 수집 시작 (URL: {BASE_URL})")
            response = requests.get(BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            body = data.get("response", {}).get("body", {})
            result_list = body.get("resultList", [])

            if not result_list:
                logger.warning("API 응답에 resultList가 없습니다.")
                return []

            all_items = []
            for res in result_list:
                download_url = res.get("url")
                if download_url and res.get("extType") == "zip":
                    logger.info(f"ZIP 데이터 다운로드 중: {download_url}")
                    zip_res = requests.get(download_url, timeout=60)
                    with zipfile.ZipFile(io.BytesIO(zip_res.content)) as z:
                        for filename in z.namelist():
                            if filename.endswith(".json"):
                                with z.open(filename) as f:
                                    content = json.load(f)
                                    if isinstance(content, list):
                                        all_items.extend(content)
                                    else:
                                        all_items.append(content)
            return all_items
        except Exception as e:
            logger.error(f"공문서 API 호출 및 다운로드 실패: {e}")
            return []

    def process_item(self, item: Dict) -> List[Dict]:
        """수집된 데이터를 LLM 학습용 Instruction 포맷으로 변환한다."""
        results = []
        doc_id = item.get("doc_id", item.get("id", "DOC"))
        title = item.get("doc_nm") or item.get("title") or "정부 공문서"
        
        # 1. 원문 및 요약 데이터 추출
        input_text = item.get("doc_cont") or item.get("content") or ""
        output_text = item.get("summary") or item.get("rewrite") or ""
        
        # 만약 요약/재구성이 없으면 본문 자체를 활용한 태스크 생성 (Fallback)
        if not output_text and input_text:
            # 본문이 충분히 길 경우 간단한 요약 태스크로 활용 가능성이 있으나,
            # 학습 품질을 위해 우선은 데이터가 있는 경우만 처리하되, 
            # 필드명이 다를 가능성을 고려하여 더 넓게 탐색
            for key in ["summaries", "content_summary", "description"]:
                if item.get(key):
                    output_text = item.get(key)
                    break

        if input_text and output_text:
            results.append({
                "instruction": "다음 정부 공문서 내용을 바탕으로 핵심 내용을 요약하고 공공기관 보고서 형식(개조식)으로 재작성하세요.",
                "input": f"제목: {title}\n본문: {input_text}",
                "output": output_text
            })
        elif input_text:
            # 요약본이 없더라도 본문이 있으면 '제목 생성' 태스크로 활용
            results.append({
                "instruction": "다음 공문서 본문을 읽고 적절한 제목을 추출하세요.",
                "input": f"본문: {input_text}",
                "output": title
            })

        # 2. 질의응답 태스크
        qa_list = item.get("qna_data", item.get("qa", []))
        if isinstance(qa_list, list):
            for qa in qa_list:
                q = qa.get("question", qa.get("q", ""))
                a = qa.get("answer", qa.get("a", ""))
                if q and a:
                    results.append({
                        "instruction": f"제공된 공문서 '{title}'의 내용을 바탕으로 다음 질문에 답하세요.",
                        "input": f"질문: {q}",
                        "output": a
                    })

        return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output", type=str, default="data/raw/manuals/public_docs.jsonl")
    args = parser.parse_args()

    api_key = os.getenv("DATA_GO_KR_API_KEY")
    if not api_key:
        logger.error("DATA_GO_KR_API_KEY 환경변수가 없습니다.")
        return

    collector = PublicDocCollector(api_key)
    raw_docs = collector.fetch_all_docs(limit=args.limit)
    
    if not raw_docs:
        logger.warning("수집된 데이터가 없습니다. API 승인 및 키 상태를 확인하세요.")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    total_records = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for d in raw_docs:
            processed_items = collector.process_item(d)
            for item in processed_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                total_records += 1
            
    logger.info(f"✅ 수집 및 변환 완료: {len(raw_docs)}개 문서로부터 {total_records}개 레코드 생성 -> {output_path}")

if __name__ == "__main__":
    main()
