"""
확장 RAG를 위한 다중 인덱스(Multi-Index) 통합 빌드 스크립트.

이슈 #157: 법령/매뉴얼/공시정보 데이터 인덱싱 완료.
ADR-004: 의미 검색(FAISS) + 키워드 검색(BM25) 인덱스 일괄 생성.

대상:
- LAW (법령): data/raw/laws.jsonl
- MANUAL (매뉴얼): data/raw/manuals/*.pdf, *.hwp, *.txt
- NOTICE (공시정보): data/raw/notices.jsonl
- CASE (유사사례): data/processed/v2_train.jsonl
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from loguru import logger

# 프로젝트 루트 추가
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data_collection_preprocessing.embedding import EmbeddingPipeline
from src.inference.bm25_indexer import BM25Indexer
from src.inference.document_processor import DocumentProcessor
from src.inference.index_manager import IndexType, MultiIndexManager


class MultiIndexBuilder:
    def __init__(self, model_name: str, index_dir: str, bm25_dir: str):
        self.embedding_pipeline = EmbeddingPipeline(model_name=model_name)
        self.doc_processor = DocumentProcessor()
        self.index_manager = MultiIndexManager(base_dir=index_dir)
        self.bm25_dir = bm25_dir
        os.makedirs(bm25_dir, exist_ok=True)

    def process_and_add(self, index_type: IndexType, data_source_path: str):
        """특정 타입의 데이터를 처리하여 인덱스에 추가한다."""
        logger.info(f"[{index_type.value}] 인덱스 빌드 시작: {data_source_path}")
        
        # 1. 문서 처리 (파싱 & 청킹)
        all_metadata = []
        path = Path(data_source_path)
        
        if path.is_file() and path.suffix == ".jsonl":
            # JSONL 처리
            import json
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    # 이미 청킹된 텍스트가 아닌 원문일 경우 청킹 수행
                    chunks = self.doc_processor.process_batch([data.get("content", "")], index_type) if "content" in data else []
                    # 단순화를 위해 여기서는 process 메서드를 활용하도록 설계 (실제 구현에 맞게 조정 가능)
                    # 임시 구현: 직접 Metadata 생성 또는 process 활용
                    ... 
            # 실제로는 JSONL 내부의 대형 텍스트를 청킹해야 하므로 로직 보강 필요
            # 여기서는 편의상 jsonl 내의 각 라인을 하나의 문서로 취급하여 처리하는 예시
            metadata_list = self._process_jsonl_source(path, index_type)
        else:
            # 디렉토리 내 파일(PDF/HWP/TXT) 처리
            files = [str(f) for f in path.glob("**/*") if f.suffix.lower() in [".pdf", ".hwp", ".txt"]]
            batch_result = self.doc_processor.process_batch(files, index_type)
            metadata_list = batch_result.succeeded

        if not metadata_list:
            logger.warning(f"처리된 문서가 없습니다: {index_type.value}")
            return

        # 2. 임베딩 생성
        texts = [m.extras.get("chunk_text") or m.title for m in metadata_list]
        embeddings = self.embedding_pipeline.embed_documents(texts)

        # 3. FAISS 저장
        self.index_manager.add_documents(index_type, embeddings, metadata_list)
        self.index_manager.save_index(index_type)

        # 4. BM25 빌드 및 저장
        bm25_indexer = BM25Indexer()
        bm25_indexer.build_index(texts)
        bm25_path = os.path.join(self.bm25_dir, f"{index_type.value}.pkl")
        bm25_indexer.save(bm25_path)
        
        logger.info(f"[{index_type.value}] 빌드 완료: {len(metadata_list)} 청크")

    def _process_jsonl_source(self, path: Path, index_type: IndexType):
        """JSONL 파일을 읽어 DocumentProcessor로 청킹 처리한다."""
        import json
        metadata_list = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                # 텍스트 파일로 임시 저장 후 process 호출 (가장 정확한 방법)
                temp_file = Path(f"data/raw/temp_{index_type.value}.txt")
                temp_file.write_text(data.get("content", ""), encoding="utf-8")
                
                chunks = self.doc_processor.process(
                    str(temp_file), 
                    index_type, 
                    source=data.get("source", ""),
                    title=data.get("title", ""),
                    category=data.get("category", "")
                )
                metadata_list.extend(chunks)
                if temp_file.exists(): temp_file.unlink()
        return metadata_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="intfloat/multilingual-e5-large")
    parser.add_argument("--index-dir", type=str, default="models/faiss_index")
    parser.add_argument("--bm25-dir", type=str, default="models/bm25_index")
    args = parser.parse_args()

    builder = MultiIndexBuilder(args.model_name, args.index_dir, args.bm25_dir)

    # 각 데이터 소스별 빌드 (수집된 파일이 있다는 전제)
    sources = [
        (IndexType.LAW, "data/raw/laws.jsonl"),
        (IndexType.MANUAL, "data/raw/manuals"),
        (IndexType.NOTICE, "data/raw/notices.jsonl"),
        (IndexType.CASE, "data/processed/v2_train.jsonl") # 기존 사례
    ]

    for itype, src in sources:
        if os.path.exists(src):
            builder.process_and_add(itype, src)
        else:
            logger.warning(f"데이터 소스가 존재하지 않아 스킵합니다: {src}")

    logger.info("모든 멀티 인덱스 빌드 작업이 완료되었습니다.")

if __name__ == "__main__":
    main()
