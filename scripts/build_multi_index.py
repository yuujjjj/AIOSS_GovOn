"""
확장 RAG를 위한 다중 인덱스(Multi-Index) 통합 빌드 스크립트.

이슈 #157: 법령/매뉴얼/공시정보 데이터 인덱싱 완료.
수정 사항: 파일 경로 불일치 해결 및 IndexType 매핑 유연성 강화.
"""

import argparse
import json
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
        
        path = Path(data_source_path)
        if not path.exists():
            logger.warning(f"데이터 소스가 존재하지 않습니다: {data_source_path}")
            return

        # 1. 문서 처리 (파싱 & 청킹)
        metadata_list = []
        
        if path.is_file() and path.suffix == ".jsonl":
            metadata_list = self._process_jsonl_source(path, index_type)
        elif path.is_dir():
            # 디렉토리 내 파일(PDF/HWP/TXT) 처리
            files = [str(f) for f in path.glob("**/*") if f.suffix.lower() in [".pdf", ".hwp", ".txt"]]
            if files:
                batch_result = self.doc_processor.process_batch(files, index_type)
                metadata_list = batch_result.succeeded
        else:
            # 단일 파일 처리
            try:
                metadata_list = self.doc_processor.process(str(path), index_type)
            except Exception as e:
                logger.error(f"단일 파일 처리 실패: {e}")

        if not metadata_list:
            logger.warning(f"[{index_type.value}] 처리된 문서가 없습니다.")
            return

        # 2. 임베딩 생성
        logger.info(f"[{index_type.value}] 임베딩 생성 중... ({len(metadata_list)} 청크)")
        texts = [m.extras.get("chunk_text") or m.title for m in metadata_list]
        embeddings = self.embedding_pipeline.embed_documents(texts)

        # 3. FAISS 저장
        self.index_manager.add_documents(index_type, embeddings, metadata_list)
        self.index_manager.save_index(index_type)

        # 4. BM25 빌드 및 저장
        logger.info(f"[{index_type.value}] BM25 인덱스 생성 중...")
        bm25_indexer = BM25Indexer()
        bm25_indexer.build_index(texts)
        bm25_path = os.path.join(self.bm25_dir, f"{index_type.value}.pkl")
        bm25_indexer.save(bm25_path)
        
        logger.info(f"[{index_type.value}] 빌드 완료: {len(metadata_list)} 청크")

    def _process_jsonl_source(self, path: Path, index_type: IndexType):
        """JSONL 파일을 읽어 DocumentProcessor로 청킹 처리한다."""
        metadata_list = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    # 텍스트 파일로 임시 저장 후 process 호출 (가장 정확한 방법)
                    # 파일명에 인덱스를 포함하여 충돌 방지
                    temp_file = Path(f"data/raw/temp_{index_type.value}_{len(metadata_list)}.txt")
                    content = data.get("content") or data.get("precedent") or ""
                    if not content: continue
                    
                    temp_file.write_text(content, encoding="utf-8")
                    
                    chunks = self.doc_processor.process(
                        str(temp_file), 
                        index_type, 
                        source=data.get("source", "Unknown"),
                        title=data.get("title") or data.get("casename") or f"{index_type.value} document",
                        category=data.get("category", "General")
                    )
                    metadata_list.extend(chunks)
                    if temp_file.exists(): temp_file.unlink()
                except Exception as e:
                    logger.error(f"JSONL 라인 처리 실패: {e}")
        return metadata_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="intfloat/multilingual-e5-large")
    parser.add_argument("--index-dir", type=str, default="models/faiss_index")
    parser.add_argument("--bm25-dir", type=str, default="models/bm25_index")
    args = parser.parse_args()

    builder = MultiIndexBuilder(args.model_name, args.index_dir, args.bm25_dir)

    # 실제 파일 경로에 맞춰 소스 리스트 정의
    sources = [
        (IndexType.LAW, "data/raw/laws.jsonl"),
        (IndexType.MANUAL, "data/raw/manuals"),
        (IndexType.NOTICE, "data/raw/notices/alio_info.jsonl"), # 경로 수정됨
        (IndexType.CASE, "data/processed/v2_train.jsonl")
    ]

    for itype, src in sources:
        builder.process_and_add(itype, src)

    logger.info("모든 멀티 인덱스 빌드 작업이 완료되었습니다.")

if __name__ == "__main__":
    main()
