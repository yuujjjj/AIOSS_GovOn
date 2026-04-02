"""
Data Collection and Preprocessing Pipeline

Main entry point for the complete data pipeline:
1. Collect data from AI Hub and Seoul Open Data API
2. Clean and validate data
3. Mask PII (Personal Identifiable Information)
4. Transform to EXAONE instruction-tuning format
5. Split into train/validation/test sets
6. Generate AWQ calibration dataset
7. Build BM25 indexes for sparse keyword retrieval

Usage:
    python -m src.data_collection_preprocessing.pipeline --help
    python -m src.data_collection_preprocessing.pipeline --mode full
    python -m src.data_collection_preprocessing.pipeline --mode collect
    python -m src.data_collection_preprocessing.pipeline --mode preprocess
    python -m src.data_collection_preprocessing.pipeline --mode bm25
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .aihub_collector import AIHubCollector, create_mock_dataset
from .calibration_dataset import CalibrationDatasetGenerator
from .collect_public_docs import CollectionResult, PublicDocumentCollector
from .config import Config, get_config
from .data_preprocessor import DataPreprocessor, ProcessedRecord
from .pii_masking import PIIMasker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("pipeline.log", encoding="utf-8")],
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of pipeline execution"""

    success: bool
    mode: str
    start_time: str
    end_time: str
    duration_seconds: float
    total_raw_records: int = 0
    total_processed_records: int = 0
    output_files: Dict[str, str] = None
    errors: List[str] = None

    def __post_init__(self):
        if self.output_files is None:
            self.output_files = {}
        if self.errors is None:
            self.errors = []


class DataPipeline:
    """
    Complete Data Collection and Preprocessing Pipeline

    Orchestrates the entire data pipeline from collection to
    generating training-ready datasets.
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config or get_config()

        # Initialize components
        self.aihub_collector = AIHubCollector(self.config.aihub)
        self.pii_masker = PIIMasker.create_strict_masker()
        self.preprocessor = DataPreprocessor(self.config.preprocessing, self.pii_masker)
        self.calibration_generator = CalibrationDatasetGenerator(self.config.calibration)
        self.public_doc_collector = PublicDocumentCollector(self.config.public_doc)

        # Pipeline state
        self.raw_data: List[Dict[str, Any]] = []
        self.processed_records: List[ProcessedRecord] = []
        self.result = None

    def collect_from_aihub(
        self, use_mock: bool = False, mock_samples: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Collect data from AI Hub.

        Args:
            use_mock: Use mock data for testing
            mock_samples: Number of mock samples

        Returns:
            List of raw data records
        """
        logger.info("Collecting data from AI Hub...")

        if use_mock:
            # Create mock data for testing
            mock_path = create_mock_dataset(
                Path(self.config.aihub.download_dir) / "mock", num_samples=mock_samples
            )
            with open(mock_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("data", [])

        # Real collection logic
        collected_data = []

        for dataset_key in self.config.aihub.dataset_keys:
            dataset_dir = Path(self.config.aihub.download_dir) / dataset_key

            if not dataset_dir.exists():
                logger.warning(
                    f"Dataset {dataset_key} not downloaded. "
                    f"Run: ./aihubshell -mode d -datasetkey {dataset_key}"
                )
                continue

            # Iterate through downloaded files
            for batch in self.aihub_collector.iterate_dataset(dataset_dir):
                collected_data.extend(batch)
                logger.info(f"Loaded batch of {len(batch)} records from {dataset_key}")

        logger.info(f"Total AI Hub records collected: {len(collected_data)}")
        return collected_data

    def collect_from_public_docs(
        self,
        use_mock: bool = False,
        min_docs: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        행안부 공공문서 API에서 학습 데이터를 수집한다.

        Args:
            use_mock: True이면 빈 목록 반환 (실제 API 미호출)
            min_docs: 최소 수집 목표 건수

        Returns:
            List of raw data records with _source="public_doc"
        """
        import asyncio

        logger.info("행안부 공공문서 API 수집 시작...")

        if use_mock:
            logger.info("mock 모드 — 공공문서 수집 건너뜀")
            return []

        if not self.config.public_doc.api_key:
            logger.warning(
                "DATA_GO_KR_API_KEY 환경변수가 설정되지 않았습니다. 공공문서 수집 건너뜀."
            )
            return []

        result: CollectionResult = asyncio.run(
            self.public_doc_collector.collect_all(min_docs=min_docs)
        )

        if not result.success:
            logger.warning(f"공공문서 수집 실패: {result.errors}")
            return []

        # JSONL 파일에서 레코드 로드
        if not result.output_path:
            return []

        records: List[Dict[str, Any]] = []
        try:
            with open(result.output_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        except Exception as exc:
            logger.error(f"공공문서 JSONL 로드 실패: {exc}")
            return []

        logger.info(f"공공문서 수집 완료: {len(records)}건")
        return records

    def collect_all(self, use_mock: bool = False, mock_samples: int = 100) -> List[Dict[str, Any]]:
        """
        Collect data from all sources (AI Hub + 공공문서).

        Args:
            use_mock: Use mock data for testing
            mock_samples: Number of mock samples

        Returns:
            List of raw data records
        """
        # Collect from AI Hub
        aihub_data = self.collect_from_aihub(use_mock, mock_samples)
        for record in aihub_data:
            record["_source"] = "aihub"

        # Collect from 공공문서 API
        public_doc_data = self.collect_from_public_docs(use_mock=use_mock)
        for record in public_doc_data:
            record["_source"] = "public_doc"

        combined = aihub_data + public_doc_data
        self.raw_data = combined
        logger.info(
            f"Total records collected: {len(combined)} "
            f"(aihub={len(aihub_data)}, public_doc={len(public_doc_data)})"
        )
        return combined

    def preprocess(self, raw_data: Optional[List[Dict[str, Any]]] = None) -> List[ProcessedRecord]:
        """
        Preprocess collected data.

        Args:
            raw_data: Raw data to process (uses self.raw_data if None)

        Returns:
            List of processed records
        """
        raw_data = raw_data or self.raw_data

        if not raw_data:
            logger.error("No raw data to preprocess")
            return []

        logger.info(f"Preprocessing {len(raw_data)} records...")

        # Group by source
        aihub_data = [r for r in raw_data if r.get("_source") == "aihub"]
        other_data = [r for r in raw_data if r.get("_source") != "aihub"]

        processed = []

        # Process AI Hub data
        if aihub_data:
            processed.extend(self.preprocessor.process_raw_data(aihub_data, source="aihub"))

        # Process other data
        if other_data:
            processed.extend(self.preprocessor.process_raw_data(other_data, source="other"))

        self.processed_records = processed
        logger.info(f"Total processed records: {len(processed)}")
        return processed

    def split_and_save(
        self,
        processed_records: Optional[List[ProcessedRecord]] = None,
        prefix: str = "civil_complaint",
    ) -> Dict[str, Path]:
        """
        Split dataset and save all files.

        Args:
            processed_records: Records to split (uses self.processed_records if None)
            prefix: Filename prefix

        Returns:
            Dictionary of saved file paths
        """
        processed_records = processed_records or self.processed_records

        if not processed_records:
            logger.error("No processed records to save")
            return {}

        # Split dataset
        train, val, test = self.preprocessor.split_dataset(
            processed_records, shuffle=True, random_seed=self.config.calibration.random_seed
        )

        # Save splits
        paths = self.preprocessor.save_all_splits(train, val, test, prefix)

        return paths

    def generate_calibration_dataset(
        self,
        processed_records: Optional[List[ProcessedRecord]] = None,
        filename: str = "calibration_dataset",
    ) -> Dict[str, Path]:
        """
        Generate AWQ calibration dataset.

        Args:
            processed_records: Source records (uses self.processed_records if None)
            filename: Output filename

        Returns:
            Dictionary of saved file paths
        """
        processed_records = processed_records or self.processed_records

        if not processed_records:
            logger.error("No processed records for calibration dataset")
            return {}

        logger.info("Generating calibration dataset...")
        return self.calibration_generator.generate_and_save(processed_records, filename)

    def run_full_pipeline(
        self,
        use_mock: bool = False,
        mock_samples: int = 100,
        output_prefix: str = "civil_complaint",
    ) -> PipelineResult:
        """
        Run the complete data pipeline.

        Args:
            use_mock: Use mock data for testing
            mock_samples: Number of mock samples
            output_prefix: Output filename prefix

        Returns:
            PipelineResult object
        """
        start_time = datetime.now()
        result = PipelineResult(
            success=False,
            mode="full",
            start_time=start_time.isoformat(),
            end_time="",
            duration_seconds=0,
        )

        try:
            # Step 1: Collect data
            logger.info("=" * 60)
            logger.info("Step 1: Data Collection")
            logger.info("=" * 60)
            raw_data = self.collect_all(use_mock, mock_samples)
            result.total_raw_records = len(raw_data)

            if not raw_data:
                raise ValueError("No data collected")

            # Step 2: Preprocess data
            logger.info("=" * 60)
            logger.info("Step 2: Data Preprocessing")
            logger.info("=" * 60)
            processed = self.preprocess()
            result.total_processed_records = len(processed)

            if not processed:
                raise ValueError("No records after preprocessing")

            # Step 3: Split and save
            logger.info("=" * 60)
            logger.info("Step 3: Dataset Splitting and Saving")
            logger.info("=" * 60)
            dataset_paths = self.split_and_save(processed, output_prefix)
            result.output_files.update({k: str(v) for k, v in dataset_paths.items()})

            # Step 4: Generate calibration dataset
            logger.info("=" * 60)
            logger.info("Step 4: Calibration Dataset Generation")
            logger.info("=" * 60)
            calibration_paths = self.generate_calibration_dataset(
                processed, f"{output_prefix}_calibration"
            )
            result.output_files.update(
                {f"calibration_{k}": str(v) for k, v in calibration_paths.items()}
            )

            # Step 5: Build BM25 indexes
            logger.info("=" * 60)
            logger.info("Step 5: BM25 Index Building")
            logger.info("=" * 60)
            bm25_paths = self.build_bm25_indexes()
            if not bm25_paths:
                logger.warning("Step 5: BM25 인덱스가 빌드되지 않았습니다 (비치명적).")
            result.output_files.update({f"bm25_{k}": v for k, v in bm25_paths.items()})

            result.success = True

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            result.errors.append(str(e))

        finally:
            end_time = datetime.now()
            result.end_time = end_time.isoformat()
            result.duration_seconds = (end_time - start_time).total_seconds()

            # Save pipeline result
            result_path = Path(self.config.preprocessing.processed_dir) / "pipeline_result.json"
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(asdict(result), f, ensure_ascii=False, indent=2)

            logger.info("=" * 60)
            logger.info("Pipeline Complete")
            logger.info("=" * 60)
            logger.info(f"Success: {result.success}")
            logger.info(f"Duration: {result.duration_seconds:.2f}s")
            logger.info(f"Raw records: {result.total_raw_records}")
            logger.info(f"Processed records: {result.total_processed_records}")
            logger.info(f"Result saved to: {result_path}")

        self.result = result
        return result

    def run_collect_only(self, use_mock: bool = False, mock_samples: int = 100) -> PipelineResult:
        """Run collection phase only"""
        start_time = datetime.now()
        result = PipelineResult(
            success=False,
            mode="collect",
            start_time=start_time.isoformat(),
            end_time="",
            duration_seconds=0,
        )

        try:
            raw_data = self.collect_all(use_mock, mock_samples)
            result.total_raw_records = len(raw_data)

            # Save raw data
            raw_path = Path(self.config.aihub.download_dir).parent / "raw_combined.json"
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(raw_data, f, ensure_ascii=False, indent=2)

            result.output_files["raw_data"] = str(raw_path)
            result.success = len(raw_data) > 0

        except Exception as e:
            logger.error(f"Collection failed: {e}")
            result.errors.append(str(e))

        finally:
            end_time = datetime.now()
            result.end_time = end_time.isoformat()
            result.duration_seconds = (end_time - start_time).total_seconds()

        return result

    def run_preprocess_only(
        self, input_file: str, output_prefix: str = "civil_complaint"
    ) -> PipelineResult:
        """Run preprocessing phase only from existing raw data"""
        start_time = datetime.now()
        result = PipelineResult(
            success=False,
            mode="preprocess",
            start_time=start_time.isoformat(),
            end_time="",
            duration_seconds=0,
        )

        try:
            # Load raw data
            with open(input_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            if isinstance(raw_data, dict) and "data" in raw_data:
                raw_data = raw_data["data"]

            self.raw_data = raw_data
            result.total_raw_records = len(raw_data)

            # Preprocess
            processed = self.preprocess()
            result.total_processed_records = len(processed)

            # Split and save
            dataset_paths = self.split_and_save(processed, output_prefix)
            result.output_files.update({k: str(v) for k, v in dataset_paths.items()})

            # Generate calibration dataset
            calibration_paths = self.generate_calibration_dataset(
                processed, f"{output_prefix}_calibration"
            )
            result.output_files.update(
                {f"calibration_{k}": str(v) for k, v in calibration_paths.items()}
            )

            result.success = len(processed) > 0

        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            result.errors.append(str(e))

        finally:
            end_time = datetime.now()
            result.end_time = end_time.isoformat()
            result.duration_seconds = (end_time - start_time).total_seconds()

        return result

    # JSONL 파일 stem → IndexType 매핑
    # 서버(api_server.py)는 IndexType.value 기준으로 BM25 인덱스를 로드하므로,
    # 빌드 시에도 동일한 네이밍 규칙을 사용해야 한다.
    _JSONL_TO_INDEX_TYPE: Dict[str, str] = {
        "v2": "case",
        "civil_complaint": "case",
        # 향후 확장
        # "law_data": "law",
        # "manual_data": "manual",
        # "notice_data": "notice",
    }

    def build_bm25_indexes(
        self,
        data_dir: Optional[str] = None,
        output_dir: str = "models/bm25_index",
    ) -> Dict[str, str]:
        """
        처리된 JSONL 파일에서 BM25 인덱스를 빌드한다.

        JSONL 파일명을 IndexType.value에 매핑하여, 서버가 로드할 수 있는
        파일명(case.pkl, law.pkl 등)으로 저장한다.
        동일 IndexType에 여러 JSONL이 매핑되면 문서를 합쳐서 하나의 인덱스로 빌드한다.

        Args:
            data_dir: 처리된 JSONL 파일이 있는 디렉토리.
                      None이면 self.config.preprocessing.processed_dir 사용.
            output_dir: BM25 인덱스 저장 디렉토리.

        Returns:
            {index_type_value: output_path} 딕셔너리
        """
        data_dir = data_dir or self.config.preprocessing.processed_dir
        result: Dict[str, str] = {}

        try:
            from src.inference.bm25_indexer import BM25Indexer
            from src.inference.index_manager import IndexType  # noqa: F401

            data_path = Path(data_dir)
            # *_train.jsonl 우선, 없으면 *.jsonl 전체
            jsonl_files = list(data_path.glob("*_train.jsonl"))
            if not jsonl_files:
                jsonl_files = list(data_path.glob("*.jsonl"))

            if not jsonl_files:
                logger.warning(f"No JSONL files found in {data_dir}")
                return result

            # IndexType.value별로 JSONL 파일을 그룹핑
            index_type_files: Dict[str, List[Path]] = {}
            unmapped_files: List[Path] = []

            for jsonl_path in jsonl_files:
                stem = (
                    jsonl_path.stem.replace("_train", "").replace("_valid", "").replace("_test", "")
                )
                index_type_value = self._JSONL_TO_INDEX_TYPE.get(stem)
                if index_type_value:
                    index_type_files.setdefault(index_type_value, []).append(jsonl_path)
                else:
                    unmapped_files.append(jsonl_path)

            if unmapped_files:
                logger.warning(
                    f"매핑되지 않은 JSONL 파일 (건너뜀): " f"{[p.name for p in unmapped_files]}"
                )

            # IndexType별로 문서를 합쳐서 인덱스 빌드
            for idx_type_value, files in index_type_files.items():
                all_documents: List[str] = []

                for jsonl_path in files:
                    try:
                        docs = self._load_documents_from_jsonl(str(jsonl_path))
                        all_documents.extend(docs)
                        logger.info(
                            f"JSONL 로드 완료: {jsonl_path.name} "
                            f"({len(docs)}건) -> {idx_type_value}"
                        )
                    except Exception as e:
                        logger.error(f"JSONL 로드 실패 (건너뜀): {jsonl_path.name}: {e}")

                if not all_documents:
                    logger.warning(f"IndexType '{idx_type_value}'에 유효한 문서가 없습니다.")
                    continue

                try:
                    indexer = BM25Indexer()
                    indexer.build_index(all_documents)
                    output_path = os.path.join(output_dir, f"{idx_type_value}.pkl")
                    indexer.save(output_path)
                    logger.info(
                        f"BM25 index built: {indexer.doc_count} documents " f"-> {output_path}"
                    )
                    result[idx_type_value] = output_path
                except Exception as e:
                    logger.error(f"BM25 인덱스 빌드 실패: {idx_type_value}: {e}")

        except ImportError as e:
            logger.error(f"BM25 모듈 임포트 실패: {e}")
        except Exception as e:
            logger.error(f"BM25 index building failed: {e}")

        return result

    @staticmethod
    def _load_documents_from_jsonl(data_path: str) -> List[str]:
        """
        JSONL 파일에서 텍스트 문서를 로드한다.

        필드 탐색 순서: text -> complaint -> input -> 템플릿 추출 fallback

        Args:
            data_path: JSONL 파일 경로.

        Returns:
            문서 텍스트 리스트.
        """
        documents: List[str] = []

        with open(data_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    if "text" in item:
                        raw = item["text"]
                        if isinstance(raw, str) and "[|user|]" in raw:
                            text = DataPipeline._extract_complaint_from_template(raw)
                        else:
                            text = raw
                    elif "complaint" in item:
                        text = item["complaint"]
                    elif "input" in item:
                        text = item["input"]
                    else:
                        text = DataPipeline._extract_complaint_from_template(item.get("text", ""))

                    if not isinstance(text, str):
                        text = str(text) if text is not None else ""
                    if text.strip():
                        documents.append(text)
                except (json.JSONDecodeError, KeyError) as e:
                    logging.getLogger(__name__).warning(
                        f"Line {line_no}: skipping due to error: {e}"
                    )

        return documents

    @staticmethod
    def _extract_complaint_from_template(text: str) -> str:
        """EXAONE 채팅 템플릿에서 민원 내용을 추출한다."""
        if not text:
            return text
        try:
            if "[|user|]" in text:
                user_part = text.split("[|user|]")[1].split("[|endofturn|]")[0]
                if "민원 내용:" in user_part:
                    return user_part.split("민원 내용:")[1].strip()
                return user_part.strip()
        except Exception:
            pass
        return text

    def run_bm25_only(
        self,
        data_dir: Optional[str] = None,
        output_dir: str = "models/bm25_index",
    ) -> PipelineResult:
        """BM25 인덱스 빌드만 실행한다."""
        start_time = datetime.now()
        result = PipelineResult(
            success=False,
            mode="bm25",
            start_time=start_time.isoformat(),
            end_time="",
            duration_seconds=0,
        )

        try:
            bm25_paths = self.build_bm25_indexes(data_dir=data_dir, output_dir=output_dir)
            result.output_files.update({f"bm25_{k}": v for k, v in bm25_paths.items()})
            result.success = len(bm25_paths) > 0

        except Exception as e:
            logger.error(f"BM25 pipeline failed: {e}")
            result.errors.append(str(e))

        finally:
            end_time = datetime.now()
            result.end_time = end_time.isoformat()
            result.duration_seconds = (end_time - start_time).total_seconds()

        return result

    def get_quality_report(self) -> str:
        """Get the data quality report as a formatted string"""
        return str(self.preprocessor.get_report())


def main():
    """Main entry point for CLI"""
    parser = argparse.ArgumentParser(
        description="Data Collection and Preprocessing Pipeline for EXAONE Fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with mock data (for testing)
  python -m src.data_collection_preprocessing.pipeline --mode full --mock

  # Run full pipeline with real data
  python -m src.data_collection_preprocessing.pipeline --mode full

  # Collect data only
  python -m src.data_collection_preprocessing.pipeline --mode collect

  # Preprocess existing data
  python -m src.data_collection_preprocessing.pipeline --mode preprocess --input raw_data.json

  # Build BM25 indexes only
  python -m src.data_collection_preprocessing.pipeline --mode bm25

  # Build BM25 indexes from specific directory
  python -m src.data_collection_preprocessing.pipeline --mode bm25 --input /path/to/jsonl --bm25-output models/bm25_index
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["full", "collect", "preprocess", "bm25", "collect-public-docs"],
        default="full",
        help=(
            "Pipeline mode: full (collect + preprocess), collect only, preprocess only, "
            "bm25 only, or collect-public-docs only"
        ),
    )

    parser.add_argument("--mock", action="store_true", help="Use mock data for testing")

    parser.add_argument(
        "--mock-samples",
        type=int,
        default=100,
        help="Number of mock samples per source (default: 100)",
    )

    parser.add_argument("--input", type=str, default=None, help="Input file for preprocess mode")

    parser.add_argument(
        "--output-prefix",
        type=str,
        default="civil_complaint",
        help="Output filename prefix (default: civil_complaint)",
    )

    parser.add_argument(
        "--bm25-output",
        type=str,
        default="models/bm25_index",
        help="BM25 인덱스 출력 디렉토리 (default: models/bm25_index)",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create and run pipeline
    pipeline = DataPipeline()

    if args.mode == "full":
        result = pipeline.run_full_pipeline(
            use_mock=args.mock, mock_samples=args.mock_samples, output_prefix=args.output_prefix
        )

    elif args.mode == "collect":
        result = pipeline.run_collect_only(use_mock=args.mock, mock_samples=args.mock_samples)

    elif args.mode == "preprocess":
        if not args.input:
            parser.error("--input is required for preprocess mode")
        result = pipeline.run_preprocess_only(
            input_file=args.input,
            output_prefix=args.output_prefix,
        )

    elif args.mode == "bm25":
        result = pipeline.run_bm25_only(
            data_dir=args.input,
            output_dir=args.bm25_output,
        )

    elif args.mode == "collect-public-docs":
        import asyncio

        start_time = datetime.now()
        _result = PipelineResult(
            success=False,
            mode="collect-public-docs",
            start_time=start_time.isoformat(),
            end_time="",
            duration_seconds=0,
        )
        try:
            pub_data = pipeline.collect_from_public_docs(
                use_mock=args.mock,
                min_docs=args.mock_samples if args.mock else 1000,
            )
            _result.total_raw_records = len(pub_data)
            if pub_data:
                raw_path = (
                    Path(pipeline.config.public_doc.output_dir).parent / "public_doc_raw.jsonl"
                )
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                with open(raw_path, "w", encoding="utf-8") as f:
                    for rec in pub_data:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                _result.output_files["public_doc_raw"] = str(raw_path)
            _result.success = len(pub_data) > 0
        except Exception as exc:
            logger.error(f"collect-public-docs 실패: {exc}")
            _result.errors.append(str(exc))
        finally:
            end_time = datetime.now()
            _result.end_time = end_time.isoformat()
            _result.duration_seconds = (end_time - start_time).total_seconds()
        result = _result

    # Print summary
    print("\n" + "=" * 60)
    print("Pipeline Execution Summary")
    print("=" * 60)
    print(f"Mode: {result.mode}")
    print(f"Success: {result.success}")
    print(f"Duration: {result.duration_seconds:.2f} seconds")
    print(f"Raw records: {result.total_raw_records}")
    print(f"Processed records: {result.total_processed_records}")

    if result.output_files:
        print("\nOutput files:")
        for key, path in result.output_files.items():
            print(f"  {key}: {path}")

    if result.errors:
        print("\nErrors:")
        for error in result.errors:
            print(f"  - {error}")

    # Print quality report
    print("\n" + pipeline.get_quality_report())

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
