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

    def collect_all(self, use_mock: bool = False, mock_samples: int = 100) -> List[Dict[str, Any]]:
        """
        Collect data from all sources (AI Hub).

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

        self.raw_data = aihub_data
        logger.info(f"Total records collected: {len(aihub_data)}")
        return aihub_data

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

    def build_bm25_indexes(
        self,
        data_dir: Optional[str] = None,
        output_dir: str = "models/bm25_index",
    ) -> Dict[str, str]:
        """
        처리된 JSONL 파일에서 BM25 인덱스를 빌드한다.

        Args:
            data_dir: 처리된 JSONL 파일이 있는 디렉토리.
                      None이면 self.config.preprocessing.processed_dir 사용.
            output_dir: BM25 인덱스 저장 디렉토리.

        Returns:
            {index_type: output_path} 딕셔너리
        """
        data_dir = data_dir or self.config.preprocessing.processed_dir
        result: Dict[str, str] = {}

        try:
            from src.inference.bm25_indexer import BM25Indexer

            data_path = Path(data_dir)
            # *_train.jsonl 우선, 없으면 *.jsonl 전체
            jsonl_files = list(data_path.glob("*_train.jsonl"))
            if not jsonl_files:
                jsonl_files = list(data_path.glob("*.jsonl"))

            if not jsonl_files:
                logger.warning(f"No JSONL files found in {data_dir}")
                return result

            for jsonl_path in jsonl_files:
                indexer = BM25Indexer()
                indexer.build_index_from_jsonl(str(jsonl_path))
                # 파일명에서 인덱스 이름 파생 (예: v2_train.jsonl -> v2)
                stem = (
                    jsonl_path.stem.replace("_train", "").replace("_valid", "").replace("_test", "")
                )
                output_path = os.path.join(output_dir, f"{stem}.pkl")
                indexer.save(output_path)
                logger.info(f"BM25 index built: {indexer.doc_count} documents -> {output_path}")
                result[stem] = output_path

        except Exception as e:
            logger.error(f"BM25 index building failed: {e}")

        return result

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
        choices=["full", "collect", "preprocess", "bm25"],
        default="full",
        help="Pipeline mode: full (collect + preprocess), collect only, or preprocess only",
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
