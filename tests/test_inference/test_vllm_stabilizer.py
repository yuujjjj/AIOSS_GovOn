"""
vllm_stabilizer 단위 테스트.

transformers / vllm 의존성을 Mock으로 대체하여 GPU 없이 실행 가능.
apply_transformers_patch()의 런타임 패치 적용을 검증한다.
"""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# 무거운 의존성 mock 등록
# ---------------------------------------------------------------------------
sys.modules.setdefault("torch", MagicMock())

# transformers 서브모듈을 실제 types.ModuleType으로 생성하여
# hasattr / setattr 동작이 정상 작동하도록 한다.
# 이미 등록된 모듈이 있으면 교체한다 (다른 테스트에서 MagicMock으로 등록했을 수 있음).
_transformers_mock = types.ModuleType("transformers")

_rope_utils = types.ModuleType("transformers.modeling_rope_utils")
_utils = types.ModuleType("transformers.utils")
_generic = types.ModuleType("transformers.utils.generic")
_modeling_utils = types.ModuleType("transformers.modeling_utils")

# 부모-자식 관계 설정 (import 문이 정상 동작하도록)
_transformers_mock.modeling_rope_utils = _rope_utils
_transformers_mock.utils = _utils
_transformers_mock.modeling_utils = _modeling_utils
_utils.generic = _generic

sys.modules["transformers"] = _transformers_mock
sys.modules["transformers.modeling_rope_utils"] = _rope_utils
sys.modules["transformers.utils"] = _utils
sys.modules["transformers.utils.generic"] = _generic
sys.modules["transformers.modeling_utils"] = _modeling_utils

# 모듈을 다시 로드하여 새 mock 모듈을 사용하도록 한다
import importlib

if "src.inference.vllm_stabilizer" in sys.modules:
    importlib.reload(sys.modules["src.inference.vllm_stabilizer"])

from src.inference.vllm_stabilizer import apply_transformers_patch


# ---------------------------------------------------------------------------
# apply_transformers_patch 테스트
# ---------------------------------------------------------------------------


class TestApplyTransformersPatch:
    def _clear_patches(self):
        """각 테스트 전에 패치 대상 속성을 제거한다."""
        for attr in ("RopeParameters",):
            if hasattr(_rope_utils, attr):
                delattr(_rope_utils, attr)
        for attr in ("check_model_inputs",):
            if hasattr(_generic, attr):
                delattr(_generic, attr)
        for attr in ("ALL_ATTENTION_FUNCTIONS",):
            if hasattr(_modeling_utils, attr):
                delattr(_modeling_utils, attr)

    def test_injects_rope_parameters(self):
        """RopeParameters가 없으면 주입한다."""
        self._clear_patches()
        assert not hasattr(_rope_utils, "RopeParameters")

        apply_transformers_patch()

        assert hasattr(_rope_utils, "RopeParameters")
        # RopeParameters는 dict 서브클래스여야 한다
        assert issubclass(_rope_utils.RopeParameters, dict)

    def test_injects_check_model_inputs(self):
        """check_model_inputs가 없으면 주입한다."""
        self._clear_patches()
        assert not hasattr(_generic, "check_model_inputs")

        apply_transformers_patch()

        assert hasattr(_generic, "check_model_inputs")
        # no-op 함수여야 한다
        assert _generic.check_model_inputs("dummy") is None

    def test_injects_all_attention_functions(self):
        """ALL_ATTENTION_FUNCTIONS가 없으면 주입한다."""
        self._clear_patches()
        assert not hasattr(_modeling_utils, "ALL_ATTENTION_FUNCTIONS")

        apply_transformers_patch()

        assert hasattr(_modeling_utils, "ALL_ATTENTION_FUNCTIONS")
        # get 메서드가 있어야 한다
        mock_attn = _modeling_utils.ALL_ATTENTION_FUNCTIONS
        assert mock_attn.get("nonexistent") is None

    def test_skips_existing_rope_parameters(self):
        """이미 RopeParameters가 있으면 덮어쓰지 않는다."""
        self._clear_patches()

        class ExistingRope(dict):
            pass

        _rope_utils.RopeParameters = ExistingRope

        apply_transformers_patch()

        # 기존 클래스가 보존되어야 한다
        assert _rope_utils.RopeParameters is ExistingRope

    def test_skips_existing_check_model_inputs(self):
        """이미 check_model_inputs가 있으면 덮어쓰지 않는다."""
        self._clear_patches()

        def existing_fn(*args, **kwargs):
            return "existing"

        _generic.check_model_inputs = existing_fn

        apply_transformers_patch()

        assert _generic.check_model_inputs is existing_fn

    def test_patch_idempotent(self):
        """두 번 호출해도 에러가 발생하지 않는다."""
        self._clear_patches()
        apply_transformers_patch()
        apply_transformers_patch()  # 두 번째 호출
        assert hasattr(_rope_utils, "RopeParameters")


# ---------------------------------------------------------------------------
# start_vllm_engine 테스트
# ---------------------------------------------------------------------------


class TestStartVllmEngine:
    def test_start_vllm_engine_calls_llm(self):
        """start_vllm_engine이 vllm.LLM을 올바른 인자로 호출한다."""
        mock_llm_class = MagicMock()
        mock_llm_instance = MagicMock()
        mock_llm_class.return_value = mock_llm_instance

        mock_vllm = MagicMock()
        mock_vllm.LLM = mock_llm_class

        with patch.dict(sys.modules, {"vllm": mock_vllm}):
            from src.inference.vllm_stabilizer import start_vllm_engine

            result = start_vllm_engine("test-model-id")

        mock_llm_class.assert_called_once()
        call_kwargs = mock_llm_class.call_args
        assert call_kwargs[1]["model"] == "test-model-id"
        assert call_kwargs[1]["trust_remote_code"] is True
        assert call_kwargs[1]["max_model_len"] == 8192
        assert result == mock_llm_instance

    def test_start_vllm_engine_gpu_utilization(self):
        """GPU memory utilization이 0.8로 설정된다."""
        mock_llm_class = MagicMock()
        mock_vllm = MagicMock()
        mock_vllm.LLM = mock_llm_class

        with patch.dict(sys.modules, {"vllm": mock_vllm}):
            from src.inference.vllm_stabilizer import start_vllm_engine

            start_vllm_engine("test-model")

        call_kwargs = mock_llm_class.call_args[1]
        assert call_kwargs["gpu_memory_utilization"] == 0.8
        assert call_kwargs["enforce_eager"] is True
