from scripts.train_fixed_view_simgen_stage2 import (
    enable_transformer_gradient_checkpointing,
    smoke_result_message,
    smoke_trace_message,
)


def test_smoke_result_message_reports_completed_update_and_loss():
    assert smoke_result_message(1, 2.5) == "Stage-2 GPU smoke test passed: completed 1 optimizer update (loss=2.500000)"


def test_smoke_trace_message_labels_execution_boundary():
    assert smoke_trace_message("components loaded") == "Stage-2 GPU smoke trace: components loaded"


def test_stage2_enables_transformer_gradient_checkpointing_when_supported():
    class Transformer:
        def __init__(self):
            self.enabled = False

        def enable_gradient_checkpointing(self):
            self.enabled = True

    transformer = Transformer()

    enable_transformer_gradient_checkpointing(transformer)

    assert transformer.enabled
