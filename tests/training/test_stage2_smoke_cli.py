from scripts.train_fixed_view_simgen_stage2 import smoke_result_message


def test_smoke_result_message_reports_completed_update_and_loss():
    assert smoke_result_message(1, 2.5) == "Stage-2 GPU smoke test passed: completed 1 optimizer update (loss=2.500000)"
