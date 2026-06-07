#!/usr/bin/env python3
import unittest
import os
import json
from twin import parse_simple_yaml, FakeLLMScaler, TraceEmulator

class TestLLMTwinCore(unittest.TestCase):

    def setUp(self):
        """Sets up temporary test inputs and configurations."""
        self.temp_yaml_path = "temp_test_config.yaml"
        self.mock_yaml_content = """
        model:
          name: "test-model-7b"
          parameters_billion: 7
        hardware:
          gpu_type: "A100"
          gpu_count: 2
          gpu_memory_gb: 40
          gpu_tflops: 156
        inference:
          batch_size: 16
          sequence_length: 2048
        """
        with open(self.temp_yaml_path, "w") as f:
            f.write(self.mock_yaml_content)

        # Mock standard Chrome trace format JSON
        self.mock_trace = {
            "traceEvents": [
                {"name": "Tokenizer", "cat": "cpu", "ph": "X", "ts": 1000, "dur": 500},
                {"name": "Attention_Kernel", "cat": "gpu", "ph": "X", "ts": 1600, "dur": 1000},
                {"name": "GPU_Memory_Used_GB", "cat": "memory", "ph": "C", "ts": 2000, "args": {"value": 90}}
            ]
        }

    def tearDown(self):
        """Cleans up temporary files after test execution."""
        if os.path.exists(self.temp_yaml_path):
            os.remove(self.temp_yaml_path)

    def test_pure_python_yaml_parser(self):
        """Verifies that the lightweight YAML parser correctly converts types and structure."""
        parsed_data = parse_simple_yaml(self.temp_yaml_path)
        self.assertIsNotNone(parsed_data)
        self.assertEqual(parsed_data["model"]["name"], "test-model-7b")
        self.assertEqual(parsed_data["hardware"]["gpu_count"], 2)
        self.assertEqual(parsed_data["inference"]["batch_size"], 16)

    def test_phase2_parallelism_and_efficiency(self):
        """Validates Phase 2 auto-parallelism scaling math and efficiency penalties."""
        parsed_data = parse_simple_yaml(self.temp_yaml_path)
        scaler = FakeLLMScaler(parsed_data)
        metrics = scaler.simulate()

        # Check topology distribution constraints (2 GPUs should map to TP=2, PP=1, DP=1)
        self.assertEqual(metrics["tp"], 2)
        self.assertEqual(metrics["pp"], 1)
        self.assertEqual(metrics["dp"], 1)
        
        # Verify communication efficiency penalty applied (0.95 - 0.03 * 2 = 89%)
        self.assertAlmostEqual(metrics["comm_efficiency_pct"], 89.0, places=1)
        self.assertTrue(metrics["throughput_tok_sec"] > 0)

    def test_phase3_trace_emulator_timeline_and_oom(self):
        """Ensures the discrete-event clock advances and flags VRAM threshold overflows."""
        parsed_data = parse_simple_yaml(self.temp_yaml_path)
        emulator = TraceEmulator(self.mock_trace, parsed_data)
        report = emulator.run()

        # Virtual clock tracking assertions (durations are converted from microseconds to ms)
        # Tokenizer: 500us = 0.5ms. Attention scaled: 1.0ms * (312 / (156 * 2)) = 1.0ms. Total = 1.5ms
        self.assertAlmostEqual(report["total_simulated_time_ms"], 1.5, places=2)
        
        # OOM Capacity Gating Check (90 GB requested vs 2x40GB = 80GB limits)
        self.assertEqual(report["peak_vram_tracked_gb"], 90)
        self.assertEqual(report["vram_limit_gb"], 80)
        
        # Ensure the OOM alert log string was correctly triggered
        oom_log_triggered = any("🚨 OVERFLOW FAILURE" in log for log in report["timeline"])
        self.assertTrue(oom_log_triggered, "Trace emulator failed to flag a critical VRAM overflow constraint.")

if __name__ == "__main__":
    unittest.main()