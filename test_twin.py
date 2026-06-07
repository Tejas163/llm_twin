#!/usr/bin/env python3
import unittest
import os
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

        # Mock standard Chrome trace format JSON matching your precise schema
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
        assert parsed_data is not None
        self.assertEqual(parsed_data["model"]["name"], "test-model-7b")
        self.assertEqual(parsed_data["hardware"]["gpu_count"], 2)

    def test_phase2_parallelism_and_efficiency(self):
        """Validates Phase 2 auto-parallelism scaling math and Phase 4 FinOps broker keys."""
        parsed_data = parse_simple_yaml(self.temp_yaml_path)
        assert parsed_data is not None
        scaler = FakeLLMScaler(parsed_data)
        metrics = scaler.simulate()

        # Assert infrastructure metadata passes
        self.assertEqual(metrics["topology"], "2x A100")
        self.assertTrue(metrics["throughput_tok_sec"] > 0)
        
        # Phase 4 FinOps Validations
        self.assertIn("hourly_cost", metrics)
        self.assertIn("cost_per_m_tokens", metrics)
        self.assertTrue(metrics["hourly_cost"] > 0)

    def test_phase3_trace_emulator_timeline_and_oom(self):
        """Ensures the discrete-event clock advances and flags VRAM threshold overflows."""
        parsed_data = parse_simple_yaml(self.temp_yaml_path)
        assert parsed_data is not None
        emulator = TraceEmulator(self.mock_trace, parsed_data)
        report = emulator.run()

        self.assertTrue(report["total_simulated_time_ms"] > 0)
        self.assertEqual(report["peak_vram_tracked_gb"], 90)
        
        # Flexible string matching looking for 'OVERFLOW', 'EXCEEDED', or 'FAILURE' keywords in logs
        oom_log_triggered = any(
            "OVERFLOW" in log.upper() or "EXCEEDED" in log.upper() or "FAILURE" in log.upper() 
            for log in report["timeline"]
        )
        self.assertTrue(oom_log_triggered, "Trace emulator failed to log a critical VRAM capacity constraint breach.")

if __name__ == "__main__":
    unittest.main()