import os
import sys
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import re

import analyze


class TestAnalyze(unittest.TestCase):
    """Unit tests for analyze.py: environment loading, file gathering, archiving, and API handling."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_load_env_file_parsing(self):
        """Verifies custom .env parser handles key=value, comments, whitespace, and quotes."""
        dotenv_path = os.path.join(self.test_dir, ".env")
        with open(dotenv_path, "w", encoding="utf-8") as f:
            f.write("# Comment line\n")
            f.write("TEST_API_KEY_CUSTOM='secret_12345'\n")
            f.write("ANOTHER_VAR = \"hello_world\"\n")
            f.write("EMPTY_VAR=\n")

        with patch.dict(os.environ, {}, clear=False):
            if "TEST_API_KEY_CUSTOM" in os.environ:
                del os.environ["TEST_API_KEY_CUSTOM"]
            analyze.load_env_file(dotenv_path)
            self.assertEqual(os.environ.get("TEST_API_KEY_CUSTOM"), "secret_12345")
            self.assertEqual(os.environ.get("ANOTHER_VAR"), "hello_world")

    def test_get_api_key_resolution(self):
        """Verifies get_api_key prioritizes GEMINI_API_KEY and falls back to GOOGLE_API_KEY."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "gemini_key_abc"}, clear=False):
            self.assertEqual(analyze.get_api_key(), "gemini_key_abc")

        with patch.dict(os.environ, {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": "google_key_xyz"}, clear=False):
            self.assertEqual(analyze.get_api_key(), "google_key_xyz")

    def test_gather_log_files_skips_subdirectories(self):
        """Verifies gather_log_files accurately classifies root logs vs brain dumps and ignores subdirectories."""
        logs_dir = os.path.join(self.test_dir, "logs")
        os.makedirs(logs_dir)

        # Create root files
        f_log1 = os.path.join(logs_dir, "logs.txt")
        f_log2 = os.path.join(logs_dir, "og_logs.txt")
        f_brain1 = os.path.join(logs_dir, "brain_id_10.txt")
        f_brain2 = os.path.join(logs_dir, "brain_id_20.txt")
        f_non_txt = os.path.join(logs_dir, "data.csv")

        for p in [f_log1, f_log2, f_brain1, f_brain2, f_non_txt]:
            with open(p, "w", encoding="utf-8") as f:
                f.write("dummy")

        # Create subdirectory with files (should NOT be collected)
        subdir = os.path.join(logs_dir, "archive_subdir")
        os.makedirs(subdir)
        with open(os.path.join(subdir, "brain_id_999.txt"), "w", encoding="utf-8") as f:
            f.write("ignored")

        logs, brains = analyze.gather_log_files(logs_dir)
        self.assertEqual(len(logs), 2)
        self.assertIn(f_log1, logs)
        self.assertIn(f_log2, logs)
        self.assertEqual(len(brains), 2)
        self.assertIn(f_brain1, brains)
        self.assertIn(f_brain2, brains)

    def test_create_archive_directory_naming_and_collision(self):
        """Verifies archive directory format HH-MM-DD-MM-YYYY-LogsArchive and duplicate resolution."""
        logs_dir = os.path.join(self.test_dir, "logs")
        os.makedirs(logs_dir)

        path1 = analyze.create_archive_directory(logs_dir)
        self.assertTrue(os.path.isdir(path1))
        folder_name1 = os.path.basename(path1)

        pattern = r"^\d{2}-\d{2}-\d{2}-\d{2}-\d{4}-LogsArchive$"
        self.assertTrue(re.match(pattern, folder_name1), f"Folder name '{folder_name1}' does not match format.")

        # Immediate second creation should handle collision by appending _1
        path2 = analyze.create_archive_directory(logs_dir)
        self.assertTrue(os.path.isdir(path2))
        folder_name2 = os.path.basename(path2)
        self.assertEqual(folder_name2, f"{folder_name1}_1")

    def test_read_file_content(self):
        """Verifies reading files normally and truncating over-sized files cleanly."""
        small_file = os.path.join(self.test_dir, "small.txt")
        with open(small_file, "w", encoding="utf-8") as f:
            f.write("Hello world\nSecond line")
        self.assertEqual(analyze.read_file_content(small_file), "Hello world\nSecond line")

        # Over-sized file
        large_file = os.path.join(self.test_dir, "large.txt")
        with open(large_file, "w", encoding="utf-8") as f:
            for i in range(2000):
                f.write(f"Line {i}\n")

        content = analyze.read_file_content(large_file, max_lines=500)
        self.assertIn("Line 0", content)
        self.assertIn("Line 1999", content)
        self.assertIn("Truncated", content)

    def test_construct_prompt_structure(self):
        """Verifies master prompt includes architectural directives, logs, and brain topologies."""
        log_data = {"logs.txt": "Peak performance: 100.0"}
        brain_data = {"brain_id_42.txt": "Node ID: 3 | tanh"}

        prompt = analyze.construct_prompt(log_data, brain_data)
        self.assertIn("Phase 9: 22 normalized inputs", prompt)
        self.assertIn("Combat Cooldown", prompt)
        self.assertIn("LOG FILE: logs.txt", prompt)
        self.assertIn("BRAIN DUMP: brain_id_42.txt", prompt)
        self.assertIn("Reverse-Engineered Neural Topologies", prompt)
        self.assertIn("Markdown", prompt)

    @patch("requests.post")
    def test_call_gemini_api_success_mock(self, mock_post):
        """Verifies call_gemini_api parses candidate response text properly."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Executive Analysis: The ecosystem shows strong evolutionary health."}
                        ]
                    }
                }
            ]
        }
        mock_post.return_value = mock_resp

        result = analyze.call_gemini_api("test prompt", "fake_key", requested_model="gemini-3.6-flash")
        self.assertIn("Executive Analysis", result)
        self.assertTrue(mock_post.called)

    @patch("requests.post")
    def test_call_gemini_api_auth_error_mock(self, mock_post):
        """Verifies authentication failure raises RuntimeError."""
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "API_KEY_INVALID"
        mock_post.return_value = mock_resp

        with self.assertRaises(RuntimeError) as ctx:
            analyze.call_gemini_api("test prompt", "invalid_key")
        self.assertIn("Authentication failure", str(ctx.exception))

    def test_main_missing_api_key(self):
        """Verifies main exits with code 1 when GEMINI_API_KEY is missing."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("analyze.load_env_file"):
                ret = analyze.main()
                self.assertEqual(ret, 1)

    def test_main_empty_logs_directory(self):
        """Verifies main exits cleanly with code 0 when logs directory has no files."""
        empty_logs = os.path.join(self.test_dir, "empty_logs")
        os.makedirs(empty_logs)

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}):
            with patch("analyze.gather_log_files", return_value=([], [])):
                with patch("os.path.exists", return_value=True):
                    ret = analyze.main()
                    self.assertEqual(ret, 0)

    @patch("analyze.call_gemini_api")
    def test_main_full_workflow_success_mock(self, mock_call_api):
        """Verifies full execution pipeline: gathers files, calls API, moves files, writes AnalyticsSummary.md."""
        mock_call_api.return_value = "AI ARCHITECT REPORT: All populations converged to optimal strategies."

        logs_dir = os.path.join(self.test_dir, "logs")
        os.makedirs(logs_dir)

        sample_log = os.path.join(logs_dir, "logs.txt")
        sample_brain = os.path.join(logs_dir, "brain_id_100.txt")

        with open(sample_log, "w", encoding="utf-8") as f:
            f.write("Generation 1..50")
        with open(sample_brain, "w", encoding="utf-8") as f:
            f.write("Genome 100 Synapses")

        with patch.dict(os.environ, {"GEMINI_API_KEY": "valid_key"}):
            with patch("analyze.gather_log_files", return_value=([sample_log], [sample_brain])):
                with patch("analyze.create_archive_directory") as mock_create_arch:
                    archive_dir = os.path.join(logs_dir, "test_archive")
                    os.makedirs(archive_dir)
                    mock_create_arch.return_value = archive_dir

                    ret = analyze.main()
                    self.assertEqual(ret, 0)

                    # Original files in root should have moved to archive
                    self.assertFalse(os.path.exists(sample_log))
                    self.assertFalse(os.path.exists(sample_brain))
                    self.assertTrue(os.path.exists(os.path.join(archive_dir, "logs.txt")))
                    self.assertTrue(os.path.exists(os.path.join(archive_dir, "brain_id_100.txt")))

                    # AnalyticsSummary.md must exist inside archive folder
                    summary_file = os.path.join(archive_dir, "AnalyticsSummary.md")
                    self.assertTrue(os.path.exists(summary_file))
                    with open(summary_file, "r", encoding="utf-8") as f:
                        summary_content = f.read()
                    self.assertIn("AI ARCHITECT REPORT", summary_content)


if __name__ == "__main__":
    unittest.main()
