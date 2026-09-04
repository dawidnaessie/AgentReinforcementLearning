import unittest
import io
import os
import sys
import tempfile
from src.stats import EvolutionTracker


class TestStats(unittest.TestCase):
    """Unit tests for statistics tracking module and evolution summary generation (Phase 4)."""

    def test_tracker_empty_summary(self):
        tracker = EvolutionTracker()
        captured_output = io.StringIO()
        sys.stdout = captured_output
        try:
            tracker.print_summary()
        finally:
            sys.stdout = sys.__stdout__

        output = captured_output.getvalue()
        self.assertIn("RAPORT PODSUMOWUJACY", output)
        self.assertIn("Symulacja zostala przerwana przed ukonczeniem pierwszej generacji", output)

    def test_tracker_records_and_summary(self):
        tracker = EvolutionTracker()
        tracker.record_generation(
            generation=1,
            best_fitness=100.0,
            avg_fitness=50.0,
            stdev=10.0,
            species_count=1,
            duration_sec=1.5,
            foods_eaten=12,
            poisons_hit=2,
            allies_saved=4,
            attacks_made=5,
            defenses_made=2,
            herd_defenses=3,
            shouts_made=10
        )
        tracker.record_generation(
            generation=2,
            best_fitness=250.0,
            avg_fitness=120.0,
            stdev=15.0,
            species_count=1,
            duration_sec=1.2,
            foods_eaten=25,
            poisons_hit=1,
            allies_saved=7,
            attacks_made=8,
            defenses_made=3,
            herd_defenses=6,
            shouts_made=15
        )

        self.assertEqual(len(tracker.generations_data), 2)
        self.assertEqual(tracker.peak_fitness, 250.0)
        self.assertEqual(tracker.peak_generation, 2)
        self.assertEqual(tracker.total_foods_collected, 37)
        self.assertEqual(tracker.total_poisons_hit, 3)
        self.assertEqual(tracker.total_allies_saved, 11)
        self.assertEqual(tracker.total_attacks_made, 13)
        self.assertEqual(tracker.total_defenses_made, 5)
        self.assertEqual(tracker.total_herd_defenses, 9)
        self.assertEqual(tracker.total_shouts_made, 25)

        captured_output = io.StringIO()
        sys.stdout = captured_output
        try:
            tracker.print_summary()
        finally:
            sys.stdout = sys.__stdout__

        output = captured_output.getvalue()
        self.assertIn("Liczba ukonczonych generacji:      2", output)
        self.assertIn("Lacznie zebrane jablka:             37 szt.", output)
        self.assertIn("Lacznie zjedzone trucizny:          3 szt.", output)
        self.assertIn("Lacznie uratowani sojusznicy (altruizm): 11 aktow", output)
        self.assertIn("Lacznie udane ataki (drapieznictwo):     13 atakow", output)
        self.assertIn("Lacznie obrony czolowe:             5 starc", output)
        self.assertIn("Lacznie odparte ataki (obrona stadna):   9 obron", output)
        self.assertIn("Lacznie wyemitowane krzyki (komunikacja):25 sygnalow", output)
        self.assertIn("Rekordowy wynik (Gen 2):             250.00 pkt", output)
        self.assertIn("Synapsy rekordzisty:                 0 aktywnych polaczen", output)
        self.assertIn("Wzrost sredniej sprawnosci:         +140.0%", output)

    def test_dump_to_file_empty(self):
        """Verifies dump of empty tracker (simulation aborted immediately) with timestamp and status."""
        tracker = EvolutionTracker()
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test_logs.txt")
            content = tracker.dump_to_file(log_path)

            self.assertTrue(os.path.exists(log_path))
            self.assertIn("SIMULATION RUN LOG - ", content)
            self.assertIn("Data rozpoczecia:", content)
            self.assertIn("Data zakonczenia:", content)
            self.assertIn("Symulacja przerwana przed ukonczeniem pierwszej generacji", content)

    def test_dump_to_file_with_generations_and_synapses(self):
        """Verifies dump of generation statistics, dates, average scores, record, and synapse count."""
        tracker = EvolutionTracker()
        tracker.record_generation(
            generation=1,
            best_fitness=120.0,
            avg_fitness=45.0,
            stdev=12.0,
            species_count=1,
            duration_sec=2.0,
            foods_eaten=10,
            poisons_hit=2,
            allies_saved=3,
            attacks_made=4,
            defenses_made=1,
            herd_defenses=2,
            shouts_made=8,
            best_synapses=26
        )
        tracker.record_generation(
            generation=2,
            best_fitness=350.0,
            avg_fitness=110.0,
            stdev=18.0,
            species_count=1,
            duration_sec=2.5,
            foods_eaten=22,
            poisons_hit=1,
            allies_saved=6,
            attacks_made=7,
            defenses_made=3,
            herd_defenses=5,
            shouts_made=12,
            best_synapses=32
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test_logs.txt")
            content = tracker.dump_to_file(log_path)

            # 1. Check header and date
            self.assertIn("SIMULATION RUN LOG - ", content)
            self.assertIn("Data rozpoczecia:", content)
            self.assertIn("Data zakonczenia:", content)

            # 2. Check peak score and record-holder synapse count
            self.assertIn("Najwyzszy fitness w ogole: 350.00 pkt", content)
            self.assertIn("Osiagniety w generacji:    Gen 2", content)
            self.assertIn("Liczba aktywnych synaps:   32 polaczen", content)

            # 3. Check average scores for each generation
            self.assertIn("SZCZEGOLOWY PRZEBIEG GENERACJA PO GENERACJI", content)
            # Gen 1 row: avg 45.00, best 120.00, synapsy 26
            self.assertIn("1     | 45.00       | 120.00      | 26", content)
            # Gen 2 row: avg 110.00, best 350.00, synapsy 32
            self.assertIn("2     | 110.00      | 350.00      | 32", content)

    def test_dump_to_file_appends_multiple_sessions(self):
        """Verifies 'append' mode — successive simulation sessions append to the end of the file."""
        tracker1 = EvolutionTracker()
        tracker1.record_generation(1, 100.0, 50.0, 5.0, 1, 1.0, best_synapses=20)

        tracker2 = EvolutionTracker()
        tracker2.record_generation(1, 200.0, 80.0, 6.0, 1, 1.2, best_synapses=25)

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test_logs.txt")
            tracker1.dump_to_file(log_path)
            tracker2.dump_to_file(log_path)

            with open(log_path, "r", encoding="utf-8") as f:
                full_text = f.read()

            # File contains two headers from distinct simulation runs
            self.assertEqual(full_text.count("SIMULATION RUN LOG - "), 2)
            self.assertIn("Najwyzszy fitness w ogole: 100.00 pkt", full_text)
            self.assertIn("Najwyzszy fitness w ogole: 200.00 pkt", full_text)
            self.assertIn("Liczba aktywnych synaps:   20 polaczen", full_text)
            self.assertIn("Liczba aktywnych synaps:   25 polaczen", full_text)

    def test_dump_to_file_creates_directory_if_missing(self):
        """Verifies automatic creation of subdirectories (e.g., logs/) if they do not yet exist."""
        tracker = EvolutionTracker()
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_log_path = os.path.join(tmpdir, "nested", "logs", "logs.txt")
            self.assertFalse(os.path.exists(os.path.dirname(nested_log_path)))

            content = tracker.dump_to_file(nested_log_path)
            self.assertTrue(os.path.exists(nested_log_path))
            self.assertIn("SIMULATION RUN LOG - ", content)

    def test_dump_to_file_auto_rotation_when_exceeds_max_bytes(self):
        """Verifies log rotation: old file receives suffix 1 (e.g., logs1.txt), and new report writes to a fresh logs.txt."""
        tracker1 = EvolutionTracker()
        tracker1.record_generation(1, 100.0, 50.0, 5.0, 1, 1.0, best_synapses=20)

        tracker2 = EvolutionTracker()
        tracker2.record_generation(1, 200.0, 80.0, 6.0, 1, 1.2, best_synapses=25)

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "logs.txt")
            # First dump - creates logs.txt
            tracker1.dump_to_file(log_path)
            self.assertTrue(os.path.exists(log_path))
            file_size = os.path.getsize(log_path)

            # Second dump with max_bytes smaller than logs.txt size -> triggers rotation
            tracker2.dump_to_file(log_path, max_bytes=file_size)

            rotated_path = os.path.join(tmpdir, "logs1.txt")
            self.assertTrue(os.path.exists(rotated_path), "Old file should be rotated to logs1.txt.")
            self.assertTrue(os.path.exists(log_path), "New logs.txt file should be created.")

            with open(rotated_path, "r", encoding="utf-8") as f:
                rotated_content = f.read()
            with open(log_path, "r", encoding="utf-8") as f:
                new_content = f.read()

            self.assertIn("100.00 pkt", rotated_content)
            self.assertNotIn("200.00 pkt", rotated_content)
            self.assertIn("200.00 pkt", new_content)
            self.assertNotIn("100.00 pkt", new_content)

    def test_dump_to_file_manual_rename_workflow(self):
        """Verifies manual rename workflow where user renames logs.txt to logs1.txt."""
        tracker1 = EvolutionTracker()
        tracker1.record_generation(1, 100.0, 50.0, 5.0, 1, 1.0)

        tracker2 = EvolutionTracker()
        tracker2.record_generation(1, 200.0, 80.0, 6.0, 1, 1.2)

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "logs.txt")
            tracker1.dump_to_file(log_path)

            # User manually renames the log file
            renamed_manual = os.path.join(tmpdir, "logs1.txt")
            os.rename(log_path, renamed_manual)
            self.assertFalse(os.path.exists(log_path))
            self.assertTrue(os.path.exists(renamed_manual))

            # Next simulation writes to logs.txt - a fresh file should be created
            tracker2.dump_to_file(log_path)
            self.assertTrue(os.path.exists(log_path))

            with open(log_path, "r", encoding="utf-8") as f:
                content = f.read()
            # logs.txt contains only session 2 data, while logs1.txt holds session 1
            self.assertEqual(content.count("SIMULATION RUN LOG - "), 1)
            self.assertIn("200.00 pkt", content)


if __name__ == '__main__':
    unittest.main()
