import unittest
import io
import sys
from src.stats import EvolutionTracker


class TestStats(unittest.TestCase):
    """Testy jednostkowe modułu zbierania statystyk i generowania podsumowania ewolucji (Faza 4)."""

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
            herd_defenses=3
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
            herd_defenses=6
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
        self.assertIn("Rekordowy wynik (Gen 2):             250.00 pkt", output)
        self.assertIn("Wzrost sredniej sprawnosci:         +140.0%", output)


if __name__ == '__main__':
    unittest.main()
