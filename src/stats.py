import time
from typing import List, Dict, Any


class EvolutionTracker:
    """Zbiera statystyki przebiegu ewolucji i generuje eleganckie podsumowanie końcowe."""

    def __init__(self):
        self.start_time = time.time()
        self.generations_data: List[Dict[str, Any]] = []
        self.peak_fitness = -999999.0
        self.peak_generation = 0

    def record_generation(
        self,
        generation: int,
        best_fitness: float,
        avg_fitness: float,
        stdev: float,
        species_count: int,
        duration_sec: float
    ):
        """Zapisuje dane pojedynczej generacji."""
        if best_fitness > self.peak_fitness:
            self.peak_fitness = best_fitness
            self.peak_generation = generation

        self.generations_data.append({
            "generation": generation,
            "best_fitness": best_fitness,
            "avg_fitness": avg_fitness,
            "stdev": stdev,
            "species_count": species_count,
            "duration": duration_sec
        })

    def print_summary(self):
        """Wyświetla czytelne, estetyczne podsumowanie sesji ewolucyjnej."""
        total_time = time.time() - self.start_time
        total_gens = len(self.generations_data)

        border = "=" * 65
        sub_border = "-" * 65

        print("\n" + border)
        print("          RAPORT PODSUMOWUJACY EWOLUCJE POPULACJI (ALife)")
        print(border)

        if total_gens == 0:
            print(" [!] Symulacja zostala przerwana przed ukonczeniem pierwszej generacji.")
            print(f" [*] Czas trwania: {total_time:.2f} s")
            print(border + "\n")
            return

        first_gen = self.generations_data[0]
        last_gen = self.generations_data[-1]

        initial_avg = first_gen["avg_fitness"]
        final_avg = last_gen["avg_fitness"]
        avg_growth = ((final_avg - initial_avg) / abs(initial_avg) * 100) if initial_avg != 0 else 0.0

        print(f" * Liczba ukonczonych generacji:    {total_gens}")
        print(f" * Czas trwania calej symulacji:     {total_time:.2f} s ({total_time / total_gens:.2f} s / generacja)")
        print(sub_border)
        print(f" * Sredni fitness na starcie (Gen 1): {initial_avg:6.2f} pkt")
        print(f" * Sredni fitness na koncu (Gen {total_gens}):  {final_avg:6.2f} pkt")
        print(f" * Wzrost sredniej sprawnosci:       {avg_growth:+6.1f}%")
        print(f" * Rekordowy wynik (Gen {self.peak_generation}):           {self.peak_fitness:6.2f} pkt")
        print(sub_border)
        print(" HISTORIA OSTATNICH GENERACJI:")
        print(f" {'Gen':<6} | {'Max Fitness':<14} | {'Sredni Fitness':<16} | {'Czas (s)':<10}")
        print(" " + "-" * 55)

        recent = self.generations_data[-10:] if total_gens > 10 else self.generations_data
        for g in recent:
            print(f" {g['generation']:<6} | {g['best_fitness']:<14.2f} | {g['avg_fitness']:<16.2f} | {g['duration']:<10.2f}")

        print(border)
        print(" Status: Ewolucja zakonczona. Wszystkie dane zostaly podsumowane.")
        print(border + "\n")
