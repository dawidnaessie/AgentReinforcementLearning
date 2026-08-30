import time
from typing import List, Dict, Any


class EvolutionTracker:
    """Zbiera statystyki przebiegu ewolucji i generuje eleganckie podsumowanie końcowe."""

    def __init__(self):
        self.start_time = time.time()
        self.generations_data: List[Dict[str, Any]] = []
        self.peak_fitness = -999999.0
        self.peak_generation = 0
        self.total_foods_collected = 0
        self.total_poisons_hit = 0
        self.total_allies_saved = 0

    def record_generation(
        self,
        generation: int,
        best_fitness: float,
        avg_fitness: float,
        stdev: float,
        species_count: int,
        duration_sec: float,
        foods_eaten: int = 0,
        poisons_hit: int = 0,
        allies_saved: int = 0
    ):
        """Zapisuje dane pojedynczej generacji."""
        if best_fitness > self.peak_fitness:
            self.peak_fitness = best_fitness
            self.peak_generation = generation

        self.total_foods_collected += foods_eaten
        self.total_poisons_hit += poisons_hit
        self.total_allies_saved += allies_saved

        self.generations_data.append({
            "generation": generation,
            "best_fitness": best_fitness,
            "avg_fitness": avg_fitness,
            "stdev": stdev,
            "species_count": species_count,
            "duration": duration_sec,
            "foods_eaten": foods_eaten,
            "poisons_hit": poisons_hit,
            "allies_saved": allies_saved
        })

    def print_summary(self):
        """Wyświetla czytelne, estetyczne podsumowanie sesji ewolucyjnej."""
        total_time = time.time() - self.start_time
        total_gens = len(self.generations_data)

        border = "=" * 80
        sub_border = "-" * 80

        print("\n" + border)
        print("          RAPORT PODSUMOWUJACY EWOLUCJE POPULACJI (ALife - Faza 2)")
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

        print(f" * Liczba ukonczonych generacji:      {total_gens}")
        print(f" * Czas trwania calej symulacji:       {total_time:.2f} s ({total_time / total_gens:.2f} s / generacja)")
        print(sub_border)
        print(f" * Sredni fitness na starcie (Gen 1):   {initial_avg:6.2f} pkt")
        print(f" * Sredni fitness na koncu (Gen {total_gens}):    {final_avg:6.2f} pkt")
        print(f" * Wzrost sredniej sprawnosci:         {avg_growth:+6.1f}%")
        print(f" * Rekordowy wynik (Gen {self.peak_generation}):             {self.peak_fitness:6.2f} pkt")
        print(sub_border)
        print(" PODSUMOWANIE ZACHOWAN SPOLECZNYCH I EKOLOGICZNYCH:")
        print(f" * Lacznie zebrane jablka:             {self.total_foods_collected} szt.")
        print(f" * Lacznie zjedzone trucizny:          {self.total_poisons_hit} szt.")
        print(f" * Lacznie uratowani sojusznicy (altruizm): {self.total_allies_saved} aktow pomocy")
        print(sub_border)
        print(" HISTORIA OSTATNICH GENERACJI:")
        print(f" {'Gen':<5} | {'Max Fit':<9} | {'Sr Fit':<9} | {'Jablka':<8} | {'Trucizny':<8} | {'Uratowani':<9} | {'Czas':<6}")
        print(" " + "-" * 76)

        recent = self.generations_data[-10:] if total_gens > 10 else self.generations_data
        for g in recent:
            print(
                f" {g['generation']:<5} | "
                f"{g['best_fitness']:<9.1f} | "
                f"{g['avg_fitness']:<9.1f} | "
                f"{g['foods_eaten']:<8} | "
                f"{g['poisons_hit']:<8} | "
                f"{g['allies_saved']:<9} | "
                f"{g['duration']:<6.2f}s"
            )

        print(border)
        print(" Status: Ewolucja zakonczona. Wszystkie dane zostaly podsumowane.")
        print(border + "\n")
