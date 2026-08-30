import time
from typing import List, Dict, Any


class EvolutionTracker:
    """Zbiera statystyki przebiegu ewolucji i generuje eleganckie podsumowanie końcowe (Faza 4: Obrona Stadna)."""

    def __init__(self):
        self.start_time = time.time()
        self.generations_data: List[Dict[str, Any]] = []
        self.peak_fitness = -999999.0
        self.peak_generation = 0
        self.total_foods_collected = 0
        self.total_poisons_hit = 0
        self.total_allies_saved = 0
        self.total_attacks_made = 0
        self.total_defenses_made = 0
        self.total_herd_defenses = 0
        self.total_shouts_made = 0

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
        allies_saved: int = 0,
        attacks_made: int = 0,
        defenses_made: int = 0,
        herd_defenses: int = 0,
        shouts_made: int = 0
    ):
        """Zapisuje dane pojedynczej generacji."""
        if best_fitness > self.peak_fitness:
            self.peak_fitness = best_fitness
            self.peak_generation = generation

        self.total_foods_collected += foods_eaten
        self.total_poisons_hit += poisons_hit
        self.total_allies_saved += allies_saved
        self.total_attacks_made += attacks_made
        self.total_defenses_made += defenses_made
        self.total_herd_defenses += herd_defenses
        self.total_shouts_made += shouts_made

        self.generations_data.append({
            "generation": generation,
            "best_fitness": best_fitness,
            "avg_fitness": avg_fitness,
            "stdev": stdev,
            "species_count": species_count,
            "duration": duration_sec,
            "foods_eaten": foods_eaten,
            "poisons_hit": poisons_hit,
            "allies_saved": allies_saved,
            "attacks_made": attacks_made,
            "defenses_made": defenses_made,
            "herd_defenses": herd_defenses,
            "shouts_made": shouts_made
        })

    def print_summary(self):
        """Wyświetla czytelne, estetyczne podsumowanie sesji ewolucyjnej."""
        total_time = time.time() - self.start_time
        total_gens = len(self.generations_data)

        border = "=" * 98
        sub_border = "-" * 98

        print("\n" + border)
        print("          RAPORT PODSUMOWUJACY EWOLUCJE POPULACJI (ALife - Faza 4: Obrona Stadna)")
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
        print(" PODSUMOWANIE ZACHOWAN SPOLECZNYCH, EKOLOGICZNYCH I KOMUNIKACJI:")
        print(f" * Lacznie zebrane jablka:             {self.total_foods_collected} szt.")
        print(f" * Lacznie zjedzone trucizny:          {self.total_poisons_hit} szt.")
        print(f" * Lacznie uratowani sojusznicy (altruizm): {self.total_allies_saved} aktow pomocy")
        print(f" * Lacznie udane ataki (drapieznictwo):     {self.total_attacks_made} atakow na samotne ofiary")
        print(f" * Lacznie obrony czolowe:             {self.total_defenses_made} starc")
        print(f" * Lacznie odparte ataki (obrona stadna):   {self.total_herd_defenses} obron grupy")
        print(f" * Lacznie wyemitowane krzyki (komunikacja):{self.total_shouts_made} sygnalow")
        print(sub_border)
        print(" HISTORIA OSTATNICH GENERACJI:")
        print(f" {'Gen':<5} | {'Max Fit':<9} | {'Sr Fit':<9} | {'Jablka':<7} | {'Trucizny':<8} | {'Uratowani':<9} | {'Ataki':<6} | {'Obrony':<6} | {'Stado':<6} | {'Czas':<6}")
        print(" " + "-" * 96)

        recent = self.generations_data[-10:] if total_gens > 10 else self.generations_data
        for g in recent:
            print(
                f" {g['generation']:<5} | "
                f"{g['best_fitness']:<9.1f} | "
                f"{g['avg_fitness']:<9.1f} | "
                f"{g['foods_eaten']:<7} | "
                f"{g['poisons_hit']:<8} | "
                f"{g['allies_saved']:<9} | "
                f"{g['attacks_made']:<6} | "
                f"{g['defenses_made']:<6} | "
                f"{g['herd_defenses']:<6} | "
                f"{g['duration']:<6.2f}s"
            )

        print(border)
        print(" Status: Ewolucja zakonczona. Wszystkie dane zostaly podsumowane.")
        print(border + "\n")
