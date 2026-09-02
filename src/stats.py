import time
from datetime import datetime
from typing import List, Dict, Any


class EvolutionTracker:
    """Zbiera statystyki przebiegu ewolucji, drukuje podsumowanie i zapisuje raport do pliku logs.txt."""

    def __init__(self):
        self.start_datetime = datetime.now()
        self.start_time = time.time()
        self.generations_data: List[Dict[str, Any]] = []
        self.peak_fitness = -999999.0
        self.peak_generation = 0
        self.peak_synapses = 0
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
        shouts_made: int = 0,
        best_synapses: int = 0
    ):
        """Zapisuje dane pojedynczej generacji."""
        if best_fitness > self.peak_fitness:
            self.peak_fitness = best_fitness
            self.peak_generation = generation
            self.peak_synapses = best_synapses

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
            "shouts_made": shouts_made,
            "best_synapses": best_synapses
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
        print(f" * Synapsy rekordzisty:                 {self.peak_synapses} aktywnych polaczen")
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

    def dump_to_file(self, filepath: str = "logs.txt") -> str:
        """
        Dopisuje (append) najważniejsze informacje z przebiegu symulacji do pliku logu.
        Zawiera datę i czas, średnie i najwyższe wyniki dla każdej generacji
        oraz rekordowy wynik w całej symulacji wraz z liczbą synaps.
        """
        end_datetime = datetime.now()
        total_time = time.time() - self.start_time
        total_gens = len(self.generations_data)

        date_str = end_datetime.strftime("%Y-%m-%d %H:%M:%S")
        start_str = self.start_datetime.strftime("%Y-%m-%d %H:%M:%S")

        lines: List[str] = [
            "=" * 98,
            f"SIMULATION RUN LOG - {date_str}",
            "=" * 98,
            f"• Data rozpoczecia:    {start_str}",
            f"• Data zakonczenia:    {date_str}",
            f"• Czas trwania:        {total_time:.2f} s ({total_time / 60.0:.2f} min)",
            f"• Ukonczone generacje: {total_gens}",
        ]

        if total_gens == 0:
            lines.append("• Status: Symulacja przerwana przed ukonczeniem pierwszej generacji.")
            lines.append("=" * 98 + "\n\n")
            content = "\n".join(lines)
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(content)
            return content

        initial_avg = self.generations_data[0]["avg_fitness"]
        final_avg = self.generations_data[-1]["avg_fitness"]
        avg_growth = ((final_avg - initial_avg) / abs(initial_avg) * 100) if initial_avg != 0 else 0.0

        lines.extend([
            "",
            "NAJLEPSZY WYNIK W CALEJ SYMULACJI (PEAK PERFORMANCE):",
            f"• Najwyzszy fitness w ogole: {self.peak_fitness:.2f} pkt",
            f"• Osiagniety w generacji:    Gen {self.peak_generation}",
            f"• Liczba aktywnych synaps:   {self.peak_synapses} polaczen",
            "",
            "PODSUMOWANIE EKOSYSTEMU I ZACHOWAN:",
            f"• Sredni fitness startowy (Gen 1):  {initial_avg:.2f} pkt",
            f"• Sredni fitness koncowy (Gen {total_gens}):  {final_avg:.2f} pkt",
            f"• Wzrost sredniej sprawnosci:       {avg_growth:+.1f}%",
            f"• Zebrane jablka:                   {self.total_foods_collected} szt.",
            f"• Zjedzone trucizny:                {self.total_poisons_hit} szt.",
            f"• Akty altruizmu (uratowani):       {self.total_allies_saved}",
            f"• Ataki drapieznikow:               {self.total_attacks_made}",
            f"• Obrony czolowe:                   {self.total_defenses_made}",
            f"• Obrony stadne:                    {self.total_herd_defenses}",
            f"• Wyemitowane krzyki:               {self.total_shouts_made}",
            "",
            "SZCZEGOLOWY PRZEBIEG GENERACJA PO GENERACJI (AVG SCORE & PEAK):",
            f"{'Gen':<5} | {'Sr Fitness':<11} | {'Max Fitness':<11} | {'Synapsy':<8} | {'Jablka':<7} | {'Trucizny':<8} | {'Altruizm':<8} | {'Ataki':<6} | {'Obrony':<6} | {'Stado':<6} | {'Czas':<7}",
            "-" * 98
        ])

        for g in self.generations_data:
            lines.append(
                f"{g['generation']:<5} | "
                f"{g['avg_fitness']:<11.2f} | "
                f"{g['best_fitness']:<11.2f} | "
                f"{g.get('best_synapses', 0):<8} | "
                f"{g['foods_eaten']:<7} | "
                f"{g['poisons_hit']:<8} | "
                f"{g['allies_saved']:<8} | "
                f"{g['attacks_made']:<6} | "
                f"{g['defenses_made']:<6} | "
                f"{g['herd_defenses']:<6} | "
                f"{g['duration']:<7.2f}s"
            )

        lines.append("=" * 98 + "\n\n")
        content = "\n".join(lines)

        with open(filepath, "a", encoding="utf-8") as f:
            f.write(content)

        return content

