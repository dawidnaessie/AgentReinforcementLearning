# Kontekst Projektu: AgentReinforcementLearning

## Cel Projektu
Stworzenie symulacji sztucznego życia (ALife), w której populacja 50 agentów AI (sterowanych przez rozwijające się sieci neuronowe NEAT) funkcjonuje we wspólnym, ciągłym środowisku 2D. Celem nadrzędnym jest zaobserwowanie emergencji zachowań społecznych, podziału na role cywilizacyjne (zbieracz, drapieżnik, obrońca) oraz współpracy (altruistyczny transfer energii) wymuszonej przez odpowiednio zaprojektowane środowisko i funkcję dopasowania (fitness function).

## Stos Technologiczny
- **Język:** Python 3.x (czysty Python, standardowa biblioteka `math`, `random`, `time`)
- **Neuroewolucja:** `neat-python` (obsługa sieci neuronowych feed-forward, mutacji wag i topologii, krzyżowania, speciacji oraz elitaryzmu Top 4).
- **Środowisko i Fizyka:** `pygame` (wbudowany `pygame.math.Vector2`, pętla symulacji, headless testy).
- **Testy:** `unittest` (TDD, pełna izolacja logiki bez wymogu okna graficznego).

## Główne Założenia Ewolucyjne
1. **Pętla Pokoleniowa:** Generacja trwa określoną liczbę klatek lub kończy się wcześniej po wymarciu populacji.
2. **Elitaryzm:** 4 najlepsze genomy (Top 4) przechodzą do kolejnego pokolenia bez żadnych mutacji.
3. **Początkowy Minimalizm:** Sieci startują z 0 warstw ukrytych (bezpośrednie połączenia wejścia-wyjścia) i samodzielnie rozbudowują topologię poprzez mutacje.
4. **Metabolizm i Zasoby:** Każdy krok kosztuje energię; zjedzenie pożywienia odnawia zasoby, trucizna i drapieżnictwo odbierają siły witalne.
5. **Autonomia:** Cały ekosystem ewoluuje bez zewnętrznej interwencji.

## Struktura Projektu
- `/src/main.py` – Inicjalizacja NEAT, punkt wejścia, zwięzłe logi w konsoli.
- `/src/environment.py` – Cykl życia Pygame, renderowanie, HUD, zarządzanie encjami (`Food`, `Poison`, `Hazard`).
- `/src/agent.py` – Klasa `Agent`, zmysły percepcyjne, metabolizm, mechaniki interakcji (altruizm, drapieżnictwo, obrona), przypisywanie fitnessu.
- `/src/entities.py` – Modularne encje świata (`Food`, `Poison`, `Hazard`).
- `/src/stats.py` – `EvolutionTracker` zbierający metryki generacji i generujący raport końcowy.
- `/config-feedforward.txt` – Parametry algorytmu NEAT.
- `/tests/` – Pełny zestaw testów jednostkowych (TDD).