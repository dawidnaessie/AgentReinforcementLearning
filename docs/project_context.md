# Kontekst Projektu: AgentReinforcementLearning

## Cel Projektu
Stworzenie symulacji sztucznego życia (ALife), w której 50 agentów AI (sterowanych przez małe sieci neuronowe) funkcjonuje we wspólnym środowisku 2D. Celem nadrzędnym jest zaobserwowanie ewolucji zachowań społecznych, podziału na role cywilizacyjne (np. zbieracz, obrońca) oraz współpracy, wymuszonej przez odpowiednio zaprojektowaną funkcję dopasowania (fitness function).

## Stos Technologiczny
- **Język:** Python 3.x
- **Neuroewolucja:** `neat-python` (obsługa sieci neuronowych, mutacji, krzyżowania i elitaryzmu Top 4).
- **Środowisko i Fizyka:** `pygame` (renderowanie 2D, pętla gry, kolizje).

## Główne Założenia Ewolucyjne
1. **Pętla Pokoleniowa:** Po ustalonej liczbie klatek (lub po wyginięciu populacji) następuje ocena.
2. **Elitaryzm:** 4 najlepsze sieci (Top 4) zawsze przechodzą do następnego pokolenia bez zmian.
3. **Mutacje:** Mutacjom ulegają wagi połączeń, dodawane są nowe węzły i nowe połączenia (początkowo agenci mają 0 warstw ukrytych).
4. **Brak interwencji zewnętrznej:** Pętla ma działać w nieskończoność i ewoluować samoistnie.

## Struktura Katalogów
- `/src/main.py` - Inicjalizacja NEAT, punkt wejścia.
- `/src/environment.py` - Logika świata, Pygame, pętla symulacji.
- `/src/agent.py` - Definicja klasy Agenta, wejścia/wyjścia sieci.
- `/config-feedforward.txt` - Parametry ewolucyjne dla neat-python.