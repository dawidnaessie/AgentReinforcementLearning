# Workflow i Testowanie (TDD & Incremental Development)

Aby zapobiec destabilizacji całego systemu, asystent musi bezwzględnie stosować podejście inkrementalne i pokrywać kod testami jednostkowymi.

## 1. Praca Segmentowa (Baby Steps)
- Nigdy nie generuj całego systemu na raz. Pisz kod małymi modułami (np. najpierw sama funkcja poruszania się agenta, potem jedzenie, potem sensory).
- Po wygenerowaniu jednego segmentu, upewnij się, że działa, zanim przejdziesz do kolejnego.

## 2. Zawsze Pisz Testy (Test-Driven Development)
- Każda nowa klasa, metoda lub funkcja logiczna musi posiadać odpowiadający jej test jednostkowy (używaj standardowej biblioteki `unittest` lub `pytest`).
- Testy przechowuj w osobnym folderze `/tests/` w korzeniu projektu (np. `tests/test_agent.py`, `tests/test_environment.py`).

## 3. Izolacja Logiki od Renderowania (Pygame)
- Aby testy mogły działać automatycznie, logika gry (matematyka, kolizje, genetyka) musi być odseparowana od renderowania (`pygame.draw`).
- Klasy takie jak `Agent` powinny pozwalać na aktualizację stanu bez konieczności inicjalizowania całego okna Pygame. Zależności od Pygame (jak `Surface` czy ekran) przekazuj tylko do metod odpowiedzialnych za rysowanie (np. `draw(screen)`).

## 4. Ochrona przed Regresją
- Modyfikując istniejący kod, najpierw zaktualizuj lub napisz nowy test. Dopiero potem zmieniaj implementację.