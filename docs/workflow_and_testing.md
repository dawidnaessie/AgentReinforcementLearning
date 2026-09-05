# Workflow and Testing (TDD & Incremental Development)

To prevent destabilizing the entire system, the assistant must strictly adopt an incremental development approach and ensure comprehensive unit test coverage.

## 1. Incremental Development (Baby Steps)
- Never generate or refactor the entire system at once. Write code in small, focused modules (e.g., first basic agent locomotion, then food consumption, then sensors).
- After implementing a single segment, ensure it passes all tests before proceeding to the next.

## 2. Always Write Tests (Test-Driven Development)
- Every new class, method, or business logic function must be backed by a corresponding unit test (using the standard `unittest` or `pytest` framework).
- Store all tests in a dedicated `/tests/` directory at the project root (e.g., `tests/test_agent.py`, `tests/test_environment.py`, `tests/test_analyze.py`).

## 3. Separation of Logic from Rendering (Pygame)
- For tests to execute automatically and headlessly, game logic (mathematics, collision detection, genetics) must be strictly decoupled from rendering routines (`pygame.draw`).
- Classes such as `Agent` must allow state updates without requiring an active Pygame window. Pass Pygame dependencies (such as `Surface` or screen displays) only to methods specifically dedicated to rendering (e.g., `draw(screen)`).

## 4. Regression Prevention
- When modifying existing code, update or write new test cases first. Only then adjust the implementation to ensure zero regressions.