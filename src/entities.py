import random
import pygame


class Food:
    """Reprezentuje zasób (pożywienie / jabłko) w świecie symulacji."""

    def __init__(self, x: float, y: float, radius: float = 4.0):
        self.pos = pygame.math.Vector2(x, y)
        self.radius = radius
        self.color = (46, 204, 113)  # Żywy zielony kolor

    def respawn(self, width: int, height: int, margin: int = 30):
        """Przemieszcza istniejący obiekt jedzenia w nowe losowe miejsce, unikając ciągłej alokacji."""
        self.pos.x = random.randint(margin, width - margin)
        self.pos.y = random.randint(margin, height - margin)

    def draw(self, screen: pygame.Surface):
        """Rysuje pożywienie jako prosty, wydajny okrąg."""
        pygame.draw.circle(screen, self.color, (int(self.pos.x), int(self.pos.y)), int(self.radius))


class Poison:
    """Reprezentuje truciznę (przeszkodę środowiskową) odbierającą energię."""

    def __init__(self, x: float, y: float, size: float = 8.0):
        self.pos = pygame.math.Vector2(x, y)
        self.size = size
        self.radius = size / 2.0  # promień efektywny do obliczeń kolizji
        self.color = (155, 89, 182)  # Fioletowy kolor

    def respawn(self, width: int, height: int, margin: int = 30):
        """Przemieszcza truciznę w nowe losowe miejsce."""
        self.pos.x = random.randint(margin, width - margin)
        self.pos.y = random.randint(margin, height - margin)

    def draw(self, screen: pygame.Surface):
        """Rysuje truciznę jako fioletowy kwadrat."""
        top_left_x = int(self.pos.x - self.radius)
        top_left_y = int(self.pos.y - self.radius)
        pygame.draw.rect(screen, self.color, (top_left_x, top_left_y, int(self.size), int(self.size)))


class Hazard:
    """Reprezentuje dynamiczne zagrożenie / drapieżnika w świecie symulacji."""

    def __init__(self, x: float, y: float, radius: float = 12.0):
        self.pos = pygame.math.Vector2(x, y)
        self.radius = radius
        # Powolne dryfowanie zagrożenia po planszy
        self.vel = pygame.math.Vector2(
            random.uniform(-1.0, 1.0),
            random.uniform(-1.0, 1.0)
        )
        if self.vel.length_squared() > 0:
            self.vel = self.vel.normalize() * random.uniform(0.5, 1.5)
        self.color = (231, 76, 60)  # Czerwony kolor ostrzegawczy

    def update(self, width: int, height: int, margin: int = 20):
        """Aktualizuje pozycję zagrożenia i odbija je od granic ekranu."""
        self.pos += self.vel

        # Odbicie od krawędzi poziomej
        if self.pos.x < margin:
            self.pos.x = margin
            self.vel.x *= -1
        elif self.pos.x > width - margin:
            self.pos.x = width - margin
            self.vel.x *= -1

        # Odbicie od krawędzi pionowej
        if self.pos.y < margin:
            self.pos.y = margin
            self.vel.y *= -1
        elif self.pos.y > height - margin:
            self.pos.y = height - margin
            self.vel.y *= -1

    def draw(self, screen: pygame.Surface):
        """Rysuje zagrożenie jako czerwony okrąg."""
        pygame.draw.circle(screen, self.color, (int(self.pos.x), int(self.pos.y)), int(self.radius))
