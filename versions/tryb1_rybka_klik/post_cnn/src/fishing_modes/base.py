"""Lekki kontrakt dla trybow minigry.

Tryby NIE musza dziedziczyc z FishingMode (to nie ABC) — wystarczy ze
implementuja te 3 metody i atrybut `name`. Tryb 2 (dymek+spacja) bedzie
mial zupelnie inna detekcje i akcje, ale ten sam interfejs zewnetrzny.

Dlaczego nie ABC z @abstractmethod?
    Trzymamy luzny kontrakt, zeby kolejne tryby mogly byc pisane od zera
    bez zaleznosci od abstrakcyjnego rodzica. Trzy metody to minimum
    wymagane przez `KosaBot.run()` — wszystko inne (dependencies, helpery,
    detekcja, klikanie) tryb organizuje sobie sam.

Przeplyw w KosaBot.run() (uproszczony):

    while bot.running:
        bot.input.start_fishing_round()         # F4 + SPACE — uniwersalne dla Metin2
        if not mode.wait_for_start(timeout=10): # czeka az minigra zacznie sie
            continue
        if not mode.play_round():               # caly cykl rundy (detekcja + akcje)
            break                               # przerwano przez stop()
        mode.wait_for_end()                     # czeka az okno minigry zniknie
        time.sleep(3.0)                         # pauza miedzy rundami
"""

from typing import Protocol


class FishingMode(Protocol):
    """Protokol jaki musi spelnic kazdy tryb minigry.

    Atrybuty:
        name: czytelna nazwa do logow (np. "Mini-gra lowienie ryb")

    Metody:
        wait_for_start: czeka az minigra rozpocznie sie po zarzuceniu wedki
        play_round: odgrywa caly cykl rundy az do jej zakonczenia
        wait_for_end: czeka az okno minigry znikinie z ekranu
    """

    name: str

    def wait_for_start(self, timeout: float = 10.0) -> bool:
        """Czeka az minigra zacznie byc widoczna.

        Returns:
            True  — minigra wykryta, mozna grac runde
            False — timeout, minigra sie nie pojawila
        """
        ...

    def play_round(self) -> bool:
        """Odgrywa jedna runde minigry.

        Tryb sam decyduje co znaczy "runda" — moze to byc petla klikniec,
        sekwencja spacji, cokolwiek. Wewnatrz powinien okresowo sprawdzac
        flage przerwania (przekazana w konstruktorze jako callable).

        Returns:
            True  — runda zakonczona normalnie (np. wyzlowiono / minigra zamknela sie)
            False — przerwano (uzytkownik wcisnal STOP, klawisz q w debug, itp.)
        """
        ...

    def wait_for_end(self, timeout: float = 5.0) -> None:
        """Czeka az okno minigry znikinie z ekranu.

        Cel: nie zaczac kolejnej rundy zanim poprzednia sie nie zwinie
        (jakies animacje, efekty wyniku HIT/MISS itd.).
        """
        ...
