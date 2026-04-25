"""Tryby minigry lowienia ryb.

Kazdy tryb to osobny modul (np. fish_click.py, bubble_space.py) z klasa
implementujaca lekki kontrakt opisany w base.py. Tryby sa niezalezne -
dodanie nowego = napisanie pliku obok, zero zmian w istniejacych.

KosaBot (src/bot.py) wybiera tryb na podstawie parametru `mode: str`
w konstruktorze i deleguje do niego cykl rundy.
"""
