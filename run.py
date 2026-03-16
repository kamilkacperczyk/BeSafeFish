"""
Launcher bota Kosa - wybiera wersje i uruchamia.

Kazda wersja to osobny folder w versions/ z wlasnym src/.
Launcher dodaje wybrany folder do sys.path, dzieki czemu
importy 'from src.X import Y' dzialaja poprawnie.

Uzycie:
  python run.py                          # menu wyboru wersji
  python run.py --version pre_cnn        # bezposredni wybor
  python run.py --version post_cnn --debug  # CNN z podgladem
  python run.py --version post_cnn --no-cnn # klasyczna detekcja (fallback)

Tworzenie nowej wersji:
  1. Skopiuj istniejacy folder: versions/pre_cnn/ -> versions/nowa/
  2. Modyfikuj pliki w versions/nowa/src/
  3. Uruchom: python run.py --version nowa --debug
"""

import sys
import os
import ctypes


def _get_versions_dir():
    """Zwraca sciezke do folderu versions/."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "versions")


def get_versions():
    """Skanuje versions/ i zwraca liste dostepnych wersji (majacych src/)."""
    versions_dir = _get_versions_dir()
    if not os.path.isdir(versions_dir):
        return []
    return sorted([
        d for d in os.listdir(versions_dir)
        if os.path.isdir(os.path.join(versions_dir, d))
        and os.path.isdir(os.path.join(versions_dir, d, "src"))
    ])


def _read_version_desc(version_name):
    """Probuje odczytac krotki opis z OPIS_WERSJI.txt."""
    opis_path = os.path.join(_get_versions_dir(), version_name, "OPIS_WERSJI.txt")
    if not os.path.isfile(opis_path):
        return ""
    try:
        with open(opis_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Pomiń puste linie i nagłówki (=== ... ===)
                if line and not line.startswith("=") and not line.startswith("#"):
                    return line[:80]
    except OSError:
        pass
    return ""


def select_version(versions):
    """Wyswietla menu i pyta o wybor wersji."""
    print()
    print("=" * 50)
    print("  KOSA BOT - Wybor wersji")
    print("=" * 50)
    print()
    for i, v in enumerate(versions, 1):
        desc = _read_version_desc(v)
        desc_str = f"  ({desc})" if desc else ""
        print(f"  {i}. {v}{desc_str}")
    print()
    try:
        choice = input(f"Wybierz wersje (1-{len(versions)}): ").strip()
        idx = int(choice) - 1
        if idx < 0 or idx >= len(versions):
            raise ValueError()
        return versions[idx]
    except (ValueError, KeyboardInterrupt, EOFError):
        print("\nAnulowano.")
        sys.exit(0)


def main():
    # --- Zbierz dostepne wersje ---
    versions = get_versions()
    if not versions:
        print("[KOSA] Brak dostepnych wersji w folderze versions/!")
        print("       Kazda wersja wymaga: versions/<nazwa>/src/")
        sys.exit(1)

    # --- Wybor wersji: z CLI lub menu ---
    selected = None
    if "--version" in sys.argv:
        idx = sys.argv.index("--version")
        if idx + 1 < len(sys.argv):
            selected = sys.argv[idx + 1]
            if selected not in versions:
                print(f"[KOSA] Wersja '{selected}' nie istnieje!")
                print(f"       Dostepne: {', '.join(versions)}")
                sys.exit(1)

    if selected is None:
        if len(versions) == 1:
            selected = versions[0]
            print(f"[KOSA] Jedyna dostepna wersja: {selected}")
        else:
            selected = select_version(versions)

    # --- Sprawdz uprawnienia Administratora ---
    if not ctypes.windll.shell32.IsUserAnAdmin():
        print("[KOSA] BLAD: Bot wymaga uprawnien Administratora!")
        print("       Uruchom ponownie jako Administrator.")
        sys.exit(1)

    # --- Ustaw sys.path na wybrana wersje ---
    version_path = os.path.join(_get_versions_dir(), selected)
    sys.path.insert(0, version_path)

    debug_mode = "--debug" in sys.argv
    no_cnn = "--no-cnn" in sys.argv
    mode_str = "DEBUG" if debug_mode else "NORMALNY"
    cnn_str = "KLASYCZNY (--no-cnn)" if no_cnn else "CNN"
    print(f"[KOSA] Wersja: {selected} | Tryb: {mode_str} | Detekcja: {cnn_str}")
    print()

    # --- Import i uruchomienie bota ---
    from src.bot import KosaBot

    # Przekaz use_cnn jesli bot go obsluguje (post_cnn)
    import inspect
    bot_params = inspect.signature(KosaBot.__init__).parameters
    kwargs = {"debug": debug_mode}
    if "use_cnn" in bot_params:
        kwargs["use_cnn"] = not no_cnn
    bot = KosaBot(**kwargs)
    bot.run()


if __name__ == "__main__":
    main()
