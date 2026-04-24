# Regulamin serwisu i RODO - wnioski i instrukcje

## Kontekst

Projekt BeSafeFish zbiera dane osobowe (email, IP, login history). RODO wymaga podstawy prawnej, informowania uzytkownika i checkboxa akceptacji przy rejestracji.

---

## 1. Jakie dane sa danymi osobowymi wg RODO

- **Email** - oczywiste
- **Adres IP** - tak, nawet dynamiczny (wyrok CJEU C-582/14)
- **Nazwa uzytkownika** - jesli mozna ja powiazac z osoba
- **User-Agent** - sam w sobie nie, ale w polaczeniu z IP moze identyfikowac
- **Zahashowane haslo** - dane osobowe (mozna je powiazac z kontem)

**Wazne:** Nawet jesli dane sa zahashowane/zanonimizowane czesciowo, RODO moze je obejmowac jesli mozna je powiazac z konkretna osoba.

---

## 2. Podstawy prawne przetwarzania (art. 6 RODO)

| Dane | Podstawa prawna | Artykul |
|------|----------------|---------|
| Email, login, haslo | Wykonanie umowy (rejestracja = umowa) | art. 6 ust. 1 lit. b |
| Historia logowan (IP) | Uzasadniony interes (bezpieczenstwo) | art. 6 ust. 1 lit. f |
| Dane uzytkowania (rundy) | Wykonanie umowy (egzekwowanie limitow planu) | art. 6 ust. 1 lit. b |

**Uzasadniony interes** (lit. f) - nie wymaga zgody, ale wymaga:
- Informowania uzytkownika (polityka prywatnosci)
- Mozliwosci sprzeciwu
- Proporcjonalnosci (nie zbieraj wiecej niz potrzebujesz)

---

## 3. Co musi zawierac regulamin (minimum)

1. **Postanowienia ogolne** - co to za serwis, charakter (edukacyjny/komercyjny)
2. **Konto uzytkownika** - wymagane dane, odpowiedzialnosc, zasady blokowania
3. **Plany/subskrypcje** - co oferuje kazdy plan, co sie dzieje po wygasnieciu
4. **Odpowiedzialnosc** - kluczowe: serwis nie odpowiada za sankcje od wydawcy gry
   - Nie pisz "nie gwarantujemy skutecznosci" wprost - to brzmi zle
   - Lepiej: "skutecznosc zalezy od wielu czynnikow (rozdzielczosc, ustawienia, obciazenie)"
   - Uzyj zwrotow neutralnych, nie obronnych
5. **Jakie dane zbieramy** - lista z opisem
6. **Cel przetwarzania** - dlaczego zbieramy kazdy typ danych + artykul RODO
7. **Przechowywanie i ochrona** - hashowanie, retencja (90 dni na logi), SSL, lokalizacja serwerow
8. **Prawa uzytkownika** - dostep, sprostowanie, usuniecie, przenoszenie, sprzeciw
9. **Zmiany regulaminu** - zastrzezenie prawa do zmian

---

## 4. Implementacja checkboxa akceptacji

### Gdzie

- **Strona WWW** - checkbox w formularzu rejestracji z linkiem do sekcji regulaminu
- **GUI desktopowe** - QCheckBox widoczny tylko w trybie rejestracji

### Walidacja

- **Po stronie klienta** (JS / PySide6) - wystarczy, checkbox to element UX
- **Po stronie backendu** - opcjonalne, ale rekomendowane jesli zalezy Ci na pewnosci
- Checkbox musi byc **domyslnie odznaczony** (RODO wymaga aktywnej zgody)

### Przyklad HTML
```html
<div class="form-group form-group-checkbox">
    <label class="checkbox-label">
        <input type="checkbox" id="reg-terms">
        <span>Akceptuje <a href="#terms">regulamin serwisu</a> i polityke prywatnosci</span>
    </label>
</div>
```

### Przyklad PySide6
```python
self._terms_checkbox = QCheckBox("Akceptuje regulamin serwisu i polityke prywatnosci")
self._terms_checkbox.setVisible(False)  # widoczny tylko w trybie rejestracji

# Walidacja przed rejstracja:
if not self._terms_checkbox.isChecked():
    self._show_error("Musisz zaakceptowac regulamin serwisu.")
    return
```

### Przyklad JS
```javascript
var terms = document.getElementById('reg-terms').checked;
if (!terms) {
    msg.textContent = 'Musisz zaakceptowac regulamin serwisu.';
    msg.classList.add('error');
    return;
}
```

---

## 5. Retencja danych (ile przechowywac)

| Dane | Okres retencji | Uzasadnienie |
|------|---------------|-------------|
| Konto (login, email, haslo) | Do usuniecia konta przez uzytkownika | Niezbedne do swiadczenia uslugi |
| Historia logowan (IP, UA) | Max 90 dni | Bezpieczenstwo - dluzej nie ma uzasadnienia |
| Dane uzytkowania (rundy) | Do konca dnia / okresu rozliczeniowego | Egzekwowanie limitow |
| Audit log | 1 rok | Rozliczalnosc (RODO art. 5 ust. 2) |

**Wazne:** Zadeklaruj retencje w regulaminie i DOTRZYMUJ jej. Najlepiej zautomatyzowac czyszczenie (SQL job lub lazy delete).

### Przyklad SQL - czyszczenie starych logow
```sql
DELETE FROM login_history WHERE created_at < now() - interval '90 days';
```

---

## 6. Przechowywanie IP - szczegoly techniczne

### Problem z X-Forwarded-For
Proxy (Render, Cloudflare) dodaje lancuch IP: `przykladowo: 84.52.177.93, 172.64.198.12, 10.23.172.5`.
Kolumna INET akceptuje tylko jeden adres.

### Rozwiazanie
```python
xff = request.headers.get("X-Forwarded-For", "")
ip_address = xff.split(",")[0].strip() if xff else request.remote_addr
```
Pierwszy IP w lancuchu = prawdziwy IP klienta.

---

## 7. Gdzie umiescic regulamin

### Opcja A: Sekcja na stronie (zastosowana w Kosa)
- Regulamin jako sekcja `#terms` na glownej stronie
- Link w nawigacji i footerze
- Checkbox z linkiem `<a href="#terms">regulamin serwisu</a>`
- **Zalety:** prosty, nie wymaga dodatkowego routingu
- **Wady:** wydluza strone

### Opcja B: Osobna podstrona `/regulamin`
- Wymaga dodatkowej trasy w backendzie
- **Zalety:** czytelniejsza struktura, latwiej linkować
- **Wady:** wiecej kodu

### GUI desktopowe (sprawdzone - QDialog)
**NIGDY nie umieszczaj regulaminu inline w formularzu** - jest za maly, nieczytelny.

Sprawdzony wzorzec:
1. Przycisk "Przeczytaj regulamin" w formularzu rejestracji
2. Klik otwiera osobne okno dialogowe (QDialog, modal, min 560x500px)
3. QTextBrowser z pelnym HTML regulaminu
4. Uzytkownik musi przewinac do konca zeby odblokowac przycisk "Akceptuje"
5. Po akceptacji przycisk zmienia tekst na "Regulamin zaakceptowany" (zielony)
6. Proba rejestracji bez akceptacji -> blad + automatyczne otwarcie dialogu

---

## 8. Checklist - dodawanie regulaminu do nowego projektu

1. [ ] Zidentyfikuj jakie dane zbierasz (email, IP, hasla, logi, cookies)
2. [ ] Dobierz podstawe prawna dla kazdego typu danych (art. 6 RODO)
3. [ ] Napisz regulamin (9 sekcji powyzej jako szablon)
4. [ ] Dodaj sekcje/podstrone z regulaminem na stronie
5. [ ] Dodaj checkbox w kazdym formularzu rejestracji (web + GUI)
6. [ ] Walidacja checkboxa po stronie klienta (domyslnie odznaczony!)
7. [ ] Link do regulaminu w nawigacji i footerze
8. [ ] Zdefiniuj i zaimplementuj retencje danych (auto-czyszczenie logow)
9. [ ] Dodaj date "Ostatnia aktualizacja" do regulaminu
10. [ ] Style dopasowane do reszty strony (karty, typografia)
