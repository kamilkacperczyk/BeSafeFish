# Struktura bazy danych - BeSafeFish

## Diagram zaleznosci

```
users (1) ──────< login_history (N)
  │
  │ (1)
  ├──────< user_subscriptions (N) >──────── subscription_plans (1)
  │                │
  │                │ (1)
  │                └──────< payments (N)
  │
  ├──────< daily_usage (N)   [user_id + usage_date = UNIQUE]
  │
  └── self-reference (created_by)

audit_log ← triggery z: users, user_subscriptions, payments
```

## Tabele

| Tabela | Opis | Triggery |
|--------|------|----------|
| users | Uzytkownicy systemu | set_updated_at, audit_users |
| subscription_plans | Plany subskrypcyjne (Darmowy, Premium) | set_updated_at |
| user_subscriptions | Subskrypcje uzytkownikow | set_updated_at, audit_subscriptions |
| payments | Historia platnosci | set_updated_at, audit_payments |
| daily_usage | Dzienne zuzycie rund per user | brak |
| login_history | Historia logowan | brak |
| audit_log | Log audytowy (auto) | brak (cel triggerow) |

## Funkcje

| Funkcja | Opis | Sec. Definer |
|---------|------|:------------:|
| create_user_short | Tworzenie usera (login, email, haslo, rola) + auto darmowy plan | tak |
| create_user_long | Pelne tworzenie usera (+ imie, nazwisko, telefon, opis) | tak |
| change_password | Zmiana hasla (admin = kazdemu, user = sobie) | tak |
| check_user_subscription | Sprawdzenie aktywnej subskrypcji (z lazy expiration) | nie |
| assign_free_subscription | Przypisanie darmowego planu uzytkownikowi | tak |
| expire_and_fallback_to_free | Wygaszenie subskrypcji + fallback na darmowy plan | tak |
| check_and_increment_rounds | Sprawdzenie limitu rund + inkrementacja (atomowe) | nie |
| update_updated_at | Trigger: auto updated_at | nie |
| audit_trigger_func | Trigger: logowanie zmian do audit_log | tak |

## Przeplywy biznesowe

### Rejestracja
1. `create_user_short(login, email, haslo, 'user')` tworzy usera
2. Automatycznie wywoluje `assign_free_subscription(new_id)` — nowy user od razu ma plan Darmowy

### Logowanie
1. Weryfikacja hasla przez `extensions.crypt()`
2. `check_user_subscription(user_id)` zwraca aktualny plan (z lazy expiration)

### Lazy expiration subskrypcji
1. `check_user_subscription()` wywoluje `expire_and_fallback_to_free()`
2. Jesli `current_period_end < now()` → status='expired' + przypisanie darmowego planu
3. Nie wymaga crona — sprawdzane przy kazdym uzyciu

### Zuzycie rund
1. `check_and_increment_rounds(user_id)` — atomowe sprawdzenie limitu + inkrementacja
2. Zwraca: (allowed, rounds_used, max_rounds, msg)
3. Limit pochodzi z `features->>'max_rounds_per_day'` w `subscription_plans`
4. Tabela `daily_usage` — jeden rekord per user per dzien (UPSERT)

## Kolejnosc tworzenia

1. Rozszerzenia (`pgcrypto`, `btree_gist`)
2. Enumy (`payment_status`, `subscription_status`)
3. Funkcje triggerowe (`update_updated_at`, `audit_trigger_func`)
4. Tabele (w kolejnosci zaleznosci):
   - `users`
   - `subscription_plans`
   - `user_subscriptions`
   - `payments`
   - `login_history`
   - `audit_log`
   - `daily_usage`
5. Funkcje biznesowe (`create_user_*`, `change_password`, `check_user_subscription`, `assign_free_subscription`, `expire_and_fallback_to_free`, `check_and_increment_rounds`)

## Rozszerzenia

| Rozszerzenie | Cel |
|--------------|-----|
| pgcrypto | Hashowanie hasel (bcrypt) — schema `extensions` na Supabase |
| btree_gist | Exclusion constraint na subskrypcjach |
