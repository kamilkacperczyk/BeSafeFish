# Zasady SQL — konwencje i checklist zmian

## Konwencje nazewnictwa

### Tabele
- Nazwy w **liczbie mnogiej**, snake_case: `users`, `subscription_plans`, `daily_usage`
- Prefiksy dla tabel powiazanych: `user_subscriptions`, `login_history`

### Kolumny
- snake_case: `first_name`, `created_at`
- Klucz glowny zawsze `id` (SERIAL)
- Klucze obce: `<tabela_w_liczbie_pojedynczej>_id` (np. `user_id`, `plan_id`)
- Daty tworzenia/modyfikacji: `created_at`, `updated_at` (TIMESTAMPTZ)
- Soft-delete: `deleted_at` (NULL = aktywny)

### Funkcje
- Nazwy opisowe: `create_user_short`, `check_and_increment_rounds`
- Parametry z prefiksem `p_`: `p_login`, `p_email`, `p_user_id`
- Zmienne lokalne z prefiksem `v_`: `v_created_by`, `v_new_id`

### Indeksy
- `idx_<tabela>_<kolumna>`: `idx_users_active`
- Unique: `<tabela>_<kolumna>_key`: `users_email_key`

### Triggery
- Nazwy opisowe: `set_updated_at`, `audit_users`

---

## Wymagania dla nowych tabel

1. Kazda tabela **MUSI** miec:
   - `id SERIAL PRIMARY KEY`
   - `created_at TIMESTAMPTZ NOT NULL DEFAULT now()`
   - `COMMENT ON TABLE` i `COMMENT ON COLUMN` dla kazdej kolumny

2. Tabele z modyfikowalnymi danymi **MUSZA** miec:
   - `updated_at TIMESTAMPTZ NOT NULL DEFAULT now()`
   - Trigger `set_updated_at`

3. Tabele z wrazliwymi danymi **MUSZA** miec:
   - Trigger `audit_*` logujacy zmiany do `audit_log`

4. Ceny/kwoty: `NUMERIC(10,2)` (nie w groszach)

---

## Wymagania dla nowych funkcji

1. Kazda funkcja **MUSI** miec `COMMENT ON FUNCTION`
2. Funkcje tworzace role PG **MUSZA** miec `SECURITY DEFINER`
3. SECURITY DEFINER: `SET search_path = public, extensions`
4. Hasla: `crypt(password, gen_salt('bf'))` — nigdy plaintext
5. `created_by` uzupelniac automatycznie

---

## Checklist przed zmiana w bazie

### Zmiana kolumny

1. **Sprawdz FK** — czy kolumna jest kluczem obcym lub jest referencjonowana?
   ```sql
   SELECT tc.table_name, kcu.column_name, ccu.table_name AS ref_table
   FROM information_schema.table_constraints tc
   JOIN information_schema.key_column_usage kcu ON tc.constraint_name = kcu.constraint_name
   JOIN information_schema.constraint_column_usage ccu ON tc.constraint_name = ccu.constraint_name
   WHERE tc.constraint_type = 'FOREIGN KEY'
     AND (kcu.column_name = 'NAZWA_KOLUMNY' OR ccu.column_name = 'NAZWA_KOLUMNY');
   ```

2. **Sprawdz indeksy**
   ```sql
   SELECT indexname, indexdef FROM pg_indexes
   WHERE tablename = 'NAZWA_TABELI' AND indexdef LIKE '%NAZWA_KOLUMNY%';
   ```

3. **Sprawdz triggery**
   ```sql
   SELECT tgname, pg_get_triggerdef(t.oid)
   FROM pg_trigger t JOIN pg_class c ON t.tgrelid = c.oid
   WHERE c.relname = 'NAZWA_TABELI' AND NOT tgisinternal;
   ```

4. **Sprawdz funkcje**
   ```sql
   SELECT proname, prosrc FROM pg_proc
   WHERE pronamespace = 'public'::regnamespace
     AND prosrc LIKE '%NAZWA_KOLUMNY%';
   ```

5. **Sprawdz CHECK constraints**
   ```sql
   SELECT conname, pg_get_constraintdef(oid)
   FROM pg_constraint
   WHERE conrelid = 'NAZWA_TABELI'::regclass AND contype = 'c';
   ```

6. **Zaktualizuj COMMENT ON COLUMN**

### Dodanie nowej tabeli

1. `id SERIAL PRIMARY KEY` + `created_at` + `updated_at` + triggery
2. `COMMENT ON TABLE` i `COMMENT ON COLUMN` dla kazdej kolumny
3. Indeksy (zwlaszcza na FK)
4. Stworz plik w `SQL/tables/`
5. Zaktualizuj `SQL/dependencies/foreign_keys.sql`
6. Zaktualizuj `docs/struktura-bazy.md`

### Zmiana/dodanie funkcji

1. Sprawdz czy zmiana sygnatury nie zlamie istniejacych wywolan
2. Jesli zmieniasz typy parametrow — moze byc potrzebny DROP + CREATE
3. Dodaj/zaktualizuj `COMMENT ON FUNCTION`
4. Zaktualizuj plik w `SQL/functions/`
5. Przetestuj z roznymi rolami (admin vs user)

### Zmiana typu ENUM

1. PostgreSQL NIE pozwala usuwac wartosci z ENUM
2. Dodawanie: `ALTER TYPE nazwa ADD VALUE 'nowa_wartosc';`
3. Usuniecie — nowy typ + migracja danych
4. Sprawdz wszystkie tabele uzywajace tego ENUM
5. Zaktualizuj `SQL/dependencies/enums.sql`

---

## Po kazdej zmianie

1. Zaktualizuj odpowiedni plik SQL w repo (`SQL/tables/`, `SQL/functions/`)
2. Zaktualizuj `docs/struktura-bazy.md`
3. Zaktualizuj `SQL/supabase_migration.sql`
4. Przetestuj — triggery, FK, audit
5. Commituj z opisowym komunikatem

---

## Zapytanie diagnostyczne — wszystkie zaleznosci tabeli

```sql
DO $$
DECLARE
    v_table TEXT := 'users';  -- <-- ZMIEN NA SWOJA TABELE
BEGIN
    RAISE NOTICE '=== FK z tej tabeli ===';
    RAISE NOTICE '%', (
        SELECT string_agg(conname || ': ' || pg_get_constraintdef(oid), E'\n')
        FROM pg_constraint WHERE conrelid = v_table::regclass AND contype = 'f'
    );
    RAISE NOTICE '=== FK DO tej tabeli ===';
    RAISE NOTICE '%', (
        SELECT string_agg(conname || ' (z ' || conrelid::regclass || ')', E'\n')
        FROM pg_constraint WHERE confrelid = v_table::regclass AND contype = 'f'
    );
    RAISE NOTICE '=== Triggery ===';
    RAISE NOTICE '%', (
        SELECT string_agg(tgname, E'\n')
        FROM pg_trigger t JOIN pg_class c ON t.tgrelid = c.oid
        WHERE c.relname = v_table AND NOT tgisinternal
    );
    RAISE NOTICE '=== Funkcje odwolujace sie do tabeli ===';
    RAISE NOTICE '%', (
        SELECT string_agg(proname, E'\n')
        FROM pg_proc WHERE pronamespace = 'public'::regnamespace AND prosrc LIKE '%' || v_table || '%'
    );
END $$;
```
