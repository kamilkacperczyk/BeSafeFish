-- Rozszerzenia PostgreSQL wymagane przez baze danych BeSafeFish
-- WAZNE: Rozszerzenia musza byc zainstalowane PRZED tworzeniem tabel i funkcji

-- pgcrypto - hashowanie hasel (bcrypt), generowanie kluczy licencyjnych
-- Uzywane w: create_user, change_password, generate_license_keys
CREATE EXTENSION IF NOT EXISTS pgcrypto;
