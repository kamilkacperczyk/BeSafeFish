-- Typy ENUM uzywane w bazie danych BeSafeFish
-- WAZNE: Enumy musza byc utworzone PRZED tabelami ktore ich uzywaja

-- Status subskrypcji (tabela: user_subscriptions)
CREATE TYPE public.subscription_status AS ENUM (
    'active',     -- Aktywna
    'expired',    -- Wygasla
    'canceled',   -- Anulowana
    'suspended'   -- Zawieszona (np. za naruszenie regulaminu)
);

-- Status klucza licencyjnego (tabela: license_keys)
CREATE TYPE public.license_key_status AS ENUM (
    'unused',     -- Nieuzywany, gotowy do aktywacji
    'activated',  -- Aktywowany przez uzytkownika
    'expired',    -- Wygasl
    'revoked'     -- Uniewazniony przez admina
);

-- Status platnosci (tabela: payments)
CREATE TYPE public.payment_status AS ENUM (
    'pending',    -- Oczekujaca
    'succeeded',  -- Zakonczona sukcesem
    'failed',     -- Nieudana
    'refunded'    -- Zwrocona
);
