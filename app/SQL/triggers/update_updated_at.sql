-- Trigger function: update_updated_at
-- Opis: Automatycznie ustawia updated_at = now() przy kazdym UPDATE
-- Przypisana do tabel: users, license_plans, user_subscriptions, license_keys, payments
--
-- Przypisanie triggera do tabeli:
--   CREATE TRIGGER set_updated_at BEFORE UPDATE ON <tabela>
--   FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();

CREATE OR REPLACE FUNCTION public.update_updated_at()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$;
