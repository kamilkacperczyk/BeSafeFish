-- Funkcja: assign_free_subscription
-- Opis: Przypisuje darmowy plan uzytkownikowi (slug='darmowy')
-- Uzywana przy: rejestracji nowego usera, fallbacku po wygasnieciu premium
-- Jesli user juz ma aktywna subskrypcje - nie robi nic (zwraca NULL)
-- SECURITY DEFINER: tak (potrzebne do INSERT)
--
-- Uzycie:
--   SELECT assign_free_subscription(42);  -- przypisz darmowy plan userowi 42

CREATE OR REPLACE FUNCTION public.assign_free_subscription(
    p_user_id integer  -- ID uzytkownika z tabeli users
)
RETURNS integer
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, extensions
AS $$
DECLARE
    v_plan_id INTEGER;
    v_new_id INTEGER;
BEGIN
    -- Sprawdz czy user juz ma aktywna subskrypcje
    IF EXISTS (
        SELECT 1 FROM user_subscriptions
        WHERE user_id = p_user_id
          AND status IN ('active', 'trialing')
    ) THEN
        RETURN NULL;
    END IF;

    -- Znajdz plan darmowy
    SELECT id INTO v_plan_id FROM subscription_plans WHERE slug = 'darmowy' AND is_active = true;
    IF v_plan_id IS NULL THEN
        RAISE EXCEPTION 'Nie znaleziono aktywnego planu darmowego (slug=darmowy)';
    END IF;

    -- Wstaw subskrypcje darmowa (current_period_end = NULL = nigdy nie wygasa)
    INSERT INTO user_subscriptions (user_id, plan_id, status, current_period_start, current_period_end, auto_renew)
    VALUES (p_user_id, v_plan_id, 'active', now(), NULL, false)
    RETURNING id INTO v_new_id;

    RETURN v_new_id;
END;
$$;

COMMENT ON FUNCTION public.assign_free_subscription IS 'Przypisuje darmowy plan uzytkownikowi. Jesli user juz ma aktywna subskrypcje - nie robi nic (zwraca NULL)';
