-- Funkcja: expire_and_fallback_to_free
-- Opis: Lazy expiration - sprawdza czy subskrypcja premium wygasla
-- Jesli tak: oznacza ja jako 'expired' i przypisuje darmowy plan
-- Wywolywana automatycznie przez check_user_subscription (lazy check)
-- Dzieki temu nie potrzeba crona do wygaszania subskrypcji
--
-- Uzycie:
--   SELECT expire_and_fallback_to_free(42);  -- sprawdz i wygasz jesli trzeba

CREATE OR REPLACE FUNCTION public.expire_and_fallback_to_free(
    p_user_id integer  -- ID uzytkownika z tabeli users
)
RETURNS boolean
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, extensions
AS $$
DECLARE
    v_expired_id INTEGER;
BEGIN
    -- Szukaj aktywnej subskrypcji ktora juz wygasla (current_period_end < now())
    SELECT id INTO v_expired_id
    FROM user_subscriptions
    WHERE user_id = p_user_id
      AND status IN ('active', 'trialing')
      AND current_period_end IS NOT NULL
      AND current_period_end < now()
    LIMIT 1;

    IF v_expired_id IS NULL THEN
        RETURN false;
    END IF;

    -- Oznacz jako expired
    UPDATE user_subscriptions
    SET status = 'expired'
    WHERE id = v_expired_id;

    -- Przypisz darmowy plan
    PERFORM assign_free_subscription(p_user_id);

    RETURN true;
END;
$$;

COMMENT ON FUNCTION public.expire_and_fallback_to_free IS 'Lazy expiration: wygasza przeterminowane subskrypcje i przypisuje darmowy plan. Wywolywana przez check_user_subscription';
