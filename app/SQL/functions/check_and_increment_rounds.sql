-- Funkcja: check_and_increment_rounds
-- Opis: Sprawdza limit rund dziennych i inkrementuje licznik
-- Zwraca: allowed (bool), rounds_used (int), max_rounds (int), msg (text)
--
-- Logika:
--   1. Pobiera max_rounds_per_day z features aktywnej subskrypcji
--   2. Jesli -1 (bez limitu) -> zawsze dozwolone
--   3. Jesli brak subskrypcji -> blokuje
--   4. Jesli rounds_used < max_rounds -> INSERT/UPDATE daily_usage, dozwolone
--   5. Jesli rounds_used >= max_rounds -> zablokowane
--
-- Uzycie:
--   SELECT * FROM check_and_increment_rounds(2);

CREATE OR REPLACE FUNCTION public.check_and_increment_rounds(
    p_user_id integer
)
RETURNS TABLE(
    allowed boolean,
    rounds_used integer,
    max_rounds integer,
    msg text
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, extensions
AS $$
DECLARE
    v_features JSONB;
    v_max_rounds INTEGER;
    v_current_used INTEGER;
BEGIN
    -- Pobierz features z aktywnej subskrypcji
    SELECT sp.features INTO v_features
    FROM user_subscriptions us
    JOIN subscription_plans sp ON sp.id = us.plan_id
    WHERE us.user_id = p_user_id
      AND us.status IN ('active', 'trialing')
      AND (us.current_period_end > now() OR us.current_period_end IS NULL)
    ORDER BY us.current_period_end DESC NULLS LAST
    LIMIT 1;

    -- Brak aktywnej subskrypcji
    IF v_features IS NULL THEN
        RETURN QUERY SELECT FALSE, 0, 0, 'Brak aktywnej subskrypcji'::TEXT;
        RETURN;
    END IF;

    -- Odczytaj limit rund (-1 = bez limitu)
    v_max_rounds := COALESCE((v_features->>'max_rounds_per_day')::INTEGER, -1);

    -- Bez limitu
    IF v_max_rounds = -1 THEN
        -- Nadal sledz zuzycie (statystyki), ale zawsze dozwol
        INSERT INTO daily_usage (user_id, usage_date, rounds_used)
        VALUES (p_user_id, CURRENT_DATE, 1)
        ON CONFLICT (user_id, usage_date)
        DO UPDATE SET rounds_used = daily_usage.rounds_used + 1,
                      updated_at = now()
        RETURNING daily_usage.rounds_used INTO v_current_used;

        RETURN QUERY SELECT TRUE, v_current_used, v_max_rounds, 'Bez limitu'::TEXT;
        RETURN;
    END IF;

    -- Pobierz aktualne zuzycie dzisiejsze
    SELECT du.rounds_used INTO v_current_used
    FROM daily_usage du
    WHERE du.user_id = p_user_id AND du.usage_date = CURRENT_DATE;

    v_current_used := COALESCE(v_current_used, 0);

    -- Sprawdz limit
    IF v_current_used >= v_max_rounds THEN
        RETURN QUERY SELECT FALSE, v_current_used, v_max_rounds,
            ('Osiagnieto limit ' || v_max_rounds || ' rund dziennie')::TEXT;
        RETURN;
    END IF;

    -- Inkrementuj
    INSERT INTO daily_usage (user_id, usage_date, rounds_used)
    VALUES (p_user_id, CURRENT_DATE, 1)
    ON CONFLICT (user_id, usage_date)
    DO UPDATE SET rounds_used = daily_usage.rounds_used + 1,
                  updated_at = now()
    RETURNING daily_usage.rounds_used INTO v_current_used;

    RETURN QUERY SELECT TRUE, v_current_used, v_max_rounds, 'OK'::TEXT;
END;
$$;

COMMENT ON FUNCTION public.check_and_increment_rounds IS 'Sprawdza limit rund dziennych i inkrementuje licznik zuzycia. Zwraca allowed/rounds_used/max_rounds/msg';
