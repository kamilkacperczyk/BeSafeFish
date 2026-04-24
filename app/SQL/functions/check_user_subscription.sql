-- Funkcja: check_user_subscription
-- Opis: Sprawdza czy uzytkownik ma aktywna subskrypcje
-- Admin = pelny dostep bez subskrypcji (full_access: true, brak daty wygasniecia)
-- User = sprawdza aktywna subskrypcje w user_subscriptions
-- Darmowy plan (current_period_end = NULL) = nigdy nie wygasa
-- Lazy expiration: automatycznie wygasza premium po dacie i nadaje darmowy plan
--
-- Uzycie:
--   SELECT * FROM check_user_subscription(2);  -- sprawdz usera o id=2

CREATE OR REPLACE FUNCTION public.check_user_subscription(
    p_user_id integer  -- ID uzytkownika z tabeli users
)
RETURNS TABLE(
    has_active boolean,
    plan_name character varying,
    features jsonb,
    expires_at timestamp with time zone
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_role VARCHAR;
BEGIN
    SELECT role INTO v_role FROM users WHERE id = p_user_id AND is_active = true;

    IF v_role = 'admin' THEN
        RETURN QUERY SELECT TRUE, 'Admin'::VARCHAR, '{"full_access": true}'::JSONB, NULL::TIMESTAMPTZ;
        RETURN;
    END IF;

    -- Lazy expiration: wygasz przeterminowane premium i nadaj darmowy
    PERFORM expire_and_fallback_to_free(p_user_id);

    RETURN QUERY
    SELECT TRUE, sp.name, sp.features, us.current_period_end
    FROM user_subscriptions us
    JOIN subscription_plans sp ON sp.id = us.plan_id
    WHERE us.user_id = p_user_id
      AND us.status IN ('active', 'trialing')
      AND (us.current_period_end > now() OR us.current_period_end IS NULL)
    ORDER BY us.current_period_end DESC NULLS LAST
    LIMIT 1;

    IF NOT FOUND THEN
        RETURN QUERY SELECT FALSE, NULL::VARCHAR, NULL::JSONB, NULL::TIMESTAMPTZ;
    END IF;
END;
$$;

COMMENT ON FUNCTION public.check_user_subscription IS 'Sprawdza dostep uzytkownika BeSafeFish. Admin = pelny dostep bez subskrypcji. User = sprawdza aktywna subskrypcje';
