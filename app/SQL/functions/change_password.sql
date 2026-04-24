-- Funkcja: change_password
-- Opis: Zmiana hasla uzytkownika
-- Admin moze zmienic haslo kazdemu, zwykly user tylko sobie
-- SECURITY DEFINER: tak (potrzebne do UPDATE z hashowaniem na Supabase)
--
-- Uzycie:
--   SELECT change_password('adm_kkacperczyk', 'testowy', 'nowehaslo123');  -- admin zmienia komus
--   SELECT change_password('testowy', 'testowy', 'mojenowehaslo');         -- user zmienia sobie

CREATE OR REPLACE FUNCTION public.change_password(
    p_admin_login character varying,   -- Login osoby wykonujacej zmiane
    p_target_login character varying,  -- Login osoby ktorej zmieniamy haslo
    p_new_password text                -- Nowe haslo (hashowane automatycznie bcrypt)
)
RETURNS boolean
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, extensions
AS $$
DECLARE
    v_admin_role VARCHAR;
BEGIN
    SELECT role INTO v_admin_role FROM users WHERE login = p_admin_login AND is_active = true;

    IF v_admin_role IS NULL THEN
        RAISE EXCEPTION 'Uzytkownik % nie istnieje lub jest nieaktywny', p_admin_login;
    END IF;

    IF v_admin_role != 'admin' AND p_admin_login != p_target_login THEN
        RAISE EXCEPTION 'Brak uprawnien - tylko admin moze zmieniac hasla innym uzytkownikom';
    END IF;

    UPDATE users
    SET password_hash = crypt(p_new_password, gen_salt('bf'))
    WHERE login = p_target_login AND is_active = true;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Uzytkownik docelowy % nie istnieje lub jest nieaktywny', p_target_login;
    END IF;

    RETURN TRUE;
END;
$$;

COMMENT ON FUNCTION public.change_password IS 'Zmiana hasla - admin moze zmienic kazdemu, zwykly user tylko sobie';
