-- Funkcja: create_user_long
-- Opis: Pelne tworzenie uzytkownika ze wszystkimi danymi osobowymi
-- Dla admina automatycznie tworzy role PostgreSQL z CREATEDB + pelne GRANT na public
-- created_by: najpierw sprawdza app.current_user_id, potem session_user (nazwa roli PG)
-- SECURITY DEFINER: tak (potrzebne do INSERT z hashowaniem i CREATE ROLE)
--
-- Uzycie:
--   SELECT create_user_long(
--       p_login := 'jan123',
--       p_email := 'jan@mail.com',
--       p_password := 'haslo',
--       p_role := 'user',
--       p_first_name := 'Jan',
--       p_last_name := 'Kowalski',
--       p_phone := '600100200',
--       p_description := 'Opis uzytkownika'
--   );

CREATE OR REPLACE FUNCTION public.create_user_long(
    p_login character varying,       -- Nazwa logowania
    p_email character varying,       -- Adres email
    p_password text,                 -- Haslo (hashowane automatycznie bcrypt)
    p_role character varying DEFAULT 'user'::character varying,   -- Rola: user/admin
    p_first_name character varying DEFAULT NULL,                  -- Imie
    p_last_name character varying DEFAULT NULL,                   -- Nazwisko
    p_phone character varying DEFAULT NULL,                       -- Numer telefonu
    p_description text DEFAULT NULL                               -- Opis / bio
)
RETURNS integer
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, extensions
AS $$
DECLARE
    v_created_by INTEGER;
    v_new_id INTEGER;
BEGIN
    SELECT id INTO v_created_by FROM users
    WHERE id = NULLIF(current_setting('app.current_user_id', true), '')::INTEGER
      AND is_active = true;

    IF v_created_by IS NULL THEN
        SELECT id INTO v_created_by FROM users
        WHERE login = session_user AND is_active = true;
    END IF;

    INSERT INTO users (login, email, password_hash, role, first_name, last_name, phone, description, created_by)
    VALUES (p_login, p_email, crypt(p_password, gen_salt('bf')), p_role, p_first_name, p_last_name, p_phone, p_description, v_created_by)
    RETURNING id INTO v_new_id;

    IF p_role = 'admin' THEN
        BEGIN
            EXECUTE format('CREATE ROLE %I WITH LOGIN PASSWORD %L CREATEDB', p_login, p_password);
            EXECUTE format('GRANT ALL ON SCHEMA public TO %I WITH GRANT OPTION', p_login);
            EXECUTE format('GRANT ALL ON ALL TABLES IN SCHEMA public TO %I WITH GRANT OPTION', p_login);
            EXECUTE format('GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO %I WITH GRANT OPTION', p_login);
            EXECUTE format('GRANT USAGE ON SCHEMA extensions TO %I', p_login);
            EXECUTE format('ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO %I WITH GRANT OPTION', p_login);
            EXECUTE format('ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO %I WITH GRANT OPTION', p_login);
        EXCEPTION WHEN OTHERS THEN
            DELETE FROM users WHERE id = v_new_id;
            RAISE EXCEPTION 'Nie udalo sie utworzyc roli PG dla admina %: %', p_login, SQLERRM;
        END;
    END IF;

    RETURN v_new_id;
END;
$$;

COMMENT ON FUNCTION public.create_user_long IS 'Pelne tworzenie uzytkownika ze wszystkimi danymi. Dla admina tworzy role PG z CREATEDB + pelne GRANT na public. created_by z app.current_user_id lub session_user';
