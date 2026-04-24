-- Tabela: users
-- Opis: Uzytkownicy systemu BeSafeFish
-- Triggery: set_updated_at, audit_users

CREATE TABLE public.users (
    id integer NOT NULL,
    login character varying(50) NOT NULL,
    email character varying(255) NOT NULL,
    email_verified boolean DEFAULT false NOT NULL,
    password_hash text NOT NULL,
    role character varying(20) DEFAULT 'user'::character varying NOT NULL,
    is_active boolean DEFAULT true NOT NULL,
    first_name character varying(100),
    last_name character varying(100),
    phone character varying(20),
    avatar_url text,
    description text,
    last_login_at timestamp with time zone,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    created_by integer,
    deleted_at timestamp with time zone,
    CONSTRAINT users_role_check CHECK (((role)::text = ANY ((ARRAY['user'::character varying, 'admin'::character varying])::text[])))
);

-- Sekwencja
CREATE SEQUENCE public.users_id_seq START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;
ALTER SEQUENCE public.users_id_seq OWNED BY public.users.id;
ALTER TABLE ONLY public.users ALTER COLUMN id SET DEFAULT nextval('public.users_id_seq'::regclass);

-- Klucze
ALTER TABLE ONLY public.users ADD CONSTRAINT users_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.users ADD CONSTRAINT users_email_key UNIQUE (email);
ALTER TABLE ONLY public.users ADD CONSTRAINT users_login_key UNIQUE (login);
ALTER TABLE ONLY public.users ADD CONSTRAINT users_created_by_fkey FOREIGN KEY (created_by) REFERENCES public.users(id);

-- Indeksy
CREATE INDEX idx_users_active ON public.users USING btree (is_active) WHERE (deleted_at IS NULL);
CREATE INDEX idx_users_email_active ON public.users USING btree (email) WHERE (deleted_at IS NULL);
CREATE INDEX idx_users_role ON public.users USING btree (role) WHERE (deleted_at IS NULL);

-- Triggery
CREATE TRIGGER set_updated_at BEFORE UPDATE ON public.users FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();
CREATE TRIGGER audit_users AFTER INSERT OR DELETE OR UPDATE ON public.users FOR EACH ROW WHEN ((pg_trigger_depth() = 0)) EXECUTE FUNCTION public.audit_trigger_func();

-- Komentarze
COMMENT ON TABLE public.users IS 'Uzytkownicy systemu BeSafeFish. Triggery: set_updated_at (auto aktualizacja updated_at), audit_users (logowanie zmian do audit_log)';
COMMENT ON COLUMN public.users.id IS 'Unikalny identyfikator uzytkownika (auto-increment)';
COMMENT ON COLUMN public.users.login IS 'Nazwa logowania (unikalna, max 50 znakow)';
COMMENT ON COLUMN public.users.email IS 'Adres email (unikalny)';
COMMENT ON COLUMN public.users.email_verified IS 'Czy email zostal zweryfikowany';
COMMENT ON COLUMN public.users.password_hash IS 'Zahashowane haslo uzytkownika (bcrypt)';
COMMENT ON COLUMN public.users.role IS 'Rola: user lub admin';
COMMENT ON COLUMN public.users.is_active IS 'Czy konto jest aktywne';
COMMENT ON COLUMN public.users.first_name IS 'Imie';
COMMENT ON COLUMN public.users.last_name IS 'Nazwisko';
COMMENT ON COLUMN public.users.phone IS 'Numer telefonu';
COMMENT ON COLUMN public.users.avatar_url IS 'URL do zdjecia profilowego';
COMMENT ON COLUMN public.users.description IS 'Opis / bio uzytkownika';
COMMENT ON COLUMN public.users.last_login_at IS 'Data ostatniego logowania';
COMMENT ON COLUMN public.users.created_at IS 'Data utworzenia konta';
COMMENT ON COLUMN public.users.updated_at IS 'Data ostatniej modyfikacji (auto przez trigger)';
COMMENT ON COLUMN public.users.created_by IS 'ID uzytkownika ktory utworzyl konto (FK -> users.id)';
COMMENT ON COLUMN public.users.deleted_at IS 'Data soft-delete (NULL = aktywny)';
