-- Tabela: login_history
-- Opis: Historia logowan uzytkownikow BeSafeFish (udane i nieudane proby)
-- Triggery: brak

CREATE TABLE public.login_history (
    id bigint NOT NULL,
    user_id integer NOT NULL,
    success boolean NOT NULL,
    ip_address inet,
    user_agent text,
    failure_reason character varying(50),
    created_at timestamp with time zone DEFAULT now() NOT NULL
);

-- Sekwencja
CREATE SEQUENCE public.login_history_id_seq START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;
ALTER SEQUENCE public.login_history_id_seq OWNED BY public.login_history.id;
ALTER TABLE ONLY public.login_history ALTER COLUMN id SET DEFAULT nextval('public.login_history_id_seq'::regclass);

-- Klucze
ALTER TABLE ONLY public.login_history ADD CONSTRAINT login_history_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.login_history ADD CONSTRAINT login_history_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id);

-- Indeksy
CREATE INDEX idx_login_history_user ON public.login_history USING btree (user_id, created_at DESC);
CREATE INDEX idx_login_history_failed ON public.login_history USING btree (user_id, created_at) WHERE (success = false);

-- Komentarze
COMMENT ON TABLE public.login_history IS 'Historia logowan uzytkownikow BeSafeFish (udane i nieudane proby)';
COMMENT ON COLUMN public.login_history.id IS 'Unikalny identyfikator wpisu (auto-increment)';
COMMENT ON COLUMN public.login_history.user_id IS 'ID uzytkownika (FK -> users.id)';
COMMENT ON COLUMN public.login_history.success IS 'Czy logowanie sie powiodlo';
COMMENT ON COLUMN public.login_history.ip_address IS 'Adres IP';
COMMENT ON COLUMN public.login_history.user_agent IS 'User-Agent przegladarki/aplikacji';
COMMENT ON COLUMN public.login_history.failure_reason IS 'Powod nieudanego logowania';
COMMENT ON COLUMN public.login_history.created_at IS 'Data proby logowania';
