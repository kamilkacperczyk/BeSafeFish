-- Tabela: daily_usage
-- Opis: Sledzenie dziennego zuzycia rund przez uzytkownikow
-- Jeden wiersz na uzytkownika na dzien

CREATE TABLE public.daily_usage (
    id integer NOT NULL,
    user_id integer NOT NULL,
    usage_date date NOT NULL DEFAULT CURRENT_DATE,
    rounds_used integer NOT NULL DEFAULT 0,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL
);

-- Sekwencja
CREATE SEQUENCE public.daily_usage_id_seq START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;
ALTER SEQUENCE public.daily_usage_id_seq OWNED BY public.daily_usage.id;
ALTER TABLE ONLY public.daily_usage ALTER COLUMN id SET DEFAULT nextval('public.daily_usage_id_seq'::regclass);

-- Klucze
ALTER TABLE ONLY public.daily_usage ADD CONSTRAINT daily_usage_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.daily_usage ADD CONSTRAINT daily_usage_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id);

-- Unikalnosc: jeden wiersz na usera na dzien
ALTER TABLE ONLY public.daily_usage ADD CONSTRAINT daily_usage_user_date_uq UNIQUE (user_id, usage_date);

-- Indeksy
CREATE INDEX idx_daily_usage_user_date ON public.daily_usage USING btree (user_id, usage_date DESC);

-- Triggery
CREATE TRIGGER set_updated_at BEFORE UPDATE ON public.daily_usage FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();

-- Komentarze
COMMENT ON TABLE public.daily_usage IS 'Dzienne zuzycie rund przez uzytkownikow. Jeden wiersz na usera na dzien';
COMMENT ON COLUMN public.daily_usage.id IS 'Unikalny identyfikator (auto-increment)';
COMMENT ON COLUMN public.daily_usage.user_id IS 'ID uzytkownika (FK -> users.id)';
COMMENT ON COLUMN public.daily_usage.usage_date IS 'Data (CURRENT_DATE, jeden wiersz na dzien)';
COMMENT ON COLUMN public.daily_usage.rounds_used IS 'Liczba wykorzystanych rund w danym dniu';
COMMENT ON COLUMN public.daily_usage.created_at IS 'Data utworzenia rekordu';
COMMENT ON COLUMN public.daily_usage.updated_at IS 'Data ostatniej modyfikacji (auto przez trigger)';
