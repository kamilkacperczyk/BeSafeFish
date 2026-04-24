-- Tabela: subscription_plans
-- Opis: Plany subskrypcyjne dostepne w systemie BeSafeFish
-- Triggery: set_updated_at

CREATE TABLE public.subscription_plans (
    id integer NOT NULL,
    name character varying(100) NOT NULL,
    slug character varying(50) NOT NULL,
    description text,
    price numeric(10,2) NOT NULL,
    currency character(3) DEFAULT 'PLN'::bpchar NOT NULL,
    billing_period character varying(20) NOT NULL,
    features jsonb DEFAULT '{}'::jsonb NOT NULL,
    trial_days integer DEFAULT 0 NOT NULL,
    is_active boolean DEFAULT true NOT NULL,
    sort_order integer DEFAULT 0 NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT subscription_plans_billing_period_check CHECK (((billing_period)::text = ANY ((ARRAY['monthly'::character varying, 'quarterly'::character varying, 'yearly'::character varying])::text[])))
);

-- Sekwencja
CREATE SEQUENCE public.subscription_plans_id_seq START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;
ALTER SEQUENCE public.subscription_plans_id_seq OWNED BY public.subscription_plans.id;
ALTER TABLE ONLY public.subscription_plans ALTER COLUMN id SET DEFAULT nextval('public.subscription_plans_id_seq'::regclass);

-- Klucze
ALTER TABLE ONLY public.subscription_plans ADD CONSTRAINT subscription_plans_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.subscription_plans ADD CONSTRAINT subscription_plans_slug_key UNIQUE (slug);

-- Indeksy
CREATE INDEX idx_plans_features ON public.subscription_plans USING gin (features);

-- Triggery
CREATE TRIGGER set_updated_at BEFORE UPDATE ON public.subscription_plans FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();

-- Komentarze
COMMENT ON TABLE public.subscription_plans IS 'Plany subskrypcyjne dostepne w systemie BeSafeFish. Trigger: set_updated_at';
COMMENT ON COLUMN public.subscription_plans.id IS 'Unikalny identyfikator planu (auto-increment)';
COMMENT ON COLUMN public.subscription_plans.name IS 'Nazwa planu wyswietlana uzytkownikowi';
COMMENT ON COLUMN public.subscription_plans.slug IS 'Slug URL-friendly (unikalny)';
COMMENT ON COLUMN public.subscription_plans.description IS 'Opis planu';
COMMENT ON COLUMN public.subscription_plans.price IS 'Cena planu (np. 19.99)';
COMMENT ON COLUMN public.subscription_plans.currency IS 'Waluta (domyslnie PLN)';
COMMENT ON COLUMN public.subscription_plans.billing_period IS 'Okres rozliczeniowy (np. monthly, yearly)';
COMMENT ON COLUMN public.subscription_plans.features IS 'Lista funkcji planu (JSON)';
COMMENT ON COLUMN public.subscription_plans.trial_days IS 'Liczba dni darmowego trialu';
COMMENT ON COLUMN public.subscription_plans.is_active IS 'Czy plan jest aktywny i dostepny do zakupu';
COMMENT ON COLUMN public.subscription_plans.sort_order IS 'Kolejnosc wyswietlania';
COMMENT ON COLUMN public.subscription_plans.created_at IS 'Data utworzenia planu';
COMMENT ON COLUMN public.subscription_plans.updated_at IS 'Data ostatniej modyfikacji (auto przez trigger)';
