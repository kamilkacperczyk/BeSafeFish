-- Tabela: user_subscriptions
-- Opis: Subskrypcje uzytkownikow BeSafeFish
-- Triggery: set_updated_at, audit_subscriptions

CREATE TABLE public.user_subscriptions (
    id integer NOT NULL,
    user_id integer NOT NULL,
    plan_id integer NOT NULL,
    status public.subscription_status DEFAULT 'trialing'::public.subscription_status NOT NULL,
    current_period_start timestamp with time zone NOT NULL,
    current_period_end timestamp with time zone,
    trial_end timestamp with time zone,
    canceled_at timestamp with time zone,
    cancel_at_period_end boolean DEFAULT false NOT NULL,
    auto_renew boolean DEFAULT true NOT NULL,
    external_id character varying(255),
    payment_provider character varying(50),
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL
);

-- Sekwencja
CREATE SEQUENCE public.user_subscriptions_id_seq START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;
ALTER SEQUENCE public.user_subscriptions_id_seq OWNED BY public.user_subscriptions.id;
ALTER TABLE ONLY public.user_subscriptions ALTER COLUMN id SET DEFAULT nextval('public.user_subscriptions_id_seq'::regclass);

-- Klucze
ALTER TABLE ONLY public.user_subscriptions ADD CONSTRAINT user_subscriptions_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.user_subscriptions ADD CONSTRAINT user_subscriptions_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id);
ALTER TABLE ONLY public.user_subscriptions ADD CONSTRAINT user_subscriptions_plan_id_fkey FOREIGN KEY (plan_id) REFERENCES public.subscription_plans(id);

-- Exclusion constraint - zapobiega nakladaniu sie aktywnych subskrypcji
ALTER TABLE ONLY public.user_subscriptions
    ADD CONSTRAINT uq_user_active_subscription EXCLUDE USING gist (user_id WITH =, tstzrange(current_period_start, current_period_end) WITH &&)
    WHERE ((status = ANY (ARRAY['active'::public.subscription_status, 'trialing'::public.subscription_status])));

-- Indeksy
CREATE INDEX idx_subscriptions_user_active ON public.user_subscriptions USING btree (user_id, status) WHERE (status = ANY (ARRAY['active'::public.subscription_status, 'trialing'::public.subscription_status]));
CREATE INDEX idx_subscriptions_expiring ON public.user_subscriptions USING btree (current_period_end) WHERE (status = ANY (ARRAY['active'::public.subscription_status, 'trialing'::public.subscription_status, 'past_due'::public.subscription_status]));
CREATE INDEX idx_subscriptions_external ON public.user_subscriptions USING btree (payment_provider, external_id);

-- Triggery
CREATE TRIGGER set_updated_at BEFORE UPDATE ON public.user_subscriptions FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();
CREATE TRIGGER audit_subscriptions AFTER INSERT OR DELETE OR UPDATE ON public.user_subscriptions FOR EACH ROW WHEN ((pg_trigger_depth() = 0)) EXECUTE FUNCTION public.audit_trigger_func();

-- Komentarze
COMMENT ON TABLE public.user_subscriptions IS 'Subskrypcje uzytkownikow BeSafeFish. Triggery: set_updated_at, audit_subscriptions (logowanie zmian do audit_log)';
COMMENT ON COLUMN public.user_subscriptions.id IS 'Unikalny identyfikator subskrypcji (auto-increment)';
COMMENT ON COLUMN public.user_subscriptions.user_id IS 'ID uzytkownika (FK -> users.id)';
COMMENT ON COLUMN public.user_subscriptions.plan_id IS 'ID planu subskrypcyjnego (FK -> subscription_plans.id)';
COMMENT ON COLUMN public.user_subscriptions.status IS 'Status: trialing, active, canceled, expired, past_due';
COMMENT ON COLUMN public.user_subscriptions.current_period_start IS 'Poczatek biezacego okresu rozliczeniowego';
COMMENT ON COLUMN public.user_subscriptions.current_period_end IS 'Koniec biezacego okresu rozliczeniowego (NULL = nigdy nie wygasa)';
COMMENT ON COLUMN public.user_subscriptions.trial_end IS 'Data zakonczenia trialu';
COMMENT ON COLUMN public.user_subscriptions.canceled_at IS 'Data anulowania subskrypcji';
COMMENT ON COLUMN public.user_subscriptions.cancel_at_period_end IS 'Czy anulowac na koniec okresu';
COMMENT ON COLUMN public.user_subscriptions.auto_renew IS 'Czy automatycznie odnawiac';
COMMENT ON COLUMN public.user_subscriptions.external_id IS 'ID w zewnetrznym systemie platnosci';
COMMENT ON COLUMN public.user_subscriptions.payment_provider IS 'Dostawca platnosci (np. stripe, przelewy24)';
COMMENT ON COLUMN public.user_subscriptions.created_at IS 'Data utworzenia subskrypcji';
COMMENT ON COLUMN public.user_subscriptions.updated_at IS 'Data ostatniej modyfikacji (auto przez trigger)';
