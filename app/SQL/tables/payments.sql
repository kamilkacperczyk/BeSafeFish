-- Tabela: payments
-- Opis: Historia platnosci BeSafeFish
-- Triggery: set_updated_at, audit_payments

CREATE TABLE public.payments (
    id integer NOT NULL,
    user_id integer NOT NULL,
    subscription_id integer,
    amount numeric(10,2) NOT NULL,
    currency character(3) DEFAULT 'PLN'::bpchar NOT NULL,
    status public.payment_status DEFAULT 'pending'::public.payment_status NOT NULL,
    payment_method character varying(50),
    provider character varying(50),
    provider_payment_id character varying(255),
    description text,
    failure_reason text,
    refunded_amount numeric(10,2) DEFAULT 0 NOT NULL,
    paid_at timestamp with time zone,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL
);

-- Sekwencja
CREATE SEQUENCE public.payments_id_seq START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;
ALTER SEQUENCE public.payments_id_seq OWNED BY public.payments.id;
ALTER TABLE ONLY public.payments ALTER COLUMN id SET DEFAULT nextval('public.payments_id_seq'::regclass);

-- Klucze
ALTER TABLE ONLY public.payments ADD CONSTRAINT payments_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.payments ADD CONSTRAINT payments_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id);
ALTER TABLE ONLY public.payments ADD CONSTRAINT payments_subscription_id_fkey FOREIGN KEY (subscription_id) REFERENCES public.user_subscriptions(id);

-- Indeksy
CREATE INDEX idx_payments_user ON public.payments USING btree (user_id, created_at DESC);
CREATE INDEX idx_payments_subscription ON public.payments USING btree (subscription_id);
CREATE INDEX idx_payments_status ON public.payments USING btree (status) WHERE (status = 'pending'::public.payment_status);
CREATE INDEX idx_payments_provider ON public.payments USING btree (provider, provider_payment_id);

-- Triggery
CREATE TRIGGER set_updated_at BEFORE UPDATE ON public.payments FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();
CREATE TRIGGER audit_payments AFTER INSERT OR DELETE OR UPDATE ON public.payments FOR EACH ROW WHEN ((pg_trigger_depth() = 0)) EXECUTE FUNCTION public.audit_trigger_func();

-- Komentarze
COMMENT ON TABLE public.payments IS 'Historia platnosci BeSafeFish. Triggery: set_updated_at, audit_payments (logowanie zmian do audit_log)';
COMMENT ON COLUMN public.payments.id IS 'Unikalny identyfikator platnosci (auto-increment)';
COMMENT ON COLUMN public.payments.user_id IS 'ID uzytkownika (FK -> users.id)';
COMMENT ON COLUMN public.payments.subscription_id IS 'ID subskrypcji (FK -> user_subscriptions.id)';
COMMENT ON COLUMN public.payments.amount IS 'Kwota platnosci (np. 19.99)';
COMMENT ON COLUMN public.payments.currency IS 'Waluta (domyslnie PLN)';
COMMENT ON COLUMN public.payments.status IS 'Status: pending, succeeded, failed, refunded, partially_refunded, disputed';
COMMENT ON COLUMN public.payments.payment_method IS 'Metoda platnosci (np. card, blik, transfer)';
COMMENT ON COLUMN public.payments.provider IS 'Dostawca platnosci';
COMMENT ON COLUMN public.payments.provider_payment_id IS 'ID transakcji u dostawcy';
COMMENT ON COLUMN public.payments.description IS 'Opis platnosci';
COMMENT ON COLUMN public.payments.failure_reason IS 'Powod niepowodzenia platnosci';
COMMENT ON COLUMN public.payments.refunded_amount IS 'Kwota zwrotu (np. 19.99)';
COMMENT ON COLUMN public.payments.paid_at IS 'Data zaksiegowania platnosci';
COMMENT ON COLUMN public.payments.created_at IS 'Data utworzenia rekordu';
COMMENT ON COLUMN public.payments.updated_at IS 'Data ostatniej modyfikacji (auto przez trigger)';
