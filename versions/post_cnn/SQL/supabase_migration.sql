-- ============================================================
-- MIGRACJA BAZY BESAFEFISH NA SUPABASE
-- Jeden plik do wklejenia w Supabase SQL Editor
-- Kolejnosc: rozszerzenia -> enumy -> trigger functions -> tabele -> funkcje biznesowe -> RLS
-- ============================================================

-- ============================================================
-- 1. ROZSZERZENIA
-- ============================================================

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS btree_gist;

-- ============================================================
-- 2. TYPY ENUM
-- ============================================================

CREATE TYPE public.payment_status AS ENUM (
    'pending',
    'succeeded',
    'failed',
    'refunded',
    'partially_refunded',
    'disputed'
);

CREATE TYPE public.subscription_status AS ENUM (
    'trialing',
    'active',
    'past_due',
    'canceled',
    'expired',
    'paused'
);

-- ============================================================
-- 3. TRIGGER FUNCTIONS
-- ============================================================

-- update_updated_at
CREATE OR REPLACE FUNCTION public.update_updated_at()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$;

-- audit_trigger_func
CREATE OR REPLACE FUNCTION public.audit_trigger_func()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    old_data JSONB;
    new_data JSONB;
    changed TEXT[];
    record_id TEXT;
    v_changed_by TEXT;
BEGIN
    v_changed_by := NULLIF(current_setting('app.current_user_id', true), '');
    IF v_changed_by IS NULL THEN
        SELECT id::TEXT INTO v_changed_by FROM users WHERE login = session_user AND is_active = true;
    END IF;

    IF TG_OP = 'DELETE' THEN
        old_data := to_jsonb(OLD);
        record_id := OLD.id::TEXT;
        old_data := old_data - 'password_hash';
        INSERT INTO audit_log (table_name, record_id, action, old_values, changed_by)
        VALUES (TG_TABLE_NAME, record_id, 'DELETE', old_data, v_changed_by);
        RETURN OLD;

    ELSIF TG_OP = 'INSERT' THEN
        new_data := to_jsonb(NEW);
        record_id := NEW.id::TEXT;
        new_data := new_data - 'password_hash';
        INSERT INTO audit_log (table_name, record_id, action, new_values, changed_by)
        VALUES (TG_TABLE_NAME, record_id, 'INSERT', new_data, v_changed_by);
        RETURN NEW;

    ELSIF TG_OP = 'UPDATE' THEN
        old_data := to_jsonb(OLD) - 'password_hash';
        new_data := to_jsonb(NEW) - 'password_hash';
        record_id := NEW.id::TEXT;

        SELECT array_agg(key) INTO changed
        FROM jsonb_each(to_jsonb(NEW)) AS n(key, value)
        WHERE n.key != 'updated_at'
          AND (NOT to_jsonb(OLD) ? n.key
               OR (to_jsonb(OLD) -> n.key) IS DISTINCT FROM n.value);

        IF changed IS NULL OR array_length(changed, 1) IS NULL THEN
            RETURN NEW;
        END IF;

        INSERT INTO audit_log (table_name, record_id, action, old_values, new_values, changed_fields, changed_by)
        VALUES (TG_TABLE_NAME, record_id, 'UPDATE', old_data, new_data, changed, v_changed_by);
        RETURN NEW;
    END IF;
END;
$$;

-- ============================================================
-- 4. TABELE (w kolejnosci zaleznosci)
-- ============================================================

-- 4a. users
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

CREATE SEQUENCE public.users_id_seq START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;
ALTER SEQUENCE public.users_id_seq OWNED BY public.users.id;
ALTER TABLE ONLY public.users ALTER COLUMN id SET DEFAULT nextval('public.users_id_seq'::regclass);

ALTER TABLE ONLY public.users ADD CONSTRAINT users_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.users ADD CONSTRAINT users_email_key UNIQUE (email);
ALTER TABLE ONLY public.users ADD CONSTRAINT users_login_key UNIQUE (login);
ALTER TABLE ONLY public.users ADD CONSTRAINT users_created_by_fkey FOREIGN KEY (created_by) REFERENCES public.users(id);

CREATE INDEX idx_users_active ON public.users USING btree (is_active) WHERE (deleted_at IS NULL);
CREATE INDEX idx_users_email_active ON public.users USING btree (email) WHERE (deleted_at IS NULL);
CREATE INDEX idx_users_role ON public.users USING btree (role) WHERE (deleted_at IS NULL);

CREATE TRIGGER set_updated_at BEFORE UPDATE ON public.users FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();
CREATE TRIGGER audit_users AFTER INSERT OR DELETE OR UPDATE ON public.users FOR EACH ROW WHEN ((pg_trigger_depth() = 0)) EXECUTE FUNCTION public.audit_trigger_func();

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

-- 4b. subscription_plans
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

CREATE SEQUENCE public.subscription_plans_id_seq START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;
ALTER SEQUENCE public.subscription_plans_id_seq OWNED BY public.subscription_plans.id;
ALTER TABLE ONLY public.subscription_plans ALTER COLUMN id SET DEFAULT nextval('public.subscription_plans_id_seq'::regclass);

ALTER TABLE ONLY public.subscription_plans ADD CONSTRAINT subscription_plans_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.subscription_plans ADD CONSTRAINT subscription_plans_slug_key UNIQUE (slug);

CREATE INDEX idx_plans_features ON public.subscription_plans USING gin (features);

CREATE TRIGGER set_updated_at BEFORE UPDATE ON public.subscription_plans FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();

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

-- 4c. user_subscriptions
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

CREATE SEQUENCE public.user_subscriptions_id_seq START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;
ALTER SEQUENCE public.user_subscriptions_id_seq OWNED BY public.user_subscriptions.id;
ALTER TABLE ONLY public.user_subscriptions ALTER COLUMN id SET DEFAULT nextval('public.user_subscriptions_id_seq'::regclass);

ALTER TABLE ONLY public.user_subscriptions ADD CONSTRAINT user_subscriptions_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.user_subscriptions ADD CONSTRAINT user_subscriptions_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id);
ALTER TABLE ONLY public.user_subscriptions ADD CONSTRAINT user_subscriptions_plan_id_fkey FOREIGN KEY (plan_id) REFERENCES public.subscription_plans(id);

ALTER TABLE ONLY public.user_subscriptions
    ADD CONSTRAINT uq_user_active_subscription EXCLUDE USING gist (user_id WITH =, tstzrange(current_period_start, current_period_end) WITH &&)
    WHERE ((status = ANY (ARRAY['active'::public.subscription_status, 'trialing'::public.subscription_status])));

CREATE INDEX idx_subscriptions_user_active ON public.user_subscriptions USING btree (user_id, status) WHERE (status = ANY (ARRAY['active'::public.subscription_status, 'trialing'::public.subscription_status]));
CREATE INDEX idx_subscriptions_expiring ON public.user_subscriptions USING btree (current_period_end) WHERE (status = ANY (ARRAY['active'::public.subscription_status, 'trialing'::public.subscription_status, 'past_due'::public.subscription_status]));
CREATE INDEX idx_subscriptions_external ON public.user_subscriptions USING btree (payment_provider, external_id);

CREATE TRIGGER set_updated_at BEFORE UPDATE ON public.user_subscriptions FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();
CREATE TRIGGER audit_subscriptions AFTER INSERT OR DELETE OR UPDATE ON public.user_subscriptions FOR EACH ROW WHEN ((pg_trigger_depth() = 0)) EXECUTE FUNCTION public.audit_trigger_func();

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

-- 4d. payments
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

CREATE SEQUENCE public.payments_id_seq START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;
ALTER SEQUENCE public.payments_id_seq OWNED BY public.payments.id;
ALTER TABLE ONLY public.payments ALTER COLUMN id SET DEFAULT nextval('public.payments_id_seq'::regclass);

ALTER TABLE ONLY public.payments ADD CONSTRAINT payments_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.payments ADD CONSTRAINT payments_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id);
ALTER TABLE ONLY public.payments ADD CONSTRAINT payments_subscription_id_fkey FOREIGN KEY (subscription_id) REFERENCES public.user_subscriptions(id);

CREATE INDEX idx_payments_user ON public.payments USING btree (user_id, created_at DESC);
CREATE INDEX idx_payments_subscription ON public.payments USING btree (subscription_id);
CREATE INDEX idx_payments_status ON public.payments USING btree (status) WHERE (status = 'pending'::public.payment_status);
CREATE INDEX idx_payments_provider ON public.payments USING btree (provider, provider_payment_id);

CREATE TRIGGER set_updated_at BEFORE UPDATE ON public.payments FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();
CREATE TRIGGER audit_payments AFTER INSERT OR DELETE OR UPDATE ON public.payments FOR EACH ROW WHEN ((pg_trigger_depth() = 0)) EXECUTE FUNCTION public.audit_trigger_func();

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

-- 4e. login_history
CREATE TABLE public.login_history (
    id bigint NOT NULL,
    user_id integer NOT NULL,
    success boolean NOT NULL,
    ip_address inet,
    user_agent text,
    failure_reason character varying(50),
    created_at timestamp with time zone DEFAULT now() NOT NULL
);

CREATE SEQUENCE public.login_history_id_seq START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;
ALTER SEQUENCE public.login_history_id_seq OWNED BY public.login_history.id;
ALTER TABLE ONLY public.login_history ALTER COLUMN id SET DEFAULT nextval('public.login_history_id_seq'::regclass);

ALTER TABLE ONLY public.login_history ADD CONSTRAINT login_history_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.login_history ADD CONSTRAINT login_history_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id);

CREATE INDEX idx_login_history_user ON public.login_history USING btree (user_id, created_at DESC);
CREATE INDEX idx_login_history_failed ON public.login_history USING btree (user_id, created_at) WHERE (success = false);

COMMENT ON TABLE public.login_history IS 'Historia logowan uzytkownikow BeSafeFish (udane i nieudane proby)';
COMMENT ON COLUMN public.login_history.id IS 'Unikalny identyfikator wpisu (auto-increment)';
COMMENT ON COLUMN public.login_history.user_id IS 'ID uzytkownika (FK -> users.id)';
COMMENT ON COLUMN public.login_history.success IS 'Czy logowanie sie powiodlo';
COMMENT ON COLUMN public.login_history.ip_address IS 'Adres IP';
COMMENT ON COLUMN public.login_history.user_agent IS 'User-Agent przegladarki/aplikacji';
COMMENT ON COLUMN public.login_history.failure_reason IS 'Powod nieudanego logowania';
COMMENT ON COLUMN public.login_history.created_at IS 'Data proby logowania';

-- 4f. audit_log
CREATE TABLE public.audit_log (
    id bigint NOT NULL,
    table_name text NOT NULL,
    record_id text NOT NULL,
    action text NOT NULL,
    old_values jsonb,
    new_values jsonb,
    changed_fields text[],
    changed_by text,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT audit_log_action_check CHECK ((action = ANY (ARRAY['INSERT'::text, 'UPDATE'::text, 'DELETE'::text])))
);

CREATE SEQUENCE public.audit_log_id_seq START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;
ALTER SEQUENCE public.audit_log_id_seq OWNED BY public.audit_log.id;
ALTER TABLE ONLY public.audit_log ALTER COLUMN id SET DEFAULT nextval('public.audit_log_id_seq'::regclass);

ALTER TABLE ONLY public.audit_log ADD CONSTRAINT audit_log_pkey PRIMARY KEY (id);

CREATE INDEX idx_audit_table_record ON public.audit_log USING btree (table_name, record_id);
CREATE INDEX idx_audit_created ON public.audit_log USING brin (created_at);

COMMENT ON TABLE public.audit_log IS 'Log audytowy zmian w tabelach BeSafeFish. Wypelniany automatycznie przez triggery audit_*';
COMMENT ON COLUMN public.audit_log.id IS 'Unikalny identyfikator wpisu (auto-increment)';
COMMENT ON COLUMN public.audit_log.table_name IS 'Nazwa tabeli w ktorej nastapila zmiana';
COMMENT ON COLUMN public.audit_log.record_id IS 'ID zmienionego rekordu';
COMMENT ON COLUMN public.audit_log.action IS 'Typ operacji: INSERT, UPDATE, DELETE';
COMMENT ON COLUMN public.audit_log.old_values IS 'Poprzednie wartosci (JSONB)';
COMMENT ON COLUMN public.audit_log.new_values IS 'Nowe wartosci (JSONB)';
COMMENT ON COLUMN public.audit_log.changed_fields IS 'Lista zmienionych kolumn';
COMMENT ON COLUMN public.audit_log.changed_by IS 'Kto dokonal zmiany';
COMMENT ON COLUMN public.audit_log.created_at IS 'Data zmiany';

-- Tabela: daily_usage (sledzenie dziennego zuzycia rund)
CREATE TABLE public.daily_usage (
    id integer NOT NULL,
    user_id integer NOT NULL,
    usage_date date NOT NULL DEFAULT CURRENT_DATE,
    rounds_used integer NOT NULL DEFAULT 0,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL
);

CREATE SEQUENCE public.daily_usage_id_seq START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;
ALTER SEQUENCE public.daily_usage_id_seq OWNED BY public.daily_usage.id;
ALTER TABLE ONLY public.daily_usage ALTER COLUMN id SET DEFAULT nextval('public.daily_usage_id_seq'::regclass);

ALTER TABLE ONLY public.daily_usage ADD CONSTRAINT daily_usage_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.daily_usage ADD CONSTRAINT daily_usage_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id);
ALTER TABLE ONLY public.daily_usage ADD CONSTRAINT daily_usage_user_date_uq UNIQUE (user_id, usage_date);

CREATE INDEX idx_daily_usage_user_date ON public.daily_usage USING btree (user_id, usage_date DESC);

CREATE TRIGGER set_updated_at BEFORE UPDATE ON public.daily_usage FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();

COMMENT ON TABLE public.daily_usage IS 'Dzienne zuzycie rund przez uzytkownikow. Jeden wiersz na usera na dzien';

-- ============================================================
-- 5. FUNKCJE BIZNESOWE
-- ============================================================

-- assign_free_subscription (musi byc przed create_user_short i check_user_subscription)
CREATE OR REPLACE FUNCTION public.assign_free_subscription(
    p_user_id integer
)
RETURNS integer
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, extensions
AS $$
DECLARE
    v_plan_id INTEGER;
    v_new_id INTEGER;
BEGIN
    IF EXISTS (
        SELECT 1 FROM user_subscriptions
        WHERE user_id = p_user_id
          AND status IN ('active', 'trialing')
    ) THEN
        RETURN NULL;
    END IF;

    SELECT id INTO v_plan_id FROM subscription_plans WHERE slug = 'darmowy' AND is_active = true;
    IF v_plan_id IS NULL THEN
        RAISE EXCEPTION 'Nie znaleziono aktywnego planu darmowego (slug=darmowy)';
    END IF;

    INSERT INTO user_subscriptions (user_id, plan_id, status, current_period_start, current_period_end, auto_renew)
    VALUES (p_user_id, v_plan_id, 'active', now(), NULL, false)
    RETURNING id INTO v_new_id;

    RETURN v_new_id;
END;
$$;

COMMENT ON FUNCTION public.assign_free_subscription IS 'Przypisuje darmowy plan uzytkownikowi. Jesli user juz ma aktywna subskrypcje - nie robi nic';

-- expire_and_fallback_to_free (musi byc przed check_user_subscription)
CREATE OR REPLACE FUNCTION public.expire_and_fallback_to_free(
    p_user_id integer
)
RETURNS boolean
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, extensions
AS $$
DECLARE
    v_expired_id INTEGER;
BEGIN
    SELECT id INTO v_expired_id
    FROM user_subscriptions
    WHERE user_id = p_user_id
      AND status IN ('active', 'trialing')
      AND current_period_end IS NOT NULL
      AND current_period_end < now()
    LIMIT 1;

    IF v_expired_id IS NULL THEN
        RETURN false;
    END IF;

    UPDATE user_subscriptions
    SET status = 'expired'
    WHERE id = v_expired_id;

    PERFORM assign_free_subscription(p_user_id);

    RETURN true;
END;
$$;

COMMENT ON FUNCTION public.expire_and_fallback_to_free IS 'Lazy expiration: wygasza przeterminowane subskrypcje i przypisuje darmowy plan';

-- create_user_short
CREATE OR REPLACE FUNCTION public.create_user_short(
    p_login character varying,
    p_email character varying,
    p_password text,
    p_role character varying DEFAULT 'user'::character varying
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

    INSERT INTO users (login, email, password_hash, role, created_by)
    VALUES (p_login, p_email, crypt(p_password, gen_salt('bf')), p_role, v_created_by)
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

    -- Dla zwyklych userow: przypisz darmowy plan subskrypcyjny
    IF p_role = 'user' THEN
        PERFORM assign_free_subscription(v_new_id);
    END IF;

    RETURN v_new_id;
END;
$$;

COMMENT ON FUNCTION public.create_user_short IS 'Tworzenie uzytkownika BeSafeFish (login, email, haslo, rola). Dla admina tworzy role PG + GRANT. Dla usera przypisuje darmowy plan';

-- create_user_long
CREATE OR REPLACE FUNCTION public.create_user_long(
    p_login character varying,
    p_email character varying,
    p_password text,
    p_role character varying DEFAULT 'user'::character varying,
    p_first_name character varying DEFAULT NULL,
    p_last_name character varying DEFAULT NULL,
    p_phone character varying DEFAULT NULL,
    p_description text DEFAULT NULL
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

COMMENT ON FUNCTION public.create_user_long IS 'Pelne tworzenie uzytkownika BeSafeFish ze wszystkimi danymi. Dla admina tworzy role PG z CREATEDB + pelne GRANT na public';

-- change_password
CREATE OR REPLACE FUNCTION public.change_password(
    p_admin_login character varying,
    p_target_login character varying,
    p_new_password text
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

-- check_user_subscription
CREATE OR REPLACE FUNCTION public.check_user_subscription(
    p_user_id integer
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

-- check_and_increment_rounds
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
    SELECT sp.features INTO v_features
    FROM user_subscriptions us
    JOIN subscription_plans sp ON sp.id = us.plan_id
    WHERE us.user_id = p_user_id
      AND us.status IN ('active', 'trialing')
      AND (us.current_period_end > now() OR us.current_period_end IS NULL)
    ORDER BY us.current_period_end DESC NULLS LAST
    LIMIT 1;

    IF v_features IS NULL THEN
        RETURN QUERY SELECT FALSE, 0, 0, 'Brak aktywnej subskrypcji'::TEXT;
        RETURN;
    END IF;

    v_max_rounds := COALESCE((v_features->>'max_rounds_per_day')::INTEGER, -1);

    IF v_max_rounds = -1 THEN
        INSERT INTO daily_usage (user_id, usage_date, rounds_used)
        VALUES (p_user_id, CURRENT_DATE, 1)
        ON CONFLICT (user_id, usage_date)
        DO UPDATE SET rounds_used = daily_usage.rounds_used + 1, updated_at = now()
        RETURNING daily_usage.rounds_used INTO v_current_used;

        RETURN QUERY SELECT TRUE, v_current_used, v_max_rounds, 'Bez limitu'::TEXT;
        RETURN;
    END IF;

    SELECT du.rounds_used INTO v_current_used
    FROM daily_usage du
    WHERE du.user_id = p_user_id AND du.usage_date = CURRENT_DATE;

    v_current_used := COALESCE(v_current_used, 0);

    IF v_current_used >= v_max_rounds THEN
        RETURN QUERY SELECT FALSE, v_current_used, v_max_rounds,
            ('Osiagnieto limit ' || v_max_rounds || ' rund dziennie')::TEXT;
        RETURN;
    END IF;

    INSERT INTO daily_usage (user_id, usage_date, rounds_used)
    VALUES (p_user_id, CURRENT_DATE, 1)
    ON CONFLICT (user_id, usage_date)
    DO UPDATE SET rounds_used = daily_usage.rounds_used + 1, updated_at = now()
    RETURNING daily_usage.rounds_used INTO v_current_used;

    RETURN QUERY SELECT TRUE, v_current_used, v_max_rounds, 'OK'::TEXT;
END;
$$;

COMMENT ON FUNCTION public.check_and_increment_rounds IS 'Sprawdza limit rund dziennych i inkrementuje licznik zuzycia';

-- ============================================================
-- 6. RLS POLITYKI
-- ============================================================

ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.subscription_plans ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.user_subscriptions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.payments ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.login_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.audit_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.daily_usage ENABLE ROW LEVEL SECURITY;

CREATE POLICY admin_full_access ON users FOR ALL USING (current_user LIKE 'adm_%') WITH CHECK (true);
CREATE POLICY admin_full_access ON subscription_plans FOR ALL USING (current_user LIKE 'adm_%') WITH CHECK (true);
CREATE POLICY admin_full_access ON user_subscriptions FOR ALL USING (current_user LIKE 'adm_%') WITH CHECK (true);
CREATE POLICY admin_full_access ON payments FOR ALL USING (current_user LIKE 'adm_%') WITH CHECK (true);
CREATE POLICY admin_full_access ON login_history FOR ALL USING (current_user LIKE 'adm_%') WITH CHECK (true);
CREATE POLICY admin_full_access ON audit_log FOR ALL USING (current_user LIKE 'adm_%') WITH CHECK (true);
CREATE POLICY admin_full_access ON daily_usage FOR ALL USING (current_user LIKE 'adm_%') WITH CHECK (true);

-- ============================================================
-- 7. DANE STARTOWE
-- ============================================================

-- Admin (created_by = NULL bo pierwszy user)
-- UWAGA: create_user_short z rola 'admin' automatycznie tworzy role PG + GRANT
-- Po uruchomieniu zmien haslo: SELECT change_password('adm_kkacperczyk', 'adm_kkacperczyk', 'PRAWDZIWE_HASLO');
-- Zmien tez haslo roli PG: ALTER ROLE adm_kkacperczyk PASSWORD 'PRAWDZIWE_HASLO';
SELECT create_user_short('adm_kkacperczyk', 'kkacperczyk@mail.com', 'ZMIEN_TO_HASLO', 'admin');

-- Plany subskrypcyjne
INSERT INTO subscription_plans (name, slug, description, price, billing_period, features, trial_days, sort_order)
VALUES
    ('Darmowy', 'darmowy', 'Plan darmowy z podstawowymi funkcjami', 0.00, 'monthly',
     '{"max_sessions": 1, "max_rounds_per_day": 50}'::jsonb, 0, 1),
    ('Premium', 'premium', 'Plan Premium z pelnym dostepem do bota', 29.99, 'monthly',
     '{"max_sessions": -1, "max_rounds_per_day": -1, "priority_support": true}'::jsonb, 7, 2),
    ('Bezplatny', 'discord-plan', 'Pelny dostep za darmo - przyznawany recznie po kontakcie na Discord', 0.00, 'monthly',
     '{"max_sessions": -1, "max_rounds_per_day": -1, "priority_support": true}'::jsonb, 0, 3);
