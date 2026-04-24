-- Tabela: audit_log
-- Opis: Log audytowy zmian w tabelach (wypelniany automatycznie przez triggery)
-- Triggery: brak (ta tabela JEST celem triggerow z innych tabel)

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

-- Sekwencja
CREATE SEQUENCE public.audit_log_id_seq START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;
ALTER SEQUENCE public.audit_log_id_seq OWNED BY public.audit_log.id;
ALTER TABLE ONLY public.audit_log ALTER COLUMN id SET DEFAULT nextval('public.audit_log_id_seq'::regclass);

-- Klucze
ALTER TABLE ONLY public.audit_log ADD CONSTRAINT audit_log_pkey PRIMARY KEY (id);

-- Indeksy
CREATE INDEX idx_audit_table_record ON public.audit_log USING btree (table_name, record_id);
CREATE INDEX idx_audit_created ON public.audit_log USING brin (created_at);

-- Komentarze
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
