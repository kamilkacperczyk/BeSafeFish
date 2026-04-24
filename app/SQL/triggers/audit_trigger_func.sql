-- Trigger function: audit_trigger_func
-- Opis: Loguje zmiany (INSERT/UPDATE/DELETE) do tabeli audit_log
-- Automatycznie usuwa password_hash z logow (bezpieczenstwo)
-- Ignoruje zmiany gdzie zmienil sie tylko updated_at
-- changed_by: najpierw app.current_user_id, potem session_user (fallback dla adminow)
-- SECURITY DEFINER: tak
-- Przypisana do tabel: users, user_subscriptions, payments
--
-- Przypisanie triggera do tabeli:
--   CREATE TRIGGER audit_<nazwa> AFTER INSERT OR DELETE OR UPDATE ON <tabela>
--   FOR EACH ROW WHEN ((pg_trigger_depth() = 0))
--   EXECUTE FUNCTION public.audit_trigger_func();

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
