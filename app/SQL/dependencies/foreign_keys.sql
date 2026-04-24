-- Mapa kluczy obcych (Foreign Keys) w bazie danych BeSafeFish
-- UWAGA: Ten plik jest dokumentacja referencyjna.
-- FK sa zdefiniowane w plikach tabel (SQL/tables/*.sql) i w supabase_migration.sql.
-- NIE uruchamiaj tego pliku osobno - spowoduje duplikaty constraintow.

-- === KOLEJNOSC TWORZENIA TABEL ===
-- 1. users (niezalezna, self-reference na created_by)
-- 2. subscription_plans (niezalezna)
-- 3. user_subscriptions (zalezy od: users, subscription_plans)
-- 4. payments (zalezy od: users, user_subscriptions)
-- 5. login_history (zalezy od: users)
-- 6. audit_log (niezalezna, wypelniana przez triggery)

-- === FOREIGN KEYS ===

-- users.created_by -> users.id (self-reference: kto utworzyl konto)
-- ALTER TABLE ONLY public.users
--     ADD CONSTRAINT users_created_by_fkey FOREIGN KEY (created_by) REFERENCES public.users(id);

-- login_history.user_id -> users.id
-- ALTER TABLE ONLY public.login_history
--     ADD CONSTRAINT login_history_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id);

-- user_subscriptions.user_id -> users.id
-- ALTER TABLE ONLY public.user_subscriptions
--     ADD CONSTRAINT user_subscriptions_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id);

-- user_subscriptions.plan_id -> subscription_plans.id
-- ALTER TABLE ONLY public.user_subscriptions
--     ADD CONSTRAINT user_subscriptions_plan_id_fkey FOREIGN KEY (plan_id) REFERENCES public.subscription_plans(id);

-- payments.user_id -> users.id
-- ALTER TABLE ONLY public.payments
--     ADD CONSTRAINT payments_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id);

-- payments.subscription_id -> user_subscriptions.id
-- ALTER TABLE ONLY public.payments
--     ADD CONSTRAINT payments_subscription_id_fkey FOREIGN KEY (subscription_id) REFERENCES public.user_subscriptions(id);
