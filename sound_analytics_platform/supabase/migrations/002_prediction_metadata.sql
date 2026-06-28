-- Extended prediction metadata for reliability, router science, and monitoring
-- Run after 001_initial_schema.sql in Supabase SQL Editor

alter table public.predictions
    add column if not exists reliability_level text check (reliability_level in ('High', 'Medium', 'Low')),
    add column if not exists is_unknown boolean not null default false,
    add column if not exists display_label text,
    add column if not exists entropy_normalized numeric(8, 6),
    add column if not exists router_metrics jsonb;

create index if not exists predictions_is_unknown_idx on public.predictions (is_unknown);
create index if not exists predictions_reliability_idx on public.predictions (reliability_level);
