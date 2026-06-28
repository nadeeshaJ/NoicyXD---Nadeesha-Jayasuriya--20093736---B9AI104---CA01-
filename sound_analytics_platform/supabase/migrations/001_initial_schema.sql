-- Sound Analytics Platform — initial Supabase schema
-- Project: fhpcrtnhqrmjsdcrpqzm
-- Run in Supabase Dashboard → SQL Editor

create extension if not exists "pgcrypto";

-- ---------------------------------------------------------------------------
-- Profiles (optional auth extension)
-- ---------------------------------------------------------------------------
create table if not exists public.profiles (
    id uuid primary key references auth.users (id) on delete cascade,
    display_name text,
    avatar_url text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

alter table public.profiles enable row level security;

create policy "profiles_select_own"
    on public.profiles for select
    using (auth.uid() = id);

create policy "profiles_update_own"
    on public.profiles for update
    using (auth.uid() = id);

create policy "profiles_insert_own"
    on public.profiles for insert
    with check (auth.uid() = id);

-- ---------------------------------------------------------------------------
-- Sound class registry
-- ---------------------------------------------------------------------------
create table if not exists public.sound_classes (
    id bigserial primary key,
    domain text not null check (domain in ('urban', 'animal')),
    class_key text not null,
    display_name text not null,
    sort_order int not null default 0,
    unique (domain, class_key)
);

alter table public.sound_classes enable row level security;

create policy "sound_classes_public_read"
    on public.sound_classes for select
    using (true);

-- ---------------------------------------------------------------------------
-- Model benchmark registry
-- ---------------------------------------------------------------------------
create table if not exists public.model_benchmarks (
    id bigserial primary key,
    model_key text not null unique,
    display_name text not null,
    total_parameters bigint not null,
    model_file_size_mb numeric(8, 2) not null,
    inference_ms_mean numeric(8, 3) not null,
    inference_ms_std numeric(8, 3),
    test_accuracy numeric(8, 6),
    test_macro_f1 numeric(8, 6),
    training_epochs int,
    device text,
    is_deployed boolean not null default false,
    notes text,
    updated_at timestamptz not null default now()
);

alter table public.model_benchmarks enable row level security;

create policy "model_benchmarks_public_read"
    on public.model_benchmarks for select
    using (true);

-- ---------------------------------------------------------------------------
-- Prediction history
-- ---------------------------------------------------------------------------
create table if not exists public.predictions (
    id uuid primary key default gen_random_uuid(),
    user_id uuid references auth.users (id) on delete set null,
    session_id text not null,
    processing_mode text not null check (processing_mode in ('urban', 'animal', 'auto')),
    routed_domain text check (routed_domain in ('urban', 'animal')),
    model_key text not null,
    input_source text not null check (input_source in ('upload', 'microphone', 'dataset')),
    original_filename text,
    top_label text not null,
    top_confidence numeric(8, 6) not null,
    probabilities jsonb not null default '{}'::jsonb,
    top_predictions jsonb not null default '[]'::jsonb,
    inference_ms numeric(10, 3),
    router_reason text,
    gradcam_enabled boolean not null default false,
    audio_storage_path text,
    device_used text,
    created_at timestamptz not null default now()
);

create index if not exists predictions_session_id_idx on public.predictions (session_id);
create index if not exists predictions_user_id_idx on public.predictions (user_id);
create index if not exists predictions_created_at_idx on public.predictions (created_at desc);

alter table public.predictions enable row level security;

create policy "predictions_insert_public"
    on public.predictions for insert
    with check (true);

create policy "predictions_select_public"
    on public.predictions for select
    using (true);

-- ---------------------------------------------------------------------------
-- Storage bucket for optional audio retention
-- ---------------------------------------------------------------------------
insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values (
    'audio-clips',
    'audio-clips',
    false,
    5242880,
    array['audio/wav', 'audio/x-wav', 'audio/wave']
)
on conflict (id) do nothing;

create policy "audio_clips_insert"
    on storage.objects for insert
    with check (bucket_id = 'audio-clips');

create policy "audio_clips_select_own_folder"
    on storage.objects for select
    using (bucket_id = 'audio-clips');

-- ---------------------------------------------------------------------------
-- Seed sound classes
-- ---------------------------------------------------------------------------
insert into public.sound_classes (domain, class_key, display_name, sort_order) values
    ('urban', 'air_conditioner', 'Air Conditioner', 1),
    ('urban', 'car_horn', 'Car Horn', 2),
    ('urban', 'children_playing', 'Children Playing', 3),
    ('urban', 'dog_bark', 'Dog Bark', 4),
    ('urban', 'drilling', 'Drilling', 5),
    ('urban', 'engine_idling', 'Engine Idling', 6),
    ('urban', 'gun_shot', 'Gun Shot', 7),
    ('urban', 'jackhammer', 'Jackhammer', 8),
    ('urban', 'siren', 'Siren', 9),
    ('urban', 'street_music', 'Street Music', 10),
    ('animal', 'dog', 'Dog', 1),
    ('animal', 'rooster', 'Rooster', 2),
    ('animal', 'pig', 'Pig', 3),
    ('animal', 'cow', 'Cow', 4),
    ('animal', 'frog', 'Frog', 5),
    ('animal', 'cat', 'Cat', 6),
    ('animal', 'hen', 'Hen', 7),
    ('animal', 'insects', 'Insects', 8),
    ('animal', 'sheep', 'Sheep', 9),
    ('animal', 'crow', 'Crow', 10)
on conflict (domain, class_key) do nothing;

-- ---------------------------------------------------------------------------
-- Seed model benchmarks (UrbanSound8K fold-10)
-- ---------------------------------------------------------------------------
insert into public.model_benchmarks (
    model_key, display_name, total_parameters, model_file_size_mb,
    inference_ms_mean, inference_ms_std, test_accuracy, test_macro_f1,
    training_epochs, device, is_deployed, notes
) values
    (
        'custom_cnn', 'Custom CNN', 6666186, 25.43,
        0.912, 0.159, 0.750299, 0.767329,
        21, 'cuda', false, 'Baseline CNN trained from scratch on Mel-spectrogram RGB images.'
    ),
    (
        'resnet50', 'ResNet50', 23528522, 90.05,
        4.408, 0.443, 0.812425, 0.811097,
        29, 'cuda', false, 'Transfer learning with ImageNet backbone.'
    ),
    (
        'mobilenetv2', 'MobileNetV2 (Deployed)', 2236682, 8.76,
        4.202, 0.374, 0.826762, 0.831002,
        30, 'cuda', true, 'Production deployment model — best accuracy/efficiency trade-off.'
    )
on conflict (model_key) do update set
    display_name = excluded.display_name,
    total_parameters = excluded.total_parameters,
    model_file_size_mb = excluded.model_file_size_mb,
    inference_ms_mean = excluded.inference_ms_mean,
    inference_ms_std = excluded.inference_ms_std,
    test_accuracy = excluded.test_accuracy,
    test_macro_f1 = excluded.test_macro_f1,
    training_epochs = excluded.training_epochs,
    device = excluded.device,
    is_deployed = excluded.is_deployed,
    notes = excluded.notes,
    updated_at = now();

-- Auto-create profile on signup
create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
    insert into public.profiles (id, display_name)
    values (new.id, coalesce(new.raw_user_meta_data->>'display_name', split_part(new.email, '@', 1)))
    on conflict (id) do nothing;
    return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
    after insert on auth.users
    for each row execute function public.handle_new_user();
