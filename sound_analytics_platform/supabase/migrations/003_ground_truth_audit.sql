-- Ground-truth auditing for dataset sample predictions
alter table public.predictions
    add column if not exists sample_id text,
    add column if not exists ground_truth_label text,
    add column if not exists dataset_domain text check (dataset_domain in ('urban', 'animal'));

create index if not exists predictions_ground_truth_idx on public.predictions (ground_truth_label)
    where ground_truth_label is not null;
