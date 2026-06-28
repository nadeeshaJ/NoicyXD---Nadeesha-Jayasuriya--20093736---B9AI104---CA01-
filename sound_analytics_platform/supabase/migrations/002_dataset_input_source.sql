-- Allow dataset-sourced predictions in history
alter table public.predictions drop constraint if exists predictions_input_source_check;
alter table public.predictions add constraint predictions_input_source_check
    check (input_source in ('upload', 'microphone', 'dataset'));
