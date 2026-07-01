-- Add test macro recall to model_benchmarks (UrbanSound8K fold-10 test set)

alter table public.model_benchmarks
    add column if not exists test_macro_recall numeric(8, 6);

update public.model_benchmarks set test_macro_recall = 0.759439 where model_key = 'custom_cnn';
update public.model_benchmarks set test_macro_recall = 0.811344 where model_key = 'resnet50';
update public.model_benchmarks set test_macro_recall = 0.827868 where model_key = 'mobilenetv2';
