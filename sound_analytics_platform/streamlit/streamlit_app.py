"""
Streamlit web app for environmental sound classification.

CA1 deployment demo (Section 6):
    - Urban mode: MobileNetV2 trained on UrbanSound8K (10 urban classes)
    - Animal mode: ImageNet-transfer MobileNetV2 on ESC-50 animals
    - Shows waveform, Mel-spectrogram, model input, top-3 probabilities
    - Optional multi-model comparison and Grad-CAM explainability

Run (from project root):
    python -m streamlit run sound_analytics_platform/streamlit/streamlit_app.py
"""



from __future__ import annotations



import sys

from pathlib import Path



ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:

    sys.path.insert(0, str(ROOT))  # allow `from src...` imports when run via streamlit



import streamlit as st

import torch



from src.domain_router import route_audio_domain

from src.predict import (

    available_models_for_mode,

    load_benchmark_table,

    load_mode_model,

    plot_mel_spectrogram,

    plot_waveform,

    predict_with_gradcam,

)

from src.utils import load_config, project_path



try:

    from streamlit_mic_recorder import mic_recorder



    MIC_RECORDER_AVAILABLE = True

except ImportError:

    MIC_RECORDER_AVAILABLE = False



st.set_page_config(

    page_title="Environmental Sound Analytics",

    page_icon="🔊",

    layout="wide",

)



PROCESSING_MODES = {

    "Urban Sound": "urban",           # UrbanSound8K MobileNetV2 expert

    "Animal Vocalization": "animal",  # ESC-50 Animals MobileNetV2 expert

    "Smart Auto-Router": "auto",      # run both experts, pick best domain

}





@st.cache_resource

def get_device() -> torch.device:

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")





@st.cache_resource

def get_benchmarks():

    # Load once per session — config and benchmark JSON don't change at runtime
    cfg = load_config()

    return load_benchmark_table(cfg), cfg





@st.cache_resource

def get_model_bundle(mode_key: str, model_name: str):

    # Cache loaded model weights — reloading .pt on every click would be slow
    cfg = load_config()

    device = get_device()

    model, class_names, deploy_cfg, chosen_name, checkpoint = load_mode_model(

        mode_key,

        cfg,

        device,

        model_name=model_name,

    )

    return model, class_names, deploy_cfg, chosen_name, checkpoint, cfg, device





@st.cache_resource

def get_router_experts():

    # Both MobileNetV2 experts loaded once for Smart Auto-Router mode
    cfg = load_config()

    device = get_device()

    urban_model, urban_classes, _, urban_name, _, _, _ = load_mode_model("urban", cfg, device, model_name="mobilenetv2")

    animal_model, animal_classes, _, animal_name, _, _, _ = load_mode_model("animal", cfg, device, model_name="mobilenetv2")

    return urban_model, urban_classes, urban_name, animal_model, animal_classes, animal_name, cfg, device





def render_mic_recorder() -> bytes | None:

    if not MIC_RECORDER_AVAILABLE:

        st.caption("Install `streamlit-mic-recorder` to enable live recording.")

        return None



    recording = mic_recorder(

        start_prompt="🎙️ Click to record",

        stop_prompt="⏹ Stop recording",

        just_once=False,

        use_container_width=True,

        key="live_mic_recorder",

    )

    if recording and recording.get("bytes"):

        st.audio(recording["bytes"], format="audio/wav")

        return recording["bytes"]

    return None





def model_label(model_name: str, cfg: dict) -> str:

    return cfg["app"]["model_labels"].get(model_name, model_name)





def render_benchmark_metrics(model_name: str, live_ms: float | None, benchmarks: dict[str, dict]) -> None:

    bench = benchmarks.get(model_name, {})

    col_a, col_b, col_c, col_d = st.columns(4)



    latency_text = f"{live_ms:.2f} ms" if live_ms is not None else "—"

    if bench:

        latency_text = f"{latency_text} (bench {bench.get('inference_ms_mean', '—')} ms)"



    col_a.metric("Live Inference Latency", latency_text)

    col_b.metric(

        "Checkpoint Size",

        f"{bench.get('model_file_size_mb', '—')} MB" if bench else "—",

    )

    col_c.metric(

        "Parameters",

        f"{bench.get('total_parameters', 0):,}" if bench else "—",

    )

    col_d.metric(

        "Urban Test Accuracy",

        f"{bench.get('test_accuracy', 0):.1%}" if bench and "test_accuracy" in bench else "—",

    )





def render_predictions(result: dict) -> None:

    top = result["predictions"][0]

    st.success(

        f"**{top['label'].replace('_', ' ').title()}** — confidence {top['confidence']:.1%}"

    )



    st.markdown("**Top 3 predictions**")

    for pred in result["predictions"]:

        st.progress(

            min(max(pred["confidence"], 0.0), 1.0),

            text=f"{pred['label']}: {pred['confidence']:.1%}",

        )



    with st.expander("All class probabilities"):

        for label, prob in sorted(result["probabilities"].items(), key=lambda x: x[1], reverse=True):

            st.write(f"{label}: {prob:.1%}")





def render_sidebar(cfg: dict, benchmarks: dict) -> tuple[str, str, bool]:

    st.sidebar.title("🎛️ Control Center")



    mode_label = st.sidebar.radio("Select Processing Mode", list(PROCESSING_MODES.keys()))

    mode_key = PROCESSING_MODES[mode_label]



    st.sidebar.markdown("---")

    st.sidebar.subheader("Backend Model Engine")



    if mode_key == "auto":

        preview_mode = "urban"

    else:

        preview_mode = mode_key



    available = available_models_for_mode(preview_mode, cfg)

    if mode_key == "animal" and len(available) == 1:

        st.sidebar.caption("Animal mode uses the trained MobileNetV2 expert. Other engines are urban benchmarks.")



    default_model = cfg["app"].get("default_model", "mobilenetv2")

    if default_model not in available and available:

        default_model = available[0]



    model_options = available or [default_model]

    model_labels = [model_label(name, cfg) for name in model_options]

    selected_label = st.sidebar.selectbox(

        "Model",

        model_labels,

        index=model_options.index(default_model) if default_model in model_options else 0,

    )

    model_name = model_options[model_labels.index(selected_label)]



    show_gradcam = st.sidebar.toggle("Grad-CAM explainability", value=True)

    show_benchmarks = st.sidebar.toggle("Show benchmark comparison cards", value=True)



    st.sidebar.markdown("---")

    st.sidebar.subheader("Preprocessing")

    st.sidebar.write(f"Sample rate: {cfg['audio']['sample_rate']} Hz")

    st.sidebar.write(f"Duration: {cfg['audio']['duration_sec']} s")

    st.sidebar.write(f"Image size: {cfg['image']['height']}×{cfg['image']['width']}")

    st.sidebar.write(f"Mel bins: {cfg['spectrogram']['n_mels']}")



    device_name = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"

    st.sidebar.write(f"Inference device: {device_name}")



    if show_benchmarks and model_name in benchmarks:

        bench = benchmarks[model_name]

        st.sidebar.markdown("---")

        st.sidebar.subheader("Static Benchmarks")

        st.sidebar.write(f"GPU latency: {bench.get('inference_ms_mean', '—')} ms")

        st.sidebar.write(f"Checkpoint: {bench.get('model_file_size_mb', '—')} MB")

        st.sidebar.write(f"Urban accuracy: {bench.get('test_accuracy', 0):.1%}")



    return mode_key, model_name, show_gradcam





def resolve_inference_context(mode_key: str, model_name: str) -> tuple[str, str]:

    if mode_key == "auto":

        return "urban", model_name

    return mode_key, model_name





def main() -> None:

    st.title("🔊 Environmental Sound Analytics Engine")

    st.caption(

        "Upload or record a 4-second clip → Mel-spectrogram RGB image → CNN inference with optional Grad-CAM and model benchmarking."

    )



    benchmarks, cfg = get_benchmarks()

    mode_key, model_name, show_gradcam = render_sidebar(cfg, benchmarks)



    st.markdown("---")

    col_upload, col_record = st.columns(2)



    with col_upload:

        st.subheader("Upload Audio Waveform (.wav)")

        uploaded = st.file_uploader("Choose a WAV file", type=["wav"], label_visibility="collapsed")



    with col_record:

        st.subheader("Record Live Mic Audio")

        recorded_bytes = render_mic_recorder()



    audio_bytes = None

    input_source = None

    if uploaded is not None:

        audio_bytes = uploaded.read()

        input_source = uploaded.name

    elif recorded_bytes is not None:

        audio_bytes = recorded_bytes

        input_source = "Live microphone recording"



    if audio_bytes is None:

        st.info("Upload a `.wav` file or record live audio to run the full pipeline.")

        with st.expander("Supported classes"):

            st.write("**Urban:** " + ", ".join(cfg["datasets"]["urbansound8k"]["classes"]))

            st.write("**Animal:** " + ", ".join(cfg["datasets"]["esc50_animals"]["classes"]))

        return



    st.success(f"Input ready: `{input_source}`")



    effective_mode = mode_key

    router_info = None



    if mode_key == "auto":

        with st.spinner("Smart Auto-Router: probing urban and animal experts..."):

            urban_model, urban_classes, urban_name, animal_model, animal_classes, animal_name, _, device = get_router_experts()

            # Run lightweight inference through both experts, compare adjusted scores
            router_info = route_audio_domain(

                urban_model,

                urban_classes,

                urban_name,

                animal_model,

                animal_classes,

                animal_name,

                audio_bytes,

                device,

                cfg,

            )

            effective_mode = router_info["domain"]

            st.info(f"**Auto-routed to {effective_mode.title()} mode.** {router_info['reason']}")



            with st.expander("Router probe details"):

                c1, c2 = st.columns(2)

                with c1:

                    st.markdown("**Urban probe**")

                    st.write(

                        f"{router_info['urban_probe']['top_label']} "

                        f"({router_info['urban_probe']['top_confidence']:.1%})"

                    )

                with c2:

                    st.markdown("**Animal probe**")

                    st.write(

                        f"{router_info['animal_probe']['top_label']} "

                        f"({router_info['animal_probe']['top_confidence']:.1%})"

                    )



    inference_mode, inference_model = resolve_inference_context(effective_mode, model_name)

    if inference_mode == "animal" and model_name not in available_models_for_mode("animal", cfg):

        inference_model = cfg["deployment"]["animal"]["model_name"]

        st.warning(

            f"`{model_label(model_name, cfg)}` is only trained for urban sounds. "

            f"Using `{model_label(inference_model, cfg)}` for the animal expert."

        )



    try:

        model, class_names, deploy_cfg, chosen_name, checkpoint, cfg, device = get_model_bundle(

            inference_mode,

            inference_model,

        )

    except FileNotFoundError as exc:

        st.error(str(exc))

        st.stop()



    with st.spinner("Running preprocessing and inference..."):

        # Grad-CAM needs a second forward+backward pass — slightly slower
        if show_gradcam:

            result = predict_with_gradcam(

                model,

                class_names,

                chosen_name,

                audio_bytes,

                device=device,

                cfg=cfg,

                top_k=3,

                measure_latency=True,

            )

        else:

            from src.predict import predict_audio



            result = predict_audio(

                model,

                class_names,

                chosen_name,

                audio_bytes,

                device=device,

                cfg=cfg,

                top_k=3,

                measure_latency=True,

            )



    st.markdown("---")

    st.subheader("Model Performance")

    render_benchmark_metrics(chosen_name, result.get("inference_ms"), benchmarks)



    if chosen_name == "mobilenetv2" and benchmarks.get("resnet50") and benchmarks.get("custom_cnn"):

        with st.expander("Efficiency vs accuracy trade-off (urban test fold)"):

            trade_cols = st.columns(3)

            for idx, name in enumerate(["custom_cnn", "resnet50", "mobilenetv2"]):

                row = benchmarks[name]

                trade_cols[idx].markdown(

                    f"**{model_label(name, cfg)}**\n\n"

                    f"Accuracy: {row['test_accuracy']:.1%}\n\n"

                    f"Latency: {row['inference_ms_mean']} ms\n\n"

                    f"Size: {row['model_file_size_mb']} MB"

                )



    st.markdown("---")

    viz1, viz2, viz3 = st.columns(3)



    with viz1:

        st.subheader("Raw Waveform")

        st.pyplot(plot_waveform(result["waveform"], result["sample_rate"]))



    with viz2:

        st.subheader("Mel-Spectrogram Input")

        st.pyplot(plot_mel_spectrogram(result["mel_spectrogram"]))



    with viz3:

        st.subheader("Model Interpretability (Grad-CAM)")

        if show_gradcam and "gradcam_figure" in result:

            st.pyplot(result["gradcam_figure"])

            st.caption(result["gradcam_summary"]["method"])

        else:

            st.image(result["rgb_image"], caption="Model input (224×224 RGB Mel)", use_container_width=True)



    with st.expander("Generated RGB model input"):

        st.image(result["rgb_image"], caption="224×224 Mel-spectrogram RGB image", use_container_width=False)



    st.markdown("---")

    st.subheader("Prediction")

    render_predictions(result)



    st.caption(

        f"Engine: {model_label(chosen_name, cfg)} | Mode: {effective_mode.title()} | "

        f"Checkpoint: `{checkpoint.relative_to(project_path())}`"

    )





if __name__ == "__main__":

    main()

