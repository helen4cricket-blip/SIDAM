from __future__ import annotations

import io
import json
from pathlib import Path

import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError
from torch import nn
from torchvision.models import ConvNeXt_Tiny_Weights

from train_damage_classifier import make_model


WORKSPACE = Path(__file__).resolve().parent
CHECKPOINT_PATH = WORKSPACE / "models" / "damage_convnext_tiny.pth"
METRICS_PATH = WORKSPACE / "models" / "training_metrics.json"


@st.cache_resource(show_spinner="Loading the trained classifier...")
def load_model(checkpoint_path: str) -> tuple[nn.Module, list[str], torch.device]:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Trained model checkpoint not found: {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    class_names = checkpoint["class_names"]
    if not isinstance(class_names, list) or len(class_names) != 2:
        raise ValueError("The checkpoint must contain the two damage class names.")

    model = make_model(len(class_names), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names, device


def main() -> None:
    st.set_page_config(
        page_title="Building Damage Classifier",
        page_icon="🏚️",
        layout="wide",
    )
    st.title("Building Damage Classifier")
    st.write(
        "Upload an aerial image to classify it as **Damaged Building** or "
        "**Undamaged Building**."
    )
    st.warning(
        "This model predicts only these two categories. It does not estimate damage "
        "severity or assess whether a building is safe."
    )

    if not CHECKPOINT_PATH.is_file():
        st.error(f"Model checkpoint not found: {CHECKPOINT_PATH}")
        st.stop()

    if METRICS_PATH.is_file():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        test_accuracy = metrics.get("test_accuracy")
        if isinstance(test_accuracy, (float, int)):
            st.sidebar.metric(
                "Held-out test accuracy",
                f"{test_accuracy:.1%}",
                help="Measured on a held-out split from this same dataset.",
            )

    st.sidebar.header("About this model")
    st.sidebar.write("Architecture: ConvNeXt-Tiny")
    st.sidebar.write("Runs locally in this workspace.")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        help="Supported formats: JPG, PNG, and TIFF.",
    )
    if uploaded_file is None:
        st.info("Choose an image to see the model's prediction.")
        return

    try:
        image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        st.error(f"Could not read the uploaded image: {error}")
        return

    try:
        model, class_names, device = load_model(str(CHECKPOINT_PATH))
        transform = ConvNeXt_Tiny_Weights.DEFAULT.transforms()
        with torch.inference_mode():
            logits = model(transform(image).unsqueeze(0).to(device))
            probabilities = torch.softmax(logits, dim=1)[0].cpu().tolist()
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        st.error(f"Could not load the model or classify this image: {error}")
        return

    predicted_index = max(range(len(probabilities)), key=probabilities.__getitem__)
    predicted_class = class_names[predicted_index]
    confidence = probabilities[predicted_index]

    image_column, result_column = st.columns([1, 1])
    with image_column:
        st.image(image, caption=uploaded_file.name, width=500)
    with result_column:
        st.subheader("Prediction")
        st.success(f"{predicted_class} — {confidence:.1%} confidence")
        st.write("Class probabilities")
        for class_name, probability in zip(class_names, probabilities):
            st.progress(probability, text=f"{class_name}: {probability:.1%}")


if __name__ == "__main__":
    main()
