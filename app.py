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
        "Upload up to 10 images at once to classify each as **Damaged Building** or "
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

    uploaded_files = st.file_uploader(
        "Choose up to 10 images",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        accept_multiple_files=True,
        help="Select up to 10 JPG, PNG, or TIFF images in one upload.",
    )
    if not uploaded_files:
        st.info("Choose one or more images to see the model's predictions.")
        return

    if len(uploaded_files) > 10:
        st.error(
            f"You selected {len(uploaded_files)} files. Please upload no more than 10 at once."
        )
        return

    valid_uploads: list[tuple[str, Image.Image]] = []
    for uploaded_file in uploaded_files:
        try:
            image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
        except (UnidentifiedImageError, OSError) as error:
            st.error(f"Could not read {uploaded_file.name}: {error}")
            continue
        valid_uploads.append((uploaded_file.name, image))

    if not valid_uploads:
        return

    try:
        model, class_names, device = load_model(str(CHECKPOINT_PATH))
        transform = ConvNeXt_Tiny_Weights.DEFAULT.transforms()
        with torch.inference_mode():
            images = torch.stack(
                [transform(image) for _, image in valid_uploads]
            ).to(device)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1).cpu().tolist()
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        st.error(f"Could not load the model or classify the images: {error}")
        return

    for (filename, image), image_probabilities in zip(valid_uploads, probabilities):
        predicted_index = max(
            range(len(image_probabilities)), key=image_probabilities.__getitem__
        )
        predicted_class = class_names[predicted_index]
        confidence = image_probabilities[predicted_index]

        with st.container(border=True):
            image_column, result_column = st.columns([1, 1])
            with image_column:
                st.image(image, caption=filename, width=500)
            with result_column:
                st.subheader("Prediction")
                st.success(f"{predicted_class} — {confidence:.1%} confidence")
                st.write("Class probabilities")
                for class_name, probability in zip(class_names, image_probabilities):
                    st.progress(probability, text=f"{class_name}: {probability:.1%}")


if __name__ == "__main__":
    main()
