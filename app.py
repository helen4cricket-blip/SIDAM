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
APP_TITLE = "Seismic Infrastructure Damage Assessment Model"
CHECKPOINT_PATH = WORKSPACE / "models" / "damage_convnext_tiny.pth"
METRICS_PATH = WORKSPACE / "models" / "training_metrics.json"

TEAM_MEMBERS: tuple[dict[str, str], ...] = (
    {
        "name": "Helen Pokkayil Eldho",
        "initials": "HE",
        "role": "SIDAM developer · Doctoral researcher",
        "affiliation": "Massey University",
        "bio": (
            "Helen developed SIDAM and is pursuing doctoral research on "
            "AI-integrated construction project management."
        ),
        "profile_url": "",
        "photo_url": str(WORKSPACE / "assets" / "helen-pokkayil-eldho.png"),
    },
    {
        "name": "Dr Chandana Siriwardana",
        "initials": "CS",
        "role": "Doctoral supervisor · Senior Lecturer",
        "affiliation": "Massey University, School of Built Environment",
        "bio": (
            "His research focuses on construction management, disaster "
            "resilience, sustainable construction, and climate adaptation."
        ),
        "profile_url": (
            "https://www.massey.ac.nz/massey/expertise/college-staff-lists/"
            "college-of-sciences/all-staff/all-staff_home.cfm?stref=116722"
        ),
        "photo_url": (
            "https://www.massey.ac.nz/massey/fms/expertise/user_images/"
            "profile_227611_319B0578-2B60-4425-8F3C017E2D081643.jpg"
        ),
    },
    {
        "name": "Dr Harold Aquino",
        "initials": "HA",
        "role": "Doctoral co-supervisor · Lecturer",
        "affiliation": "Massey University, School of Built Environment",
        "bio": (
            "His work brings together resilience engineering, infrastructure "
            "planning, construction management, data analytics, and AI."
        ),
        "profile_url": (
            "https://www.massey.ac.nz/massey/expertise/college-staff-lists/"
            "college-of-sciences/school-of-built-environment-staff/"
            "all-staff_home.cfm?stref=834922"
        ),
        "photo_url": (
            "https://www.massey.ac.nz/massey/fms/expertise/user_images/"
            "profile_229438_3A3E0A31-8E5A-4B14-B977D800B51AEB7F.jpg"
        ),
    },
    {
        "name": "Dr Surangika Ranathunga",
        "initials": "SR",
        "role": "Doctoral supervisor · Senior Lecturer",
        "affiliation": (
            "Massey University, School of Mathematical and Computational Sciences"
        ),
        "bio": (
            "Her research includes natural language processing, computational "
            "linguistics, large language models, and machine learning."
        ),
        "profile_url": (
            "https://www.massey.ac.nz/massey/expertise/college-staff-lists/"
            "college-of-sciences/school-of-mathematical-and-computational-"
            "sciences-staff.cfm?stref=319722"
        ),
        "photo_url": (
            "https://www.massey.ac.nz/massey/fms/expertise/user_images/"
            "profile_227913_DD2144CE-2E00-44DE-ACA8BB5BD57D2048.jpg"
        ),
    },
    {
        "name": "Dr Tariq Maqsood",
        "initials": "TM",
        "role": "Academic supervisor · Associate Professor",
        "affiliation": "RMIT University, School of Engineering, Australia",
        "bio": (
            "His research focuses on natural-hazard vulnerability and risk "
            "assessment of the built environment and disaster risk reduction."
        ),
        "profile_url": "https://www.rmit.edu.au/profiles/m/tariq-maqsood",
        "photo_url": (
            "https://www.rmit.edu.au/assets/au/en/profile-images/"
            "m/tariq-maqsood.jpeg"
        ),
    },
)


APP_STYLES = """
<style>
:root {
    --plum: #5B214E;
    --magenta: #B82F6C;
    --terracotta: #C56A4A;
    --ivory: #F7F1E3;
    --charcoal: #302A2D;
}

.stApp,
[data-testid="stAppViewContainer"] {
    background-color: var(--ivory);
    color: var(--charcoal);
}

[data-testid="stHeader"] {
    background-color: transparent;
}

h1, h2, h3 {
    color: var(--plum) !important;
}

[data-testid="stMain"] h1 {
    font-size: clamp(1rem, 2.5vw, 2.25rem) !important;
    line-height: 1.2;
    white-space: nowrap;
}

.sidam-acronym {
    color: var(--magenta);
    font-size: 1.3rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    line-height: 1.2;
    margin-bottom: 1rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--plum), var(--charcoal));
}

[data-testid="stSidebar"] [data-testid="stElementContainer"]:has([data-testid="stRadio"]) {
    width: 100%;
}

[data-testid="stSidebar"] [data-testid="stRadio"],
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] {
    align-items: stretch;
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
    width: 100%;
}

[data-testid="stSidebar"] [data-testid="stRadioOption"] {
    background-color: rgba(247, 241, 227, 0.08);
    border: 1px solid rgba(247, 241, 227, 0.45);
    border-radius: 0.4rem;
    box-sizing: border-box;
    padding: 0.7rem 0.9rem;
    transition: background-color 140ms ease, border-color 140ms ease;
    width: 100%;
}

[data-testid="stSidebar"] [data-testid="stRadioOption"]:hover {
    background-color: rgba(184, 47, 108, 0.35);
    border-color: var(--terracotta);
}

[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] {
    background-color: var(--magenta);
    border-color: var(--ivory);
}

[data-testid="stSidebar"] * {
    color: var(--ivory);
}

[data-testid="stMetricValue"] {
    color: var(--plum);
}

[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    color: var(--ivory);
}

[data-testid="stFileUploader"] section {
    background-color: #FFF9F0;
    border-color: var(--terracotta);
}

[data-testid="stFileUploader"] button {
    background-color: var(--magenta) !important;
    border-color: var(--magenta) !important;
    color: var(--ivory) !important;
}

[data-testid="stFileUploader"] button:hover {
    background-color: var(--plum) !important;
    border-color: var(--plum) !important;
    color: var(--ivory) !important;
}

.stButton button[kind="primary"] {
    background-color: var(--magenta);
    border-color: var(--magenta);
    color: var(--ivory);
}

[data-testid="stProgressBar"] > div > div {
    background-color: var(--magenta);
}

[data-testid="stAlert"] {
    background-color: #F0E2D6 !important;
    border-left: 0.35rem solid var(--terracotta) !important;
}

[data-testid="stAlert"] * {
    color: var(--charcoal) !important;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFCF6;
    border-color: #D9C7BD;
}

.team-avatar {
    align-items: center;
    aspect-ratio: 1;
    background: linear-gradient(135deg, var(--magenta), var(--plum));
    border: 3px solid var(--terracotta);
    border-radius: 50%;
    color: var(--ivory);
    display: flex;
    font-size: 2rem;
    font-weight: 700;
    justify-content: center;
    max-width: 150px;
}
</style>
"""


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


def render_page_title() -> None:
    st.title(APP_TITLE)
    st.markdown('<div class="sidam-acronym">SIDAM</div>', unsafe_allow_html=True)


def render_team_page() -> None:
    render_page_title()
    st.subheader("The SIDAM team")
    st.write(
        "Meet the developer and academic supervisors behind this building "
        "damage classifier."
    )

    for member in TEAM_MEMBERS:
        with st.container(border=True):
            photo_column, details_column = st.columns([1, 4])
            with photo_column:
                if member["photo_url"]:
                    st.image(
                        member["photo_url"],
                        caption=member["name"],
                        width=150,
                    )
                else:
                    st.markdown(
                        f'<div class="team-avatar" role="img" '
                        f'aria-label="Initials {member["initials"]}">'
                        f'{member["initials"]}</div>',
                        unsafe_allow_html=True,
                    )
            with details_column:
                st.subheader(member["name"])
                st.caption(f'{member["role"]} | {member["affiliation"]}')
                st.write(member["bio"])
                if member["profile_url"]:
                    st.markdown(f'[View official profile]({member["profile_url"]})')


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🏚️",
        layout="wide",
    )
    st.markdown(APP_STYLES, unsafe_allow_html=True)

    selected_page = st.sidebar.radio(
        "Navigation",
        ("Model", "TEAM"),
        label_visibility="collapsed",
    )
    st.sidebar.divider()
    if selected_page == "TEAM":
        render_team_page()
        return

    render_page_title()
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
