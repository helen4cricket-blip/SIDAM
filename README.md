# Building Damage Classifier

A Streamlit app that uses a trained ConvNeXt-Tiny model to classify an aerial
image as `Damaged Building` or `Undamaged Building`.

## Run locally

From the project directory in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Deploy for a public URL

This project is prepared for Streamlit Community Cloud. The app uses the model
checkpoint in `models/damage_convnext_tiny.pth`; the extracted training images
are not needed by the deployed app.

The model checkpoint is larger than GitHub's 100 MiB regular-file limit.
Git LFS is configured in `.gitattributes`, and Streamlit Community Cloud
supports Git LFS repositories.

1. Change the existing
   [SIDAM repository](https://github.com/helen4cricket-blip/SIDAM) to **Private**
   in GitHub under **Settings > General > Danger Zone > Change repository
   visibility**. Install Git and Git LFS. Do not upload the `data` or `.venv`
   folders; `.gitignore` excludes them.
2. In PowerShell, from this project directory, run:

   ```powershell
   git init
   git lfs install
   git lfs track "models/damage_convnext_tiny.pth"
   git add .gitignore .gitattributes README.md requirements.txt app.py train_damage_classifier.py models/damage_convnext_tiny.pth models/training_metrics.json
   git commit -m "Prepare building damage classifier for deployment"
   ```

3. Connect this project to the existing repository and push it:

   ```powershell
   git remote add origin https://github.com/helen4cricket-blip/SIDAM.git
   git branch -M main
   git push -u origin main
   ```

4. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/) with
   GitHub, create an app, and select this repository, the `main` branch, and
   `app.py` as the entrypoint. In Advanced settings, choose Python 3.15 to
   match the environment used to train the model, then deploy.
5. Because the repository is private, the app starts private. In the deployed
   app's **Settings > Sharing**, set **Who can view this app** to **This app is
   public and searchable**. Community Cloud will then provide a shareable
   `https://...streamlit.app` URL.

The deployed app classifies only damaged versus undamaged buildings. It does
not estimate damage severity or assess structural safety.
