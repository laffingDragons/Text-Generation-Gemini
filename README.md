# Story Generator

A Streamlit app that generates hooking stories based on user-provided titles using Google Generative AI (Gemini 1.5 Flash). Users can input their own Gemini API key directly in the app, making it easy to use without modifying configuration files.

## Features

- Enter a Gemini API key securely via the app interface.
- Input a story title to generate a unique, engaging story.
- Simple and intuitive UI built with Streamlit.
- Deployable on Streamlit Community Cloud.

## Setup

### Prerequisites

- Python 3.12 or higher.
- A Google Generative AI API key (obtain one from [Google AI Studio](https://aistudio.google.com/app/apikey)).
- Git installed for version control.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/streamlit-story-generator.git
   cd streamlit-story-generator
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app locally:
   ```bash
   streamlit run app.py
   ```

4. Open the provided URL (e.g., `http://localhost:8501`) in your browser.

### Obtaining a Gemini API Key

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Sign in with your Google account.
3. Create a new API key or copy an existing one.
4. Paste the API key into the app's input field when prompted.

## Deployment

Deploy the app on [Streamlit Community Cloud](https://streamlit.io/cloud):

1. Push the repository to GitHub (see instructions below).
2. Sign in to Streamlit Community Cloud with your GitHub account.
3. Create a new app, select your repository, and set `app.py` as the main file.
4. Deploy the app and access it via the provided URL.

### Pushing to GitHub

1. Initialize a Git repository:
   ```bash
   git init
   ```

2. Add files:
   ```bash
   git add app.py requirements.txt .gitignore README.md
   ```

3. Commit changes:
   ```bash
   git commit -m "Initial commit of Streamlit story generator"
   ```

4. Link to a GitHub repository:
   ```bash
   git remote add origin https://github.com/<your-username>/streamlit-story-generator.git
   git branch -M main
   git push -u origin main
   ```

## Problem Faced and Solution

**Problem**: During development, I was unable to install the `google-generativeai` library using `uvicorn` or the `uv` package manager, which caused errors when trying to run the app.

**Solution**: The issue was resolved by installing `google-generativeai` using the standard `pip` command:
```bash
pip install google-generativeai
```
This ensured the library was correctly installed and compatible with the Python environment. For deployment, all dependencies are listed in `requirements.txt` to avoid similar issues.

## Dependencies

Listed in `requirements.txt`:
- `streamlit>=1.38.0`
- `google-generativeai>=0.8.5`

## Contributing

Feel free to submit issues or pull requests to improve the app!

## License

MIT License