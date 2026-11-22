# WaitWise — Streamlit + SQLite + local ML

## Run in PyCharm (Windows PowerShell)

1. Open this folder in PyCharm.
2. Open Terminal in the project folder.
3. Create and activate a virtual environment (recommended):

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

4. Install packages:

   ```powershell
   python -m pip install -r requirements.txt
   ```

5. Start the app:

   ```powershell
   python -m streamlit run app.py
   ```

6. Open the local URL shown in the terminal, usually http://localhost:8501.

## Files

- `app.py`: Streamlit customer, staff, and admin dashboards.
- `database.py`: SQLite ticket and feedback storage.
- `ai_engine.py`: waiting-time regression model and local chatbot intent classifier.
- `requirements.txt`: Python dependencies.

The database file `waitwise.db` is created automatically on first run.

## AI notes

- Wait prediction uses a Random Forest regressor. It starts with synthetic demo examples, then can train on recorded completed-ticket durations after at least five completed tickets exist.
- The chatbot uses TF-IDF + Logistic Regression to recognize common queue-related questions. It is a small local ML assistant, not a general-purpose LLM.
- Predictions are approximate. A real deployment should collect more history, evaluate prediction error, add authentication, and protect personal data.
