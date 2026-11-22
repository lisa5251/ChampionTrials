# WaitWise — Streamlit + SQLite + local ML

Smart queue management: customers join from their phone, staff run counters, a TV board shows who is next,
and a small ML model predicts waiting times.

## Run in PyCharm (Windows PowerShell)

1. Open this folder in PyCharm and open the Terminal.
2. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install packages: `python -m pip install -r requirements.txt`
4. Start the app: `python -m streamlit run app.py`
5. Open http://localhost:8501

**Let customers' phones reach it (QR code):** run
`python -m streamlit run app.py --server.address 0.0.0.0`, find your PC's Wi-Fi address (`ipconfig`),
then in **Admin → Settings & Outbox** set the public address to e.g. `http://192.168.1.20:8501`.
Print the QR code shown there.

## Features

**Customers**
- Join by QR code or the portal; optional phone number and a **priority** box (seniors, pregnant, disability, with a baby)
- Live ticket: position, people ahead, **estimated wait range** (e.g. 8–14 min), auto-refresh every 8 s, "you're next" and "it's your turn" alerts
- Bookmarkable ticket link (`?ticket=WW-0007`), cancel, one review per ticket
- Live waiting time per service before you join; double sign-up protection
- Assistant chatbot that answers with your real position and wait

**Staff (Admin Panel, signed-in admin or manager)**
- Call next (priority first, then first-come; two counters can never take the same ticket)
- Complete, **Recall**, **No-show**, **Transfer** to another service
- Analytics: avg wait, avg service time, no-show rate, satisfaction, tickets by hour, counter performance, CSV export
- Settings: pause the queue, per-service capacity limit, announcement banner, QR code, notification outbox

**Manager (signed-in manager)** — edit any ticket (incl. priority/phone), add tickets, see feedback, and manage staff accounts. Nothing is ever deleted, only flagged.

**TV board** — now serving, up next with ★ priority, live waits, announcement, join QR.

## Files

- `app.py` — Streamlit pages
- `database.py` — SQLite storage (auto-migrates older `waitwise.db` files)
- `ai_engine.py` — wait-time model (service + hour of day, rush-hour aware, low/high range) and intent chatbot
- `notifier.py` — notifications; logs always, real SMS if Twilio is configured
- `requirements.txt`

`waitwise.db` is created automatically. `__pycache__` / `.pyc` files are generated and can be ignored.

## Who needs to log in?

| Role | Login? | Sees |
|---|---|---|
| Customer | **No** | Queue Display, Customer Portal |
| Admin | Yes (sidebar → Staff sign-in) | the above + Admin Panel |
| Manager | Yes | everything, plus Staff accounts (add users, reset passwords, disable) |

Customers never see a login box or the staff pages. Passwords are stored hashed (PBKDF2), 5 wrong tries locks sign-in for 60 s,
and a disabled account is signed out immediately.

First run creates two accounts: **admin / 1234** and **manager / 5678** (or the passwords in `ADMIN_PIN` / `MANAGER_PIN`
if set before the first start). Change them right away in Manager → Staff accounts; a warning shows until you do.

## Settings

| Variable | Purpose |
|---|---|
| `ADMIN_PIN`, `MANAGER_PIN` | Initial passwords of the `admin` / `manager` accounts (first run only) |
| `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM` | Turn on real SMS (also `pip install twilio`) |

## AI notes

- Wait model: Random Forest on service + hour of day. It starts with demo data and switches to your real completed
  tickets after 5 exist (tickets over 2 h are ignored). The wait = typical service time × people ahead ÷ open counters.
  With 12+ real tickets the Analytics tab shows the cross-validated error.
- Chatbot: TF-IDF + Logistic Regression over 12 intents. Local, not a general LLM.
- Estimates are approximate. For a real deployment add proper authentication and protect personal data.
