import html
import inspect
import os

import streamlit as st
import pandas as pd
from datetime import datetime
import database as db
from ai_engine import (
    SERVICES,
    train_wait_model,
    predict_wait,
    build_chatbot,
    chatbot_reply
)

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="WaitWise | Intelligent Queue Platform",
    page_icon="🎟️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------
# HTML RENDERING FIX
# Markdown turns any line indented 4+ spaces (after a blank line) into a code
# block, which made our indented HTML show up as raw code. For every
# st.markdown(..., unsafe_allow_html=True) call we strip the indentation and
# blank lines first, so the HTML always renders as HTML.
# -----------------------------------------------------

_st_markdown = st.markdown


def _html_safe_markdown(body, *args, **kwargs):
    if kwargs.get("unsafe_allow_html"):
        body = "\n".join(
            line.strip() for line in str(body).splitlines() if line.strip()
        )
    return _st_markdown(body, *args, **kwargs)


st.markdown = _html_safe_markdown


db.init_db()


def _stretch(fn):
    """Streamlit >= 1.50 uses width="stretch"; older versions use use_container_width=True."""
    return {"width": "stretch"} if "width" in inspect.signature(fn).parameters else {"use_container_width": True}


STRETCH_BTN = _stretch(st.button)
STRETCH_DF = _stretch(st.dataframe)
STRETCH_ED = _stretch(st.data_editor)


# =====================================================
# LOAD AI MODELS
# =====================================================

@st.cache_resource
def load_ai():
    durations = db.completed_durations()
    wait_model, using_real_history = train_wait_model(durations)
    chatbot = build_chatbot()

    return wait_model, using_real_history, chatbot


wait_model, using_real_history, chatbot = load_ai()


# =====================================================
# CUSTOM DESIGN / CSS
# =====================================================

# -----------------------------------------------------
# THEMES (pick one in the sidebar; add your own by copying a dict)
# -----------------------------------------------------

THEMES = {
    "Ocean": dict(ink="#0B2E33", muted="#4F6F73", line="#CDEBE6", a="#0D9488", b="#0EA5E9", c="#F59E0B",
                  d="#065F5B", e="#0284C7", side="#062B2E", bg="#F0FBF9", ta="#CCFBF1", tb="#E0F2FE",
                  tc="#FEF3C7", soft="#D5F5F0"),
    "Aurora": dict(ink="#1E1B4B", muted="#6B6890", line="#E4DFFB", a="#7C3AED", b="#EC4899", c="#06B6D4",
                   d="#3B2FBF", e="#BE3FA6", side="#130F33", bg="#F7F5FF", ta="#F3E8FF", tb="#FCE7F3",
                   tc="#CFFAFE", soft="#EDE9FE"),
    "Sunset": dict(ink="#2D1B2E", muted="#7A6272", line="#F6DDD2", a="#EA580C", b="#DB2777", c="#FACC15",
                   d="#9A3412", e="#BE185D", side="#2A1226", bg="#FFF7F2", ta="#FFEDD5", tb="#FCE7F3",
                   tc="#FEF9C3", soft="#FFE4D6"),
}

THEME = THEMES.get(st.session_state.get("theme"), THEMES["Ocean"])

CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;700&family=Outfit:wght@600;700;800&display=swap');
:root { --ink:@ink@; --muted:@muted@; --line:@line@; --a:@a@; --b:@b@; --c:@c@;
  --grad:linear-gradient(110deg,@d@ 0%,@a@ 55%,@b@ 100%); }
html, body, [class*="css"] { font-family: 'Figtree', sans-serif; }
.stApp { color: var(--ink);
  background: radial-gradient(900px 500px at 5% -5%, @ta@ 0%, transparent 60%),
              radial-gradient(800px 500px at 100% 0%, @tb@ 0%, transparent 55%),
              radial-gradient(700px 500px at 60% 100%, @tc@ 0%, transparent 60%), @bg@; }
.block-container { padding-top:1.7rem; padding-bottom:3rem; max-width:1440px; }

[data-testid="stSidebar"] { background: radial-gradient(500px 300px at 0% 0%, @a@66, transparent 60%),
  radial-gradient(400px 300px at 100% 100%, @b@44, transparent 60%), @side@; border-right:1px solid @a@44; }
[data-testid="stSidebar"] * { color:@soft@ !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { background:#ffffff0d; border:1px solid #ffffff1c;
  border-radius:14px; padding:10px 14px; margin-bottom:8px; transition:all .15s ease; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background:#ffffff1c; transform:translateX(3px); }
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) { background:var(--grad);
  border-color:transparent; box-shadow:0 8px 22px @a@66; }

.brand-mark { width:46px; height:46px; border-radius:15px; background:var(--grad); display:flex; align-items:center;
  justify-content:center; color:white; font-family:Outfit,sans-serif; font-size:23px; font-weight:800;
  box-shadow:0 10px 26px @a@88; }
.brand-row { display:flex; align-items:center; gap:12px; margin:5px 0 24px; }
.brand-name { color:white; font-family:Outfit,sans-serif; font-size:22px; font-weight:800; }
.brand-sub { font-size:11px; letter-spacing:1.4px; text-transform:uppercase; }

.hero { background: radial-gradient(420px 260px at 90% 10%, @c@88, transparent 65%),
  radial-gradient(380px 260px at 55% 125%, @b@88, transparent 65%),
  linear-gradient(115deg, @d@ 0%, @a@ 58%, @e@ 100%);
  border-radius:28px; padding:34px 38px; color:white; position:relative; overflow:hidden;
  box-shadow:0 24px 60px @a@40; margin-bottom:26px; }
.hero:after { content:'◈'; position:absolute; right:48px; top:-45px; font-size:230px; color:#ffffff14; transform:rotate(-15deg); }
.hero h1 { font-family:Outfit,sans-serif; color:white; font-size:38px; font-weight:800; line-height:1.12; margin:0 0 10px; }
.hero p { color:@soft@; font-size:15px; margin:0; max-width:650px; }
.hero-chip { display:inline-block; background:#ffffff24; border:1px solid #ffffff46; backdrop-filter:blur(6px);
  color:white; border-radius:30px; padding:6px 13px; font-size:11px; font-weight:700; letter-spacing:.6px; margin-bottom:16px; }

.metric-card { position:relative; overflow:hidden; background:#ffffffd9; backdrop-filter:blur(8px); border:1px solid #fff;
  border-radius:20px; padding:19px 20px; min-height:118px; box-shadow:0 10px 30px @a@18;
  transition:transform .18s ease, box-shadow .18s ease; }
.metric-card:before { content:''; position:absolute; left:0; right:0; top:0; height:4px; background:var(--grad); }
.metric-card:hover { transform:translateY(-3px); box-shadow:0 16px 38px @a@2a; }
.metric-label { color:var(--muted); font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:1px; }
.metric-value { font-family:Outfit,sans-serif; font-size:34px; font-weight:800; margin-top:8px;
  background:var(--grad); -webkit-background-clip:text; background-clip:text; color:transparent; }
.metric-foot { color:var(--muted); font-size:12px; margin-top:2px; }

.panel, div[class*="st-key-panel"] { background:#ffffffe0; backdrop-filter:blur(8px); border:1px solid #fff;
  border-radius:22px; padding:23px 24px; margin-bottom:16px; box-shadow:0 12px 34px @a@14; }
.panel-title { color:var(--ink); font-family:Outfit,sans-serif; font-weight:700; font-size:17px; margin-bottom:4px; }
.panel-caption { color:var(--muted); font-size:12.5px; margin-bottom:14px; }
.section-title { font-family:Outfit,sans-serif; font-size:21px; font-weight:800; color:var(--ink); margin:10px 0 3px; }
.section-sub { color:var(--muted); font-size:13px; margin-bottom:15px; }

.ticket-code { font-family:Outfit,sans-serif; font-size:34px; font-weight:800; letter-spacing:1px;
  background:var(--grad); -webkit-background-clip:text; background-clip:text; color:transparent; }
.status-pill { display:inline-block; border-radius:30px; padding:6px 12px; font-size:11px; font-weight:800;
  letter-spacing:.5px; background:linear-gradient(100deg,@ta@,@tb@); color:var(--a); }
.ai-banner { background:linear-gradient(110deg,@ta@,@tb@ 55%,@tc@); border:1px solid @line@; border-radius:18px;
  padding:16px 18px; color:var(--ink); margin:12px 0; }
.ai-dot { display:inline-block; width:9px; height:9px; border-radius:50%; background:var(--c); margin-right:7px;
  box-shadow:0 0 0 4px @c@44; }

div.stButton > button, div.stFormSubmitButton > button { border-radius:14px; border:0; background:var(--grad);
  background-size:140% 100%; color:white; font-weight:700; padding:10px 18px; box-shadow:0 10px 22px @a@44;
  transition:transform .15s ease, box-shadow .15s ease, background-position .3s ease; }
div.stButton > button:hover, div.stFormSubmitButton > button:hover { background-position:100% 0; color:white;
  transform:translateY(-2px); box-shadow:0 14px 28px @a@66; }
[data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea, [data-testid="stSelectbox"] div { border-radius:12px; }
[data-testid="stTextInput"] input:focus, [data-testid="stTextArea"] textarea:focus {
  border-color:var(--a) !important; box-shadow:0 0 0 3px @a@33 !important; }
button[data-baseweb="tab"][aria-selected="true"] { color:var(--a) !important; font-weight:700; }
[data-baseweb="tab-highlight"] { background:var(--grad) !important; height:3px !important; }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:16px; overflow:hidden; }
div[data-testid="stMetric"] { background:#ffffffd9; border:1px solid #fff; padding:15px 17px; border-radius:18px;
  box-shadow:0 10px 28px @a@14; }
hr { border-color:var(--line); } .small-note { color:var(--muted); font-size:11px; } footer { visibility:hidden; }

.tv-card { background:var(--grad); color:white; border-radius:24px; padding:24px; text-align:center; box-shadow:0 18px 44px @a@44; }
.tv-card .metric-label { color:#ffffffcc; }
.tv-num { font-family:Outfit,sans-serif; font-size:68px; font-weight:800; line-height:1.1; margin:6px 0; }
.tv-go { font-size:24px; font-weight:700; }
.tv-row { display:flex; justify-content:space-between; gap:12px; background:#ffffffe0; border-radius:14px;
  padding:12px 16px; margin-bottom:8px; font-weight:600; border-left:5px solid var(--a); }
.tv-code { font-family:Outfit,sans-serif; color:var(--a); font-weight:800; }
</style>
"""


def build_css():
    import re
    return re.sub(r"@(\w+)@", lambda m: THEME[m.group(1)], CSS)


st.markdown(build_css(), unsafe_allow_html=True)


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def queue_ahead(ticket):

    rows = db.list_tickets(
        service=ticket["service"],
        statuses=["Waiting", "Serving"]
    )

    ahead = 0

    for row in rows:

        if row["ticket_code"] == ticket["ticket_code"]:
            break

        if row["status"] in ("Waiting", "Serving"):
            ahead += 1

    return ahead


def metric_card(label, value, foot, icon):

    st.markdown(f"""
    <div class="metric-card">

        <div style="display:flex;justify-content:space-between;align-items:center">

            <div class="metric-label">{label}</div>

            <div style="font-size:19px">{icon}</div>

        </div>

        <div class="metric-value">{value}</div>

        <div class="metric-foot">{foot}</div>

    </div>
    """, unsafe_allow_html=True)


def section(title, subtitle=None):

    st.markdown(
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True
    )

    if subtitle:

        st.markdown(
            f'<div class="section-sub">{subtitle}</div>',
            unsafe_allow_html=True
        )


def ticket_details(code):

    ticket = db.get_ticket(code)

    if not ticket:

        st.error("We couldn't find that ticket. Check the code and try again.")

        return None

    if ticket["is_deleted"]:

        st.warning("This ticket was removed by staff. Please ask at the service desk.")

        return None

    ahead = queue_ahead(ticket) if ticket["status"] == "Waiting" else 0

    estimate = (
        predict_wait(wait_model, ticket["service"], ahead)
        if ticket["status"] == "Waiting"
        else 0
    )

    with st.container(key="panel_1"):


        left, right = st.columns([2, 1])

        with left:

            st.caption("YOUR QUEUE TICKET")

            st.markdown(
                f'<div class="ticket-code">{ticket["ticket_code"]}</div>',
                unsafe_allow_html=True
            )

            st.write(f"**{ticket['service']}** · {ticket['customer_name']}")

        with right:

            st.caption("CURRENT STATUS")

            st.markdown(
                f'<span class="status-pill">{ticket["status"].upper()}</span>',
                unsafe_allow_html=True
            )

        if ticket["status"] == "Waiting":

            a, b = st.columns(2)

            a.metric("People ahead", ahead)

            b.metric("AI wait estimate", f"{estimate} min")

            st.markdown("""
            <div class="ai-banner">
                <span class="ai-dot"></span>
                <b>AI prediction</b><br>
                <span style="font-size:12px">
                Your estimated wait is based on the selected service
                and the current queue. Actual times may vary.
                </span>
            </div>
            """, unsafe_allow_html=True)

        elif ticket["status"] == "Serving":

            st.success(f"It's your turn. Please go to {ticket['counter'] or 'the service desk'}.")

        elif ticket["status"] == "Completed":

            st.success("Your service is complete. Thank you for choosing WaitWise.")

        else:

            st.warning("This ticket has been cancelled.")


    return ticket


COUNTERS = ["Counter 1", "Counter 2", "Counter 3", "Counter 4"]


def hero(chip, title, text):

    st.markdown(f"""
    <div class="hero">
        <div class="hero-chip">{chip}</div>
        <h1>{title}</h1>
        <p>{text}</p>
    </div>
    """, unsafe_allow_html=True)


def require_pin(role, env_name, default):
    """Simple PIN gate. Change the PIN with an environment variable (ADMIN_PIN / MANAGER_PIN)."""

    key = f"auth_{role}"

    if st.session_state.get(key):
        return True

    hero(role.upper(), f"{role.capitalize()} sign-in", "Enter your PIN to continue.")

    pin = st.text_input(f"{role.capitalize()} PIN", type="password", key=f"pin_{role}")

    if st.button("Unlock"):

        if pin == os.getenv(env_name, default):
            st.session_state[key] = True
            st.rerun()
        else:
            st.error("Wrong PIN. Try again.")

    return False


@st.fragment(run_every=5)
def tv_board():
    """Public TV screen. Refreshes every 5 seconds. Shows ticket numbers only (no customer names)."""

    serving = db.list_tickets(statuses=["Serving"])
    waiting = db.list_tickets(statuses=["Waiting"])

    hero("NOW SERVING", "Please go to your counter when your number appears",
         "This screen updates automatically.")

    if serving:

        cols = st.columns(min(len(serving), 4))

        for col, t in zip(cols, serving[:4]):

            with col:

                st.markdown(f"""
                <div class="tv-card">
                    <div class="metric-label">{html.escape(t["service"])}</div>
                    <div class="tv-num">{html.escape(t["ticket_code"])}</div>
                    <div class="tv-go">➜ {html.escape(t["counter"] or "Service desk")}</div>
                </div>
                """, unsafe_allow_html=True)

    else:

        st.info("No one is being served right now.")

    st.write("")

    left, right = st.columns([1.3, 1], gap="large")

    with left:

        st.markdown("**Up next**")

        if waiting:

            for t in waiting[:6]:

                est = predict_wait(wait_model, t["service"], queue_ahead(t))

                st.markdown(f"""
                <div class="tv-row">
                    <span class="tv-code">{html.escape(t["ticket_code"])}</span>
                    <span>{html.escape(t["service"])}</span>
                    <span>~{est} min</span>
                </div>
                """, unsafe_allow_html=True)

        else:

            st.caption("The queue is empty.")

    with right:

        st.markdown("**Waiting by service**")

        for service in SERVICES:

            n = sum(1 for t in waiting if t["service"] == service)
            metric_card(service, n, "people waiting", "⏳")


# =====================================================
# SIDEBAR NAVIGATION
# =====================================================

with st.sidebar:

    st.markdown("""
    <div class="brand-row">
        <div class="brand-mark">W</div>

        <div>
            <div class="brand-name">WaitWise</div>
            <div class="brand-sub">Intelligent queue platform</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='color:#91a2c2;font-size:10px;
    font-weight:800;letter-spacing:1.5px;margin:8px 0 10px'>
        WORKSPACE
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "Queue Display",
            "Customer Portal",
            "Admin Panel",
            "Manager Database"
        ],
        label_visibility="collapsed"
    )

    st.markdown("""
    <div style='color:#91a2c2;font-size:10px;font-weight:800;letter-spacing:1.5px;margin:18px 0 10px'>
        THEME
    </div>
    """, unsafe_allow_html=True)

    st.radio("Theme", list(THEMES), key="theme", label_visibility="collapsed")

    st.markdown("""
    <div style='height:15px'></div>
    <div style='border-top:1px solid #2b3853;
    padding-top:15px;color:#91a2c2;
    font-size:10px;font-weight:800;letter-spacing:1.4px'>
        PLATFORM STATUS
    </div>

    <div style='margin-top:12px;color:#d8e2f5;font-size:12px'>
        🟢 Queue engine online
    </div>

    <div style='margin-top:8px;color:#d8e2f5;font-size:12px'>
        🧠 AI assistant ready
    </div>

    <div style='margin-top:8px;color:#d8e2f5;font-size:12px'>
        🗄️ SQLite connected
    </div>

    <div style='height:28px'></div>

    <div style='color:#8292b0;font-size:10px'>
        WAITWISE · HACKATHON EDITION<br>
        Smart service. Less waiting.
    </div>
    """, unsafe_allow_html=True)


# =====================================================
# PAGE 1: OVERVIEW
# =====================================================

# =====================================================
# PANEL 1: QUEUE DISPLAY (public TV screen)
# =====================================================

if page == "Queue Display":

    tv_board()


# =====================================================
# PAGE 2: CUSTOMER PORTAL
# =====================================================

elif page == "Customer Portal":

    st.markdown("""
    <div class="hero">

        <div class="hero-chip">CUSTOMER EXPERIENCE</div>

        <h1>Your time matters.</h1>

        <p>
        Join a queue digitally, keep your ticket close,
        and get a smarter estimate of when you'll be served.
        </p>

    </div>
    """, unsafe_allow_html=True)

    create_col, track_col = st.columns(
        [1, 1.15],
        gap="large"
    )

    with create_col:

        with st.container(key="panel_3"):

            st.markdown("""

                <div class="panel-title">Create your ticket</div>

                <div class="panel-caption">
                Choose a service and join the virtual queue.
                </div>
            """, unsafe_allow_html=True)

            with st.form("new_ticket_form", clear_on_submit=True):

                name = st.text_input(
                    "Full name",
                    placeholder="Enter your name"
                )

                service = st.selectbox(
                    "Service category",
                    SERVICES
                )

                submitted = st.form_submit_button(
                    "Create my ticket  →",
                    **STRETCH_BTN
                )

            if submitted:

                if not name.strip():

                    st.error(
                        "Please enter your name before continuing."
                    )

                else:

                    code = db.create_ticket(name, service)

                    st.session_state["last_ticket"] = code
                    st.session_state["lookup_ticket"] = code

                    st.success(f"Ticket created: {code}")


        with st.container(key="panel_4"):

            st.markdown("""

                <div class="panel-title">Available services</div>

                <div class="panel-caption">
                Select the service that best matches your needs.
                </div>
            """, unsafe_allow_html=True)

            services_info = [
                (
                    "General Inquiry",
                    "Information and general assistance"
                ),
                (
                    "Payments",
                    "Billing and payment support"
                ),
                (
                    "Technical Support",
                    "Help with technical issues"
                )
            ]

            for item, detail in services_info:

                st.markdown(f"""
                <div style="padding:10px 0;border-bottom:1px solid #eef1f6">

                    <div style="font-size:13px;font-weight:700;color:#263653">
                        {item}
                    </div>

                    <div style="font-size:11px;color:#8b98ad">
                        {detail}
                    </div>

                </div>
                """, unsafe_allow_html=True)


    with track_col:

        with st.container(key="panel_5"):

            st.markdown("""

                <div class="panel-title">Track your ticket</div>

                <div class="panel-caption">
                Enter your ticket code to see the latest queue status.
                </div>
            """, unsafe_allow_html=True)

            default_code = st.session_state.get("last_ticket", "")

            code = st.text_input(
                "Ticket reference",
                value=default_code,
                placeholder="WW-0001",
                label_visibility="collapsed"
            )

            if st.button(
                "Find my ticket",
                **STRETCH_BTN
            ):

                st.session_state["lookup_ticket"] = code

            current_ticket = None

            if st.session_state.get("lookup_ticket"):

                current_ticket = ticket_details(
                    st.session_state["lookup_ticket"]
                )

            else:

                st.markdown("""
                <div style="padding:28px 12px;text-align:center;color:#95a1b4">

                    <div style="font-size:32px">🎫</div>

                    <b style="color:#596983">
                        Your ticket details will appear here
                    </b>

                    <div style="font-size:12px;margin-top:5px">
                        Create a ticket or enter an existing reference.
                    </div>

                </div>
                """, unsafe_allow_html=True)

            if current_ticket and current_ticket["status"] == "Waiting":

                if st.button("Cancel this ticket"):

                    if db.cancel_ticket(current_ticket["ticket_code"]):

                        st.success("Your ticket has been cancelled.")

                        st.rerun()

                    else:

                        st.error(
                            "This ticket can no longer be cancelled."
                        )

            if current_ticket and current_ticket["status"] == "Completed":

                st.markdown("**How was your experience?**")

                with st.form("feedback_form"):

                    rating = st.slider(
                        "Rating",
                        1,
                        5,
                        5
                    )

                    comment = st.text_area(
                        "Share feedback (optional)"
                    )

                    if st.form_submit_button("Submit feedback"):

                        db.add_feedback(
                            current_ticket["ticket_code"],
                            rating,
                            comment
                        )

                        st.success(
                            "Thanks for helping us improve!"
                        )


    st.write("")

    with st.container(key="panel_chat"):

            st.markdown("""

                <div class="panel-title">
                    ✦ Ask the WaitWise assistant
                </div>

                <div class="panel-caption">
                Get quick help with ticket creation, queue position,
                waiting times, and services.
                </div>
            """, unsafe_allow_html=True)

            prompt_col, button_col = st.columns([5, 1])

            with prompt_col:

                prompt = st.text_input(
                    "Ask a question",
                    key="chat_prompt",
                    placeholder="Example: How long will I wait?",
                    label_visibility="collapsed"
                )

            with button_col:

                ask = st.button(
                    "Ask AI",
                    **STRETCH_BTN
                )

            if ask:

                ticket_for_chat = (
                    db.get_ticket(st.session_state.get("lookup_ticket", ""))
                    if st.session_state.get("lookup_ticket")
                    else None
                )

                if prompt.strip():

                    st.session_state["chat_answer"] = chatbot_reply(
                        chatbot,
                        prompt,
                        ticket_for_chat
                    )

                else:

                    st.session_state["chat_answer"] = (
                        "Type a question to start chatting."
                    )

            if st.session_state.get("chat_answer"):

                st.markdown(f"""
                <div class="ai-banner">

                    <b>WaitWise AI</b><br>

                    <span style="font-size:13px">
                        {st.session_state["chat_answer"]}
                    </span>

                </div>
                """, unsafe_allow_html=True)

            st.markdown("""
            <div class="small-note">
                Local machine-learning assistant ·
                Your message is processed within this application.
            </div>
            """, unsafe_allow_html=True)


# =====================================================
# PANEL 2: ADMIN PANEL
# =====================================================

elif page == "Admin Panel":

    if not require_pin("admin", "ADMIN_PIN", "1234"):
        st.stop()

    hero("ADMIN", "See everything. Keep service moving.",
         "Run your counter, watch the analytics and review every record, including the ones flagged as deleted.")

    desk, analytics, records = st.tabs(["Counter desk", "Analytics", "All records"])

    with desk:

        c1, c2 = st.columns(2)
        service = c1.selectbox("Service queue", SERVICES)
        counter = c2.selectbox("Your counter", COUNTERS)

        waiting = db.list_tickets(service=service, statuses=["Waiting"])
        mine = [t for t in db.list_tickets(service=service, statuses=["Serving"]) if t["counter"] == counter]

        m1, m2 = st.columns(2)
        with m1:
            metric_card("Waiting", len(waiting), "In this queue", "⏳")
        with m2:
            metric_card("Next ticket", waiting[0]["ticket_code"] if waiting else "—", "First in line", "🎟️")

        st.write("")

        if mine:
            st.markdown(f'<div class="ticket-code">{mine[0]["ticket_code"]}</div>', unsafe_allow_html=True)
            st.write(f"**{mine[0]['customer_name']}** at {counter}")
            if st.button("✓ Complete service", **STRETCH_BTN):
                if db.complete_ticket(mine[0]["ticket_code"]):
                    st.cache_resource.clear()
                    st.rerun()
        else:
            st.info(f"{counter} has no active customer.")

        if st.button("Call next customer  →", type="primary", **STRETCH_BTN):
            code = db.call_next(service, counter)
            if code:
                st.rerun()
            else:
                st.warning("There are no waiting tickets for this service.")

        if waiting:
            st.dataframe(pd.DataFrame([dict(r) for r in waiting])[["ticket_code", "customer_name", "created_at"]],
                         hide_index=True, **STRETCH_DF)

    with analytics:

        stats = db.get_stats()
        cols = st.columns(5)
        for col, (label, key, icon) in zip(cols, [("Total", "total", "🎟️"), ("Waiting", "waiting", "⏳"),
                                                  ("Serving", "serving", "🟢"), ("Completed", "completed", "✓"),
                                                  ("Flagged deleted", "deleted", "🗑")]):
            with col:
                metric_card(label, stats[key], "Active records" if key != "deleted" else "Kept in database", icon)

        rows = db.list_tickets(limit=5000)
        if rows:
            df = pd.DataFrame([dict(r) for r in rows])
            a, b = st.columns(2)
            a.bar_chart(df.groupby("service").size().rename("Tickets"), color=THEME["a"])
            b.bar_chart(df["status"].value_counts().rename("Tickets"), color=THEME["b"])
        else:
            st.info("Analytics will appear after customers start creating tickets.")

        if using_real_history:
            st.success("Wait-time model is trained on completed tickets from this database.")
        else:
            st.info("Wait-time model uses demo data until 5 real completed tickets exist.")

    with records:

        show = st.selectbox("Show", ["All records", "Active only", "Flagged as deleted only"])
        rows = db.list_tickets(include_deleted=True, limit=5000)
        if rows:
            df = pd.DataFrame([dict(r) for r in rows])
            df["Deleted"] = df["is_deleted"].map({1: "🗑 DELETED", 0: ""})
            if show == "Active only":
                df = df[df["is_deleted"] == 0]
            elif show == "Flagged as deleted only":
                df = df[df["is_deleted"] == 1]
            st.caption("Deleted rows are never removed. They stay in the database and are only flagged here.")
            st.dataframe(df.drop(columns=["is_deleted"]), hide_index=True, **STRETCH_DF)
        else:
            st.info("No records yet.")


# =====================================================
# PANEL 3: MANAGER DATABASE
# =====================================================

elif page == "Manager Database":

    if not require_pin("manager", "MANAGER_PIN", "5678"):
        st.stop()

    hero("MANAGER", "Your database, fully in your hands.",
         "Edit any ticket, add new ones, and flag records as deleted. Nothing is ever erased.")

    if st.session_state.get("flash"):
        st.success(st.session_state.pop("flash"))

    t_edit, t_add, t_fb = st.tabs(["Edit tickets", "Add ticket", "Feedback"])

    with t_edit:

        rows = db.list_tickets(include_deleted=True, limit=5000)

        if rows:

            cols = ["id", "ticket_code", "customer_name", "service", "status", "counter",
                    "service_minutes", "created_at", "is_deleted", "deleted_at", "deleted_by"]
            df = pd.DataFrame([dict(r) for r in rows])[cols]
            df["is_deleted"] = df["is_deleted"].astype(bool)

            st.caption("Double-click a cell to edit. Tick “Flag as deleted” instead of deleting: "
                       "the row stays in the database and the admin sees it marked as deleted. Untick to restore.")

            edited = st.data_editor(
                df, hide_index=True, key="mgr_editor", **STRETCH_ED,
                disabled=["id", "ticket_code", "created_at", "deleted_at", "deleted_by"],
                column_config={
                    "service": st.column_config.SelectboxColumn("Service", options=SERVICES, required=True),
                    "status": st.column_config.SelectboxColumn("Status", options=db.STATUSES, required=True),
                    "counter": st.column_config.SelectboxColumn("Counter", options=COUNTERS),
                    "service_minutes": st.column_config.NumberColumn("Service minutes", min_value=0.0),
                    "is_deleted": st.column_config.CheckboxColumn("Flag as deleted"),
                })

            if st.button("Save changes", type="primary"):

                def clean(rec):
                    return {k: (None if pd.isna(v) else v) for k, v in rec.items()}

                old = {r["id"]: r for r in map(clean, df.to_dict("records"))}
                changed, errors = 0, []

                for new in map(clean, edited.to_dict("records")):
                    before = old[new["id"]]
                    if new == before:
                        continue
                    try:
                        fields = {k: new[k] for k in ("customer_name", "service", "status", "counter", "service_minutes")
                                  if new[k] != before[k]}
                        db.update_ticket(int(new["id"]), fields)
                        if bool(new["is_deleted"]) != bool(before["is_deleted"]):
                            db.set_deleted(int(new["id"]), bool(new["is_deleted"]), "Manager")
                        changed += 1
                    except ValueError as e:
                        errors.append(f"{new['ticket_code']}: {e}")

                for e in errors:
                    st.error(e)

                if changed and not errors:
                    st.session_state.pop("mgr_editor", None)
                    st.cache_resource.clear()
                    st.session_state["flash"] = f"Saved {changed} change(s)."
                    st.rerun()
                elif not changed and not errors:
                    st.info("No changes to save.")
        else:
            st.info("The database has no tickets yet. Add one in the next tab.")

    with t_add:

        with st.form("mgr_add", clear_on_submit=True):
            n_name = st.text_input("Customer name")
            n_service = st.selectbox("Service", SERVICES)
            n_status = st.selectbox("Status", db.STATUSES)
            n_counter = st.selectbox("Counter (optional)", ["—"] + COUNTERS)
            if st.form_submit_button("Add ticket", **STRETCH_BTN):
                if not n_name.strip():
                    st.error("Enter the customer name.")
                else:
                    code = db.add_ticket(n_name, n_service, n_status, None if n_counter == "—" else n_counter)
                    st.session_state["flash"] = f"Added ticket {code}."
                    st.rerun()

    with t_fb:

        fb = db.all_feedback()
        if fb:
            st.dataframe(pd.DataFrame([dict(r) for r in fb]), hide_index=True, **STRETCH_DF)
        else:
            st.info("No feedback yet.")


# =====================================================
# FOOTER
# =====================================================

st.markdown("""
<div style="
    text-align:center;
    color:#9aa6b8;
    font-size:11px;
    padding:20px 0 4px
">
    WAITWISE · SMARTER QUEUES, BETTER EXPERIENCES
</div>
""", unsafe_allow_html=True)