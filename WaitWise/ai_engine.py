import random
import re
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import FeatureUnion, Pipeline

from config import SERVICES, COUNTERS, SERVICE_INFO  # noqa: F401  (re-exported for app.py)


# ==========================================
# AI WAITING-TIME PREDICTION
# ==========================================

BASE_MINUTES = {
    "General Inquiry": 5.0,
    "Payments": 7.0,
    "Technical Support": 10.0
}
DEFAULT_BASE_MINUTES = 7.0   # for a service added in config.py that has no entry above

MAX_REAL_MINUTES = 120   # a ticket left "Serving" overnight must not poison the model
MIN_REAL_MINUTES = 0.5   # a few seconds is an accidental click, not a real service
MIN_REAL_ROWS = 5

REAL_WEIGHT = 3.0        # one real ticket counts as three demo tickets
DEMO_TARGET = 30         # demo rows per service shrink as real rows arrive (gone at 10 real rows)


def _hour_of(called_at):
    """Hour of day (0-23) a ticket was called; 12 if unknown."""
    try:
        return datetime.fromisoformat(called_at).hour
    except (TypeError, ValueError):
        return 12


def _usable_rows(completed_rows):
    rows = []
    for row in completed_rows:
        service, minutes = row["service"], row["service_minutes"]
        if service in SERVICES and minutes is not None and MIN_REAL_MINUTES <= minutes <= MAX_REAL_MINUTES:
            try:
                hour = _hour_of(row["called_at"])
            except (KeyError, IndexError):
                hour = 12
            rows.append((service, hour, float(minutes)))
    return rows


def train_wait_model(completed_rows):
    """
    Train a Random Forest on completed-ticket durations.

    Features: which service, and the hour of day it was called (rush hours are slower).
    - Rows with an unknown service, a duration under 30 s or over 2 h are ignored.
    - Demo data fills the gaps smoothly: a service with no real tickets gets 30 demo rows,
      and every real ticket replaces three of them (none left at 10 real tickets).
      Real tickets also count three times as much as demo rows when the forest is trained.
    - using_real_history is True only when at least 5 usable real rows exist.
    """
    real_rows = _usable_rows(completed_rows)
    rng = random.Random(17)
    rows = [(s, h, m, REAL_WEIGHT) for s, h, m in real_rows]

    for service in SERVICES:
        have = sum(1 for s, _, _ in real_rows if s == service)
        base_minutes = BASE_MINUTES.get(service, DEFAULT_BASE_MINUTES)
        for _ in range(max(0, DEMO_TARGET - 3 * have)):
            hour = rng.randint(8, 18)
            rush = 1.15 if 11 <= hour <= 14 else 1.0
            base = base_minutes * rush
            rows.append((service, hour, max(2.0, rng.gauss(base, base * 0.22)), 1.0))

    X = pd.DataFrame({
        "service_code": [SERVICES.index(s) for s, _, _, _ in rows],
        "hour": [h for _, h, _, _ in rows],
    })
    y = [m for _, _, m, _ in rows]
    weights = [w for _, _, _, w in rows]

    model = RandomForestRegressor(n_estimators=80, min_samples_leaf=2, random_state=17)
    model.fit(X, y, sample_weight=weights)

    return model, len(real_rows) >= MIN_REAL_ROWS


def wait_model_mae(completed_rows):
    """Cross-validated mean absolute error (minutes) on REAL data, or None if too little data."""
    real_rows = _usable_rows(completed_rows)
    if len(real_rows) < 12:
        return None
    X = pd.DataFrame({"service_code": [SERVICES.index(s) for s, _, _ in real_rows],
                      "hour": [h for _, h, _ in real_rows]})
    y = [m for _, _, m in real_rows]
    folds = min(5, len(real_rows) // 3)
    scores = cross_val_score(RandomForestRegressor(n_estimators=40, min_samples_leaf=2, random_state=17), X, y,
                             cv=folds, scoring="neg_mean_absolute_error")
    return float(-scores.mean())


# ==========================================
# PREDICT WAITING TIME
# ==========================================

def _unit_table(model):
    """Minutes per person (low, expected, high) for every service x hour, computed once per model.

    Walking all 80 trees for every ticket on every 5-second TV refresh was wasteful: there are
    only services x 24 hours distinct questions, so we answer them all up front.
    """
    table = getattr(model, "_ww_table", None)
    if table is None:
        combos = [(s, h) for s in range(len(SERVICES)) for h in range(24)]
        raw = np.array(combos, dtype=float)
        per_tree = np.vstack([tree.predict(raw) for tree in model.estimators_])
        lo, hi = np.percentile(per_tree, [20, 80], axis=0)
        mid = per_tree.mean(axis=0)
        table = {c: (float(lo[i]), float(mid[i]), float(hi[i])) for i, c in enumerate(combos)}
        model._ww_table = table
    return table


def predict_wait_range(model, service, load, counters=1, hour=None):
    """
    Returns (low, expected, high) wait in whole minutes.

    load     - people ahead of the customer (a person already being served can count as 0.5)
    counters - how many counters are working this service in parallel
    The low/high band comes from the spread between the forest's trees (20th-80th percentile).
    """
    if service not in SERVICES:
        service = SERVICES[0]
    hour = (datetime.now().hour if hour is None else int(hour)) % 24

    lo, mid, hi = _unit_table(model)[(SERVICES.index(service), hour)]

    scale = max(0.0, float(load)) / max(1, int(counters))
    low, expected, high = (max(1, round(v * scale)) for v in (lo, mid, hi))
    return low, expected, max(high, expected)


def predict_wait(model, service, people_ahead, counters=1, hour=None):
    """Single-number estimate (minutes). Kept for backwards compatibility."""
    return predict_wait_range(model, service, people_ahead, counters, hour)[1]


# ==========================================
# CHATBOT TRAINING DATA
# ==========================================

INTENTS = {

    "greeting": [
        "hello", "hi", "good morning", "hey there", "good afternoon", "hey",
        "good evening", "hi there", "hiya", "greetings"
    ],

    "wait_time": [
        "how long will i wait", "what is my waiting time", "estimate my wait",
        "how much longer", "when will it be my turn", "how many minutes left",
        "how long is the wait", "what is the estimated waiting time",
        "how long until i am served", "when will they call me"
    ],

    "position": [
        "what is my queue position", "how many people are ahead",
        "where am i in the queue", "check my ticket position", "am i next",
        "what number am i", "how many people are before me",
        "am i next in line", "how far am i from the front"
    ],

    "new_ticket": [
        "how do i get a ticket", "create a queue ticket", "i need a ticket",
        "how can i join the queue", "join the line", "get in line",
        "take a number", "how do i sign up", "where do i register",
        "i want to register in the queue", "i would like a ticket please",
        "can i have a ticket", "i want a ticket"
    ],

    "services": [
        "what services are available", "which services do you offer",
        "show me the service options", "what can i get help with",
        "what departments do you have", "list your services", "what do you offer"
    ],

    "cancel": [
        "how do i cancel my ticket", "cancel my queue ticket",
        "i cannot wait anymore", "remove me from the queue", "leave the queue",
        "i want to cancel", "delete my ticket", "i do not want to wait any more",
        "withdraw my ticket"
    ],

    "priority": [
        "i am pregnant", "i am elderly", "i have a disability",
        "do you have priority service", "i need priority", "i am in a wheelchair",
        "senior citizen priority", "i am with a small baby",
        "i am over 65", "i have a newborn with me", "i am expecting a baby",
        "my father is old and cannot stand", "my mother is elderly",
        "i cannot stand for long", "i have trouble walking"
    ],

    "no_show": [
        "what if i miss my turn", "i was not here when called",
        "what happens if i am late", "i missed my number", "my number was called",
        "what if i am not here when my number is called",
        "they called my number but i was not there", "i was late and missed my turn"
    ],

    "notifications": [
        "will i get a text message", "can you notify me", "send me an sms",
        "how will i know it is my turn", "do you send alerts", "can i leave and come back",
        "do you send text messages", "can you message me when it is my turn",
        "do i need to stay here", "can i wait outside",
        "do i get a notification", "will i be notified", "do you send notifications"
    ],

    "open_hours": [
        "are you open", "is the queue open", "can i take a ticket now",
        "is the queue paused", "are you taking new customers",
        "is the queue closed", "can i still join", "are tickets paused"
    ],

    "human": [
        "i want to talk to a person", "speak to a human", "i need a real person",
        "talk to staff", "i have a complaint",
        "i want to speak to someone", "give me a real person",
        "connect me to an agent", "staff please"
    ],

    "thanks": [
        "thank you", "thanks", "that helped", "great thanks", "perfect thank you",
        "thanks a lot", "cheers", "thank you so much", "appreciate it"
    ]
}


# ==========================================
# CHATBOT RESPONSES
# ==========================================

def _service_list():
    if len(SERVICES) == 1:
        return SERVICES[0]
    return ", ".join(SERVICES[:-1]) + ", and " + SERVICES[-1]


RESPONSES = {

    "greeting":
        "Hello! I'm the WaitWise AI assistant. "
        "I can help you with tickets, queue positions, "
        "waiting times, and available services.",

    "wait_time":
        "Enter your ticket code in the Customer Portal "
        "to see your estimated waiting time.",

    "position":
        "Open the Customer Portal and enter your ticket "
        "code to check how many people are ahead of you.",

    "new_ticket":
        "Open the Customer Portal, enter your name, "
        "choose a service, and click Create My Ticket.",

    "services":
        f"WaitWise currently offers {_service_list()}.",

    "cancel":
        "You can cancel your ticket while it is still waiting: open it from the link you got when you "
        "created it (or on the device you used) and press Cancel. "
        "If you lost the link, ask at the service desk.",

    "priority":
        "When you create your ticket, tick the priority box if you are a senior, pregnant, "
        "have a disability or are with a young child. Priority tickets are called first, "
        "and staff may ask to confirm.",

    "no_show":
        "If you are called and don't arrive, staff can re-announce your number. "
        "If you still don't show up the ticket is closed and you can simply take a new one.",

    "notifications":
        "Add your phone number when creating a ticket to be told when you're next and when it's your turn. "
        "You can also keep the Customer Portal open: it refreshes by itself.",

    "open_hours":
        "The Customer Portal tells you right away if the queue is paused or full. "
        "If you can create a ticket, we are accepting customers.",

    "human":
        "Of course. Please speak to a team member at the service desk and they will help you in person.",

    "thanks":
        "You're welcome! Let me know if you need "
        "anything else."
}


# ==========================================
# BUILD AI CHATBOT
# ==========================================

CONFIDENCE_THRESHOLD = 0.30


def _normalize(message):
    """Lower-case, keep letters/digits/apostrophes only, cap the length."""
    text = str(message or "").lower()[:300]
    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", text).split())


def build_chatbot():

    training_texts = []
    training_labels = []

    for intent, examples in INTENTS.items():

        for example in examples:

            training_texts.append(_normalize(example))
            training_labels.append(intent)

    chatbot_model = Pipeline([

        (
            "features",
            # words understand meaning, character pieces forgive typos ("wating tme")
            FeatureUnion([
                ("words", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
                ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True)),
            ])
        ),

        (
            "classifier",
            LogisticRegression(
                C=20,
                max_iter=1000
            )
        )

    ])

    chatbot_model.fit(
        training_texts,
        training_labels
    )

    return chatbot_model


# ==========================================
# GENERATE CHATBOT RESPONSE
# ==========================================

def _status_line(ticket):
    code, status = ticket["ticket_code"], ticket["status"]
    if status == "Serving":
        return f"Ticket {code} is being called now. Please go to {ticket['counter'] or 'the service desk'}."
    if status == "Completed":
        return f"Ticket {code} is complete. Thank you!"
    if status == "No-show":
        return f"Ticket {code} was closed because it was called and nobody came. You can take a new one."
    if status == "Cancelled":
        return f"Ticket {code} was cancelled."
    return f"Your ticket {code} is currently {status}."


def chatbot_reply(model, message, ticket=None, info=None):
    """
    info (optional) is live data about the customer's ticket, e.g.
    {"ahead": 3, "wait_low": 8, "wait_high": 14} so the bot can give real numbers.
    """

    message = _normalize(message)

    if not message:
        return "Type a question to start chatting."

    intent = model.predict([message])[0]

    probabilities = model.predict_proba([message])[0]

    confidence = max(probabilities)

    if confidence < CONFIDENCE_THRESHOLD:

        return (
            "I'm not sure I understood. "
            "Try asking about tickets, waiting times, "
            "queue positions, services, priority or cancellations."
        )

    info = info or {}

    if intent == "wait_time" and ticket:

        if ticket["status"] == "Waiting" and "wait_low" in info and "ahead" in info:
            low, high = info["wait_low"], info["wait_high"]
            span = f"{low}-{high} minutes" if high > low else f"{low} {'minute' if low == 1 else 'minutes'}"
            if info["ahead"] == 0:
                return f"Ticket {ticket['ticket_code']}: you're next! Expect to be called in about {span}."
            return (
                f"Ticket {ticket['ticket_code']}: about {span} "
                f"to go, with {info['ahead']} {'person' if info['ahead'] == 1 else 'people'} ahead of you."
            )

        return _status_line(ticket)

    if intent == "position" and ticket:

        if ticket["status"] == "Waiting" and "ahead" in info:
            return (
                "You're next!" if info["ahead"] == 0 else
                f"Ticket {ticket['ticket_code']} has {info['ahead']} "
                f"{'person' if info['ahead'] == 1 else 'people'} ahead of it."
            )

        return _status_line(ticket)

    return RESPONSES.get(
        intent,
        "I can help you with WaitWise queue services."
    )