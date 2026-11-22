import random
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


# ==========================================
# AVAILABLE SERVICES
# ==========================================

SERVICES = [
    "General Inquiry",
    "Payments",
    "Technical Support"
]


# ==========================================
# AI WAITING-TIME PREDICTION
# ==========================================

BASE_MINUTES = {
    "General Inquiry": 5.0,
    "Payments": 7.0,
    "Technical Support": 10.0
}

MAX_REAL_MINUTES = 120   # a ticket left "Serving" overnight must not poison the model
MIN_REAL_ROWS = 5


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
        if service in SERVICES and minutes is not None and 0 < minutes <= MAX_REAL_MINUTES:
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
    - Rows with an unknown service, non-positive or absurd (>2h) duration are ignored.
    - Any service with fewer than 3 real examples is topped up with synthetic
      demo data so every service always gets a sensible estimate.
    - using_real_history is True only when at least 5 usable real rows exist.
    """
    real_rows = _usable_rows(completed_rows)
    rng = random.Random(17)
    rows = list(real_rows)

    for service in SERVICES:
        have = sum(1 for s, _, _ in real_rows if s == service)
        if have < 3:
            for _ in range(35):
                hour = rng.randint(8, 18)
                rush = 1.15 if 11 <= hour <= 14 else 1.0
                base = BASE_MINUTES[service] * rush
                rows.append((service, hour, max(2.0, rng.gauss(base, base * 0.22))))

    X = pd.DataFrame({
        "service_code": [SERVICES.index(s) for s, _, _ in rows],
        "hour": [h for _, h, _ in rows],
    })
    y = [m for _, _, m in rows]

    model = RandomForestRegressor(n_estimators=80, random_state=17)
    model.fit(X, y)

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
    scores = cross_val_score(RandomForestRegressor(n_estimators=40, random_state=17), X, y,
                             cv=folds, scoring="neg_mean_absolute_error")
    return float(-scores.mean())


# ==========================================
# PREDICT WAITING TIME
# ==========================================

def predict_wait_range(model, service, load, counters=1, hour=None):
    """
    Returns (low, expected, high) wait in whole minutes.

    load     - people ahead of the customer (a person already being served can count as 0.5)
    counters - how many counters are working this service in parallel
    The low/high band comes from the spread between the forest's trees (20th-80th percentile).
    """
    if service not in SERVICES:
        service = SERVICES[0]
    if hour is None:
        hour = datetime.now().hour

    features = pd.DataFrame({"service_code": [SERVICES.index(service)], "hour": [hour]})
    mid = float(model.predict(features)[0])

    raw = features.to_numpy(dtype=float)
    per_tree = [tree.predict(raw)[0] for tree in model.estimators_]
    lo, hi = (float(v) for v in np.percentile(per_tree, [20, 80]))

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
        "hello", "hi", "good morning", "hey there", "good afternoon", "hey"
    ],

    "wait_time": [
        "how long will i wait", "what is my waiting time", "estimate my wait",
        "how much longer", "when will it be my turn", "how many minutes left"
    ],

    "position": [
        "what is my queue position", "how many people are ahead",
        "where am i in the queue", "check my ticket position", "am i next"
    ],

    "new_ticket": [
        "how do i get a ticket", "create a queue ticket", "i need a ticket",
        "how can i join the queue", "join the line", "get in line"
    ],

    "services": [
        "what services are available", "which services do you offer",
        "show me the service options", "what can i get help with"
    ],

    "cancel": [
        "how do i cancel my ticket", "cancel my queue ticket",
        "i cannot wait anymore", "remove me from the queue", "leave the queue"
    ],

    "priority": [
        "i am pregnant", "i am elderly", "i have a disability",
        "do you have priority service", "i need priority", "i am in a wheelchair",
        "senior citizen priority", "i am with a small baby"
    ],

    "no_show": [
        "what if i miss my turn", "i was not here when called",
        "what happens if i am late", "i missed my number", "my number was called"
    ],

    "notifications": [
        "will i get a text message", "can you notify me", "send me an sms",
        "how will i know it is my turn", "do you send alerts", "can i leave and come back"
    ],

    "open_hours": [
        "are you open", "is the queue open", "can i take a ticket now",
        "is the queue paused", "are you taking new customers"
    ],

    "human": [
        "i want to talk to a person", "speak to a human", "i need a real person",
        "talk to staff", "i have a complaint"
    ],

    "thanks": [
        "thank you", "thanks", "that helped", "great thanks", "perfect thank you"
    ]
}


# ==========================================
# CHATBOT RESPONSES
# ==========================================

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
        "WaitWise currently offers General Inquiry, "
        "Payments, and Technical Support.",

    "cancel":
        "You can cancel your ticket while it is still "
        "waiting by opening the Customer Portal.",

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

def build_chatbot():

    training_texts = []
    training_labels = []

    for intent, examples in INTENTS.items():

        for example in examples:

            training_texts.append(example)
            training_labels.append(intent)

    chatbot_model = Pipeline([

        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                lowercase=True
            )
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

def chatbot_reply(model, message, ticket=None, info=None):
    """
    info (optional) is live data about the customer's ticket, e.g.
    {"position": 3, "wait_low": 8, "wait_high": 14} so the bot can give real numbers.
    """

    intent = model.predict([message])[0]

    probabilities = model.predict_proba([message])[0]

    confidence = max(probabilities)

    if confidence < 0.30:

        return (
            "I'm not sure I understood. "
            "Try asking about tickets, waiting times, "
            "queue positions, services, priority or cancellations."
        )

    info = info or {}

    if intent == "wait_time" and ticket:

        if ticket["status"] == "Waiting" and "wait_low" in info:
            return (
                f"Ticket {ticket['ticket_code']}: about {info['wait_low']}-{info['wait_high']} minutes "
                f"to go, with {info['ahead']} people ahead of you."
            )

        return (
            f"Your ticket {ticket['ticket_code']} "
            f"is currently {ticket['status']}."
        )

    if intent == "position" and ticket:

        if ticket["status"] == "Waiting" and "ahead" in info:
            return (
                "You're next!" if info["ahead"] == 0 else
                f"Ticket {ticket['ticket_code']} has {info['ahead']} "
                f"{'person' if info['ahead'] == 1 else 'people'} ahead of it."
            )

        return (
            f"Your ticket is {ticket['ticket_code']} "
            f"and its current status is {ticket['status']}."
        )

    return RESPONSES.get(
        intent,
        "I can help you with WaitWise queue services."
    )