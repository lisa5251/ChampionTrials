
import random
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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

def train_wait_model(completed_rows):
    """
    Train a Random Forest on completed-ticket durations.

    - Rows with an unknown service or a non-positive duration are ignored.
    - Any service with fewer than 3 real examples is topped up with
      synthetic demo data so every service always gets a sensible estimate.
    - using_real_history is True only when at least 5 usable real rows exist.
    """

    real_rows = []

    for row in completed_rows:

        service = row["service"]
        minutes = row["service_minutes"]

        if service in SERVICES and minutes is not None and minutes > 0:

            real_rows.append((service, float(minutes)))

    base = {
        "General Inquiry": 5.0,
        "Payments": 7.0,
        "Technical Support": 10.0
    }

    rng = random.Random(17)
    rows = list(real_rows)

    for service in SERVICES:

        have = sum(1 for s, _ in real_rows if s == service)

        if have < 3:

            for _ in range(35):

                rows.append((
                    service,
                    max(2.0, rng.gauss(base[service], base[service] * 0.22))
                ))

    X = pd.DataFrame({
        "service_code": [SERVICES.index(service) for service, _ in rows]
    })

    y = [duration for _, duration in rows]

    model = RandomForestRegressor(n_estimators=80, random_state=17)
    model.fit(X, y)

    using_real_history = len(real_rows) >= 5

    return model, using_real_history


# ==========================================
# PREDICT WAITING TIME
# ==========================================

def predict_wait(model, service, people_ahead):

    if service not in SERVICES:

        service = SERVICES[0]

    service_code = SERVICES.index(service)

    prediction_data = pd.DataFrame({
        "service_code": [service_code]
    })

    estimated_service_time = model.predict(
        prediction_data
    )[0]

    estimated_wait = (
        estimated_service_time * (people_ahead + 1)
    )

    return max(1, round(float(estimated_wait)))


# ==========================================
# CHATBOT TRAINING DATA
# ==========================================

INTENTS = {

    "greeting": [
        "hello",
        "hi",
        "good morning",
        "hey there",
        "good afternoon"
    ],

    "wait_time": [
        "how long will i wait",
        "what is my waiting time",
        "estimate my wait",
        "how much longer",
        "when will it be my turn"
    ],

    "position": [
        "what is my queue position",
        "how many people are ahead",
        "where am i in the queue",
        "check my ticket position"
    ],

    "new_ticket": [
        "how do i get a ticket",
        "create a queue ticket",
        "i need a ticket",
        "how can i join the queue"
    ],

    "services": [
        "what services are available",
        "which services do you offer",
        "show me the service options",
        "what can i get help with"
    ],

    "cancel": [
        "how do i cancel my ticket",
        "cancel my queue ticket",
        "i cannot wait anymore",
        "remove me from the queue"
    ],

    "thanks": [
        "thank you",
        "thanks",
        "that helped",
        "great thanks"
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

def chatbot_reply(model, message, ticket=None):

    intent = model.predict([message])[0]

    probabilities = model.predict_proba([message])[0]

    confidence = max(probabilities)

    if confidence < 0.40:

        return (
            "I'm not sure I understood. "
            "Try asking about tickets, waiting times, "
            "queue positions, services, or cancellations."
        )

    if intent == "wait_time" and ticket:

        return (
            f"Your ticket {ticket['ticket_code']} "
            f"is currently {ticket['status']}. "
            "Check your ticket details for the latest estimate."
        )

    if intent == "position" and ticket:

        return (
            f"Your ticket is {ticket['ticket_code']} "
            f"and its current status is {ticket['status']}."
        )

    return RESPONSES.get(
        intent,
        "I can help you with WaitWise queue services."
    )
