"""Outgoing notifications.

Every notification is always saved in the `notifications` table (shown in the Admin
"Settings & Outbox" tab). If you also want REAL text messages, install twilio
(`pip install twilio`) and set these environment variables before starting the app:

    TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM   (your Twilio number)

Without them nothing is sent and the app keeps working normally.
"""
import logging
import os

log = logging.getLogger("waitwise.notifier")


def sms_enabled() -> bool:
    return all(os.getenv(k) for k in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM"))


def send(phone: str, message: str) -> bool:
    """Best-effort SMS. Returns True only if a real message was handed to Twilio."""
    if not phone or not sms_enabled():
        return False
    try:
        from twilio.rest import Client  # imported lazily so it stays optional

        Client(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"]).messages.create(
            to=phone, from_=os.environ["TWILIO_FROM"], body=message
        )
        return True
    except Exception:  # never let a failed SMS break the queue
        log.exception("SMS delivery failed")
        return False
