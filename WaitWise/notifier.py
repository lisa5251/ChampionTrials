"""Outgoing notifications.

Every notification is always saved in the `notifications` table (shown in the Admin
"Settings & Outbox" tab). If you also want REAL text messages, install twilio
(`pip install twilio`) and set these environment variables before starting the app:

    TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM   (your Twilio number)

Optional: DEFAULT_COUNTRY_CODE (e.g. 383) turns local numbers such as "044 123 456"
into international format (+38344123456), which Twilio requires.

Without the Twilio variables nothing is sent and the app keeps working normally.
"""
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor

log = logging.getLogger("waitwise.notifier")

# SMS is sent in the background so a slow provider never freezes the counter screen.
_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="waitwise-sms")


def sms_enabled() -> bool:
    return all(os.getenv(k) for k in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM"))


def normalize_phone(phone):
    """Best-effort conversion to E.164 (+<country><number>). Returns None if impossible."""
    p = (phone or "").strip()
    digits = re.sub(r"\D", "", p)
    if not digits:
        return None
    if p.startswith("+"):
        return "+" + digits
    if p.startswith("00"):
        return "+" + digits[2:]
    country = re.sub(r"\D", "", os.getenv("DEFAULT_COUNTRY_CODE", ""))
    if country:
        return "+" + country + digits.lstrip("0")
    return None  # cannot guess the country: better to skip than to text a stranger


def send(phone: str, message: str) -> bool:
    """Best-effort SMS. Returns True only if a real message was handed to Twilio."""
    if not phone or not sms_enabled():
        return False
    number = normalize_phone(phone)
    if not number:
        log.warning("SMS skipped: cannot turn %r into an international number "
                    "(set DEFAULT_COUNTRY_CODE or ask for +<country> numbers)", phone)
        return False
    try:
        from twilio.rest import Client  # imported lazily so it stays optional

        Client(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"]).messages.create(
            to=number, from_=os.environ["TWILIO_FROM"], body=message
        )
        return True
    except Exception:  # never let a failed SMS break the queue
        log.exception("SMS delivery failed")
        return False


def send_async(phone: str, message: str) -> None:
    """Queue an SMS in the background (fire and forget)."""
    if phone and sms_enabled():
        _pool.submit(send, phone, message)