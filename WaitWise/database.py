"""SQLite storage for WaitWise tickets, feedback, settings and notifications."""
import hashlib
import hmac
import os
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import notifier

DB_PATH = Path(__file__).parent / "waitwise.db"

ROLES = ["admin", "manager"]
DEFAULT_LOGINS = {"admin": "1234", "manager": "5678"}  # first-run only; change them in Manager > Staff accounts

STATUSES = ["Waiting", "Serving", "Completed", "Cancelled", "No-show"]

DEFAULT_SETTINGS = {
    "queue_open": "1",          # "1" = accepting new tickets, "0" = paused
    "max_waiting": "0",         # 0 = unlimited
    "announcement": "",         # shown on the TV board and customer portal
    "public_url": "http://localhost:8501",  # used for the QR code
}

# Waiting tickets are ordered: priority first, then first-come-first-served.
ORDER = "priority DESC, id ASC"


class QueueClosed(ValueError):
    """Raised when a customer tries to join a paused or full queue."""


@contextmanager
def connect():
    """Open a connection, commit on success, roll back on error, ALWAYS close.
    (sqlite3's own `with conn:` never closes the connection, which leaks handles.)"""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _now():
    return datetime.now().isoformat(timespec="seconds")


def _ensure_valid_db():
    """If waitwise.db exists but is not a real SQLite file (corrupt, partial copy,
    edited in a text editor...), move it aside as waitwise.db.corrupt-<time> and
    let a fresh database be created instead of crashing."""
    if not DB_PATH.exists() or DB_PATH.stat().st_size == 0:
        return
    try:
        check = sqlite3.connect(DB_PATH, timeout=10)
        try:
            ok = check.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        finally:
            check.close()
    except sqlite3.DatabaseError:
        ok = False
    if not ok:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        DB_PATH.rename(DB_PATH.with_name(f"{DB_PATH.name}.corrupt-{stamp}"))
        for extra in ("-journal", "-wal", "-shm"):
            leftover = DB_PATH.with_name(DB_PATH.name + extra)
            if leftover.exists():
                leftover.unlink()


def init_db():
    _ensure_valid_db()
    _create_tables()
    _migrate()
    _seed_users()


def _create_tables():
    with connect() as conn:
        conn.execute("PRAGMA journal_mode=WAL")  # readers (TV board) never block writers (counters)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_code TEXT UNIQUE NOT NULL,
                customer_name TEXT NOT NULL,
                service TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'Waiting',
                created_at TEXT NOT NULL,
                called_at TEXT,
                completed_at TEXT,
                service_minutes REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_code TEXT NOT NULL,
                rating INTEGER NOT NULL,
                comment TEXT,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                role TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_code TEXT NOT NULL,
                channel TEXT NOT NULL,
                phone TEXT,
                message TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)


def _migrate():
    """Add newer columns to databases created by older versions (never loses data)."""
    with connect() as conn:
        have = {r["name"] for r in conn.execute("PRAGMA table_info(tickets)")}
        for name, ddl in [("counter", "TEXT"), ("is_deleted", "INTEGER NOT NULL DEFAULT 0"),
                          ("deleted_at", "TEXT"), ("deleted_by", "TEXT"), ("updated_at", "TEXT"),
                          ("priority", "INTEGER NOT NULL DEFAULT 0"), ("phone", "TEXT"),
                          ("notified_next", "INTEGER NOT NULL DEFAULT 0")]:
            if name not in have:
                conn.execute(f"ALTER TABLE tickets ADD COLUMN {name} {ddl}")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tickets_queue ON tickets(service, status, is_deleted)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_notif_code ON notifications(ticket_code)")


# ---------- settings ----------

def get_settings():
    with connect() as conn:
        stored = {r["key"]: r["value"] for r in conn.execute("SELECT key, value FROM settings")}
    return {**DEFAULT_SETTINGS, **stored}


def set_setting(key, value):
    if key not in DEFAULT_SETTINGS:
        raise ValueError(f"Unknown setting: {key}")
    with connect() as conn:
        conn.execute("INSERT INTO settings(key, value) VALUES(?, ?) "
                     "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, str(value)))


# ---------- notifications ----------

def _record(conn, outbox, code, phone, message):
    conn.execute(
        "INSERT INTO notifications(ticket_code, channel, phone, message, created_at) VALUES(?,?,?,?,?)",
        (code, "sms" if phone else "screen", phone, message, _now()))
    outbox.append((phone, message))


def _flush(outbox):
    """Send real SMS only AFTER the database transaction has been committed."""
    for phone, message in outbox:
        if phone:
            notifier.send(phone, message)


def _promote_next(conn, outbox, service):
    """Tell the first waiting person in a service that they are next (once)."""
    row = conn.execute(
        f"SELECT id, ticket_code, phone, notified_next FROM tickets "
        f"WHERE service=? AND status='Waiting' AND is_deleted=0 ORDER BY {ORDER} LIMIT 1", (service,)).fetchone()
    if row and not row["notified_next"]:
        conn.execute("UPDATE tickets SET notified_next=1 WHERE id=?", (row["id"],))
        _record(conn, outbox, row["ticket_code"], row["phone"],
                f"{row['ticket_code']}: you're next for {service}. Please make your way to the service area.")


def recent_notifications(limit=100):
    with connect() as conn:
        return conn.execute("SELECT * FROM notifications ORDER BY id DESC LIMIT ?", (limit,)).fetchall()


def ticket_notifications(code, limit=5):
    with connect() as conn:
        return conn.execute("SELECT * FROM notifications WHERE ticket_code=? ORDER BY id DESC LIMIT ?",
                            (code.upper().strip(), limit)).fetchall()


# ---------- tickets ----------

def create_ticket(name, service, priority=False, phone=None, enforce=True):
    name = (name or "").strip()
    if not name:
        raise ValueError("Customer name cannot be empty.")
    if len(name) > 60:
        raise ValueError("Name is too long (60 characters max).")
    phone = (phone or "").strip() or None
    if phone and (len(phone) > 20 or not all(c.isdigit() or c in "+- ()" for c in phone)):
        raise ValueError("Enter a valid phone number (digits, +, spaces and dashes only).")

    if enforce:
        cfg = get_settings()
        if cfg["queue_open"] != "1":
            raise QueueClosed("The queue is temporarily paused. Please ask at the service desk.")
        cap = int(cfg["max_waiting"] or 0)
        if cap and len(list_tickets(service=service, statuses=["Waiting"])) >= cap:
            raise QueueClosed(f"The {service} queue is full right now. Please try again shortly.")

    outbox = []
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO tickets(ticket_code, customer_name, service, status, created_at, priority, phone) "
            "VALUES(?,?,?,?,?,?,?)",
            (f"NEW-{uuid.uuid4().hex}", name, service, "Waiting", _now(), 1 if priority else 0, phone)
        )
        ticket_id = cur.lastrowid
        code = f"WW-{ticket_id:04d}"
        conn.execute("UPDATE tickets SET ticket_code=? WHERE id=?", (code, ticket_id))
        _record(conn, outbox, code, phone,
                f"{code}: you're in the {service} queue" + (" (priority)." if priority else "."))
    _flush(outbox)
    return code


def find_active_ticket(name, service):
    """Used to warn about accidental double sign-ups by the same person."""
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM tickets WHERE lower(customer_name)=lower(?) AND service=? "
            "AND status IN ('Waiting','Serving') AND is_deleted=0 LIMIT 1", ((name or "").strip(), service)).fetchone()


def get_ticket(code):
    with connect() as conn:
        return conn.execute("SELECT * FROM tickets WHERE ticket_code=?", ((code or "").upper().strip(),)).fetchone()


def list_tickets(service=None, statuses=None, limit=1000, include_deleted=False):
    query = "SELECT * FROM tickets"
    clauses, params = [], []
    if not include_deleted:
        clauses.append("is_deleted=0")
    if service and service != "All":
        clauses.append("service=?")
        params.append(service)
    if statuses:
        placeholders = ",".join("?" for _ in statuses)
        clauses.append(f"status IN ({placeholders})")
        params.extend(statuses)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    # Waiting lines are shown in the order they will really be served (priority first).
    query += f" ORDER BY {ORDER} LIMIT ?" if statuses and set(statuses) == {"Waiting"} else " ORDER BY id ASC LIMIT ?"
    params.append(limit)
    with connect() as conn:
        return conn.execute(query, params).fetchall()


def position_info(ticket):
    """How many people are ahead of a ticket, and how loaded the service is.

    ahead   - people shown to the customer (waiting ahead + currently being served)
    load    - what the wait estimate uses (a person already being served counts as half)
    counters- distinct counters currently serving this service (at least 1)
    """
    with connect() as conn:
        waiting_ahead = conn.execute(
            "SELECT COUNT(*) FROM tickets WHERE service=? AND status='Waiting' AND is_deleted=0 "
            "AND (priority > ? OR (priority = ? AND id < ?))",
            (ticket["service"], ticket["priority"], ticket["priority"], ticket["id"])).fetchone()[0]
        serving = conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT counter) FROM tickets "
            "WHERE service=? AND status='Serving' AND is_deleted=0", (ticket["service"],)).fetchone()
    return {"ahead": waiting_ahead + serving[0], "waiting_ahead": waiting_ahead,
            "load": waiting_ahead + 0.5 * serving[0], "counters": max(1, serving[1])}


def call_next(service, counter=None):
    outbox = []
    with connect() as conn:
        conn.execute("BEGIN IMMEDIATE")  # two counters can never grab the same ticket
        row = conn.execute(
            f"SELECT * FROM tickets WHERE service=? AND status='Waiting' AND is_deleted=0 ORDER BY {ORDER} LIMIT 1",
            (service,)).fetchone()
        if not row:
            return None
        conn.execute("UPDATE tickets SET status='Serving', called_at=?, counter=? WHERE id=?",
                     (_now(), counter, row["id"]))
        _record(conn, outbox, row["ticket_code"], row["phone"],
                f"{row['ticket_code']}: it's your turn! Please go to {counter or 'the service desk'}.")
        _promote_next(conn, outbox, service)
    _flush(outbox)
    return row["ticket_code"]


def recall_ticket(code):
    """Re-announce a ticket that is being served but whose customer hasn't shown up."""
    outbox = []
    with connect() as conn:
        row = conn.execute("SELECT * FROM tickets WHERE ticket_code=? AND status='Serving'",
                           ((code or "").upper().strip(),)).fetchone()
        if not row:
            return False
        _record(conn, outbox, row["ticket_code"], row["phone"],
                f"Reminder: {row['ticket_code']} is being called at {row['counter'] or 'the service desk'}.")
    _flush(outbox)
    return True


def complete_ticket(code):
    now = datetime.now()
    outbox = []
    with connect() as conn:
        row = conn.execute("SELECT * FROM tickets WHERE ticket_code=?", ((code or "").upper().strip(),)).fetchone()
        if not row or row["status"] != "Serving":
            return False
        start = datetime.fromisoformat(row["called_at"]) if row["called_at"] else now
        minutes = max(0.1, (now - start).total_seconds() / 60)
        conn.execute(
            "UPDATE tickets SET status='Completed', completed_at=?, service_minutes=? WHERE ticket_code=?",
            (now.isoformat(timespec="seconds"), minutes, row["ticket_code"])
        )
        _record(conn, outbox, row["ticket_code"], row["phone"],
                f"{row['ticket_code']}: thank you! We'd love your feedback on how it went.")
        _promote_next(conn, outbox, row["service"])
    _flush(outbox)
    return True


def mark_no_show(code):
    """Customer was called but never came. Frees the counter. NOT used to train the wait model."""
    outbox = []
    with connect() as conn:
        row = conn.execute("SELECT * FROM tickets WHERE ticket_code=? AND status='Serving'",
                           ((code or "").upper().strip(),)).fetchone()
        if not row:
            return False
        conn.execute("UPDATE tickets SET status='No-show', completed_at=?, updated_at=? WHERE id=?",
                     (_now(), _now(), row["id"]))
        _record(conn, outbox, row["ticket_code"], row["phone"],
                f"{row['ticket_code']} was called but not answered, so it was closed. Please take a new ticket.")
        _promote_next(conn, outbox, row["service"])
    _flush(outbox)
    return True


def transfer_ticket(code, new_service):
    """Move a waiting/serving customer to another service queue (keeps priority)."""
    with connect() as conn:
        row = conn.execute("SELECT * FROM tickets WHERE ticket_code=? AND status IN ('Waiting','Serving')",
                           ((code or "").upper().strip(),)).fetchone()
        if not row or row["service"] == new_service:
            return False
        conn.execute(
            "UPDATE tickets SET service=?, status='Waiting', called_at=NULL, counter=NULL, notified_next=0, "
            "updated_at=? WHERE id=?", (new_service, _now(), row["id"]))
    outbox = []
    with connect() as conn:
        _promote_next(conn, outbox, row["service"])
        _promote_next(conn, outbox, new_service)
    _flush(outbox)
    return True


def cancel_ticket(code):
    with connect() as conn:
        row = conn.execute("SELECT service FROM tickets WHERE ticket_code=? AND status='Waiting' AND is_deleted=0",
                           ((code or "").upper().strip(),)).fetchone()
        if not row:
            return False
        conn.execute("UPDATE tickets SET status='Cancelled', updated_at=? WHERE ticket_code=?",
                     (_now(), (code or "").upper().strip()))
    outbox = []
    with connect() as conn:
        _promote_next(conn, outbox, row["service"])
    _flush(outbox)
    return True


# ---------- feedback ----------

def has_feedback(code):
    with connect() as conn:
        return conn.execute("SELECT 1 FROM feedback WHERE ticket_code=?", ((code or "").upper().strip(),)).fetchone() is not None


def add_feedback(code, rating, comment):
    """One review per ticket. Returns False if this ticket already left feedback."""
    code = (code or "").upper().strip()
    if has_feedback(code):
        return False
    rating = min(5, max(1, int(rating)))
    with connect() as conn:
        conn.execute(
            "INSERT INTO feedback(ticket_code,rating,comment,created_at) VALUES(?,?,?,?)",
            (code, rating, (comment or "").strip()[:1000], _now())
        )
    return True


def all_feedback():
    with connect() as conn:
        return conn.execute(
            "SELECT f.id, f.ticket_code, t.service, f.rating, f.comment, f.created_at "
            "FROM feedback f LEFT JOIN tickets t ON t.ticket_code = f.ticket_code "
            "ORDER BY f.id DESC LIMIT 200").fetchall()


# ---------- stats / data for AI ----------

def get_stats():
    with connect() as conn:
        def n(where):
            return conn.execute(f"SELECT COUNT(*) FROM tickets WHERE {where}").fetchone()[0]
        return {"total": n("is_deleted=0"), "waiting": n("is_deleted=0 AND status='Waiting'"),
                "serving": n("is_deleted=0 AND status='Serving'"),
                "completed": n("is_deleted=0 AND status='Completed'"),
                "no_show": n("is_deleted=0 AND status='No-show'"), "deleted": n("is_deleted=1")}


def completed_durations():
    with connect() as conn:
        return conn.execute(
            "SELECT service, service_minutes, called_at FROM tickets "
            "WHERE is_deleted=0 AND status='Completed' AND service_minutes IS NOT NULL"
        ).fetchall()


# ---------- manager tools ----------
# NOTE: there is intentionally NO "DELETE FROM" on tickets anywhere in this project.
# "Deleting" only sets is_deleted=1; the row stays in the database and the admin sees it flagged.

def set_deleted(ticket_id, deleted, by):
    now = _now()
    with connect() as conn:
        conn.execute(
            "UPDATE tickets SET is_deleted=?, deleted_at=?, deleted_by=?, updated_at=? WHERE id=?",
            (1 if deleted else 0, now if deleted else None, by if deleted else None, now, ticket_id))


def update_ticket(ticket_id, fields):
    allowed = {"customer_name", "service", "status", "counter", "service_minutes", "priority", "phone"}
    fields = {k: v for k, v in fields.items() if k in allowed}
    if not fields:
        return
    if "customer_name" in fields and not str(fields["customer_name"] or "").strip():
        raise ValueError("Customer name cannot be empty.")
    if "status" in fields and fields["status"] not in STATUSES:
        raise ValueError(f"Status must be one of {STATUSES}.")
    if fields.get("service_minutes") is not None:
        fields["service_minutes"] = float(fields["service_minutes"])
    if "priority" in fields:
        fields["priority"] = 1 if fields["priority"] else 0
    fields["updated_at"] = _now()
    sets = ", ".join(f"{k}=?" for k in fields)
    with connect() as conn:
        conn.execute(f"UPDATE tickets SET {sets} WHERE id=?", (*fields.values(), ticket_id))


def add_ticket(name, service, status="Waiting", counter=None, priority=False):
    code = create_ticket(name, service, priority=priority, enforce=False)
    if status != "Waiting" or counter:
        with connect() as conn:
            row = conn.execute("SELECT id FROM tickets WHERE ticket_code=?", (code,)).fetchone()
        update_ticket(row["id"], {"status": status, "counter": counter})
    return code



# ---------- staff accounts (customers never need one) ----------

def _hash(password, salt_hex):
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), 200_000).hex()


def create_user(username, password, role, min_len=6):
    username = (username or "").strip().lower()
    if not re.fullmatch(r"[a-z0-9._-]{3,30}", username):
        raise ValueError("Username: 3-30 letters, digits, dot, dash or underscore.")
    if role not in ROLES:
        raise ValueError(f"Role must be one of {ROLES}.")
    if len(password or "") < min_len:
        raise ValueError(f"Password must be at least {min_len} characters.")
    salt = os.urandom(16).hex()
    try:
        with connect() as conn:
            conn.execute("INSERT INTO users(username, password_hash, salt, role, created_at) VALUES(?,?,?,?,?)",
                         (username, _hash(password, salt), salt, role, _now()))
    except sqlite3.IntegrityError:
        raise ValueError("That username already exists.")


def _seed_users():
    """First run only: create one admin and one manager (ADMIN_PIN / MANAGER_PIN env vars or the defaults)."""
    with connect() as conn:
        if conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]:
            return
    for role, default in DEFAULT_LOGINS.items():
        create_user(role, os.getenv(f"{role.upper()}_PIN", default), role, min_len=4)


def verify_user(username, password):
    """Returns the role if the login is valid and the account is active, else None."""
    with connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE username=? AND is_active=1",
                           ((username or "").strip().lower(),)).fetchone()
    if not row:
        _hash(password or "", "00" * 16)  # keep timing similar whether or not the user exists
        return None
    ok = hmac.compare_digest(_hash(password or "", row["salt"]), row["password_hash"])
    return row["role"] if ok else None


def user_role(username):
    """Current role of an ACTIVE account (used to drop sessions of disabled accounts)."""
    with connect() as conn:
        row = conn.execute("SELECT role FROM users WHERE username=? AND is_active=1", (username,)).fetchone()
    return row["role"] if row else None


def list_users():
    with connect() as conn:
        return conn.execute("SELECT id, username, role, is_active, created_at FROM users ORDER BY id").fetchall()


def set_password(username, password):
    if len(password or "") < 6:
        raise ValueError("Password must be at least 6 characters.")
    salt = os.urandom(16).hex()
    with connect() as conn:
        cur = conn.execute("UPDATE users SET password_hash=?, salt=? WHERE username=?",
                           (_hash(password, salt), salt, username))
        if cur.rowcount == 0:
            raise ValueError("Unknown account.")


def set_user_active(username, active):
    with connect() as conn:
        if not active:
            row = conn.execute("SELECT role FROM users WHERE username=?", (username,)).fetchone()
            others = conn.execute("SELECT COUNT(*) FROM users WHERE role='manager' AND is_active=1 AND username<>?",
                                  (username,)).fetchone()[0]
            if row and row["role"] == "manager" and others == 0:
                raise ValueError("You can't disable the last active manager.")
        conn.execute("UPDATE users SET is_active=? WHERE username=?", (1 if active else 0, username))


def defaults_in_use():
    """True while any account still has its first-run default password."""
    return any(verify_user(u, p) for u, p in DEFAULT_LOGINS.items())