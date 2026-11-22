"""SQLite storage for WaitWise tickets and feedback."""
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "waitwise.db"


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


STATUSES = ["Waiting", "Serving", "Completed", "Cancelled"]


def init_db():
    _ensure_valid_db()
    _create_tables()
    _migrate()


def _migrate():
    """Add the soft-delete + counter columns to databases created by older versions."""
    with connect() as conn:
        have = {r["name"] for r in conn.execute("PRAGMA table_info(tickets)")}
        for name, ddl in [("counter", "TEXT"), ("is_deleted", "INTEGER NOT NULL DEFAULT 0"),
                          ("deleted_at", "TEXT"), ("deleted_by", "TEXT"), ("updated_at", "TEXT")]:
            if name not in have:
                conn.execute(f"ALTER TABLE tickets ADD COLUMN {name} {ddl}")


def _create_tables():
    with connect() as conn:
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

def create_ticket(name, service):
    now = datetime.now().isoformat(timespec="seconds")
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO tickets(ticket_code, customer_name, service, status, created_at) VALUES(?,?,?,?,?)",
            (f"NEW-{uuid.uuid4().hex}", name.strip(), service, "Waiting", now)
        )
        ticket_id = cur.lastrowid
        code = f"WW-{ticket_id:04d}"
        conn.execute("UPDATE tickets SET ticket_code=? WHERE id=?", (code, ticket_id))
    return code

def get_ticket(code):
    with connect() as conn:
        return conn.execute("SELECT * FROM tickets WHERE ticket_code=?", (code.upper().strip(),)).fetchone()

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
    query += " ORDER BY id ASC LIMIT ?"
    params.append(limit)
    with connect() as conn:
        return conn.execute(query, params).fetchall()

def call_next(service, counter=None):
    now = datetime.now().isoformat(timespec="seconds")
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM tickets WHERE service=? AND status='Waiting' AND is_deleted=0 ORDER BY id ASC LIMIT 1",
            (service,)
        ).fetchone()
        if row:
            conn.execute("UPDATE tickets SET status='Serving', called_at=?, counter=? WHERE id=?", (now, counter, row["id"]))
            return row["ticket_code"]
    return None

def complete_ticket(code):
    now = datetime.now()
    with connect() as conn:
        row = conn.execute("SELECT * FROM tickets WHERE ticket_code=?", (code.upper().strip(),)).fetchone()
        if not row or row["status"] != "Serving":
            return False
        start = datetime.fromisoformat(row["called_at"]) if row["called_at"] else now
        minutes = max(0.1, (now - start).total_seconds() / 60)
        conn.execute(
            "UPDATE tickets SET status='Completed', completed_at=?, service_minutes=? WHERE ticket_code=?",
            (now.isoformat(timespec="seconds"), minutes, row["ticket_code"])
        )
    return True

def cancel_ticket(code):
    with connect() as conn:
        cur = conn.execute(
            "UPDATE tickets SET status='Cancelled' WHERE ticket_code=? AND status='Waiting'",
            (code.upper().strip(),)
        )
        return cur.rowcount > 0

def add_feedback(code, rating, comment):
    with connect() as conn:
        conn.execute(
            "INSERT INTO feedback(ticket_code,rating,comment,created_at) VALUES(?,?,?,?)",
            (code, int(rating), comment.strip(), datetime.now().isoformat(timespec="seconds"))
        )

def get_stats():
    with connect() as conn:
        total = conn.execute("SELECT COUNT(*) FROM tickets WHERE is_deleted=0").fetchone()[0]
        waiting = conn.execute("SELECT COUNT(*) FROM tickets WHERE is_deleted=0 AND status='Waiting'").fetchone()[0]
        serving = conn.execute("SELECT COUNT(*) FROM tickets WHERE is_deleted=0 AND status='Serving'").fetchone()[0]
        completed = conn.execute("SELECT COUNT(*) FROM tickets WHERE is_deleted=0 AND status='Completed'").fetchone()[0]
        deleted = conn.execute("SELECT COUNT(*) FROM tickets WHERE is_deleted=1").fetchone()[0]
        return {"total": total, "waiting": waiting, "serving": serving, "completed": completed, "deleted": deleted}

def completed_durations():
    with connect() as conn:
        return conn.execute(
            "SELECT service, service_minutes FROM tickets WHERE is_deleted=0 AND status='Completed' AND service_minutes IS NOT NULL"
        ).fetchall()

def all_feedback():
    with connect() as conn:
        return conn.execute("SELECT * FROM feedback ORDER BY id DESC LIMIT 100").fetchall()


# ---------- manager tools ----------
# NOTE: there is intentionally NO "DELETE FROM" anywhere in this project.
# "Deleting" only sets is_deleted=1; the row stays in the database and the admin sees it flagged.

def set_deleted(ticket_id, deleted, by):
    now = datetime.now().isoformat(timespec="seconds")
    with connect() as conn:
        conn.execute(
            "UPDATE tickets SET is_deleted=?, deleted_at=?, deleted_by=?, updated_at=? WHERE id=?",
            (1 if deleted else 0, now if deleted else None, by if deleted else None, now, ticket_id))


def update_ticket(ticket_id, fields):
    allowed = {"customer_name", "service", "status", "counter", "service_minutes"}
    fields = {k: v for k, v in fields.items() if k in allowed}
    if not fields:
        return
    if "customer_name" in fields and not str(fields["customer_name"] or "").strip():
        raise ValueError("Customer name cannot be empty.")
    if "status" in fields and fields["status"] not in STATUSES:
        raise ValueError(f"Status must be one of {STATUSES}.")
    if fields.get("service_minutes") is not None:
        fields["service_minutes"] = float(fields["service_minutes"])
    fields["updated_at"] = datetime.now().isoformat(timespec="seconds")
    sets = ", ".join(f"{k}=?" for k in fields)
    with connect() as conn:
        conn.execute(f"UPDATE tickets SET {sets} WHERE id=?", (*fields.values(), ticket_id))


def add_ticket(name, service, status="Waiting", counter=None):
    code = create_ticket(name, service)
    if status != "Waiting" or counter:
        with connect() as conn:
            row = conn.execute("SELECT id FROM tickets WHERE ticket_code=?", (code,)).fetchone()
        update_ticket(row["id"], {"status": status, "counter": counter})
    return code