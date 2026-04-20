import sqlite3
import json
import datetime
from .config import DB_PATH


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prescriptions (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                mode              TEXT    NOT NULL,
                analyzed_at       TEXT    NOT NULL,
                image_filename    TEXT,
                output_dir        TEXT,
                doctor_name       TEXT,
                doctor_quals      TEXT,
                doctor_specialty  TEXT,
                doctor_clinic     TEXT,
                doctor_address    TEXT,
                doctor_contact    TEXT,
                doctor_reg_no     TEXT,
                patient_name      TEXT,
                patient_age       TEXT,
                patient_gender    TEXT,
                prescription_date TEXT,
                languages         TEXT,
                diagnosis         TEXT,
                medications       TEXT,
                investigations    TEXT,
                summary           TEXT,
                confidence_notes  TEXT,
                raw_json          TEXT    NOT NULL
            )
        """)
        conn.commit()


def save_to_db(mode: str, image_filename: str, output_dir: str, data: dict) -> int:
    doc  = data.get("doctor")  or {}
    pat  = data.get("patient") or {}
    clin = data.get("clinical") or {}
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("""
            INSERT INTO prescriptions (
                mode, analyzed_at, image_filename, output_dir,
                doctor_name, doctor_quals, doctor_specialty, doctor_clinic,
                doctor_address, doctor_contact, doctor_reg_no,
                patient_name, patient_age, patient_gender, prescription_date,
                languages, diagnosis, medications, investigations,
                summary, confidence_notes, raw_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            mode,
            datetime.datetime.now().isoformat(timespec="seconds"),
            image_filename,
            output_dir,
            doc.get("name"),            doc.get("qualifications"), doc.get("specialty"),
            doc.get("clinic_hospital"), doc.get("address"),        doc.get("contact"),
            doc.get("registration_number"),
            pat.get("name"), pat.get("age"), pat.get("gender"), pat.get("date"),
            json.dumps(data.get("languages_detected") or [], ensure_ascii=False),
            json.dumps(clin.get("diagnosis")          or [], ensure_ascii=False),
            json.dumps(data.get("medications")        or [], ensure_ascii=False),
            json.dumps(data.get("investigations")     or [], ensure_ascii=False),
            data.get("summary"),
            data.get("confidence_notes"),
            json.dumps(data, indent=2, ensure_ascii=False),
        ))
        conn.commit()
        return cur.lastrowid


init_db()
