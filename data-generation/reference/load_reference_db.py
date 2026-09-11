"""
load_reference_db.py

Loads the CSV reference tables (built by build_reference_data.py) into a
local SQLite database. This database is the "ground-truth rule engine":
the synthetic bill generator and, later, the anomaly detection / RAG agent
both read from it rather than hard-coding rule logic inline.
"""

import csv
import os
import sqlite3

REF_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(REF_DIR, "..", "wc_bill_review.db")

SCHEMA = """
DROP TABLE IF EXISTS cpt_hcpcs_codes;
DROP TABLE IF EXISTS icd10_codes;
DROP TABLE IF EXISTS ncci_ptp_edits;
DROP TABLE IF EXISTS mue_table;

CREATE TABLE cpt_hcpcs_codes (
    code        TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    category    TEXT NOT NULL,
    unit_basis  TEXT NOT NULL
);

CREATE TABLE icd10_codes (
    code        TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    body_region TEXT NOT NULL
);

CREATE TABLE ncci_ptp_edits (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    column1_code       TEXT NOT NULL REFERENCES cpt_hcpcs_codes(code),
    column2_code       TEXT NOT NULL REFERENCES cpt_hcpcs_codes(code),
    modifier_indicator TEXT NOT NULL,
    rationale          TEXT NOT NULL,
    source             TEXT NOT NULL,
    UNIQUE(column1_code, column2_code)
);

CREATE TABLE mue_table (
    code       TEXT PRIMARY KEY REFERENCES cpt_hcpcs_codes(code),
    mue_units  INTEGER NOT NULL,
    rationale  TEXT NOT NULL
);
"""


def load_csv(cursor, csv_name, insert_sql):
    path = os.path.join(REF_DIR, csv_name)
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    cursor.executemany(insert_sql, rows)
    return len(rows)


def main():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.executescript(SCHEMA)

    n1 = load_csv(cur, "cpt_hcpcs_codes.csv", "INSERT INTO cpt_hcpcs_codes VALUES (?,?,?,?)")
    n2 = load_csv(cur, "icd10_codes.csv", "INSERT INTO icd10_codes VALUES (?,?,?)")
    n3 = load_csv(cur, "ncci_ptp_edits.csv", "INSERT INTO ncci_ptp_edits (column1_code, column2_code, modifier_indicator, rationale, source) VALUES (?,?,?,?,?)")
    n4 = load_csv(cur, "mue_table.csv", "INSERT INTO mue_table VALUES (?,?,?)")

    conn.commit()

    print(f"loaded {n1} cpt_hcpcs_codes, {n2} icd10_codes, {n3} ncci_ptp_edits, {n4} mue_table rows")
    print(f"db -> {os.path.abspath(DB_PATH)}")

    # sanity queries
    cur.execute("""
        SELECT e.column1_code, c1.description, e.column2_code, c2.description, e.modifier_indicator, e.source
        FROM ncci_ptp_edits e
        JOIN cpt_hcpcs_codes c1 ON c1.code = e.column1_code
        JOIN cpt_hcpcs_codes c2 ON c2.code = e.column2_code
        LIMIT 3
    """)
    print("\nsample NCCI PTP edits (joined):")
    for row in cur.fetchall():
        print(" ", row)

    conn.close()


if __name__ == "__main__":
    main()
