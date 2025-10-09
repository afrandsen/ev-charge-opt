from dotenv import load_dotenv
load_dotenv('.env.local')

import sys
import os
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email_notification(subject: str, body: str, sender: str, recipient: str, smtp_server: str, smtp_port: int, username: str, password: str):
    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()  # secure connection
            server.login(username, password)
            server.sendmail(sender, recipient, msg.as_string())
        log("📧 Email notification sent.")
    except Exception as e:
        log(f"⚠️ Failed to send email: {e}")

# --- Logging ---
log_lines = []

def log(msg):
    print(msg)
    log_lines.append(str(msg))

try:
    CHARGING = sys.argv[1] == "f"
except ValueError:
    log("CHARGING is not false!")
    sys.exit(1)

try:
    SOC_PCT = float(sys.argv[2])
    if SOC_PCT > 1:
        SOC_PCT /= 100.0
except ValueError:
    log("SOC is not a valid number!")
    sys.exit(1)

# --- Notification Logic ---
STATE_FILE = os.path.expanduser("~/repos/ev-charge-opt/tmp/ev_charging_state.json")

def load_last_state():
    try:
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
            last_amp = state.get("last_amp", 0)
            target_soc = state.get("target_soc", 0)
            return last_amp, target_soc
    except FileNotFoundError:
        return 0

last_amp, target_soc = load_last_state()

if last_amp > 0 and SOC_PCT < target_soc:
    subject = "NOT CHARGING"

    body = "NOT CHARGING"

    send_email_notification(
        subject=subject,
        body=body,
        sender=os.getenv("EMAIL_SENDER"),
        recipient=os.getenv("EMAIL_RECIPIENT"),
        smtp_server=os.getenv("SMTP_SERVER"),
        smtp_port=int(os.getenv("SMTP_PORT", "587")),
        username=os.getenv("SMTP_USER"),
        password=os.getenv("SMTP_PASS"),
    )
else:
    sys.exit(1)