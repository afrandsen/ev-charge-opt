import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def send_email_notification(
    subject: str,
    body: str,
    sender: str,
    recipient: str,
    smtp_server: str,
    smtp_port: int,
    username: str,
    password: str,
    log,
) -> None:
    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(username, password)
            server.sendmail(sender, recipient, msg.as_string())
        log("📧 Email notification sent.")
    except Exception as e:
        log(f"⚠️ Failed to send email: {e}")
