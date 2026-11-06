"""
Notification service for 0DTE Trading Bot.
Sends alerts via email and/or webhook on trade events and LLM disagreements.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class NotificationService:
    """Sends notifications on trade events, disagreements, and errors."""

    def __init__(
        self,
        email_config: Optional[Dict] = None,
        webhook_config: Optional[Dict] = None,
    ):
        self.email_config = email_config or {}
        self.webhook_config = webhook_config or {}

    async def notify_disagreement(
        self,
        disagreement_reason: str,
        primary_decision: Optional[Dict] = None,
        secondary_decision: Optional[Dict] = None,
    ):
        """Notify user that LLMs disagreed — trade blocked."""
        subject = "0DTE Bot: LLM Disagreement — Trade Blocked"
        body = (
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"Dual-LLM validation FAILED — no trade executed.\n\n"
            f"Reason: {disagreement_reason}\n\n"
            f"Primary decision: {json.dumps(primary_decision, indent=2, default=str)}\n\n"
            f"Secondary decision: {json.dumps(secondary_decision, indent=2, default=str)}"
        )
        await self._send(subject, body)

    async def notify_error(self, error_reason: str):
        """Notify user of a system error that blocked execution."""
        subject = "0DTE Bot: System Error — Trade Blocked"
        body = (
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"System error — no trade executed.\n\nReason: {error_reason}"
        )
        await self._send(subject, body)

    async def notify_trade_executed(self, trade_details: Dict[str, Any]):
        """Notify user that a trade was executed."""
        subject = "0DTE Bot: Trade Executed"
        body = (
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"Iron condor executed.\n\n"
            f"{json.dumps(trade_details, indent=2, default=str)}"
        )
        await self._send(subject, body)

    async def notify_trade_closed(self, trade_details: Dict[str, Any]):
        """Notify user that a trade was closed."""
        subject = "0DTE Bot: Trade Closed"
        body = (
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"Position closed.\n\n"
            f"{json.dumps(trade_details, indent=2, default=str)}"
        )
        await self._send(subject, body)

    async def _send(self, subject: str, body: str):
        """Send notification via configured channels."""
        logger.info("NOTIFICATION — %s", subject)
        logger.debug(body)

        if self.email_config.get("enabled"):
            await self._send_email(subject, body)

        if self.webhook_config.get("enabled"):
            await self._send_webhook(subject, body)

    async def _send_email(self, subject: str, body: str):
        try:
            import smtplib
            from email.mime.text import MIMEText

            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"] = self.email_config["sender"]
            msg["To"] = ", ".join(self.email_config["recipients"])

            with smtplib.SMTP(
                self.email_config["smtp_server"],
                self.email_config.get("smtp_port", 587),
            ) as smtp:
                smtp.starttls()
                if self.email_config.get("password"):
                    smtp.login(self.email_config["sender"], self.email_config["password"])
                smtp.send_message(msg)
        except Exception:
            logger.exception("Failed to send email notification")

    async def _send_webhook(self, subject: str, body: str):
        try:
            import aiohttp

            payload = {"subject": subject, "body": body, "timestamp": datetime.now().isoformat()}
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_config["url"],
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status >= 400:
                        logger.error("Webhook returned status %d", resp.status)
        except Exception:
            logger.exception("Failed to send webhook notification")
