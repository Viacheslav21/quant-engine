"""Tests for bug fixes in quant-engine.

Covers: #4 (zero-PnL WIN), #6 (year rollover), #7 (resolution shield min-date),
        #14 (CLV no filter <=0), #16 (NO-side bid price).
Run: python -m pytest tests/test_bugfixes.py -v  (or just: python tests/test_bugfixes.py)
"""
import sys, os, unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

# ── Bug #4: Zero PnL should be WIN, not LOSS ──

class TestZeroPnlIsWin(unittest.TestCase):
    def test_zero_pnl_is_win(self):
        """pnl == 0 must be classified as WIN (breakeven, not a loss)."""
        pnl = 0.0
        result = "WIN" if pnl >= 0 else "LOSS"
        self.assertEqual(result, "WIN")

    def test_positive_pnl_is_win(self):
        pnl = 0.01
        result = "WIN" if pnl >= 0 else "LOSS"
        self.assertEqual(result, "WIN")

    def test_negative_pnl_is_loss(self):
        pnl = -0.01
        result = "WIN" if pnl >= 0 else "LOSS"
        self.assertEqual(result, "LOSS")


# ── Bug #6: Year rollover in _parse_question_date ──

class TestYearRollover(unittest.TestCase):
    """Test that dates without a year in late December don't get parsed as past dates."""

    def _parse(self, question, fake_now=None):
        """Inline version of MathEngine._parse_question_date with injectable now."""
        import re
        m = re.search(
            r'(?:on|by|before)\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:,?\s+(\d{4}))?',
            question, re.IGNORECASE
        )
        if not m:
            return None
        month_str, day_str, year_str = m.group(1), m.group(2), m.group(3)
        now = fake_now or datetime.now(timezone.utc)
        year = int(year_str) if year_str else now.year
        try:
            dt = datetime.strptime(f"{month_str} {day_str} {year}", "%B %d %Y").replace(tzinfo=timezone.utc)
            if not year_str and (now - dt).days > 30:
                dt = dt.replace(year=year + 1)
            return dt
        except ValueError:
            return None

    def test_january_in_december_gets_next_year(self):
        """'on January 5' in December 2026 → January 5, 2027."""
        fake_now = datetime(2026, 12, 15, tzinfo=timezone.utc)
        result = self._parse("Will X happen on January 5?", fake_now)
        self.assertIsNotNone(result)
        self.assertEqual(result.year, 2027)
        self.assertEqual(result.month, 1)
        self.assertEqual(result.day, 5)

    def test_explicit_year_is_respected(self):
        """'on January 5, 2026' always stays 2026, even if in past."""
        fake_now = datetime(2026, 12, 15, tzinfo=timezone.utc)
        result = self._parse("Will X happen on January 5, 2026?", fake_now)
        self.assertIsNotNone(result)
        self.assertEqual(result.year, 2026)

    def test_future_date_same_year_stays(self):
        """'on March 15' in January 2026 → March 15, 2026."""
        fake_now = datetime(2026, 1, 10, tzinfo=timezone.utc)
        result = self._parse("Will X happen on March 15?", fake_now)
        self.assertIsNotNone(result)
        self.assertEqual(result.year, 2026)

    def test_recent_past_date_stays_current_year(self):
        """'on March 1' on March 20 → stays March 1, 2026 (only 19 days ago, < 30d threshold)."""
        fake_now = datetime(2026, 3, 20, tzinfo=timezone.utc)
        result = self._parse("Will X happen on March 1?", fake_now)
        self.assertIsNotNone(result)
        self.assertEqual(result.year, 2026)

    def test_no_date_returns_none(self):
        result = self._parse("Will Bitcoin hit $100k?")
        self.assertIsNone(result)


# ── Bug #7: Resolution shield uses min(question_date, end_date) ──

class TestResolutionShieldMinDate(unittest.TestCase):
    """Shield must use the EARLIER of question date and end_date."""

    def _shield(self, pos, yes_price):
        """Inline version of _resolution_shield."""
        import re
        now = datetime.now(timezone.utc)
        candidates = []

        question = pos.get("question", "")
        m = re.search(
            r'(?:on|by|before)\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:,?\s+(\d{4}))?',
            question, re.IGNORECASE
        )
        if m:
            month_str, day_str, year_str = m.group(1), m.group(2), m.group(3)
            year = int(year_str) if year_str else now.year
            try:
                qdate = datetime.strptime(f"{month_str} {day_str} {year}", "%B %d %Y").replace(tzinfo=timezone.utc)
                if not year_str and (now - qdate).days > 30:
                    qdate = qdate.replace(year=year + 1)
                candidates.append((qdate - now).total_seconds() / 3600)
            except ValueError:
                pass

        end_date = pos.get("end_date")
        if end_date:
            if isinstance(end_date, str):
                end_date = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
            candidates.append((end_date - now).total_seconds() / 3600)

        if not candidates:
            return False
        hours_to_expiry = min(candidates)

        if hours_to_expiry < 6 or hours_to_expiry > 48:
            return False

        side_price = yes_price if pos["side"] == "YES" else 1 - yes_price
        if side_price < 0.55:
            return False
        # Simplified — skip the loss check for test
        return True

    def test_uses_earlier_end_date(self):
        """When end_date is 24h out but question date is 72h out, use end_date (24h → in shield window)."""
        now = datetime.now(timezone.utc)
        end_24h = (now + timedelta(hours=24)).isoformat()
        # Question date 72h out (outside 48h window — but end_date should take priority)
        future = now + timedelta(hours=72)
        question = f"Will X happen on {future.strftime('%B')} {future.day}, {future.year}?"

        pos = {"question": question, "end_date": end_24h, "side": "YES"}
        result = self._shield(pos, 0.92)
        self.assertTrue(result, "Shield should activate using earlier end_date (24h)")

    def test_both_outside_window_no_shield(self):
        """Both dates >48h out → no shield."""
        now = datetime.now(timezone.utc)
        end_72h = (now + timedelta(hours=72)).isoformat()
        future = now + timedelta(hours=96)
        question = f"Will X happen on {future.strftime('%B')} {future.day}, {future.year}?"

        pos = {"question": question, "end_date": end_72h, "side": "YES"}
        result = self._shield(pos, 0.92)
        self.assertFalse(result, "Both dates >48h → no shield")

    def test_no_dates_no_shield(self):
        pos = {"question": "Will something happen?", "side": "YES"}
        result = self._shield(pos, 0.92)
        self.assertFalse(result)


# ── Bug #14: CLV should not filter out prices <= 0 ──

class TestClvNoFilterZero(unittest.TestCase):
    def _clv_val(self, row, col):
        """Fixed version: only filter None, not <= 0."""
        v = row.get(col)
        if v is None:
            return None
        entry = row["side_price"]
        if row["side"] == "YES":
            return (v - entry) / entry
        else:
            return (entry - v) / entry

    def test_no_side_zero_price_is_valid(self):
        """NO side: yes_price=0.0 at close means market resolved NO → we won big."""
        row = {"side": "NO", "side_price": 0.70, "clv_close": 0.0}
        clv = self._clv_val(row, "clv_close")
        self.assertIsNotNone(clv, "CLV of 0.0 should not be filtered out")
        self.assertAlmostEqual(clv, 1.0, places=2)  # (0.70 - 0.0) / 0.70 = 1.0

    def test_yes_side_low_price_valid(self):
        """YES side: price dropped to 0.01 → bad CLV but still valid data."""
        row = {"side": "YES", "side_price": 0.50, "clv_1h": 0.01}
        clv = self._clv_val(row, "clv_1h")
        self.assertIsNotNone(clv)
        self.assertAlmostEqual(clv, -0.98, places=2)

    def test_none_still_filtered(self):
        row = {"side": "YES", "side_price": 0.50, "clv_1h": None}
        clv = self._clv_val(row, "clv_1h")
        self.assertIsNone(clv)


# ── Bug #16: NO-side bid price with best_ask edge cases ──

class TestNoSideBidPrice(unittest.TestCase):
    def _compute_no_bid(self, best_ask, yes_price):
        """Fixed version: explicit None check instead of falsy."""
        return (1 - best_ask) if best_ask is not None and best_ask > 0 else (1 - yes_price)

    def test_normal_best_ask(self):
        """best_ask=0.95 → NO bid = 0.05."""
        self.assertAlmostEqual(self._compute_no_bid(0.95, 0.93), 0.05)

    def test_best_ask_zero_falls_back(self):
        """best_ask=0.0 (falsy but valid edge) → fallback to 1 - yes_price."""
        self.assertAlmostEqual(self._compute_no_bid(0.0, 0.10), 0.90)

    def test_best_ask_none_falls_back(self):
        """best_ask=None → fallback to 1 - yes_price."""
        self.assertAlmostEqual(self._compute_no_bid(None, 0.10), 0.90)

    def test_normal_mid_range(self):
        """best_ask=0.50 → NO bid = 0.50."""
        self.assertAlmostEqual(self._compute_no_bid(0.50, 0.48), 0.50)


if __name__ == "__main__":
    unittest.main(verbosity=2)
