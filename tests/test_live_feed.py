import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from live_feed import get_first_valid_price


class GetFirstValidPriceTests(unittest.TestCase):
    def test_uses_first_provider_and_does_not_call_backup(self):
        primary = Mock(return_value="101.25")
        backup = Mock()

        price, provider, errors = get_first_valid_price([
            ("Kraken", primary),
            ("Binance", backup),
        ])

        self.assertEqual(price, 101.25)
        self.assertEqual(provider, "Kraken")
        self.assertEqual(errors, [])
        backup.assert_not_called()

    def test_uses_backup_and_returns_primary_error(self):
        primary = Mock(side_effect=TimeoutError("request timed out"))
        backup = Mock(return_value=99.5)

        price, provider, errors = get_first_valid_price([
            ("Kraken", primary),
            ("Binance", backup),
        ])

        self.assertEqual(price, 99.5)
        self.assertEqual(provider, "Binance")
        self.assertEqual(errors, ["Kraken: TimeoutError: request timed out"])
        backup.assert_called_once_with()

    def test_preserves_http_status_for_fallback_diagnostics(self):
        failure = RuntimeError("request failed")
        failure.response = SimpleNamespace(status_code=451, reason="Unavailable")
        primary = Mock(side_effect=failure)
        backup = Mock(return_value=99.5)

        price, provider, errors = get_first_valid_price([
            ("Kraken", primary),
            ("Binance", backup),
        ])

        self.assertEqual(price, 99.5)
        self.assertEqual(provider, "Binance")
        self.assertEqual(errors, ["Kraken: HTTP 451 Unavailable"])

    def test_rejects_non_positive_and_non_finite_prices(self):
        for invalid_price in (0, -1, math.nan, math.inf):
            with self.subTest(invalid_price=invalid_price):
                first = Mock(return_value=invalid_price)
                backup = Mock(return_value=12.0)

                price, provider, errors = get_first_valid_price([
                    ("Kraken", first),
                    ("Binance", backup),
                ])

                self.assertEqual(price, 12.0)
                self.assertEqual(provider, "Binance")
                self.assertIn("Kraken: ValueError", errors[0])

    def test_raises_with_provider_details_if_all_fail(self):
        primary = Mock(side_effect=TimeoutError("request timed out"))
        backup = Mock(return_value=math.nan)

        with self.assertRaisesRegex(RuntimeError, "Kraken: TimeoutError.*Binance: ValueError"):
            get_first_valid_price([
                ("Kraken", primary),
                ("Binance", backup),
            ])


if __name__ == "__main__":
    unittest.main()
