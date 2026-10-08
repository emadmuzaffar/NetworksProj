from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from networks_proj.parsing import StructuredResponseError, parse_json_object


class ParsingTests(unittest.TestCase):
    def test_accepts_fenced_json(self) -> None:
        self.assertEqual(parse_json_object('```json\n{"warmth": 5}\n```'), {"warmth": 5})

    def test_accepts_short_leading_prose(self) -> None:
        self.assertEqual(parse_json_object('Result: {"competence": 4}'), {"competence": 4})

    def test_rejects_non_object(self) -> None:
        with self.assertRaises(StructuredResponseError):
            parse_json_object("[1, 2, 3]")


if __name__ == "__main__":
    unittest.main()
