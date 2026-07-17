import asyncio
import unittest

from backend.api import run_ingestion_background


class IngestionTaskTests(unittest.TestCase):
    def test_run_ingestion_background_is_async(self):
        self.assertTrue(asyncio.iscoroutinefunction(run_ingestion_background))


if __name__ == "__main__":
    unittest.main()
