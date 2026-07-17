import asyncio
import unittest

from core.handlers.identity_handler import handle_identity_query
from core.handlers.seat_handler import handle_seat_query


class HandlerStreamingTests(unittest.TestCase):
    def test_identity_handler_supports_async_iteration(self):
        async def collect_chunks():
            return [chunk async for chunk in handle_identity_query("who are you")]

        chunks = asyncio.run(collect_chunks())
        self.assertTrue(chunks)
        self.assertIn("Prospectus AI", chunks[0])

    def test_seat_handler_supports_async_iteration(self):
        async def collect_chunks():
            return [chunk async for chunk in handle_seat_query("give me seat distribution", "SEAT")]

        chunks = asyncio.run(collect_chunks())
        self.assertTrue(chunks)
        self.assertTrue(any("Undergraduate" in chunk for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
