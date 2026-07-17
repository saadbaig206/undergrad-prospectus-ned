import os
import asyncio
from dotenv import load_dotenv
load_dotenv()
from core.chatbot import route_chat_stream

async def test():
    async for t in route_chat_stream('chairperson of civil engineering', []):
        print(t, end='')
    print()

asyncio.run(test())
