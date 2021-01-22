import asyncio
import logging
import time

from websockets.exceptions import ConnectionClosed

log = logging.getLogger(__name__)


class Server:
    clients = set()

    def __init__(self, msg_queue: asyncio.Queue):
        self.msg_queue = msg_queue
        self.start_timestamp = time.time()

    @property
    def has_clients(self):
        return bool(self.clients)

    async def register(self, ws):
        self.clients.add(ws)

    async def unregister(self, ws):
        self.clients.remove(ws)

    @classmethod
    async def send_to_client(cls, client, msg):
        try:
            await client.send(msg)
        except ConnectionClosed:
            cls.clients.remove(client)
        except Exception as e:
            log.error(e)
            cls.clients.remove(client)

    async def send_to_clients(self, msg):
        if self.clients:
            await asyncio.wait(
                [self.send_to_client(client, msg) for client in self.clients])

    async def ws_handler(self, ws, uri):
        print('wowowowowow')
        await self.register(ws)
        print('NEW CLIENT')
        try:
            await self.distribute()
        finally:
            await self.unregister(ws)

    async def distribute(self):
        await self.send_to_clients('hi')
        while True:
            msg = await self.msg_queue.get()
            if msg is not None:
                await self.send_to_clients(msg)
