import asyncio

import logging.config
import random
from time import sleep

import websockets
from aiohttp import web

from spark_lp.myserver.ws_server import Server

log = logging.getLogger('producer')
msg_queue = asyncio.Queue()
server = Server(msg_queue)


async def producer():
    words = ['w1', 'w2', 'w3']
    while True:
        msg = random.choice(words)
        server.latest_version = msg
        if server.has_clients:
            await msg_queue.put(msg)
            print(f'Put msg {msg} to queque')
        else:
            print(f'Server has no clients. Msg {msg} has been saved.')
        await asyncio.sleep(random.randint(1, 10))


async def produce(host, port):
    while True:
        async with websockets.connect(f'ws://{host}:{port}') as ws:
            print('emm')
            await ws.send('sdfjhghjgj')
            print('sent')
            await asyncio.sleep(3)


if __name__ == "__main__":
    # app = web.Application()
    # loop = asyncio.get_event_loop()
    # start_server = websockets.serve(server.ws_handler,
    #                                 port=8080)
    # loop.run_until_complete(start_server)
    #
    # loop.run_until_complete(producer())

    data = 'abcdefg\n'
    to_be_sent = data.encode('utf-8')
    main_path = '/home/alex/PycharmProjects/spark_lp/texts'
    files = ['hamster0.txt', 'hamster1.txt', 'hamster2.txt', 'hamster3.txt',
             'opera.txt', 'opera1.txt']

    import socket

    # Create a socket with the socket() system call
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Bind the socket to an address using the bind() system call
    s.bind(('0.0.0.0', 8080))
    # Enable a server to accept connections.
    # It specifies the number of unaccepted connections
    # that the system will allow before refusing new connections
    print('listen')
    s.listen(1)
    # Accept a connection with the accept() system call
    conn, addr = s.accept()
    print('accept')
    # Send data. It continues to send data from bytes until either
    # all data has been sent or an error occurs. It returns None.
    try:
        i = 0
        while True:
            filename = files[i]
            i = (i + 1) % len(files)
            with open(f'{main_path}/{filename}') as f:
                text = f.read()
                text = text.replace('\n', ' ')
                text += '\n'
                conn.sendall(text.encode('utf-8'))
                # conn.sendall('ываправ\n'.encode('utf-8'))
                print('sent', filename)
            sleep(3)
    finally:
        conn.close()
