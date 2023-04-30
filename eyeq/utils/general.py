# SuperFastPython.com
# example of using the asyncio priority queue
from time import sleep
from random import random
from random import randint
import asyncio


# generate work
async def producer(queue):
    print('Producer: Running')
    # generate work
    for i in range(10):
        # generate a value
        value = random()
        # generate a priority
        priority = randint(0, 10)
        # create an item
        item = (priority, value)
        # add to the queue
        print(f'>put {item}')
        await queue.put(item)
    # wait for all items to be processed
    await queue.join()
    # send sentinel value
    await queue.put(None)
    print('Producer: Done')


# consume work
async def consumer(queue):
    print('Consumer: Running')
    # consume work
    while True:
        # get a unit of work
        item = await queue.get()
        # check for stop
        if item is None:
            break
        # block
        await asyncio.sleep(item[1])
        # report
        print(f'>got {item}')
        # mark it as processed
        queue.task_done()
    # all done
    print('Consumer: Done')


# entry point coroutine
async def main():
    # create the shared queue
    queue = asyncio.PriorityQueue()
    # run the producer and consumers
    await asyncio.gather(producer(queue), consumer(queue))


# start the asyncio program
asyncio.run(main())