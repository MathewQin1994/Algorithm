#!/usr/bin/python
# -*- coding: utf-8 -*-
from LinkedList import LNode


class QueueUnderflow(ValueError):
    pass


# Sequential table implementation
class SQueue:
    def __init__(self, init_len=8):
        self._len = init_len
        self._elems = [0]*init_len
        self._head = 0
        self._num = 0

    def is_empty(self):
        return self._num == 0

    def peek(self):
        if self._num == 0:
            raise QueueUnderflow("in SQueue.peek()")
        return self._elems[self._head]

    def dequeue(self):
        if self._num == 0:
            raise QueueUnderflow("in SQueue.dequeue()")
        elif self._num == int(self._len/4) and self._len > 8:
            old_len = self._len
            self._len = int(self._len / 2)
            new_elems = [0] * self._len
            j = 0
            e = self._elems[self._head]
            for i in range(self._head+1, self._head + self._num):
                new_elems[j] = self._elems[i % old_len]
                j += 1
            self._elems, self._head = new_elems, 0
        else:
            e = self._elems[self._head]
            self._head = (self._head+1) % self._len
        self._num -= 1
        return e

    def enqueue(self, e):
        if self._num == self._len:
            self.__extend()
        self._elems[(self._head + self._num) % self._len] = e
        self._num += 1

    def __extend(self):
        old_len = self._len
        self._len *= 2
        new_elems = [0]*self._len
        for i in range(old_len):
            new_elems[i] = self._elems[(self._head + i) % old_len]
        self._elems, self._head = new_elems, 0


# Linked list implementation
class LQueue:
    def __init__(self):
        self._first = None
        self._last = None

    def is_empty(self):
        return self._first == None

    def peek(self):
        if self._first == None:
            raise QueueUnderflow("in LQueue.peek()")
        return self._first.elem

    def dequeue(self):
        if self._first == None:
            raise QueueUnderflow("in LQueue.peek()")
        e = self._first.elem
        self._first = self._first.next
        if self._first ==None:
            self._last = self._first
        return e

    def enqueue(self, e):
        if self._first == None:
            self._first = LNode(e)
            self._last = self._first
        else:
            new_last = LNode(e)
            self._last.next = new_last
            self._last = new_last


def testSQueue():
    SQ=SQueue()
    for i in range(13):
        SQ.enqueue(i)
        print(SQ._elems)
    for i in range(9):
        SQ.dequeue()
        SQ.enqueue(i+13)
        print(SQ._elems)
    while not SQ.is_empty():
        SQ.dequeue()
        print(SQ._head)

def testLQueue():
    LQ=LQueue()
    for i in range(13):
        LQ.enqueue(i)
        print(LQ._first.elem,LQ._last.elem)

    for i in range(13):
        print(LQ._first.elem,LQ._last.elem)
        LQ.dequeue()


if __name__ == "__main__":
    testSQueue()