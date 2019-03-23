#!/usr/bin/python
# -*- coding: utf-8 -*-


class LinkedListUnderflow(ValueError):
    pass


class LNode:
    def __init__(self, elem, next_=None):
        self.elem = elem
        self.next = next_


class LList:
    def __init__(self):
        self._head = None

    def is_empty(self):
        return self._head is None

    def prepend(self, elem):
        self._head = LNode(elem, self._head)

    def pop(self):
        if self._head is None:
            raise LinkedListUnderflow('in pop()')
        e = self._head.elem
        self._head = self._head.next
        return e

    def append(self, elem):
        if self._head is None:
            self._head = LNode(elem)
            return
        p = self._head
        while p.next is not None:
            p = p.next
        p.next = LNode(elem)

    def pop_last(self):
        if self._head is None:
            raise LinkedListUnderflow("in pop_last()")
        p = self._head
        if p.next is None:
            self._head = None
            return p.elem
        while p.next.next is not None:
            p = p.next
        e = p.next.elem
        p.next = None
        return e

    def find(self, pred):
        p = self._head
        while p is not None:
            if pred(p.elem):
                return p.elem
            p = p.next

    def elements(self):
        p = self._head
        while p is not None:
            yield p.elem
            p = p.next

    def filter(self, pred):
        p = self._head
        while p is not None:
            if pred(p.elem):
                yield p.elem
            p = p.next

    def __iter__(self):
        p = self._head
        print('iter')
        while p is not None:
            yield p.elem
            p = p.next

    def __contains__(self,item):
        p=self._head
        print('contains')
        while p is not None:
            if p.elem==item:
                return True
            p=p.next
        return False


if __name__ == "__main__":
    ll1=LList()
    for i in range(10):
        ll1.prepend(i)

    for i in ll1:
        print(i)

    print(5 in ll1)