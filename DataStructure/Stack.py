#!/usr/bin/python
# -*- coding: utf-8 -*-
from LinkedList import LNode


class StackUnderflow(ValueError):  # empty stack
    pass


# Sequential table implementation
class SStack:
    def __init__(self):
        self._elems = []

    def is_empty(self):
        return self._elems == []

    def top(self):
        if self._elems == []:
            raise StackUnderflow("in SStack.top()")
        return self._elems[-1]

    def push(self, elem):
        self._elems.append(elem)

    def pop(self):
        if self._elems == []:
            raise StackUnderflow("in SStack.pop()")
        return self._elems.pop()


# Linked list implementation
class LStack:
    def __init__(self):
        self._top = None

    def is_empty(self):
        return self._top is None

    def top(self):
        if self._top is None:
            raise StackUnderflow("in LStack.top()")
        return self._top.elem

    def push(self, elem):
        self._top = LNode(elem, self._top)

    def pop(self):
        if self._top is None:
            raise StackUnderflow("in LStack.top()")
        p = self._top
        self._top = p.next
        return p.elem


# Applications: Parenthesis Matching
def ParenMatch(text):
    parens = "()[]{}"
    open_parens = "([{"
    opposite = {")":"(","]":"[","}":"{"}

    def parentheses(text):
        for i in range(len(text)):
            if text[i] in parens:
                yield text[i], i

    st=SStack()
    for pr, i in parentheses(text):
        if pr in open_parens:
            st.push(pr)
        elif st.is_empty():
            print("Unmatching is found at {} for {}".format(i, pr))
            return False
        elif st.pop() != opposite[pr]:
            print("Unmatching is found at {} for {}".format(i, pr))
            return False

    print("All parentheses are correctly matched")
    return True



def evalRPN(tokens):
    """
    :type tokens: List[str]
    :rtype: int
    """
    operator={"+","-","*","/"}
    st=list()
    for i in tokens:
        if i in operator:
            a=st.pop()
            b=st.pop()
            st.append(str(eval(b + i + a)).split(".")[0])
        else:
            st.append(i)
    return int(st[0])

if __name__ == "__main__":
    # text1="{}(){}(())}(){)(}{"
    # print(ParenMatch(text1))
    print(evalRPN(["10","6","9","3","+","-11","*","/","*","17","+","5","+"]))
