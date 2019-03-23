#!/usr/bin/python
# -*- coding: utf-8 -*-
import random
import time


def isSorted(lst):
    for i in range(len(lst)-1):
        if lst[i + 1] < lst[i]:
            return False
    return True


def swap(lst, i, j):
    lst[i], lst[j] = lst[j], lst[i]


def selection_sort(lst):
    n = len(lst)
    for i in range(n-1):
        minindex = i
        for j in range(i + 1, n):
            if lst[j] < lst[minindex]:
                minindex = j
        swap(lst, i, minindex)


def bubble_sort(lst):
    n = len(lst)
    for i in range(n - 1, 0, -1):
        for j in range(i):
            if lst[j + 1] < lst[j]:
                swap(lst, j + 1, j)


def insertion_sort(lst):
    n = len(lst)
    for i in range(1, n):
        for j in range(i, 0, -1):
            if lst[j] < lst[j - 1]:
                swap(lst, j, j-1)
            else:
                break


def shell_sort(lst):
    n = len(lst)
    h = 1
    while h < n//3:
        h = h * 3 + 1
    while h >= 1:
        for i in range(h, n):
            for j in range(i, h - 1, -h):
                if lst[j] < lst[j - h]:
                    swap(lst, j, j - h)
                else:
                    break
        h = h//3


def merge_sort(lst):
    lst_copy = [0]*len(lst)
    _merge_sort(lst, lst_copy, 0, len(lst)-1)


def _merge_sort(lst, lst_copy, lo, hi):
    if hi <= lo + 63:
        insertion_sort(lst[lo:hi+1])
        return
    mid = (lo + hi) // 2
    _merge_sort(lst, lst_copy, lo, mid)
    _merge_sort(lst, lst_copy, mid + 1, hi)
    if lst[mid] <= lst[mid + 1]:
        return
    merge(lst, lst_copy, lo, mid, hi)


def merge(lst, lst_copy, lo, mid, hi):
    for k in range(lo, hi + 1):
        lst_copy[k] = lst[k]
    i = lo
    j = mid + 1
    for k in range(lo, hi + 1):
        if i > mid:
            lst[k] = lst_copy[j]
            j += 1
        elif j > hi:
            lst[k] = lst_copy[i]
            i += 1
        elif lst_copy[j] < lst_copy[i]:
            lst[k] = lst_copy[j]
            j += 1
        else:
            lst[k] = lst_copy[i]
            i += 1


def quick_sort(lst):
    _quick_sort(lst, 0, len(lst) - 1)


def _quick_sort(lst, lo, hi):
    if hi <= lo:
        return
    lt = lo
    gt = hi
    v = lst[lo]
    i = lo
    while i <= gt:
        if lst[i] < v:
            swap(lst, lt, i)
            lt += 1
            i += 1
        elif lst[i] > v:
            swap(lst, i, gt)
            gt -= 1
        else:
            i += 1
    _quick_sort(lst, lo, lt - 1)
    _quick_sort(lst, gt + 1, hi)


if __name__ == "__main__":
    random.seed(2)
    test_list = random.choices(range(10000), k=10000)
    # print(test_list[0:10])
    start_time = time.time()
    # selection_sort(test_list)
    # insertion_sort(test_list)
    # bubble_sort(test_list)
    # shell_sort(test_list)
    merge_sort(test_list.copy())
    # quick_sort(test_list.copy())
    print(time.time() - start_time)
    test_list.sort()
    print(time.time() - start_time)
    print(isSorted(test_list))