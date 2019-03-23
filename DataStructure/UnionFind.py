#!/usr/bin/python
# -*- coding: utf-8 -*-
import random
from math import sqrt
class QuickUnionUF:
    def __init__(self, num):
        self._id = list(range(num))
        self._sz = [1] * num

    def root(self, i):
        while i != self._id[i]:
            self._id[i] = self._id[self._id[i]]
            i = self._id[i]
        return i

    def connected(self, p, q):
        return self.root(p) == self.root(q)

    def union(self, p, q):
        i = self.root(p)
        j = self.root(q)
        if i == j:
            return
        if self._sz[i] < self._sz[j]:
            self._id[i] = j
            self._sz[j] += self._sz[i]
        else:
            self._id[j] = i
            self._sz[i] += self._sz[j]


class Percolation:
    def __init__(self, n):
        self.sites = [[0 for i in range(n)] for i in range(n)]
        self.qu = QuickUnionUF(n ** 2 + 2)
        self.n = n
        self.count_opensites = 0

    def open(self, row, col):
        if self.isOpen(row, col):
            return
        self.sites[row - 1][col - 1] = 1
        self.count_opensites += 1
        index = (row - 1) * self.n + col - 1
        if row == 1:
            self.qu.union(index, self.n ** 2)
            if self.isOpen(row + 1, col):
                self.qu.union(index, index + self.n)
        elif row == self.n:
            self.qu.union(index, self.n ** 2 + 1)
            if self.isOpen(row - 1, col):
                self.qu.union(index, index - self.n)
        else:
            if self.isOpen(row - 1, col):
                self.qu.union(index, index - self.n)
            if self.isOpen(row + 1, col):
                self.qu.union(index, index + self.n)

        if col == 1:
            if self.isOpen(row, col + 1):
                self.qu.union(index, index + 1)
        elif col == self.n:
            if self.isOpen(row, col - 1):
                self.qu.union(index, index - 1)
        else:
            if self.isOpen(row, col + 1):
                self.qu.union(index, index + 1)
            if self.isOpen(row, col - 1):
                self.qu.union(index, index - 1)

    def isOpen(self, row, col):
        return self.sites[row - 1][col - 1] == 1

    def isFull(self, row, col):
        index = (row - 1) * self.n + col - 1
        return self.qu.connected(index, self.n ** 2)

    def numberofOpenSites(self):
        return self.count_opensites

    def percolates(self):
        return self.qu.connected(self.n ** 2, self.n ** 2 + 1)


class PercolationStats:
    def __init__(self, n, trials):
        self.x=[0]*trials
        for i in range(trials):
            if i%1000==0:
                print("times {}".format(i))
            pe=Percolation(n)
            while not pe.percolates():
                a=random.randint(0,n**2-1)
                row =a//n+1
                col=a%n+1
                pe.open(row, col)
            self.x[i]=pe.numberofOpenSites()/n**2

    def mean(self):
        return sum(self.x)/len(self.x)

    def stddev(self):
        mean=self.mean()
        return sqrt(sum([(i-mean)**2 for i in self.x])/(len(self.x)-1))

    def confidenceLo(self):
        return self.mean()-1.96*self.stddev()/sqrt(len(self.x))

    def confidenceHi(self):
        return self.mean()+1.96*self.stddev()/sqrt(len(self.x))

if __name__ == "__main__":
    # qu = QuickUnionUF(10)
    # qu.union(6, 5)
    # qu.union(5, 0)
    # qu.union(0, 1)
    # qu.union(2, 1)
    # qu.union(7, 1)
    # qu.union(4, 3)
    # qu.union(9, 8)
    # qu.union(3, 8)
    PS=PercolationStats(2,10000)
    print("mean:",PS.mean())
    print("stddev:",PS.stddev())
    print("95% confidence interval:",[PS.confidenceLo(),PS.confidenceHi()])

