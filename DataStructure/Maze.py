#!/usr/bin/python
# -*- coding: utf-8 -*-
from random import randint
import copy
# recursion algorithm
dirs = [(0,1),(1,0),(0,-1),(-1,0)]

def mark(maze, pos):
    maze[pos[0]][pos[1]] = 2

def passable(maze,pos):
    return maze[pos[0]][pos[1]] == 0

def find_path(maze,pos,end):
    mark(maze, pos)
    if pos == end:
        print(pos,end=" ")
        return True
    for i in range(4):
        nextp = pos[0]+dirs[i][0],pos[1]+dirs[i][1]
        try:
            p = passable(maze,nextp)
            if p:
                if find_path(maze, nextp, end):
                    print(pos, end=" ")
                    return True
        except:
            pass
    return False

def generate_maze(a,b):
    maze=[]
    for i in range(a):
        row=[]
        for j in range(b):
            row.append(randint(0,1))
        maze.append(row)
    maze[0][0]=0
    maze[a-1][b-1]=0
    return maze

if __name__ == "__main__":

    shape=(4,4)
    # maze0=[[0,1,1,1],[0,0,0,1],[0,0,0,0],[1,1,1,0]]
    maze0=generate_maze(shape[0],shape[1])
    start = (0,0)
    end = (shape[0]-1,shape[1]-1)
    find_path(copy.deepcopy(maze0),start,end)