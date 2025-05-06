import numpy as np
import math
# from tqdm import tqdm 
import pandas as pd
import random
import networkx as nx
# from networkx.algorithms.assortativity import neighbor_degree

# import itertools
# import os
# from glob import glob
from sklearn.metrics import mean_absolute_error

from utils import *

# import argparse


def make_lattice(t_radius=394, h=950, lattice_type='triangle'):
    """
    make triangular lattice with circle shape

    Parameters
    -----------
    t_radius : total radius 
    h : height h+1만큼의 row 개수를 가진다. 
    w : width (w+1)//2만큼 columns 개수를 가진다.
    Return
    -----------
    Triangular lattice : H and position : pos
    Example
    -----------
    H, pos = make_lattice()
    """
    # 실제거리면 별로 안좋은데 노드로 세는게 더 좋을 수도
    w = 2*h
    if lattice_type == 'triangle':
        G = nx.triangular_lattice_graph(m=h, n=w, periodic=False, with_positions=True, create_using=None)
        pos = nx.get_node_attributes(G, 'pos')

        # 중심을 어디서 잡을거냐
        center = (((w+1)//2)//2,(h+1)//2)

    elif lattice_type == 'rectangle':
        G = nx.grid_2d_graph(h,h, periodic=False)
        pos = {(i,j):(i,j) for i in range(h) for j in range(w)}
        nx.set_node_attributes(G,pos,'pos')
        center = (h//2,h//2)
    
    center_pos = G.nodes[center]['pos']

    circle_nodes = set()

    # 원형의 network를 구현하기 위해서 이렇게 진행하였음
    for node, poss in pos.items():
        # center pos와의 거리가 원형으로 일정 이내로 들어오도록 해주었다.
        distance = math.sqrt((poss[0] - center_pos[0])**2 + (poss[1] - center_pos[1])**2)
        if distance <= t_radius:
            circle_nodes |= {node}

    H = nx.subgraph(G, circle_nodes)
    pos = nx.get_node_attributes(H, 'pos')

    return H, pos


def set_initial(H, i_radius, interval,beta=0.5, h= 950):
    """
    setting initial radius

    Parameters
    -----------
    H : graph 
    i_radius : 초기 감염 반지름 크기
    beta : 초기 반지름까지 성장시킬 때의 감염율 (조금 높게 해도 괜찮다) 
    interval : height h+1만큼의 row 개수를 가진다. 
    Return
    -----------
    S,E,I,R 각 상태에 해당하는 노드의 set반환
    Example
    -----------
    S,E,I,R = set_initial(H,i_radius,beta, interval)
    """
    # single seed에서 성장해서 나타나는 부분이 필요하다 
    w= 2*h

    center = (((w+1)//2)//2,(h+1)//2)
    infected= set()
    infected.add(center)
    S = set(H.nodes())-infected
    E = set()
    I = infected
    D = set()

    # beta = betaa * (24/(interval-1))
    # E -> I paramter 잠복기? 기간의 역수 -> 요것도 비율로 하는게 아니라 몇일이 지나면 바이러스 내보낼 수 있는 시간으로 

    # alpha = 1
    alpha = 1 * (24/(interval-1))# 초단위로 바꿔보자
    
    # I -> D parameter 회복률 -> 몇일 지나면 더이상 바이러스 못 시키는지로 바꿔야됨
    # gamma = 1/27 * (24/(interval-1))
    gamma = 1 * 4.5 / 27 * (24/(interval-1))


    while radius_cal(len(D))<i_radius:
        
        new_exposed = set()
        for i in I:
                # s의 이웃들에 대해서
                # 6n 보다
                # link based로 물어보는게 더 좋다
                # S I 연결되어 있는 link만 확인하자

            for neighbor in H.neighbors(i):
                # neighbor에서 E 혹은 I가 있으면 감염시켜라 그러면 E 빼야되는거 아닌가? 이건 고민좀 해봐야될 듯?
                # if neighbor in I or neighbor in E:
                if neighbor in S:
                    #if np.random.choice([0,1], 1, p = [1-beta, beta]):
                    if np.random.uniform() < beta:
                        new_exposed.add(neighbor)
                        # break 들어가는 이유 그 다음 스텝 추가 했으면 그건 다시 감염 안시켜서 한번에 하나만 감염시킬 수 있게 해주기 위해서 들어감
                        # 한step에 빨간 녀석은 한명만 감염시킬 수 있음.
                        # break 없앰


        # E -> I process
        new_infections = set()

        for e in E:
            if np.random.uniform() < alpha: 
                new_infections.add(e)

        # I -> D process
        new_recoveries = set()

        for ii in I:
            if np.random.uniform() < gamma:
                new_recoveries.add(ii)
                
        I -= new_recoveries
        D |= new_recoveries        

        E -= new_infections
        I |= new_infections
                
        S -= new_exposed
        E |= new_exposed
        
    return S, E, I, D



def SEID_model(H, S, E, I, D, interval, beta, ps=None):
    """
    infection process simulation

    Parameters
    -----------
    H : graph 
    S,E,I,D : each cell state set
    interval : simulation step divided by 24 hour
    beta : cell-to-cell infection probability (adjusted)
    ps : how long simulate 
    Return
    -----------
    sus_list : 시뮬레이션 스텝당 감염 가능한 세포의 수가 순차적으로 기록
    exposed_list : 시뮬레이션 스텝당 잠복기인 세포의 수가 순차적으로 기록
    infected_list : 시뮬레이션 스텝당 감염된 세포의 수가 순차적으로 기록
    dead_list : 시뮬레이션 스텝당 죽은 세포의 수가 순차적으로 기록
    Example
    -----------
    sus_list, exposed_list, infected_list, dead_list = SEID_model(H,pos,S,E,I,D, interval=73, beta = 0.08)

    sus_list, exposed_list, infected_list, dead_list = SEID_model(H,pos,S,E,I,D, interval=73, beta = 0.08, ps =  10)
    """

    # E -> I paramter 잠복기? 기간의 역수 -> 요것도 비율로 하는게 아니라 몇일이 지나면 바이러스 내보낼 수 있는 시간으로 
    # alpha = 24/(4.5*(interval-1))
    # # I -> D parameter 회복률 -> 몇일 지나면 더이상 바이러스 못 시키는지로 바꿔야됨
    # gamma = 24/(27*(interval-1))

    alpha = 2 * (24/(interval-1))# 초단위로 바꿔보자
    # I -> D parameter 회복률 -> 몇일 지나면 더이상 바이러스 못 시키는지로 바꿔야됨
    # gamma =4.5 * (24/(interval-1))/27 # 27시간으로 가정했으면

    gamma =2*4.5/27 * (24/(interval-1)) # 27시간으로 가정했으면

    

    if ps is not None:
        num_steps = interval*ps - (ps-1)
    else:
        num_steps = interval*4-3


    exposed_list = []
    infected_list = []
    dead_list = []

    # 이거 순서도 생각해야됨
    for _ in range(num_steps):

        # S -> E process
        new_exposed = set()
        new_recoveries = set()

        for i in I:

            # s의 이웃들에 대해서
            # 6n 보다
            # link based로 물어보는게 더 좋다
            # S I 연결되어 있는 link만 확인하자

            for neighbor in H.neighbors(i):
                # neighbor에서 E 혹은 I가 있으면 감염시켜라 그러면 E 빼야되는거 아닌가? 이건 고민좀 해봐야될 듯?
                # if neighbor in I or neighbor in E:
                if neighbor in S:
                    #if np.random.choice([0,1], 1, p = [1-beta, beta]):
                    if np.random.uniform() < beta:
                        new_exposed.add(neighbor)
                        # break 들어가는 이유 그 다음 스텝 추가 했으면 그건 다시 감염 안시켜서 한번에 하나만 감염시킬 수 있게 해주기 위해서 들어감
                        # 한step에 빨간 녀석은 한명만 감염시킬 수 있음.
                        # break 없앰

        # E -> I process
        new_infections = set()

        for e in E:
            # if np.random.choice([0,1], 1, p = [1-alpha, alpha]):
            if np.random.uniform() < alpha: 
                new_infections.add(e)
        # I -> D processs
        for ii in I:
            if np.random.uniform() < gamma:
                new_recoveries.add(ii)

        I -= new_recoveries
        D |= new_recoveries        
        E -= new_infections
        I |= new_infections
        S -= new_exposed
        E |= new_exposed

        # 각 step에서의 갯수 기록

        exposed_list.append(len(E))
        infected_list.append(len(I))
        dead_list.append(len(D))

    return exposed_list, infected_list, dead_list