import random as r
import numpy as np
import math

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import networkx as nx
from networkx.algorithms.assortativity import neighbor_degree


import missingno as msno
import pandas as pd
import itertools
import cv2
import imageio
import os
from glob import glob
import natsort


# 실제 plaque 데이터 가져오기
path = '../data/'

im_path = '../data/Figure/'
plaque = pd.read_excel(path + 'plaque_processed.xlsx')
plaque.fillna(method='ffill', inplace=True)
df = plaque.groupby(['Virus','Day','SNPs'])

# 그룹별 데이터프레임을 생성 후 dict에 저장
output = dict(list(df))
output.keys()

day_list = ['Day2', 'Day3', 'Day4', 'Day5', 'Day6']
plaque_list = ['2NPC1','Y420S', 'P424A', 'S425L', 'D502E','D508N']


# SNPs별로 볼라고 만들어놓음
def S_type(virus:str,Day : str):
    temp = pd.DataFrame()
    for i in plaque_list:
        temp = pd.concat([temp, output[(virus, Day, i)]])
    return temp

# Day 별로 보려고 
def D_type(virus, snps):
    temp = pd.DataFrame()
    for i in day_list:
        temp = pd.concat([temp, output[(virus,i , snps)]])
    return temp






# step 수랑 
def SEID_model(num, radius, beta):
    path = '/Users/juseongkim/Documents/대학원/epidemic spreading/Epidemic_speading/result/Fig/'

    v_type = 'EBOV'
    # v_type = 'MARV'

    num_steps = num
    # 시뮬레이션 시간 한스텝이 일어나는 시간을 정해야되나? 어떻게 3시간뒤에 감염되고 24시간 이후에는 감염안되게 하는지 생각해보깆

    # 확률로 하는데 맞나?
    # S -> E paramter  감염율
    if v_type == 'EBOV':
        beta = beta
    elif v_type == 'MARV':
        beta = 0.41
    # E -> I paramter 잠복기? 기간의 역수 -> 요것도 비율로 하는게 아니라 몇일이 지나면 바이러스 내보낼 수 있는 시간으로 
    if v_type == 'EBOV':
        # 1로 다음에 시간을 맞춘다 1시간에
        alpha=1
    elif v_type == 'MARV':
        alpha=0.22
    # I -> D parameter 회복률 -> 몇일 지나면 더이상 바이러스 못 시키는지로 바꿔야됨
    if v_type == 'EBOV':
        gamma = 0.091
    elif v_type == 'MARV':
        gamma = 0.2

    radius = radius
    # 세로
    h = 500
    # 가로
    w = 1000


    G = nx.triangular_lattice_graph(m=h, n=w, periodic=False, with_positions=True, create_using=None)
    pos = nx.get_node_attributes(G, 'pos')


    # center는 무조건 걸리게 해뒀음 나중에는 바꿀 꺼임
    # infected = {infected_node} dictionary
    # seed 가 3개다
    # seed에서 반지름 길이에 해당하는 세포수는 0.23 / 0.004 = 57.5개 정도

    ##############
    # 랜덤하게 여러개의 시드를 넣어주자
    ##############
    # for i in range(seed_number):
    #     # 지금은 가장자리에 잘 안 분포되게 해놓는게 좋을 거 같아서 1 l로 범위를 잡아높고 나중에 uniform distribution같은걸로 값을 넣어주는게 필요할 듯
    #     G.nodes[(r.randrange(1,l),r.randrange(1,l))]['occupied'] = 1



    # 반으로 줄였으니까 한 28개정도로 
    center = (h//2+1, h//2+1)
    center_pos = G.nodes[center]['pos']

    circle_nodes = set()

    for node, poss in pos.items():
        distance = math.sqrt((poss[0] - center_pos[0])**2 + (poss[1] - center_pos[1])**2)
        if distance <= radius:
            circle_nodes |= {node}

    H = nx.subgraph(G, circle_nodes)


    # initial radius cell 갯수
    infected = set()
    # initail_radius는 위의 전체 radius보다는 항상 작아야됨
    # 0.23mm가 기준인데 0.004mm가 세포 하나당 반지름이니까 57.5개 정도의 반지름이어야됨
    initial_radius = 5.75
    # 이 안의 범위에 들어오는것은 저렇게 해줘
    for node, poss in pos.items():
        distance = math.sqrt((poss[0] - center_pos[0])**2 + (poss[1] - center_pos[1])**2)
        if distance <= initial_radius:
            infected |= {node}


    # initial population setting
    S = set(H.nodes()) - infected
    E = set()
    I = infected
    D = set()

    # for record population
    sus_list = []
    exposed_list = []
    infected_list = []
    dead_list = []



    # 이거 순서도 생각해야됨
    for t in range(num_steps):

        # # E -> I initial time
        # new_infections = set()

        # for e in E:
        #     if np.random.choice([0,1], 1, p = [1-alpha, alpha]):
        #     # if np.random.uniform() > beta:
        #         new_infections.add(e)

        # E -= new_infections
        # I |= new_infections

        # S -> E process
        new_exposed = set()

        for s in S:
            # s의 이웃들에 대해서
            for neighbor in H.neighbors(s):
                # neighbor에서 E 혹은 I가 있으면 감염시켜라 그러면 E 빼야되는거 아닌가? 이건 고민좀 해봐야될 듯?
                # if neighbor in I or neighbor in E:
                if neighbor in I:
                    #if np.random.choice([0,1], 1, p = [1-beta, beta]):
                    if np.random.uniform() < beta:
                        new_exposed.add(s)
                        # break 들어가는 이유 그 다음 스텝 추가 했으면 그건 다시 감염 안시켜서 한번에 하나만 감염시킬 수 있게 해주기 위해서 들어감
                        # 한step에 빨간 녀석은 한명만 감염시킬 수 있음.
                        # break 없앰

        S -= new_exposed
        E |= new_exposed
        
        # I -> D process
        new_recoveries = set()

        for i in I:
            # if np.random.choice([0,1], 1, p = [1-gamma, gamma]):
            if np.random.uniform() < gamma:
                new_recoveries.add(i)

        I -= new_recoveries
        D |= new_recoveries



        # E -> I process
        new_infections = set()

        for e in E:
            # if np.random.choice([0,1], 1, p = [1-alpha, alpha]):
            if np.random.uniform() < alpha: 
                new_infections.add(e)

        E -= new_infections
        I |= new_infections

        # 각 step에서의 갯수 기록
        sus_list.append(len(S))
        exposed_list.append(len(E))
        infected_list.append(len(I))
        dead_list.append(len(D))

        # Draw Figures
        # fig, ax = plt.subplots()
        # ax.set_aspect('equal')
        # nx.draw(H, pos, node_size=10, node_color='yellow', with_labels=False)
        # nx.draw_networkx_nodes(H, pos, nodelist=S, node_size=8, node_color='blue')     
        # nx.draw_networkx_nodes(H, pos, nodelist=E, node_size=8, node_color='green')
        # nx.draw_networkx_nodes(H, pos, nodelist=I, node_size=8, node_color='red')
        # nx.draw_networkx_nodes(H, pos, nodelist=D, node_size=8, node_color='black')
            # plt.savefig(path + f'SEIR_{t}.png')
    return sus_list, exposed_list, infected_list, dead_list




day_list = ['Day2', 'Day3', 'Day4', 'Day5', 'Day6']


def radius_cal(circle_number):
        n = circle_number
        r = 0.004
        # 실제보다 1/10 작은 사이즈기 때문에 10배 해준다
        return 10*(r*np.sqrt(n))

v_type = 'Angola'
s_type = '2NPC1'
x= np.arange(0.3,0.7,0.01)
for j in x:
    # dead_list라는 개수를 가져오자
    _, _, _, dead_list = SEID_model(42,216.9, j)


    # 개수를 반지름으로 변환
    rad_list = []
    for i in dead_list:
        rad_list.append(radius_cal(i))

    test_list = []
    sd_list = []

    df=D_type(v_type,s_type)

    for k in day_list:
        test_list.append(df[df['Day']==k]['Size [sqmm]'].mean())
        sd_list.append(df[df['Day']==k]['Size [sqmm]'].sem())

    test_array=np.sqrt(np.array(test_list) / np.pi)
    
    x=0
    x_list = []
    for i in range(1,6):
        x = abs(rad_list[8*i+1] - test_array[i-1])
        x += x
    x_list.append(x)

    # 어떤 값이 제일 작냐?
    plt.figure()
    plt.plot(np.arange(0,5,0.125),rad_list[:40], marker='o', label='simulation', color='black', markersize=2)
    plt.scatter(np.arange(1,len(test_array)+1,1),list(test_array), marker='o', color='blue', alpha=0.5, label='experiment')
    plt.errorbar(np.arange(1,len(test_array)+1,1), test_array, yerr = sd_list, color='blue', alpha=0.5, linestyle='')
    plt.xlabel('steps', fontsize=15)
    plt.title(f'beta = {j} and {x}')
    plt.legend()
    plt.ylabel('radius(mm)',fontsize=15)
    plt.ylim(0,3)
    plt.xticks(np.arange(1,len(test_array)+1,1), labels=day_list)
    plt.savefig(f'../result/radius_fit/fit_{j}.png', dpi=200)
