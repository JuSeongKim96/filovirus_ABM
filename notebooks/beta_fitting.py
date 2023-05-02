import random as r
import numpy as np
import math

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import networkx as nx
from networkx.algorithms.assortativity import neighbor_degree

from tqdm import tqdm 
import missingno as msno
import pandas as pd
import itertools
import cv2
import imageio
import os
from glob import glob
import natsort


def SEID_model(num, interval, t_radius, i_radius, beta):

    num_steps = num
    # 시뮬레이션 시간 한스텝이 일어나는 시간을 정해야되나? 어떻게 3시간뒤에 감염되고 24시간 이후에는 감염안되게 하는지 생각해보깆

    # 확률로 하는데 맞나?
    # S -> E paramter  감염율
    beta = beta
    # E -> I paramter 잠복기? 기간의 역수 -> 요것도 비율로 하는게 아니라 몇일이 지나면 바이러스 내보낼 수 있는 시간으로 
    alpha = interval / 3
    # I -> D parameter 회복률 -> 몇일 지나면 더이상 바이러스 못 시키는지로 바꿔야됨
    gamma = interval / 33

    # 전체 lattice의 크기를 결정
    radius = t_radius
    # 세로
    h = 500
    # 가로
    w = 1000

    # networkx triangular lattice를 가지고 진행
    G = nx.triangular_lattice_graph(m=h, n=w, periodic=False, with_positions=True, create_using=None)
    pos = nx.get_node_attributes(G, 'pos')



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
    
    # 원형의 network를 구현하기 위해서 이렇게 진행하였음
    for node, poss in pos.items():
        distance = math.sqrt((poss[0] - center_pos[0])**2 + (poss[1] - center_pos[1])**2)
        if distance <= radius:
            circle_nodes |= {node}

    H = nx.subgraph(G, circle_nodes)
    # infection
    # # 직접적인 지정
    # center1 = (h//2+1, h//2+1)
    # center2 = (h//2-1, h//2-1)
    # center3 = (h//2+1, h//2-1)
    # center4 = (h//2-1, h//2+1)
    # infected = {center}
    # infected = {center, center1, center2, center3, center4}

    # initial radius cell 갯수
    infected = set()
    dead = set()
    # initail_radius는 위의 전체 radius보다는 항상 작아야됨
    # 0.23mm가 기준인데 0.004mm가 세포 하나당 반지름이니까 57.5개 정도의 반지름이어야됨
    initial_radius = i_radius
    initial_dead_radius = i_radius - 1
    # 이 안의 범위에 들어오는것은 저렇게 해줘
    for node, poss in pos.items():
        distance = math.sqrt((poss[0] - center_pos[0])**2 + (poss[1] - center_pos[1])**2)
        if (distance <= initial_radius) & (distance > initial_dead_radius):
            infected |= {node}
        elif distance <= initial_dead_radius:
            dead |= {node}

    S = set(H.nodes()) - infected
    E = set()
    I = infected - dead
    D = dead
    # for record population


    sus_list = []
    exposed_list = []
    infected_list = []
    dead_list = []

    # 처음 모습
    # fig, ax = plt.subplots()
    # ax.set_aspect('equal')
    # nx.draw(H, pos, node_size=10, node_color='yellow', with_labels=False)
    # nx.draw_networkx_nodes(H, pos, nodelist=S, node_size=8, node_color='blue')     
    # nx.draw_networkx_nodes(H, pos, nodelist=E, node_size=8, node_color='green')
    # nx.draw_networkx_nodes(H, pos, nodelist=I, node_size=8, node_color='red')
    # nx.draw_networkx_nodes(H, pos, nodelist=D, node_size=8, node_color='black')

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


        # E -> I process
        new_infections = set()

        for e in E:
            # if np.random.choice([0,1], 1, p = [1-alpha, alpha]):
            if np.random.uniform() < alpha: 
                new_infections.add(e)

        # I -> D process
        new_recoveries = set()

        
        for i in I:
            # if np.random.choice([0,1], 1, p = [1-gamma, gamma]):
            if np.random.uniform() < gamma:
                new_recoveries.add(i)
                
        I -= new_recoveries
        D |= new_recoveries        

        E -= new_infections
        I |= new_infections
                
        S -= new_exposed
        E |= new_exposed

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

def radius_cal(circle_number):
    # n pi r^2 = pi R^2 
    # r sqrt(n) = R
    n = circle_number
    r = 0.004
    # 실제보다 1/10 작은 사이즈기 때문에 10배 해준다
    return 10*(r*np.sqrt(n))

plaque = pd.read_excel('./plaque_processed.xlsx')
plaque.fillna(method='ffill', inplace=True)
df = plaque.groupby(['Virus','Day','SNPs'])

# 그룹별 데이터프레임을 생성 후 dict에 저장
output = dict(list(df))
output.keys()

day_list = ['Day2', 'Day3', 'Day4', 'Day5', 'Day6']
plaque_list = ['2NPC1','Y420S', 'P424A', 'S425L', 'D502E','D508N']

def S_type(virus:str,Day : str):
    temp = pd.DataFrame()
    for i in plaque_list:
        temp = pd.concat([temp, output[(virus, Day, i)]])
    return temp

def D_type(virus, snps):
    temp = pd.DataFrame()
    for i in day_list:
        temp = pd.concat([temp, output[(virus,i , snps)]])
    return temp



snp_list = ['P424A', 'S425L', 'D508N', '2NPC1', 'D502E', 'Y420S']
virus_list = ['Angola', 'Zaire']

r_dict_M = dict()
r_dict_E = dict()

for i in virus_list:
    for j in snp_list:
        df = D_type(i, j)


        r_list = []
        sd_list = []

        for k in day_list:
            r_list.append(df[df['Day']==k]['Size [sqmm]'].mean())
            sd_list.append(df[df['Day']==k]['Size [sqmm]'].sem())

        r_array=np.sqrt(np.array(r_list) / np.pi)
        
        if i == 'Angola':
            r_dict_M[j]=r_array[0]
        elif i == 'Zaire':
            r_dict_E[j] = r_array[0]

ini_rad_a = np.arange(4.5,8.5,0.1)

r_dict_M1 = dict()
r_dict_E1 = dict()

# snp 별로 돈다 각각
for v in virus_list: 
    # snp별로
    for s in snp_list:

        # 차이를 넣어둘거야
        ini_rad = []
        # 내가 정한 initial radius 값에서 구해보자
        for q in ini_rad_a:

            # 한번만 돌아도 됨 beta는 상관없으므로 임의의 값 아무거나 넣어도 됨
            _, _, _, dead_list = SEID_model(1, 1, 216.9, q, 0.5)

            # ini_rad라는 것에 대해서 넣어준다
            ini_rad.append(abs(radius_cal(dead_list[0]) - r_dict_E[s]))


        if v == 'Angola':
            r_dict_M1[s] = ini_rad_a[np.argmin(ini_rad)]
        if v == 'Zaire':
            r_dict_E1[s] = ini_rad_a[np.argmin(ini_rad)]



v_type = 'Zaire'
s_type = 'Y420S'

# 시간 간격을 어떻게 잡을 거냐
interval = 1

p = np.arange(0.2,0.35,0.01)

df_tot = pd.DataFrame(columns=['zero'])

n_ensemble = 10

for u in tqdm(range(n_ensemble)):
    x_list = []
    
    for b in p:
        
        # dead_list라는 개수를 가져오자
        if v_type == 'Zaire':
            _, _, _, dead_list = SEID_model(102, interval, 216.9, r_dict_E1[s_type], b)
        elif v_type == 'Angola':
            _, _, _, dead_list = SEID_model(102, interval, 216.9, r_dict_M1[s_type], b)

        rad_list = []

        # 개수를 가지고 반지름 계산
        for i in dead_list:
            rad_list.append(radius_cal(i))

        test_list = []
        sd_list = []

        df = D_type(v_type,s_type)

        for k in day_list:
            # 실제 값은 어떤지
            test_list.append(df[df['Day']==k]['Size [sqmm]'].mean())
            sd_list.append(df[df['Day']==k]['Size [sqmm]'].sem())

        # pi r^2 이 우리가 얻은 값이기 때문에 pi 로 나누고 루프 씌어준다
        test_array=np.sqrt(np.array(test_list) / np.pi)

        x=0
        b=0
        for j in range(1,5):
            # 102로 하니까 25인데  1시간 간격
            # 32 로 하니까 8 인데 3시간 가격
            x = abs(rad_list[25*j+1] - test_array[j])
            b += x
        x_list.append(b)

    df = pd.DataFrame([p,x_list])
    df = df.transpose()
    df_tot = pd.concat([df, df_tot], axis=1)


df_tot.columns = ['x1','e1','x2','e2','x3','e3','x4','e4','x5','e5', 'x6', 'e6', 'x7','e7','x8','e8','x9','e9','x10','e10', 'zero']
df_tot.to_csv(f'./{v_type}_{s_type}_total.csv')