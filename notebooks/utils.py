import pandas as pd
import numpy as np


day_list = ['Day2', 'Day3', 'Day4', 'Day5', 'Day6']
snp_list = ['2NPC1','Y420S', 'P424A', 'S425L', 'D502E','D508N']

plaque = pd.read_excel('../data/plaque_processed.xlsx')
plaque.fillna(method='ffill', inplace=True)
df = plaque.groupby(['Virus','Day','SNPs'])

# 그룹별 데이터프레임을 생성 후 dict에 저장
output = dict(list(df))

def S_type(virus:str,Day : str):
    temp = pd.DataFrame()
    for i in snp_list:
        temp = pd.concat([temp, output[(virus, Day, i)]])
    return temp

def D_type(virus, snps):
    temp = pd.DataFrame()
    for i in day_list:
        temp = pd.concat([temp, output[(virus,i , snps)]])
    return temp

def plaque_f(df):
    '''
    real plaque data radius and standard deviation

    Parameters
    -----------
    df : dataframe real plaque data
    Return
    -----------
    r_list : plaque radius 
    sd_list : std palque with specific SNP
    Example
    -----------
    H, pos = make_lattice()
    '''
    day_list = ['Day2', 'Day3', 'Day4', 'Day5', 'Day6']
    # snp_list = ['2NPC1','Y420S', 'P424A', 'S425L', 'D502E','D508N']

    df['Radius [mm]'] = np.sqrt(df['Size [sqmm]']/ np.pi)

    r_list = []
    sd_list = []

    for k in day_list:
        r_list.append(df[df['Day']==k]['Radius [mm]'].mean())
        sd_list.append(df[df['Day']==k]['Radius [mm]'].std())
    return r_list, sd_list

def radius_cal(arr):
    return 0.004*np.sqrt(arr)


def raw_radi(v_type, s_type):
    day_list = ['Day2', 'Day3', 'Day4', 'Day5', 'Day6']
    df = D_type(v_type, s_type)
    df.loc[:,'Radius [mm]'] = np.sqrt(df['Size [sqmm]']/ np.pi)

    day_radius=[]
    for i in day_list:
        filtered_df = df[df['Day'] ==i]
        day_radius.append(filtered_df['Radius [mm]'].values)
    return day_radius[1:]


def error_func(radius_raw, rad_list):
    results = []
    for i in range(len(radius_raw)):
        diff = abs(radius_raw[i] - rad_list[i])/len(radius_raw[i])
        results.append(diff)
    total_error = np.sum([np.sum(result) for result in results]) /4
    return total_error


def cal_beta(interval: int, beta_probability: float, multiplier=1):
    return beta_probability * interval/ 24 / 4.5  / multiplier