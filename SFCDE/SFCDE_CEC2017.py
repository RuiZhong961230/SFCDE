import os
from copy import deepcopy
from scipy.stats import cauchy
from cec17_functions import cec17_test_func
import numpy as np
import warnings

warnings.filterwarnings("ignore")

PopSize = 50
DimSize = 10
LB = [-100] * DimSize
UB = [100] * DimSize
TrialRuns = 20
curFEs = 0
MaxFEs = DimSize * 1000

Pop = np.zeros((PopSize, DimSize))
FitPop = np.zeros(PopSize)

FuncNum = 0

H = 5
muF = [0.3] * H
muCr = [0.5] * H

Fail_F = []
Fail_Cr = []


def fitness(X):
    global DimSize, FuncNum
    f = [0]
    cec17_test_func(X, f, DimSize, 1, FuncNum)
    return f[0]


def meanL(arr):
    numer = 0
    denom = 0
    for var in arr:
        numer += var ** 2
        denom += var
    return numer / denom


def Initialization():
    global Pop, FitPop, curFEs, DimSize, muF, muCr, H, Fail_F, Fail_Cr
    for i in range(PopSize):
        for j in range(DimSize):
            Pop[i][j] = LB[j] + (UB[j] - LB[j]) * np.random.rand()
        FitPop[i] = fitness(Pop[i])
        curFEs += 1
    muF = [0.5] * H
    muCr = [0.5] * H
    Fail_F, Fail_Cr = [], []


def SFCDE():
    global Pop, FitPop, LB, UB, PopSize, DimSize, curFEs, Fail_F, Fail_Cr
    Off = np.zeros((PopSize, DimSize))
    FitOff = np.zeros(PopSize)
    F_idx, Cr_idx = np.random.randint(H), np.random.randint(H)
    mu_Fail_F = -1
    if len(Fail_F) != 0:
        mu_Fail_F = np.mean(Fail_F)
    mu_Fail_Cr = -1
    if len(Fail_Cr) != 0:
        mu_Fail_Cr = np.mean(Fail_Cr)

    Fail_F, Fail_Cr = [], []  # Empty the failure history
    Success_F, Success_Cr = [], []
    for i in range(PopSize):
        IDX = np.random.randint(0, PopSize)
        while IDX == i:
            IDX = np.random.randint(0, PopSize)
        candi = list(range(0, PopSize))
        candi.remove(i)
        candi.remove(IDX)
        r1, r2 = np.random.choice(candi, 2, replace=False)

        F = cauchy.rvs(muF[F_idx], 0.1)
        Cr = np.clip(np.random.normal(muCr[Cr_idx], 0.1), 0, 1)
        while True:  # F determination
            if F > 1:
                F = 1
                break
            elif F < 0:
                F = cauchy.rvs(muF[F_idx], 0.1)
            break

        while True:
            if abs(F - mu_Fail_F) < abs(F - muF[F_idx]):  # Be close to the failure F
                D1 = abs(F - mu_Fail_F)
                D2 = abs(F - muF[F_idx])
                prob = np.exp(-np.sqrt(D2 / D1))
                if np.random.rand() < prob:
                    break
                else:
                    F = cauchy.rvs(muF[F_idx], 0.1)
                    while True:
                        if F > 1:
                            F = 1
                            break
                        elif F < 0:
                            F = cauchy.rvs(muF[F_idx], 0.1)
                        break
            else:
                break

        while True:
            if abs(Cr - mu_Fail_Cr) < abs(Cr - muCr[Cr_idx]):  # Be close to the failure Cr
                D1 = abs(Cr - mu_Fail_Cr)
                D2 = abs(Cr - muCr[Cr_idx])
                prob = np.exp(-np.sqrt(D2 / D1))
                if np.random.rand() < prob:
                    break
                else:
                    Cr = np.clip(np.random.normal(muCr[Cr_idx], 0.1), 0, 1)
            else:
                break

        if FitPop[IDX] < FitPop[i]:  # DE/winner-to-best/1
            Off[i] = Pop[i] + F * (Pop[np.argmin(FitPop)] - Pop[i]) + F * (Pop[r1] - Pop[r2])
        else:
            Off[i] = Pop[IDX] + F * (Pop[np.argmin(FitPop)] - Pop[IDX]) + F * (Pop[r1] - Pop[r2])

        jrand = np.random.randint(0, DimSize)  # bin crossover
        for j in range(DimSize):
            if np.random.rand() < Cr or j == jrand:
                pass
            else:
                Off[i][j] = Pop[i][j]

        for j in range(DimSize):
            if Off[i][j] < LB[j] or Off[i][j] > UB[j]:
                Off[i][j] = np.random.uniform(LB[j], UB[j])

        FitOff[i] = fitness(Off[i])
        curFEs += 1
        if FitOff[i] < FitPop[i]:
            Success_F.append(F)
            Success_Cr.append(Cr)
        else:
            Fail_F.append(F)
            Fail_Cr.append(Cr)

    for i in range(PopSize):
        if FitOff[i] < FitPop[i]:
            Pop[i] = deepcopy(Off[i])
            FitPop[i] = FitOff[i]

    c = 0.1
    if len(Success_F) == 0:
        pass
    else:
        muF[F_idx] = (1 - c) * muF[F_idx] + c * meanL(Success_F)
    if len(Success_Cr) == 0:
        pass
    else:
        muCr[Cr_idx] = (1 - c) * muCr[Cr_idx] + c * np.mean(Success_Cr)


def RunSFCDE():
    global curFEs, MaxFEs, TrialRuns, Pop, FitPop, DimSize
    All_Trial_Best = []
    for i in range(TrialRuns):
        Best_list = []
        curFEs = 0
        np.random.seed(4 + 28 * i)
        Initialization()
        Best_list.append(min(FitPop))
        while curFEs < MaxFEs:
            SFCDE()
            Best_list.append(min(FitPop))
        All_Trial_Best.append(Best_list)
    np.savetxt("./SFCDE_Data/CEC2017/F" + str(FuncNum) + "_" + str(DimSize) + "D.csv", All_Trial_Best, delimiter=",")


def main(dim):
    global FuncNum, DimSize, MaxFEs, MaxIter, Pop, LB, UB
    DimSize = dim
    Pop = np.zeros((PopSize, dim))
    MaxFEs = dim * 1000
    LB = [-100] * dim
    UB = [100] * dim

    for i in range(1, 31):
        if i == 2:
            continue
        FuncNum = i
        RunSFCDE()


if __name__ == "__main__":
    if os.path.exists('./SFCDE_Data/CEC2017') == False:
        os.makedirs('./SFCDE_Data/CEC2017')
    Dims = [30, 50]
    for Dim in Dims:
        main(Dim)
