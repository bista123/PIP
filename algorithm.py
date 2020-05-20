
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import copy
import random
from scipy.spatial import distance
from scipy.stats import levy
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist


# df = pd.DataFrame({
#     'x': [12, 20, 28, 18, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72],
#     'y': [39, 36, 30, 52, 54, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24]
# })


class Nest:

    def __init__(self, nest_centroids, nest_clusters, closest_distance, fitness):
        self.nest_centroids = nest_centroids
        self.nest_clusters = nest_clusters
        self.closest_distance = closest_distance
        self.fitness = fitness

    def set_centroids(self, df, k):
        self.nest_centroids = gen_centroids(df, k)
        # print(self.nest_centroids)


def gen_centroids(d_frame, k):

    max_df = max(pd.DataFrame.max(d_frame, axis=0)) * 1.1
    min_df = min(pd.DataFrame.min(d_frame, axis=0)) * 0.9

    centroids = {
        i + 1: [random.uniform(min_df, max_df) for j in range(len(d_frame.columns))]
        for i in range(k)
    }

    return centroids


def column(matrix, i):
    return [row[i] for row in matrix]


def assignment(d_frame, centroids, dim):

    colmap = {1: 'r', 2: 'g', 3: 'b'}

    pole = []

    x = 0
    for i, j in d_frame.iterrows():
        pole.append([])
        for k in centroids.keys():
            pole[x].append(distance.euclidean(j.tolist(), centroids[k]))
        x += 1

    for i in centroids.keys():
        d_frame.insert(i+dim-1, 'distance_from_{}'.format(i), column(pole, i-1))

    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
    d_frame['closest'] = d_frame.loc[:, centroid_distance_cols].idxmin(axis=1)
    d_frame['min_dist'] = d_frame.loc[:, centroid_distance_cols].min(axis=1)
    d_frame['closest'] = d_frame['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    # df['color'] = df['closest'].map(lambda x: colmap[x])

    return d_frame


def levy_step(beta, dim):

    s1 = np.power((math.gamma(1 + beta) * np.sin((np.pi * beta) / 2)) / math.gamma((1 + beta) / 2) * 2 * np.power(2, (beta - 1) / 2), 1 / beta)
    s2 = 1
    u = np.random.normal(0, s1, size=dim)
    v = np.random.normal(0, s2, size=dim)
    step = u / np.power(np.fabs(v), 1 / beta)

    return step


def main():

    # Inicializacia datasetu
    df = pd.DataFrame({
        'x': [12, 20, 28, 18, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72],
        'y': [39, 36, 30, 52, 54, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24]
    })

    k = 3
    Beta = 1.5 #integrating clanok
    step_size = 1 #CS clanok
    Pa = 0.25 #sanca na najdenie cudzich vajicok v hniezde - integrating clanok
    max_gen = 10
    dimensions = len(df.columns)
    cl = list(df.columns.values)
    cols = cl.copy()

    # Vytvor pociatocne hniezda, inicializuj ich centroidy a pridaj ich do zoznamu hniezd
    nests = []
    for i in range(k):
        nest = Nest([], [], [], 0)
        nest.set_centroids(df, k)
        nests.append(nest)

    # Pre kazde hniezdy prirad body k centroidom a vypocitaj fitness hniezd
    for nest in nests:
        d_frame = copy.deepcopy(df)
        nest_df = assignment(d_frame, nest.nest_centroids, dimensions)
        nest.nest_clusters = nest_df['closest']
        nest.closest_distance = nest_df['min_dist']
        # print(nest_df['min_dist'].mean())
        nest.fitness = nest_df['min_dist'].mean()

    # Uloz najlepsie hniezdo
    nests.sort(key=lambda x: x.fitness)
    best_nest = nests[0]

    # Inicializacia Levyho letu
    new_nest = Nest([], [], [], 0)
    new_nest.set_centroids(df, k)

    # Vstup do hlavne cyklu algoritmu - prechod hniezdami
    for itr in range(max_gen):

        # Nechaj kukucku Lévyho letom najst nove hniezdo
        # new_nest = copy.deepcopy(nest)
        step = levy_step(Beta, dimensions) * step_size
        for key, value in new_nest.nest_centroids.items():
            new_nest.nest_centroids[key] = value + step

        # Prirad prvky klastrom a vypocitaj fitness noveho hniezda
        d_frame = copy.deepcopy(df)
        new_nest_df = assignment(d_frame, new_nest.nest_centroids, dimensions)
        new_nest.nest_clusters = new_nest_df['closest']
        new_nest.fitness = new_nest_df['min_dist'].mean()

        # Vyber nahodne hniezda z pola
        position = random.randint(0, len(nests) - 1)
        old_nest = copy.deepcopy(nests[position])

        # Ak ma nove hniezdo lepsiu fitness ako to stare nahrad ho
        # Ak nove hniezdo nema mensiu fitness ako to stare nic sa nestane
        if new_nest.fitness < old_nest.fitness:
            nests.pop(position)
            nests.insert(position, new_nest)

        # Prepocitaj centroidy a prirad pozorovania k novym centroidom pre vsetky hniezda
        for nest in nests:

            # Prepocet centroidov hniezd
            for i in nest.nest_centroids.keys():
                num = 0
                for j in cols:
                    suma = 0
                    counter = 0
                    for m in range(len(df.index)):
                        if nest.nest_clusters[m] == i:
                            counter += 1
                            suma += df[j][m]
                    nest.nest_centroids[i][num] = suma/counter
                    num += 1

            # Priradenie pozorovani k centroidom
            d_frame = copy.deepcopy(df)
            new_nest_df = assignment(d_frame, nest.nest_centroids, dimensions)
            nest.nest_clusters = new_nest_df['closest']
            nest.fitness = new_nest_df['min_dist'].mean()
            print('l')


    return
    
   
if __name__ == '__main__':
    main()
