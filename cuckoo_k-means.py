import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import copy
import random
from scipy.spatial import distance
from sklearn.metrics import davies_bouldin_score


# Trieda nest predstavuje jedno konkretne hniezdo v algoritme, teda mozne riesenie
class Nest:

    def __init__(self, nest_centroids, nest_clusters, closest_distance, fitness):
        self.nest_centroids = nest_centroids
        self.nest_clusters = nest_clusters
        self.closest_distance = closest_distance
        self.fitness = fitness

    def set_centroids(self, df, k):
        self.nest_centroids = gen_centroids(df, k)


# Metoda, ktora sluzi na nahodne generovanie centroidov zhlukov
def gen_centroids(d_frame, k):
    max_df = max(pd.DataFrame.max(d_frame, axis=0)) * 1.1

    centroids = {
        i + 1: [random.uniform(0, max_df) for j in range(len(d_frame.columns))]
        for i in range(k)
    }

    return centroids


# Metoda na vratenie stlpca matice
def column(matrix, i):
    return [row[i] for row in matrix]


# Metoda sluziaca na priradenie bodov jednotlivym zhlukom podla vzdialenosti od centroidov
def assign(d_frame, centroids, dim):

    assigned = []

    x = 0
    for i, j in d_frame.iterrows():
        assigned.append([])
        for k in centroids.keys():
            assigned[x].append(distance.euclidean(j.tolist(), centroids[k]))
        x += 1

    for i in centroids.keys():
        d_frame.insert(i + dim - 1, 'distance_from_{}'.format(i), column(assigned, i - 1))

    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
    d_frame['closest'] = d_frame.loc[:, centroid_distance_cols].idxmin(axis=1)
    d_frame['min_dist'] = d_frame.loc[:, centroid_distance_cols].min(axis=1)
    d_frame['closest'] = d_frame['closest'].map(lambda x: int(x.lstrip('distance_from_')))

    return d_frame


# Metoda generujuca jeden krok Levyho letu kukucky
def levy_step(beta, dim):
    x = np.power((math.gamma(1 + beta) * np.sin((np.pi * beta) / 2)) /
                 math.gamma((1 + beta) / 2) * beta * np.power(2, (beta - 1) / 2), 1 / beta)
    a = np.random.normal(0, x, size=dim)
    b = np.random.normal(0, 1, size=dim)
    step = a / np.power(np.fabs(b), 1 / beta)

    return step


# Hlavna metoda algoritmu
def main():

    # Nacitanie datasetu
    df = pd.read_excel('My_data.xlsx', sheet_name='Wine')

    # Inicializacia premennych
    # Premenne kukucieho k-means algoritmu
    k = 3 # Pocet zhlukov
    pop = 10 # Pocet hniezd, standardne oznacovane N
    Beta = 1.5  # Parameter pre Levyho let - integrating clanok
    step_size = 1  # Hodnota, ktorou sa nasobi generovana dlzka Levyho letu z distribucie - CS clanok
    Pa = 0.25  # Pravdepodobnost najdenia cudzich vajicok v hniezde - integrating clanok
    max_gen = 10 # Maximalny pocet generacii
    # Pomocne premenne
    stopping = 0
    same = 0
    dimensions = len(df.columns)
    cl = list(df.columns.values)
    cols = cl.copy()

    # Vytvor pociatocne hniezda, inicializuj ich centroidy a pridaj ich do zoznamu hniezd
    nests = []
    for i in range(pop):
        nest = Nest([], [], [], 0)
        nest.set_centroids(df, k)
        nests.append(nest)

    # Pre kazde hniezdo prirad body k centroidom a vypocitaj fitness hniezd
    for nest in nests:
        d_frame = copy.deepcopy(df)
        nest_df = assign(d_frame, nest.nest_centroids, dimensions)
        nest.nest_clusters = nest_df['closest']
        nest.closest_distance = nest_df['min_dist']
        nest.fitness = nest_df['min_dist'].mean()

    # Uloz najlepsie hniezdo
    nests.sort(key=lambda x: x.fitness)
    best_nest = copy.deepcopy(nests[0])

    # Uloz list hniezd kvoli overeniu stopping criteria
    old_nests = copy.deepcopy(nests)

    # Vstup do hlavne cyklu algoritmu - ukoncuje sa ked sa dosiahne maximalny pocet generacii
    # alebo kriterium zastavenia
    for itr in range(max_gen):

        # Pre vsetky hniezda v zozname hniezd
        for i in range(0, len(nests)):

            # Vygeneruj kro Levyho letu, prelet kukuckou z hniezda na nove miesto
            levy_nest = copy.deepcopy(nests[i])
            step = levy_step(Beta, dimensions) * step_size
            for key, value in levy_nest.nest_centroids.items():
                levy_nest.nest_centroids[key] = value + step

            # Prirad prvky zhlukom a vypocitaj fitness noveho hniezda
            d_frame = copy.deepcopy(df)
            new_nest_df = assign(d_frame, levy_nest.nest_centroids, dimensions)
            levy_nest.nest_clusters = new_nest_df['closest']
            levy_nest.closest_distance = new_nest_df['min_dist']
            levy_nest.fitness = new_nest_df['min_dist'].mean()

            # Vyber nahodne hniezdo z pola
            position = random.randint(0, len(nests) - 1)
            old_nest = copy.deepcopy(nests[position])

            # Ak ma nove hniezdo lepsiu fitness ako to stare nahrad ho
            # Ak nove hniezdo nema mensiu fitness ako to stare nic sa nestane
            if levy_nest.fitness < old_nest.fitness:
                nests.pop(position)
                nests.insert(position, levy_nest)

        # Zorad hniezda podla fitness a uloz najlepsie
        nests.sort(key=lambda x: x.fitness)
        best_nest = copy.deepcopy(nests[0])

        # Zahod hniezda na zaklade pravdepodonosti Pa okrem najlepsieho
        # Ak si hniezdo zahodil nahodne vygeneruj nove
        for i in range(1, len(nests)):
            probability = random.randint(0, 100)
            if probability < Pa * 100:
                nests.pop(i)
                new_nest = Nest([], [], [], 0)
                new_nest.set_centroids(df, k)
                d_frame = copy.deepcopy(df)
                new_nest_df = assign(d_frame, new_nest.nest_centroids, dimensions)
                new_nest.nest_clusters = new_nest_df['closest']
                new_nest.closest_distance = new_nest_df['min_dist']
                new_nest.fitness = new_nest_df['min_dist'].mean()
                nests.insert(i, new_nest)

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
                    if suma != 0:
                        nest.nest_centroids[i][num] = suma / counter
                    num += 1

            # Priradenie pozorovani k centroidom
            d_frame = copy.deepcopy(df)
            new_nest_df = assign(d_frame, nest.nest_centroids, dimensions)
            nest.nest_clusters = new_nest_df['closest']
            nest.closest_distance = new_nest_df['min_dist']
            nest.fitness = new_nest_df['min_dist'].mean()

        # Zorad hniezda podla fitness a uloz najlepsie
        nests.sort(key=lambda x: x.fitness)
        best_nest = copy.deepcopy(nests[0])

        # Porovnaj fitness stareho a noveho zoznamu hniezd
        fit_dist = 0
        for nest_old, nest_new in zip(old_nests, nests):
            if nest_old.fitness == nest_new.fitness:
                fit_dist += 1
            else:
                break

        if fit_dist == pop:
            same = 1
        else:
            same = 0

        if same == 1:
            stopping += 1
        else:
            stopping = 0

        # Stopping kriterium - moje kriterium je, ked sa 3-krat po sebe nezmeni fitness hodnota
        if stopping == 3:
            break

    # Uloz vysledok
    np.savetxt(r'result1.txt', best_nest.nest_clusters, fmt='%d')

    # Ohodnot riesenie ulohy zhlukovania Davies-Bouldinovym indexom
    print(davies_bouldin_score(df, best_nest.nest_clusters))

    # Vizualizuj zhluky
    # colors = {1: 'r', 2: 'g', 3: 'b'}
    # df['color'] = best_nest.nest_clusters.map(lambda x: colors[x])
    # fig = plt.figure(figsize=(5, 5))
    # plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
    # for i in best_nest.nest_centroids.keys():
    #     plt.scatter(*best_nest.nest_centroids[i], color=colors[i])
    # plt.show()

    return

# Volanie hlavnej metody algoritmu
if __name__ == '__main__':
    main()
