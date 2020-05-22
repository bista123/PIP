import pandas as pd
import numpy as np
import copy
from scipy.spatial import distance
import random
from sklearn.metrics import davies_bouldin_score


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
        d_frame.insert(i+dim-1, 'distance_from_{}'.format(i), column(assigned, i-1))

    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
    d_frame['closest'] = d_frame.loc[:, centroid_distance_cols].idxmin(axis=1)
    d_frame['closest'] = d_frame['closest'].map(lambda x: int(x.lstrip('distance_from_')))

    return d_frame


# Hlavna metoda algoritmu
def main():

    # Nacitanie datasetu
    df = pd.read_excel('My_data.xlsx', sheet_name='Haberman')

    # Inicializacia premennych
    k = 2  # Pocet zhlukov
    itr = 10 # Maximalny pocet iteracii
    max_df = max(pd.DataFrame.max(df, axis=0)) * 1.1
    cl = list(df.columns.values)
    cols = cl.copy()
    last = str(cl.pop())
    dimensions = len(df.columns)
    save_df = copy.deepcopy(df)

    # Inicializuj centroidy
    centroids = {}
    keys = range(k)
    for i in keys:
        possit = random.randint(0, len(df)-1)
        centroids[i+1] = df.loc[possit].values.tolist()

    # Prirad body k centroidom
    df = assign(df, centroids, dimensions)

    # Vypocitaj poluhu novych centroidov
    for i in centroids.keys():
        num = 0
        for j in cols:
            suma = 0
            counter = 0
            for m in range(len(df.index)):
                if df['closest'][m] == i:
                    counter += 1
                    suma += df[j][m]
            if suma != 0:
                centroids[i][num] = suma / counter
            num += 1

    iterator = 0
    # Hlavny cyklus algoritmu - ide po kriterium zastavenia
    while True:

        iterator += 1

        # Uloz povodne centroidy
        closest_centroids = df['closest'].copy(deep=True)

        # Vypocitaj novu polohu centroidov, prirad centroidom body
        for i in centroids.keys():
            num = 0
            for j in cols:
                suma = 0
                counter = 0
                for m in range(len(df.index)):
                    if df['closest'][m] == i:
                        counter += 1
                        suma += df[j][m]
                if suma != 0:
                    centroids[i][num] = suma / counter
                num += 1
        df = df.loc[:, :last]
        df = assign(df, centroids, dimensions)

        # Ak je splnene kriterium zastavenia
        if (closest_centroids.equals(df['closest'])) or (iterator == itr):
            # Uloz vysledky
            df = df.iloc[:, -1]
            np.savetxt(r'result.txt', df.values, fmt='%d')
            break

    # Ohodnot riesenie ulohy zhlukovania Davies-Bouldinovym indexom
    print(davies_bouldin_score(save_df, df))

# Volanie hlavnej metody algoritmu
if __name__ == '__main__':
    main()
    
