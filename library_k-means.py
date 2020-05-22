import pandas as pd
import numpy as np
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import KMeans

# Hlavna metoda algoritmu
def main():

    # Nacitanie datasetu
    df = pd.read_excel('My_data.xlsx', sheet_name='Wine')

    # Pouzitie kniznicneho k-means algoritmu
    kmeans = KMeans(n_clusters=2, init='random', max_iter=10, n_init=1, random_state=1)
    y_kmeans = kmeans.fit_predict(df)
    np.savetxt(r'result.txt', y_kmeans, fmt='%d')
    labels = kmeans.labels_
    print(davies_bouldin_score(df, labels))
    
    return

if __name__ == '__main__':
    main()
