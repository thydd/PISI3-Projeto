import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def carregar_dados(caminho_dataset="../DataSet/spotify_songs.csv"):
    if not os.path.exists(caminho_dataset):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_dataset}")
    
    print("[*] Carregando dataset...")
    df = pd.read_csv(caminho_dataset)
    
    features = ['valence', 'energy', 'danceability', 'tempo', 'acousticness']
    df_features = df[features].copy()
    
    print(f"[+] Dataset carregado com {len(df)} registros.")
    return df, df_features

def preprocessar_dados(df_features):
    print("\n[*] Padronizando dados...")
    scaler = StandardScaler()
    dados_normalizados = scaler.fit_transform(df_features)
    print("[+] Dados normalizados.")
    return dados_normalizados, scaler

def escolher_numero_clusters(dados_normalizados, max_clusters=10):
    wcss = []
    for i in range(1, max_clusters + 1):
        km = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
        km.fit(dados_normalizados)
        wcss.append(km.inertia_)
    
    plt.figure(figsize=(8,5))
    plt.plot(range(1, max_clusters + 1), wcss, marker='o')
    plt.xlabel("Número de clusters")
    plt.ylabel("WCSS (Inércia)")
    plt.title("Método do Cotovelo para Escolher o Número de Clusters")
    plt.grid(True, alpha=0.3)
    plt.show()

def aplicar_kmeans(dados_normalizados, n_clusters=5):
    print(f"\n[*] Aplicando K-Means++ com {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
    clusters = kmeans.fit_predict(dados_normalizados)
    print("[+] K-Means++ concluído.")
    return clusters, kmeans

def analisar_clusters(df, df_features, clusters, kmeans, features, scaler):
    df_resultado = df.copy()
    df_resultado['cluster'] = clusters

    # Médias das features por cluster
    medias = df_resultado.groupby('cluster')[features].mean()
    print("\n[*] Médias das features por cluster:\n")
    print(medias)

    # Plotagem 2D (valence x energy)
    x_feat, y_feat = 'valence', 'energy'
    plt.figure(figsize=(8,6))
    plt.scatter(
        df_features[x_feat],
        df_features[y_feat],
        c=clusters,
        cmap='tab10',
        alpha=0.6,
        s=30
    )

    # Centròides nos valores originais
    centros_originais = scaler.inverse_transform(kmeans.cluster_centers_)
    plt.scatter(
        centros_originais[:, features.index(x_feat)],
        centros_originais[:, features.index(y_feat)],
        c='black',
        s=200,
        alpha=0.9,
        marker='X',
        label='Centróides'
    )

    plt.xlabel(x_feat)
    plt.ylabel(y_feat)
    plt.title("Clusterização de Músicas (K-Means++)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return df_resultado, medias

def main():
    caminho = os.path.join(os.path.dirname(__file__), "..", "DataSet", "spotify_songs.csv")
    caminho = os.path.abspath(caminho)
    print("[*] Caminho do dataset:", caminho)

    df, df_features = carregar_dados(caminho)
    features = ['valence', 'energy', 'danceability', 'tempo', 'acousticness']

    # Normalização
    dados_normalizados, scaler = preprocessar_dados(df_features)

    # Cotovelo para decidir número de clusters
    escolher_numero_clusters(dados_normalizados, max_clusters=10)
    # Após analisar o gráfico, escolha o número de clusters desejado:
    n_clusters = 3

    # K-Means++
    clusters, kmeans = aplicar_kmeans(dados_normalizados, n_clusters=n_clusters)

    # Análise e plot
    df_resultado, medias = analisar_clusters(df, df_features, clusters, kmeans, features, scaler)

    # Salvar resultados
    df_resultado.to_csv("resultados_clusterizacao.csv", index=False)
    print("\n[*] Resultados salvos em 'resultados_clusterizacao.csv'.")

if __name__ == "__main__":
    main()

