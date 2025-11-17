import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

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
    """
    Aplica dois métodos para determinar o número ótimo de clusters:
    1. Método do Cotovelo (Elbow Method) - minimiza WCSS
    2. Método da Silhueta (Silhouette Method) - maximiza coesão intra-cluster e separação inter-cluster
    """
    print("\n[*] Calculando métricas para escolha do número de clusters...")
    wcss = []
    silhouette_scores = []
    
    for i in range(1, max_clusters + 1):
        km = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
        km.fit(dados_normalizados)
        wcss.append(km.inertia_)
        
        # Silhouette score só funciona com 2+ clusters
        if i > 1:
            score = silhouette_score(dados_normalizados, km.labels_)
            silhouette_scores.append(score)
            print(f"    K={i}: WCSS={km.inertia_:.2f}, Silhouette={score:.4f}")
        else:
            silhouette_scores.append(0)
            print(f"    K={i}: WCSS={km.inertia_:.2f}")
    
    # Criar figura com 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Método do Cotovelo
    ax1.plot(range(1, max_clusters + 1), wcss, marker='o', linewidth=2, markersize=8, color='steelblue')
    ax1.set_xlabel("Número de clusters", fontsize=12)
    ax1.set_ylabel("WCSS (Inércia)", fontsize=12)
    ax1.set_title("Método do Cotovelo\n(Menor WCSS = Melhor)", fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(1, max_clusters + 1))
    
    # Plot 2: Método da Silhueta
    ax2.plot(range(1, max_clusters + 1), silhouette_scores, marker='s', linewidth=2, markersize=8, color='coral')
    ax2.set_xlabel("Número de clusters", fontsize=12)
    ax2.set_ylabel("Silhouette Score", fontsize=12)
    ax2.set_title("Método da Silhueta\n(Maior Score = Melhor)", fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(1, max_clusters + 1))
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Destacar o melhor score de silhueta
    if len(silhouette_scores) > 1:
        best_k = np.argmax(silhouette_scores[1:]) + 2  # +2 porque começamos do índice 1 e K começa em 1
        best_score = silhouette_scores[best_k - 1]
        ax2.scatter([best_k], [best_score], c='red', s=200, marker='*', 
                   edgecolors='black', linewidths=2, zorder=5, label=f'Melhor K={best_k}')
        ax2.legend()
    
    plt.tight_layout()
    plt.savefig('metricas_clusters_dna.png', dpi=300, bbox_inches='tight')
    print("\n[+] Gráficos de métricas salvos em 'metricas_clusters_dna.png'")
    plt.show()
    
    return silhouette_scores

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

    # Métodos do Cotovelo e Silhueta para decidir número de clusters
    silhouette_scores = escolher_numero_clusters(dados_normalizados, max_clusters=13)
    
    # Sugestão automática baseada no melhor silhouette score
    best_k = np.argmax(silhouette_scores[1:]) + 2
    print(f"\n[*] Sugestão baseada no Silhouette Score: {best_k} clusters")
    print("[*] Após analisar os gráficos, escolha o número de clusters desejado.")
    n_clusters = 3  # Ajuste conforme necessário

    # K-Means++
    clusters, kmeans = aplicar_kmeans(dados_normalizados, n_clusters=n_clusters)

    # Análise e plot
    df_resultado, medias = analisar_clusters(df, df_features, clusters, kmeans, features, scaler)

    # Salvar resultados
    df_resultado.to_csv("resultados_clusterizacao.csv", index=False)
    print("\n[*] Resultados salvos em 'resultados_clusterizacao.csv'.")

if __name__ == "__main__":
    main()

