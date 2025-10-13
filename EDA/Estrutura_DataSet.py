import pandas as pd
from pathlib import Path

try:
    # Carregue o dataset a partir do arquivo CSV
    csv_path = Path(__file__).resolve().parent.parent / 'DataSet' / 'spotify_songs.csv'
    df = pd.read_csv(csv_path)

    print("--- Análise Exploratória da Estrutura do Dataset Spotify ---")

    num_rows, num_cols = df.shape
    print(f"\n[ESTRUTURA GERAL]")
    print(f"O dataset possui {num_rows} linhas (representando músicas) e {num_cols} colunas (características).")

    print("\n[INFORMAÇÕES DAS COLUNAS E TIPOS DE DADOS]")
    df.info()

    print("\n[VERIFICAÇÃO DE DADOS FALTANTES POR COLUNA]")
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    
    if not missing_values.empty:
        print("Foram encontradas as seguintes colunas com dados faltantes:")
        print(missing_values)
    else:
        print("Ótima notícia! Nenhuma coluna com dados faltantes foi encontrada.")

    print("\n[CONTAGEM DE ITENS ÚNICOS]")
    num_songs = len(df)
    num_artists = df['track_artist'].nunique()
    num_albums = df['track_album_name'].nunique()
    
    print(f"Número total de entradas de músicas: {num_songs}")
    print(f"Número de artistas únicos: {num_artists}")
    print(f"Número de álbuns únicos: {num_albums}")

    print("\n[AMOSTRA DOS DADOS - PRIMEIRAS 5 MÚSICAS]")
    print(df.head())

except FileNotFoundError:
    print("\nErro: O arquivo 'spotify_songs.csv' não foi encontrado.")
    print("Por favor, verifique se o nome do arquivo está correto e se ele está no diretório esperado.")
except Exception as e:
    print(f"\nOcorreu um erro inesperado: {e}")