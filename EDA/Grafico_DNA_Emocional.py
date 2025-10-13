import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

try:
    csv_path = Path(__file__).resolve().parent.parent / 'DataSet' / 'spotify_songs.csv'
    df = pd.read_csv(csv_path)

    # --- Gráfico Analítico: O "DNA Emocional" de Cada Gênero ---
    print("\nGerando o painel 'DNA Emocional' por gênero...")

    sns.set_theme(style="white")

    g = sns.FacetGrid(df, col="playlist_genre", col_wrap=3, height=4)
    
    g.map_dataframe(sns.kdeplot, x="valence", y="energy", fill=True, cmap='viridis', thresh=0.05)
    
    for ax in g.axes.flat:
        ax.axvline(x=0.5, ls='--', color='white', lw=1.5, alpha=0.8)
        ax.axhline(y=0.5, ls='--', color='white', lw=1.5, alpha=0.8)
        
        ax.text(0.75, 0.75, "Feliz", color="white", ha="center", va="center", fontsize=10, bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.4))
        ax.text(0.25, 0.25, "Triste", color="white", ha="center", va="center", fontsize=10, bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.4))
        ax.text(0.25, 0.75, "Intenso", color="white", ha="center", va="center", fontsize=10, bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.4))
        ax.text(0.75, 0.25, "Calmo", color="white", ha="center", va="center", fontsize=10, bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.4))

    g.fig.suptitle('O "DNA Emocional" de Cada Gênero Musical', fontsize=20, y=1.03)
    g.set_axis_labels("Valência (Positividade)", "Energia")
    plt.tight_layout()
    plt.savefig("dna_emocional_generos.png", bbox_inches='tight')
    print("--> Gráfico 'DNA Emocional' salvo como 'dna_emocional_generos.png'")

except FileNotFoundError:
    print("\nErro: O arquivo 'spotify_songs.csv' não foi encontrado.")
except Exception as e:
    print(f"\nOcorreu um erro inesperado: {e}")