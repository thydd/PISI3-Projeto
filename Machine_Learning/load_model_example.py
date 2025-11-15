"""
🎵 Script de Exemplo: Carregamento e Uso do Modelo Salvo
=========================================================

Este script demonstra como carregar um modelo treinado e fazer predições.
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def carregar_modelo_mais_recente():
    """Carrega o modelo, scaler e metadados mais recentes."""
    models_dir = Path(__file__).resolve().parent / 'saved_models'
    
    if not models_dir.exists():
        raise FileNotFoundError("Diretório 'saved_models' não encontrado. Execute primeiro Mode_Classification_Enhanced.py")
    
    # Encontrar arquivos mais recentes
    model_files = sorted(models_dir.glob('mode_classifier_*.pkl'))
    scaler_files = sorted(models_dir.glob('scaler_*.pkl'))
    metadata_files = sorted(models_dir.glob('metadata_*.pkl'))
    
    if not model_files:
        raise FileNotFoundError("Nenhum modelo encontrado. Execute primeiro Mode_Classification_Enhanced.py")
    
    # Carregar mais recentes
    model_path = model_files[-1]
    scaler_path = scaler_files[-1]
    metadata_path = metadata_files[-1]
    
    print("=" * 80)
    print("📦 CARREGANDO MODELO SALVO")
    print("=" * 80)
    
    with open(model_path, 'rb') as f:
        modelo = pickle.load(f)
    print(f"\n✓ Modelo carregado: {model_path.name}")
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"✓ Scaler carregado: {scaler_path.name}")
    
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    print(f"✓ Metadados carregados: {metadata_path.name}")
    
    return modelo, scaler, metadata


def exibir_info_modelo(metadata):
    """Exibe informações sobre o modelo carregado."""
    print("\n" + "=" * 80)
    print("ℹ️  INFORMAÇÕES DO MODELO")
    print("=" * 80)
    
    print(f"\n🤖 Modelo: {metadata['modelo']}")
    print(f"📊 Acurácia no Teste: {metadata['acuracia_teste']:.4f}")
    
    if metadata.get('roc_auc'):
        print(f"🎯 ROC-AUC Score: {metadata['roc_auc']:.4f}")
    
    print(f"\n📅 Data de Treinamento: {metadata['timestamp']}")
    
    print(f"\n📋 Features Necessárias ({len(metadata['features'])}):")
    for i, feat in enumerate(metadata['features'], 1):
        print(f"   {i:2d}. {feat}")
    
    print(f"\n⚙️  Hiperparâmetros Otimizados:")
    print(f"   {metadata['melhores_parametros']}")


def prever_modo(modelo, scaler, features, dados_musica):
    """
    Faz predição do modo musical.
    
    Args:
        modelo: Modelo treinado
        scaler: Scaler para preprocessamento
        features: Lista de nomes das features
        dados_musica: DataFrame com os dados da música
    
    Returns:
        dict: Resultado da predição
    """
    # Verificar se todas as features estão presentes
    missing = set(features) - set(dados_musica.columns)
    if missing:
        raise ValueError(f"Features faltantes: {missing}")
    
    # Garantir ordem correta das features
    dados_musica = dados_musica[features]
    
    # Preprocessar
    dados_scaled = scaler.transform(dados_musica)
    
    # Prever
    predicao = modelo.predict(dados_scaled)[0]
    
    # Obter probabilidades
    if hasattr(modelo, 'predict_proba'):
        probabilidades = modelo.predict_proba(dados_scaled)[0]
        prob_menor = probabilidades[0]
        prob_maior = probabilidades[1]
    else:
        prob_menor = 1 - predicao
        prob_maior = predicao
    
    return {
        'predicao': predicao,
        'modo': 'Maior (Major)' if predicao == 1 else 'Menor (Minor)',
        'prob_menor': prob_menor,
        'prob_maior': prob_maior,
        'confianca': max(prob_menor, prob_maior)
    }


def exibir_resultado(resultado, nome_musica=None):
    """Exibe o resultado da predição de forma formatada."""
    print("\n" + "=" * 80)
    if nome_musica:
        print(f"🎵 RESULTADO DA PREDIÇÃO: {nome_musica}")
    else:
        print("🎵 RESULTADO DA PREDIÇÃO")
    print("=" * 80)
    
    # Emoji baseado no resultado
    emoji = "😊" if resultado['predicao'] == 1 else "😔"
    
    print(f"\n{emoji} Modo Predito: {resultado['modo']}")
    print(f"\n📊 Probabilidades:")
    print(f"   Menor (Minor): {resultado['prob_menor']:.2%} {'█' * int(resultado['prob_menor'] * 50)}")
    print(f"   Maior (Major): {resultado['prob_maior']:.2%} {'█' * int(resultado['prob_maior'] * 50)}")
    print(f"\n✅ Confiança: {resultado['confianca']:.2%}")


def main():
    """Função principal com exemplos de uso."""
    
    # 1. Carregar modelo
    try:
        modelo, scaler, metadata = carregar_modelo_mais_recente()
    except FileNotFoundError as e:
        print(f"\n❌ Erro: {e}")
        return
    
    # 2. Exibir informações
    exibir_info_modelo(metadata)
    
    # 3. Exemplos de Predição
    print("\n" + "=" * 80)
    print("🧪 EXEMPLOS DE PREDIÇÃO")
    print("=" * 80)
    
    # Exemplo 1: Música alegre e energética (esperado: Maior)
    musica_alegre = pd.DataFrame({
        'danceability': [0.735],
        'energy': [0.826],
        'key': [1],
        'loudness': [-6.340],
        'speechiness': [0.0461],
        'acousticness': [0.0514],
        'instrumentalness': [0.000902],
        'liveness': [0.159],
        'valence': [0.824],  # Alta valência (alegre)
        'tempo': [128.002],
        'duration_ms': [255349],
        'track_popularity': [76]
    })
    
    resultado1 = prever_modo(modelo, scaler, metadata['features'], musica_alegre)
    exibir_resultado(resultado1, "Música Alegre e Energética")
    
    # Exemplo 2: Música triste e lenta (esperado: Menor)
    musica_triste = pd.DataFrame({
        'danceability': [0.435],
        'energy': [0.226],
        'key': [0],
        'loudness': [-18.840],
        'speechiness': [0.0361],
        'acousticness': [0.814],
        'instrumentalness': [0.0902],
        'liveness': [0.0959],
        'valence': [0.224],  # Baixa valência (triste)
        'tempo': [78.002],
        'duration_ms': [285349],
        'track_popularity': [54]
    })
    
    resultado2 = prever_modo(modelo, scaler, metadata['features'], musica_triste)
    exibir_resultado(resultado2, "Música Triste e Lenta")
    
    # Exemplo 3: Carregar do dataset real
    print("\n" + "=" * 80)
    print("🎵 TESTANDO COM DADOS REAIS DO DATASET")
    print("=" * 80)
    
    csv_path = Path(__file__).resolve().parent.parent / 'DataSet' / 'spotify_songs.csv'
    
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        
        # Pegar 5 músicas aleatórias
        amostras = df[metadata['features'] + ['track_name', 'track_artist', 'mode']].dropna().sample(5, random_state=42)
        
        print(f"\nTestando {len(amostras)} músicas aleatórias do dataset:\n")
        
        acertos = 0
        for idx, row in amostras.iterrows():
            musica_data = pd.DataFrame([row[metadata['features']]])
            resultado = prever_modo(modelo, scaler, metadata['features'], musica_data)
            
            modo_real = 'Maior' if row['mode'] == 1 else 'Menor'
            correto = "✅" if resultado['predicao'] == row['mode'] else "❌"
            
            print(f"{correto} {row['track_name'][:40]:40s} - {row['track_artist'][:20]:20s}")
            print(f"   Real: {modo_real:5s} | Predito: {resultado['modo']:13s} | Conf: {resultado['confianca']:.1%}")
            
            if resultado['predicao'] == row['mode']:
                acertos += 1
        
        print(f"\n📊 Acurácia nesta amostra: {acertos}/{len(amostras)} ({acertos/len(amostras):.1%})")
    
    # 4. Instruções para uso personalizado
    print("\n" + "=" * 80)
    print("💡 COMO USAR COM SUAS PRÓPRIAS MÚSICAS")
    print("=" * 80)
    
    print("""
Para fazer predições com suas próprias músicas:

1. Crie um DataFrame com as features necessárias:
   
   from modo_predictor import carregar_modelo_mais_recente, prever_modo
   
   minha_musica = pd.DataFrame({
       'danceability': [0.7],
       'energy': [0.8],
       'key': [5],
       'loudness': [-5.0],
       'speechiness': [0.05],
       'acousticness': [0.1],
       'instrumentalness': [0.0],
       'liveness': [0.1],
       'valence': [0.8],
       'tempo': [120.0],
       'duration_ms': [200000],
       'track_popularity': [70]
   })
   
2. Carregue o modelo e faça a predição:
   
   modelo, scaler, metadata = carregar_modelo_mais_recente()
   resultado = prever_modo(modelo, scaler, metadata['features'], minha_musica)
   
3. Use o resultado:
   
   print(f"Modo: {resultado['modo']}")
   print(f"Confiança: {resultado['confianca']:.2%}")
""")
    
    print("\n" + "=" * 80)
    print("✅ DEMONSTRAÇÃO CONCLUÍDA!")
    print("=" * 80)


if __name__ == "__main__":
    main()
