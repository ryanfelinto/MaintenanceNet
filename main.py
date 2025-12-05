"""
Script principal de execução do Sistema de Manutenção Preditiva DDQN.
Versão Otimizada para CPU e Teste Rápido.
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Importação dos módulos locais
from config import Config
from trainer import DDQNTrainer, BaselineComparison
from visualizer import MaintenanceVisualizer

def main():
    # Cria diretório para salvar resultados
    os.makedirs("results", exist_ok=True)
    
    print("="*60)
    print("INICIALIZANDO SISTEMA DDQN DE MANUTENÇÃO")
    print("="*60)
    
    # 1. Configuração Inicial
    config = Config()
    
    # ==============================================================================
    # CONFIGURAÇÕES DE OTIMIZAÇÃO (PARA CORRIGIR O TRAVAMENTO NA CPU)
    # ==============================================================================
    # O problema anterior era que o episódio durava 3000 passos.
    # Se o agente decidisse fazer 3000 preventivas, o computador travava calculando.
    # Reduzimos para 200 passos por episódio para garantir velocidade.
    config.env.episode_max_steps = 200 
    
    # Reduzimos o tamanho do "lote" de aprendizado para a CPU não sofrer
    config.ddqn.batch_size = 32
    
    # Rede neural leve para teste rápido
    config.ddqn.hidden_layers = [64, 64]
    
    # Número de episódios total (suficiente para ver resultado em poucos minutos)
    config.training.n_episodes = 300
    # ==============================================================================

    # Configurações do Cenário (Custos e Máquina)
    config.weibull.beta = 2.5       # Desgaste acentuado (peça envelhece)
    config.weibull.eta = 1000.0     # Vida média da peça ~1000h
    
    # Detecção automática de Hardware
    if torch.cuda.is_available():
        config.device = "cuda"
    else:
        config.device = "cpu"
        print("! AVISO: GPU não detectada. Rodando em CPU com configurações otimizadas.")

    print(f"Hardware: {config.device.upper()}")
    print(f"Episódios: {config.training.n_episodes}")
    print(f"Max Steps por Episódio: {config.env.episode_max_steps}")
    
    # 2. Treinamento do Agente
    print("\n--- [1/3] Iniciando Treinamento ---")
    trainer = DDQNTrainer(config)
    
    # O treinamento retorna o histórico para plotagem
    history = trainer.train()
    
    # Salva gráfico de evolução
    print("Gerando gráficos de aprendizado...")
    viz = MaintenanceVisualizer()
    viz.plot_training_curves(history, save_path="results/training_curves.png")
    
    # 3. Teste e Comparação
    print("\n--- [2/3] Comparando Estratégias ---")
    baseline = BaselineComparison(config)
    
    # Estratégia 1: Intervalo Fixo (Preventiva tradicional)
    # Define troca a cada 700h (70% da vida útil média)
    fixed_interval = 700
    print(f"Rodando simulação: Troca Fixa a cada {fixed_interval}h...")
    res_fixed = baseline.run_fixed_interval(interval=fixed_interval, n_episodes=100)
    
    # Estratégia 2: Agente Inteligente (DDQN)
    print("Rodando simulação: Agente Inteligente (DDQN)...")
    res_agent = trainer.test_policy(n_episodes=100)
    
    # 4. Resultados Finais
    print("\n--- [3/3] Resultados Finais e Política ---")
    
    # Análise da "Fronteira de Decisão" (Q-Values)
    # Mostra em qual idade o agente aprendeu que vale a pena trocar a peça
    threshold = viz.plot_policy_structure(trainer.agent, 
                                          max_age=2000, 
                                          save_path="results/policy_structure.png")
    
    if threshold:
        print(f"\n>>> INSIGHT: O Agente aprendeu a trocar a peça com aprox. {threshold:.0f} horas.")
    else:
        print("\n>>> INSIGHT: O Agente ainda não definiu um ponto fixo de troca (pode precisar de mais treino).")

    # Comparativo Financeiro
    avg_cost_fixed = np.mean(res_fixed['costs'])
    avg_cost_agent = np.mean(res_agent['costs'])
    
    print("\n" + "="*40)
    print("       RESUMO DE CUSTOS (MÉDIA)")
    print("="*40)
    print(f"Manutenção Fixa ({fixed_interval}h):  R$ {avg_cost_fixed:10.2f}")
    print(f"Agente Inteligente (IA):    R$ {avg_cost_agent:10.2f}")
    print("-" * 40)
    
    if avg_cost_agent < avg_cost_fixed:
        economy = ((avg_cost_fixed - avg_cost_agent) / avg_cost_fixed) * 100
        print(f"✅ SUCESSO: A IA reduziu os custos em {economy:.2f}%!")
    else:
        print("⚠️ AVISO: A IA empatou ou perdeu. Tente aumentar 'n_episodes' para 500 ou 1000.")
        
    print("="*40)
    print("Gráficos salvos na pasta 'results/'")

if __name__ == "__main__":
    main()