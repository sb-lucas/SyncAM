"""Script para agregar resultados de múltiplos CSVs em um único arquivo consolidado.

Agrega os arquivos de resultados MAE e MSE, ignorando os arquivos de detalhes individuais
(note_stats e all_note_errors).
"""

import pandas as pd
import os
import glob

def aggregate_results():
    """Agregar todos os resultados finais em um único CSV."""
    
    # Padrões de arquivos a incluir (MAE e MSE)
    patterns_to_include = [
        'results_*_mae.csv',
        'results_*_mse.csv'
    ]
    
    # Padrões de arquivos a excluir (detalhes individuais)
    patterns_to_exclude = [
        '*note_stats*',
        '*all_note_errors*'
    ]
    
    all_results = []
    
    for pattern in patterns_to_include:
        files = glob.glob(pattern)
        
        for file in files:
            # Verificar se o arquivo deve ser excluído
            should_exclude = any(excl in file for excl in patterns_to_exclude)
            if should_exclude:
                continue
            
            print(f"Processando: {file}")
            
            # Ler o CSV
            df = pd.read_csv(file)
            
            # Extrair informações do nome do arquivo
            # Formato: results_{dataset}_{model}_{method}_{metric}.csv
            parts = file.replace('results_', '').replace('.csv', '').split('_')
            
            # Identificar dataset, model, method e metric
            if 'musicnet' in file:
                dataset = 'musicnet'
                rest = file.replace('results_musicnet_', '').replace('.csv', '')
            elif 'smd' in file:
                dataset = 'smd'
                rest = file.replace('results_smd_', '').replace('.csv', '')
            else:
                dataset = 'unknown'
                rest = file.replace('results_', '').replace('.csv', '')
            
            # Identificar modelo
            if 'basicpitch' in file or 'basicpith' in file:
                model = 'basic_pitch'
            elif 'bytedance' in file:
                model = 'bytedance'
            else:
                model = 'unknown'
            
            # Métrica é a última parte
            metric = rest.split('_')[-1]  # mae ou mse
            
            # Adicionar colunas de contexto
            df['dataset'] = dataset
            df['model'] = model
            df['metric'] = metric
            
            all_results.append(df)
    
    if not all_results:
        print("Nenhum arquivo encontrado para agregar.")
        return
    
    # Concatenar todos os resultados
    final_df = pd.concat(all_results, ignore_index=True)
    
    # Reorganizar colunas para melhor visualização
    columns_order = ['dataset', 'model', 'method', 'metric', 'mae', 'mse', 'rmse', 'files_processed']
    
    # Adicionar apenas as colunas que existem
    final_columns = [col for col in columns_order if col in final_df.columns]
    final_df = final_df[final_columns]
    
    # Ordenar por dataset, model e method
    final_df = final_df.sort_values(['dataset', 'model', 'method', 'metric'])
    
    # Salvar resultado agregado
    output_file = 'aggregated_transcription_results.csv'
    final_df.to_csv(output_file, index=False)
    
    print(f"\n{'='*60}")
    print(f"Resultados agregados salvos em: {output_file}")
    print(f"Total de registros: {len(final_df)}")
    print(f"\nResumo:")
    print(final_df.to_string(index=False))
    print(f"{'='*60}")

if __name__ == "__main__":
    aggregate_results()
