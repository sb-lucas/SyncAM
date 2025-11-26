#!/usr/bin/env python3
"""
Script para criar tabelas em imagem dos resultados de transcrição
agrupados por dataset e instrumento
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import argparse

def parse_dataset_instrument(path_str):
    """
    Extrair dataset e instrumento do caminho
    
    Retorna:
        (dataset, instrument) onde dataset pode ser:
        - 'audio_original'
        - 'FluidR3_GM'
        - 'pianoteq'
        - 'pro_gm_samples_studio'
    """
    path = Path(path_str)
    parts = path.parts
    
    if len(parts) == 1 or 'audio_original' in str(path):
        return ('audio_original', 'audio_original')
    
    # Procurar pelos datasets conhecidos
    if 'FluidR3_GM' in parts:
        idx = parts.index('FluidR3_GM')
        if idx + 1 < len(parts):
            return ('FluidR3_GM', parts[idx + 1])
        return ('FluidR3_GM', 'FluidR3_GM')
    
    if 'pianoteq' in parts:
        idx = parts.index('pianoteq')
        if idx + 1 < len(parts):
            return ('pianoteq', parts[idx + 1])
        return ('pianoteq', 'pianoteq')
    
    if 'pro_gm_samples_studio' in parts:
        idx = parts.index('pro_gm_samples_studio')
        if idx + 1 < len(parts):
            return ('pro_gm_samples_studio', parts[idx + 1])
        return ('pro_gm_samples_studio', 'pro_gm_samples_studio')
    
    return ('unknown', 'unknown')

def load_and_process_csv(csv_path, filter_type='filtered'):
    """
    Carregar CSV e processar dados agrupando por dataset/instrumento
    """
    df = pd.read_csv(csv_path)
    
    # Adicionar colunas de dataset e instrumento
    df['dataset'] = df['path'].apply(lambda x: parse_dataset_instrument(x)[0])
    df['instrument'] = df['path'].apply(lambda x: parse_dataset_instrument(x)[1])
    
    # Agrupar por dataset e instrumento
    grouped = df.groupby(['dataset', 'instrument'])
    
    # Calcular estatísticas
    results = []
    for (dataset, instrument), group in grouped:
        # Calcular métricas se não existirem
        if 'precision' in group.columns:
            precision_mean = group['precision'].mean()
            precision_std = group['precision'].std()
        else:
            # Calcular precision a partir dos dados
            precision_mean = (group['matched_count'] / group['total_estimated']).mean()
            precision_std = (group['matched_count'] / group['total_estimated']).std()
        
        if 'recall' in group.columns:
            recall_mean = group['recall'].mean()
            recall_std = group['recall'].std()
        else:
            # Calcular recall a partir dos dados
            recall_mean = (group['matched_count'] / group['total_reference']).mean()
            recall_std = (group['matched_count'] / group['total_reference']).std()
        
        if 'f1_score' in group.columns:
            f1_mean = group['f1_score'].mean()
            f1_std = group['f1_score'].std()
        else:
            # Calcular F1 a partir de precision e recall
            precisions = group['matched_count'] / group['total_estimated']
            recalls = group['matched_count'] / group['total_reference']
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
            f1_scores = f1_scores.fillna(0.0)
            f1_mean = f1_scores.mean()
            f1_std = f1_scores.std()
        
        result = {
            'dataset': dataset,
            'instrument': instrument,
            'count': len(group),
            'mean_error_mean': group['mean_error'].mean(),
            'mean_error_std': group['mean_error'].std(),
            'precision_mean': precision_mean,
            'precision_std': precision_std,
            'recall_mean': recall_mean,
            'recall_std': recall_std,
            'f1_score_mean': f1_mean,
            'f1_score_std': f1_std,
            'matched_count_mean': group['matched_count'].mean(),
            'total_reference_mean': group['total_reference'].mean(),
            'total_estimated_mean': group['total_estimated'].mean(),
        }
        results.append(result)
    
    return pd.DataFrame(results)

def create_table_image(df, title, output_path, filter_type='filtered'):
    """
    Criar tabela em imagem usando matplotlib
    """
    # Ordenar por dataset e instrumento
    df = df.sort_values(['dataset', 'instrument'])
    
    # Preparar dados para a tabela
    # Agrupar por dataset primeiro - apenas audio_original e FluidR3_GM
    datasets_order = ['audio_original', 'FluidR3_GM']
    
    # Filtrar apenas os datasets desejados
    df = df[df['dataset'].isin(datasets_order)]
    
    # Criar figura
    fig, ax = plt.subplots(figsize=(16, max(8, len(df) * 0.4)))
    ax.axis('tight')
    ax.axis('off')
    
    # Preparar dados da tabela
    table_data = []
    numeric_values = []  # Armazenar valores numéricos para heatmap
    
    # Cabeçalho
    headers = ['Dataset', 'Instrument', 'N', 'Mean Error (s)', 'Precision', 'Recall', 'F1-Score', 
               'Matched/Ref', 'Est/Ref']
    table_data.append(headers)
    
    # Índices das colunas numéricas (Mean Error, Precision, Recall, F1-Score)
    numeric_cols = [3, 4, 5, 6]  # Índices das colunas com valores numéricos
    
    # Dados
    for dataset in datasets_order:
        dataset_df = df[df['dataset'] == dataset]
        if len(dataset_df) == 0:
            continue
        
        for _, row in dataset_df.iterrows():
            # Formatar valores
            mean_error_str = f"{row['mean_error_mean']:.4f} ± {row['mean_error_std']:.4f}"
            precision_str = f"{row['precision_mean']:.3f} ± {row['precision_std']:.3f}" if row['precision_mean'] > 0 else "N/A"
            recall_str = f"{row['recall_mean']:.3f} ± {row['recall_std']:.3f}" if row['recall_mean'] > 0 else "N/A"
            f1_str = f"{row['f1_score_mean']:.3f} ± {row['f1_score_std']:.3f}" if row['f1_score_mean'] > 0 else "N/A"
            matched_ref_str = f"{row['matched_count_mean']:.0f}/{row['total_reference_mean']:.0f}"
            est_ref_str = f"{row['total_estimated_mean']:.0f}/{row['total_reference_mean']:.0f}"
            
            table_data.append([
                row['dataset'],
                row['instrument'],
                f"{row['count']:.0f}",
                mean_error_str,
                precision_str,
                recall_str,
                f1_str,
                matched_ref_str,
                est_ref_str
            ])
            
            # Armazenar valores numéricos para heatmap
            numeric_values.append({
                'mean_error': row['mean_error_mean'],
                'precision': row['precision_mean'] if row['precision_mean'] > 0 else None,
                'recall': row['recall_mean'] if row['recall_mean'] > 0 else None,
                'f1_score': row['f1_score_mean'] if row['f1_score_mean'] > 0 else None
            })
    
    # Criar tabela
    table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                     colWidths=[0.12, 0.15, 0.05, 0.12, 0.12, 0.12, 0.12, 0.10, 0.10])
    
    # Criar colormap verde para vermelho
    colors = ['#2ecc71', '#f39c12', '#e74c3c']  # Verde, Laranja, Vermelho
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('green_red', colors, N=n_bins)
    
    # Calcular min/max para normalização
    # Mean Error: valores baixos são melhores (verde), altos são piores (vermelho)
    mean_errors = [v['mean_error'] for v in numeric_values]
    mean_error_min, mean_error_max = min(mean_errors), max(mean_errors)
    
    # Precision, Recall, F1: valores altos são melhores (verde), baixos são piores (vermelho)
    precisions = [v['precision'] for v in numeric_values if v['precision'] is not None]
    recalls = [v['recall'] for v in numeric_values if v['recall'] is not None]
    f1_scores = [v['f1_score'] for v in numeric_values if v['f1_score'] is not None]
    
    precision_min, precision_max = (min(precisions), max(precisions)) if precisions else (0, 1)
    recall_min, recall_max = (min(recalls), max(recalls)) if recalls else (0, 1)
    f1_min, f1_max = (min(f1_scores), max(f1_scores)) if f1_scores else (0, 1)
    
    def get_text_color(bg_color):
        """Determinar cor do texto baseado no brilho do fundo"""
        # Converter hex para RGB
        bg_color = bg_color.lstrip('#')
        r, g, b = tuple(int(bg_color[i:i+2], 16) for i in (0, 2, 4))
        # Calcular brilho (luminância relativa)
        brightness = (r * 299 + g * 587 + b * 114) / 1000
        return 'white' if brightness < 128 else 'black'
    
    # Estilizar cabeçalho
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
        table[(0, i)].set_height(0.05)
    
    # Estilizar células de dados com heatmap
    for i in range(1, len(table_data)):
        row_idx = i - 1  # Índice na lista numeric_values
        numeric_row = numeric_values[row_idx] if row_idx < len(numeric_values) else None
        
        for j in range(len(headers)):
            # Colunas de texto (Dataset, Instrument, N, Matched/Ref, Est/Ref)
            if j in [0, 1, 2, 7, 8]:
                # Cores alternadas para linhas
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
                else:
                    table[(i, j)].set_facecolor('white')
                table[(i, j)].set_text_props(color='black')
            # Colunas numéricas com heatmap
            elif j in numeric_cols and numeric_row:
                if j == 3:  # Mean Error (invertido: baixo = verde, alto = vermelho)
                    if mean_error_max > mean_error_min:
                        normalized = 1.0 - (numeric_row['mean_error'] - mean_error_min) / (mean_error_max - mean_error_min)
                    else:
                        normalized = 0.5
                    color = cmap(normalized)
                    hex_color = '#{:02x}{:02x}{:02x}'.format(
                        int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
                    )
                elif j == 4:  # Precision (alto = verde, baixo = vermelho)
                    if numeric_row['precision'] is not None:
                        if precision_max > precision_min:
                            normalized = (numeric_row['precision'] - precision_min) / (precision_max - precision_min)
                        else:
                            normalized = 0.5
                        color = cmap(normalized)
                        hex_color = '#{:02x}{:02x}{:02x}'.format(
                            int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
                        )
                    else:
                        hex_color = '#f0f0f0' if i % 2 == 0 else 'white'
                elif j == 5:  # Recall (alto = verde, baixo = vermelho)
                    if numeric_row['recall'] is not None:
                        if recall_max > recall_min:
                            normalized = (numeric_row['recall'] - recall_min) / (recall_max - recall_min)
                        else:
                            normalized = 0.5
                        color = cmap(normalized)
                        hex_color = '#{:02x}{:02x}{:02x}'.format(
                            int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
                        )
                    else:
                        hex_color = '#f0f0f0' if i % 2 == 0 else 'white'
                elif j == 6:  # F1-Score (alto = verde, baixo = vermelho)
                    if numeric_row['f1_score'] is not None:
                        if f1_max > f1_min:
                            normalized = (numeric_row['f1_score'] - f1_min) / (f1_max - f1_min)
                        else:
                            normalized = 0.5
                        color = cmap(normalized)
                        hex_color = '#{:02x}{:02x}{:02x}'.format(
                            int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
                        )
                    else:
                        hex_color = '#f0f0f0' if i % 2 == 0 else 'white'
                else:
                    hex_color = '#f0f0f0' if i % 2 == 0 else 'white'
                
                table[(i, j)].set_facecolor(hex_color)
                text_color = get_text_color(hex_color)
                table[(i, j)].set_text_props(color=text_color, weight='bold')
            else:
                # Fallback para outras colunas
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
                else:
                    table[(i, j)].set_facecolor('white')
                table[(i, j)].set_text_props(color='black')
            
            table[(i, j)].set_height(0.04)
    
    # Ajustar tamanho da fonte
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    
    # Ajustar layout
    plt.subplots_adjust(top=0.98, bottom=0.02, left=0.05, right=0.95)
    
    # Salvar
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Tabela salva em: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Criar tabelas em imagem dos resultados")
    parser.add_argument('--filtered-csv', type=str, 
                       default='results_sintetizados_by_note.csv',
                       help='CSV com resultados filtrados (padrão: results_sintetizados_by_note.csv)')
    parser.add_argument('--no-filter-csv', type=str,
                       default='results_no_filter_sintetizados_by_note.csv',
                       help='CSV com resultados sem filtro (padrão: results_no_filter_sintetizados_by_note.csv)')
    parser.add_argument('--audio-original-filtered', type=str,
                       default='results_audio_original_by_note.csv',
                       help='CSV com audio_original filtrado')
    parser.add_argument('--audio-original-no-filter', type=str,
                       default='results_no_filter_audio_original_by_note.csv',
                       help='CSV com audio_original sem filtro')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Diretório para salvar as imagens (padrão: .)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Processar resultados COM filtro
    print("📊 Processando resultados COM filtro de logits...")
    
    # Carregar e combinar CSVs filtrados
    df_filtered_sint = load_and_process_csv(args.filtered_csv, 'filtered')
    df_filtered_audio = load_and_process_csv(args.audio_original_filtered, 'filtered')
    df_filtered = pd.concat([df_filtered_audio, df_filtered_sint], ignore_index=True)
    
    # Filtrar apenas audio_original e FluidR3_GM
    df_filtered = df_filtered[df_filtered['dataset'].isin(['audio_original', 'FluidR3_GM'])]
    
    create_table_image(
        df_filtered,
        'Results with Logits Filter',
        output_dir / 'results_table_filtered.png',
        'filtered'
    )
    
    # Processar resultados SEM filtro
    print("📊 Processando resultados SEM filtro...")
    
    df_no_filter_sint = load_and_process_csv(args.no_filter_csv, 'no_filter')
    df_no_filter_audio = load_and_process_csv(args.audio_original_no_filter, 'no_filter')
    df_no_filter = pd.concat([df_no_filter_audio, df_no_filter_sint], ignore_index=True)
    
    # Filtrar apenas audio_original e FluidR3_GM
    df_no_filter = df_no_filter[df_no_filter['dataset'].isin(['audio_original', 'FluidR3_GM'])]
    
    create_table_image(
        df_no_filter,
        'Results without Filter',
        output_dir / 'results_table_no_filter.png',
        'no_filter'
    )
    
    print("\n✅ Todas as tabelas foram criadas com sucesso!")

if __name__ == "__main__":
    main()

