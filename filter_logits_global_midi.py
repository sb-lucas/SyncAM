#!/usr/bin/env python3
"""
Script para filtrar logits do Basic Pitch usando filtro global baseado no MIDI
Cria um filtro com todas as notas presentes no MIDI e aplica top-k apenas entre essas notas
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import logging
from datetime import datetime

try:
    import basic_pitch
    from basic_pitch.inference import predict
    import pretty_midi
    print("✅ Bibliotecas importadas com sucesso!")
except ImportError as e:
    print(f"❌ Erro ao importar bibliotecas: {e}")
    print("Execute: conda activate staffspy")
    print("pip install pretty_midi matplotlib")
    sys.exit(1)

def setup_logging():
    """Configurar logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def create_global_midi_filter(midi_path, logger):
    """
    Criar filtro global com todas as notas presentes no arquivo MIDI
    
    Args:
        midi_path (str): Caminho para arquivo MIDI
        logger: Logger
        
    Returns:
        set: Conjunto de pitches MIDI presentes no arquivo
    """
    logger.info(f"Criando filtro global do MIDI: {midi_path}")
    
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        logger.info(f"MIDI carregado - duração: {midi_data.get_end_time():.2f}s")
        
        # Extrair todas as notas de todos os instrumentos
        all_pitches = set()
        total_notes = 0
        
        for instrument in midi_data.instruments:
            if not instrument.is_drum:  # Ignorar percussão
                for note in instrument.notes:
                    all_pitches.add(note.pitch)
                    total_notes += 1
        
        logger.info(f"Total de notas no MIDI: {total_notes}")
        logger.info(f"Pitches únicos encontrados: {len(all_pitches)}")
        
        # Mostrar as notas do filtro
        sorted_pitches = sorted(all_pitches)
        logger.info("Notas no filtro global:")
        
        # Agrupar por oitavas para melhor visualização
        pitch_names = [pretty_midi.note_number_to_name(p) for p in sorted_pitches]
        logger.info(f"  Range: {pitch_names[0]} (MIDI {sorted_pitches[0]}) até {pitch_names[-1]} (MIDI {sorted_pitches[-1]})")
        
        # Mostrar algumas notas como exemplo
        if len(sorted_pitches) <= 20:
            logger.info(f"  Todas as notas: {', '.join(pitch_names)}")
        else:
            logger.info(f"  Primeiras 10: {', '.join(pitch_names[:10])}")
            logger.info(f"  Últimas 10: {', '.join(pitch_names[-10:])}")
        
        return all_pitches
        
    except Exception as e:
        logger.error(f"Erro ao criar filtro MIDI: {e}")
        return None

def apply_global_filter_with_topk(model_output, midi_filter, topk=3, logger=None):
    """
    Aplicar filtro global com seleção top-k
    
    Args:
        model_output: Saída do Basic Pitch
        midi_filter (set): Conjunto de pitches MIDI permitidos
        topk (int): Número máximo de notas a selecionar por frame
        logger: Logger
        
    Returns:
        dict: Logits filtrados
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Aplicando filtro global com top-k={topk}")
    
    if not isinstance(model_output, dict) or 'note' not in model_output:
        logger.error("model_output deve ser um dict com chave 'note'")
        return None
    
    note_logits = model_output['note']  # Shape: (frames, 88)
    num_frames, num_pitches = note_logits.shape
    
    logger.info(f"Logits shape: {note_logits.shape}")
    logger.info(f"Processando {num_frames} frames")
    
    # Converter pitches MIDI para índices Basic Pitch
    allowed_indices = []
    for midi_pitch in midi_filter:
        bp_index = midi_pitch - 21  # A0 = MIDI 21 = índice 0
        if 0 <= bp_index < num_pitches:
            allowed_indices.append(bp_index)
    
    allowed_indices = sorted(allowed_indices)
    logger.info(f"Índices permitidos no Basic Pitch: {len(allowed_indices)} de {num_pitches}")
    logger.info(f"Range de índices: {min(allowed_indices)} a {max(allowed_indices)}")
    
    # Criar array filtrado (inicialmente zeros)
    filtered_logits = np.zeros_like(note_logits)
    
    # Estatísticas
    total_notes_selected = 0
    frames_with_selection = 0
    
    # Processar cada frame
    for frame_idx in range(num_frames):
        # Obter logits apenas para os pitches permitidos
        frame_logits = note_logits[frame_idx]
        allowed_logits = frame_logits[allowed_indices]
        
        # Aplicar top-k entre os pitches permitidos
        if len(allowed_logits) > 0:
            # Selecionar top-k
            num_to_select = min(topk, len(allowed_logits))
            top_k_indices_in_allowed = np.argsort(allowed_logits)[-num_to_select:]
            # Mapear de volta para índices globais e copiar valores
            for i, allowed_idx in enumerate(top_k_indices_in_allowed):
                global_idx = allowed_indices[allowed_idx]
                filtered_logits[frame_idx, global_idx] = note_logits[frame_idx, global_idx]
                total_notes_selected += 1
            
            if len(top_k_indices_in_allowed) > 0:
                frames_with_selection += 1
    
    logger.info(f"Frames com seleções: {frames_with_selection}/{num_frames}")
    logger.info(f"Total de notas selecionadas: {total_notes_selected}")
    logger.info(f"Média de notas por frame ativo: {total_notes_selected/max(frames_with_selection, 1):.2f}")
    
    # Criar resultado filtrado
    filtered_output = model_output.copy()
    filtered_output['note'] = filtered_logits
    
    return filtered_output

def demonstrate_filtering_example(model_output, midi_filter, topk, logger):
    """
    Demonstrar o filtro com exemplos específicos
    """
    try:
        print("\n" + "="*60)
        print("DEMONSTRAÇÃO DO FILTRO")
        print("="*60)
        logger.info("\n" + "="*60)
        logger.info("DEMONSTRAÇÃO DO FILTRO")
        logger.info("="*60)
        
        note_logits = model_output['note']
        logger.info(f"Shape dos logits: {note_logits.shape}")
        
        # Converter filtro MIDI para índices Basic Pitch
        allowed_indices = []
        allowed_pitches_info = []
        for midi_pitch in sorted(midi_filter):
            bp_index = midi_pitch - 21
            if 0 <= bp_index < 88:
                allowed_indices.append(bp_index)
                note_name = pretty_midi.note_number_to_name(midi_pitch)
                allowed_pitches_info.append(f"{note_name}(MIDI{midi_pitch}→idx{bp_index})")
        
        print(f"Filtro permite {len(allowed_indices)} notas:")
        logger.info(f"Filtro permite {len(allowed_indices)} notas:")
        if len(allowed_pitches_info) <= 10:
            print(f"  {', '.join(allowed_pitches_info)}")
            logger.info(f"  {', '.join(allowed_pitches_info)}")
        else:
            print(f"  Primeiras 5: {', '.join(allowed_pitches_info[:5])}")
            print(f"  Últimas 5: {', '.join(allowed_pitches_info[-5:])}")
            logger.info(f"  Primeiras 5: {', '.join(allowed_pitches_info[:5])}")
            logger.info(f"  Últimas 5: {', '.join(allowed_pitches_info[-5:])}")
        
        # Encontrar alguns frames interessantes para demonstrar
        logger.info(f"\nExemplos de filtragem (top-k={topk}):")
        logger.info("-" * 40)
        
        # Procurar frames com atividade
        logger.info("Procurando frames com atividade...")
        frame_activities = np.sum(note_logits > 0.3, axis=1)
        active_frames = np.where(frame_activities > 0)[0]
        logger.info(f"Encontrados {len(active_frames)} frames ativos")
        
        if len(active_frames) == 0:
            logger.warning("Nenhum frame ativo encontrado! Usando frames aleatórios...")
            active_frames = np.random.choice(note_logits.shape[0], min(10, note_logits.shape[0]), replace=False)
        
        # Mostrar alguns exemplos - evitar divisão por zero
        if len(active_frames) >= 5:
            step = len(active_frames) // 5
            example_frames = active_frames[::step][:5]
        else:
            example_frames = active_frames[:5]
        
        logger.info(f"Analisando {len(example_frames)} frames de exemplo...")
        
        for i, frame_idx in enumerate(example_frames):
            logger.info(f"Processando exemplo {i+1}/{len(example_frames)}...")
            frame_logits = note_logits[frame_idx]
            
            # Top 5 predições originais
            top_5_original = np.argsort(frame_logits)[-5:][::-1]
            
            # Predições permitidas pelo filtro
            allowed_logits = frame_logits[allowed_indices]
            if len(allowed_logits) > 0:
                top_k_in_allowed = np.argsort(allowed_logits)[-topk:][::-1]
                selected_global_indices = [allowed_indices[i] for i in top_k_in_allowed]
            else:
                selected_global_indices = []
            
            logger.info(f"\nFrame {frame_idx} (tempo ~{frame_idx*0.0116:.2f}s):")
            
            # Mostrar top 5 originais
            logger.info("  Top 5 predições originais:")
            for j, idx in enumerate(top_5_original):
                midi_pitch = idx + 21
                note_name = pretty_midi.note_number_to_name(midi_pitch)
                prob = frame_logits[idx]
                in_filter = "✓" if idx in allowed_indices else "✗"
                logger.info(f"    {j+1}. {note_name} (MIDI{midi_pitch}): {prob:.3f} {in_filter}")
            
            # Mostrar seleções finais
            logger.info(f"  Seleções finais (top-{topk} do filtro):")
            if selected_global_indices:
                for j, idx in enumerate(selected_global_indices):
                    midi_pitch = idx + 21
                    note_name = pretty_midi.note_number_to_name(midi_pitch)
                    prob = frame_logits[idx]
                    logger.info(f"    {j+1}. {note_name} (MIDI{midi_pitch}): {prob:.3f} ✓")
            else:
                logger.info("    Nenhuma nota selecionada")
        
        logger.info("="*60)
        logger.info("Demonstração concluída com sucesso!")
        
    except Exception as e:
        logger.error(f"Erro na demonstração de filtragem: {e}")
        import traceback
        traceback.print_exc()

def create_piano_roll_comparison(original_logits, filtered_logits, midi_filter, 
                                output_path, midi_path=None, logger=None):
    """
    Criar comparação visual dos piano rolls com melhor contraste e notas MIDI
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("Criando piano roll comparison...")
    
    try:
        # Preparar dados
        original_notes = original_logits['note']
        filtered_notes = filtered_logits['note']
        
        # Limitar visualização para primeiros N frames para clareza
        max_frames = min(2000, original_notes.shape[0])  # ~23 segundos
        original_subset = original_notes[:max_frames]
        filtered_subset = filtered_notes[:max_frames]
        
        # Criar máscara do filtro MIDI
        filter_mask = np.zeros(88, dtype=bool)
        for midi_pitch in midi_filter:
            bp_index = midi_pitch - 21
            if 0 <= bp_index < 88:
                filter_mask[bp_index] = True
        
        # Melhorar contraste aplicando transformações
        # Aplicar threshold e realçar valores altos
        original_enhanced = np.where(original_subset > 0.3, original_subset, 0)
        original_enhanced = np.power(original_enhanced, 0.7)  # Gamma correction
        
        filtered_enhanced = np.where(filtered_subset > 0.3, filtered_subset, 0)
        filtered_enhanced = np.power(filtered_enhanced, 0.7)
        
        # Criar figura com subplots usando constrained_layout
        fig = plt.figure(figsize=(18, 20), constrained_layout=True)
        gs = fig.add_gridspec(5, 1, height_ratios=[0.4, 0.8, 1, 1, 1])
        
        # 1. Filtro MIDI (quais notas são permitidas)
        ax1 = fig.add_subplot(gs[0])
        filter_display = np.tile(filter_mask, (max_frames//10, 1)).T  # Usar frames reais
        im1 = ax1.imshow(filter_display, aspect='auto', origin='lower', 
                        cmap='Blues', vmin=0, vmax=1, interpolation='nearest')
        ax1.set_title('Filtro MIDI Global (Notas Permitidas)', fontsize=16, fontweight='bold', pad=20)
        ax1.set_ylabel('Pitch', fontsize=12)
        ax1.set_xticks([])
        
        # 2. Notas MIDI de referência (ground truth)
        ax2 = fig.add_subplot(gs[1])
        midi_ground_truth = create_midi_ground_truth_visualization(midi_filter, max_frames, midi_path, logger)
        im2 = ax2.imshow(midi_ground_truth, aspect='auto', origin='lower', 
                        cmap='viridis', vmin=0, vmax=1, interpolation='nearest')
        ax2.set_title('MIDI Ground Truth (Notas Reais)', fontsize=16, fontweight='bold', pad=20)
        ax2.set_ylabel('Pitch', fontsize=12)
        
        # 3. Piano roll Basic Pitch original (com melhor contraste)
        ax3 = fig.add_subplot(gs[2])
        im3 = ax3.imshow(original_enhanced.T, aspect='auto', origin='lower', 
                        cmap='Reds', vmin=0, vmax=1, interpolation='bilinear')
        ax3.set_title('Basic Pitch Original (Todas as Predições)', fontsize=16, fontweight='bold', pad=20)
        ax3.set_ylabel('Pitch', fontsize=12)
        
        # 4. Piano roll filtrado (com melhor contraste)
        ax4 = fig.add_subplot(gs[3])
        im4 = ax4.imshow(filtered_enhanced.T, aspect='auto', origin='lower', 
                        cmap='Greens', vmin=0, vmax=1, interpolation='bilinear')
        ax4.set_title('Basic Pitch Filtrado (Apenas Notas do MIDI)', fontsize=16, fontweight='bold', pad=20)
        ax4.set_ylabel('Pitch', fontsize=12)
        
        # 5. Diferença (o que foi removido) com melhor contraste
        ax5 = fig.add_subplot(gs[4])
        difference = original_enhanced - filtered_enhanced
        difference = np.where(difference > 0.1, difference, 0)  # Threshold para clareza
        im5 = ax5.imshow(difference.T, aspect='auto', origin='lower', 
                        cmap='Oranges', vmin=0, vmax=1, interpolation='bilinear')
        ax5.set_title('Diferença (Notas Removidas pelo Filtro)', fontsize=16, fontweight='bold', pad=20)
        ax5.set_ylabel('Pitch', fontsize=12)
        ax5.set_xlabel('Frame (tempo)', fontsize=12)
        
        axes = [ax1, ax2, ax3, ax4, ax5]
        
        # Adicionar colorbar para cada subplot
        for i, (ax, im) in enumerate(zip(axes, [im1, im2, im3, im4, im5])):
            cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.01)
            if i == 0:
                cbar.set_label('Filtro (0=Não permitido, 1=Permitido)', fontsize=10)
            elif i == 1:
                cbar.set_label('MIDI (0=Silêncio, 1=Nota ativa)', fontsize=10)
            else:
                cbar.set_label('Probabilidade/Ativação', fontsize=10)
        
        # Adicionar marcações de tempo melhoradas
        time_ticks = np.arange(0, max_frames, 200)  # Mais marcações
        time_labels = [f"{t*0.0116:.1f}s" for t in time_ticks]
        
        for ax in axes[1:]:  # Pular apenas o primeiro (filtro estático)
            ax.set_xticks(time_ticks)
            ax.set_xticklabels(time_labels, fontsize=10)
            ax.grid(True, alpha=0.2, linestyle='--')
        
        # Adicionar marcações de pitch melhoradas
        pitch_ticks = np.arange(0, 88, 12)  # A cada oitava
        pitch_labels = [pretty_midi.note_number_to_name(21 + p) for p in pitch_ticks]
        
        for ax in axes:
            ax.set_yticks(pitch_ticks)
            ax.set_yticklabels(pitch_labels, fontsize=10)
            
            # Destacar notas do filtro MIDI
            for midi_pitch in sorted(midi_filter):
                bp_index = midi_pitch - 21
                if 0 <= bp_index < 88:
                    ax.axhline(y=bp_index, color='yellow', alpha=0.3, linewidth=0.5)
        
        # Adicionar legenda
        fig.suptitle('Comparação: Filtro Global de Logits Basic Pitch', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Piano roll salvo: {output_path}")
        
    except Exception as e:
        logger.error(f"Erro ao criar piano roll: {e}")
        logger.info("Tentando criar versão simplificada...")
        
        # Versão simplificada em caso de erro
        try:
            create_simple_comparison(original_logits, filtered_logits, output_path.replace('.png', '_simple.png'), logger)
        except Exception as e2:
            logger.error(f"Erro também na versão simplificada: {e2}")

def create_midi_ground_truth_visualization(midi_filter, max_frames, midi_path, logger):
    """
    Criar visualização temporal real das notas MIDI
    """
    try:
        if midi_path is None:
            logger.warning("Caminho MIDI não fornecido, usando visualização simplificada")
            return create_simple_midi_visualization(midi_filter, max_frames)
        
        # Carregar arquivo MIDI
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        
        # Criar array para visualização temporal
        midi_viz = np.zeros((88, max_frames))
        
        # Calcular duração por frame (Basic Pitch usa ~86 fps)
        frame_duration = 0.0116  # segundos por frame
        total_duration = max_frames * frame_duration
        
        logger.info(f"Criando visualização MIDI temporal: {total_duration:.2f}s, {max_frames} frames")
        
        # Processar todas as notas de todos os instrumentos
        for instrument in midi_data.instruments:
            if not instrument.is_drum:  # Ignorar percussão
                for note in instrument.notes:
                    # Verificar se a nota está no filtro
                    if note.pitch in midi_filter:
                        bp_index = note.pitch - 21  # Converter para índice Basic Pitch
                        if 0 <= bp_index < 88:
                            # Converter tempos para frames
                            start_frame = int(note.start / frame_duration)
                            end_frame = int(note.end / frame_duration)
                            
                            # Limitar aos frames disponíveis
                            start_frame = max(0, min(start_frame, max_frames - 1))
                            end_frame = max(0, min(end_frame, max_frames - 1))
                            
                            if start_frame < end_frame:
                                # Criar gradiente de velocidade para melhor visualização
                                velocity_normalized = note.velocity / 127.0
                                intensity = 0.3 + 0.7 * velocity_normalized  # Entre 0.3 e 1.0
                                
                                # Preencher frames da nota
                                midi_viz[bp_index, start_frame:end_frame] = intensity
        
        logger.info("Visualização MIDI temporal criada com sucesso")
        return midi_viz
        
    except Exception as e:
        logger.error(f"Erro ao criar visualização MIDI temporal: {e}")
        logger.info("Usando visualização simplificada como fallback")
        return create_simple_midi_visualization(midi_filter, max_frames)

def create_simple_midi_visualization(midi_filter, max_frames):
    """Visualização MIDI simplificada baseada apenas no filtro"""
    midi_viz = np.zeros((88, max_frames))
    
    for midi_pitch in midi_filter:
        bp_index = midi_pitch - 21
        if 0 <= bp_index < 88:
            # Criar padrão que varia no tempo para mostrar que são notas ativas
            pattern = np.sin(np.linspace(0, 4*np.pi, max_frames)) * 0.3 + 0.7
            pattern = np.where(pattern > 0.5, pattern, 0)
            midi_viz[bp_index, :] = pattern
    
    return midi_viz

def create_simple_comparison(original_logits, filtered_logits, output_path, logger):
    """Versão simplificada da comparação"""
    original_notes = original_logits['note']
    filtered_notes = filtered_logits['note']
    
    max_frames = min(1000, original_notes.shape[0])
    original_subset = original_notes[:max_frames]
    filtered_subset = filtered_notes[:max_frames]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Original
    ax1.imshow(original_subset.T, aspect='auto', origin='lower', cmap='Reds')
    ax1.set_title('Basic Pitch Original')
    ax1.set_ylabel('Pitch')
    
    # Filtrado
    ax2.imshow(filtered_subset.T, aspect='auto', origin='lower', cmap='Greens')
    ax2.set_title('Basic Pitch Filtrado')
    ax2.set_ylabel('Pitch')
    ax2.set_xlabel('Frame')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    logger.info(f"Piano roll simplificado salvo: {output_path}")

def convert_filtered_logits_to_midi(filtered_output, output_path, threshold=0.3, frame_duration=0.0116, logger=None):
    """
    Converter logits filtrados em arquivo MIDI
    
    Args:
        filtered_output: Saída filtrada do Basic Pitch
        output_path: Caminho para salvar o MIDI
        threshold: Limiar de confiança para considerar uma nota ativa
        frame_duration: Duração de cada frame em segundos (~0.0116s para Basic Pitch)
        logger: Logger
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Convertendo logits filtrados em MIDI: {output_path}")
        
        note_logits = filtered_output['note']  # Shape: (frames, 88)
        num_frames, num_pitches = note_logits.shape
        
        # Criar objeto PrettyMIDI
        midi_data = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0, is_drum=False)  # Piano
        
        # Processar cada pitch
        for pitch_idx in range(num_pitches):
            midi_pitch = pitch_idx + 21  # Converter índice Basic Pitch para MIDI pitch
            
            # Obter ativações deste pitch ao longo do tempo
            pitch_activations = note_logits[:, pitch_idx]
            
            # Encontrar regiões onde a ativação está acima do threshold
            active_frames = np.where(pitch_activations > threshold)[0]
            
            if len(active_frames) == 0:
                continue
            
            # Agrupar frames consecutivos em notas
            note_starts = []
            note_ends = []
            current_start = active_frames[0]
            
            for i in range(1, len(active_frames)):
                if active_frames[i] - active_frames[i-1] > 1:  # Gap detectado
                    note_starts.append(current_start)
                    note_ends.append(active_frames[i-1])
                    current_start = active_frames[i]
            
            # Adicionar última nota
            note_starts.append(current_start)
            note_ends.append(active_frames[-1])
            
            # Criar notas MIDI
            for start_frame, end_frame in zip(note_starts, note_ends):
                start_time = start_frame * frame_duration
                end_time = (end_frame + 1) * frame_duration  # +1 para incluir o último frame
                
                # Calcular velocity baseada na confiança média
                confidence = np.mean(pitch_activations[start_frame:end_frame+1])
                velocity = int(np.clip(confidence * 127, 1, 127))
                
                # Criar nota
                note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=midi_pitch,
                    start=start_time,
                    end=end_time
                )
                instrument.notes.append(note)
        
        midi_data.instruments.append(instrument)
        midi_data.write(output_path)
        
        logger.info(f"MIDI salvo com {len(instrument.notes)} notas: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Erro ao converter logits em MIDI: {e}")
        import traceback
        traceback.print_exc()
        return False

def find_matching_midi(audio_path, midi_dir, logger=None):
    """
    Encontrar arquivo MIDI correspondente ao áudio
    
    Args:
        audio_path: Caminho para arquivo de áudio
        midi_dir: Diretório contendo arquivos MIDI
        logger: Logger
        
    Returns:
        str: Caminho para o MIDI correspondente ou None
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    audio_name = Path(audio_path).stem
    
    # Limpar nome do áudio removendo extensões e sufixos comuns
    clean_name = audio_name
    # Remover sufixos comuns de arquivos de áudio
    for suffix in ['_wav', '_WAV', '-AUDIO', '_audio', '_AUDIO']:
        if clean_name.endswith(suffix):
            clean_name = clean_name[:-len(suffix)]
    
    # Tentar diferentes variações de nome
    possible_names = [
        audio_name,  # Nome exato
        clean_name,  # Nome limpo
        audio_name.replace('_wav', ''),
        audio_name.replace('_WAV', ''),
        audio_name.replace('-AUDIO', ''),
        audio_name.replace('_AUDIO', ''),
        # Para arquivos como "MIDI-Unprocessed_Chamber6_MID--AUDIO_20_R3_2018_wav--1"
        # tentar encontrar "MIDI-Unprocessed_Chamber6_MID--AUDIO_20_R3_2018_wav--1.midi"
        audio_name.replace('--1', ''),
    ]
    
    # Remover duplicatas mantendo ordem
    seen = set()
    unique_names = []
    for name in possible_names:
        if name not in seen:
            seen.add(name)
            unique_names.append(name)
    
    possible_extensions = ['.mid', '.midi', '.MID', '.MIDI']
    
    # Buscar recursivamente no diretório MIDI
    midi_dir_path = Path(midi_dir)
    if not midi_dir_path.exists():
        logger.warning(f"Diretório MIDI não existe: {midi_dir}")
        return None
    
    for name in unique_names:
        for ext in possible_extensions:
            # Buscar exatamente no diretório
            midi_path = midi_dir_path / f"{name}{ext}"
            if midi_path.exists():
                logger.info(f"MIDI correspondente encontrado: {midi_path}")
                return str(midi_path)
            
            # Buscar recursivamente
            for midi_file in midi_dir_path.rglob(f"{name}{ext}"):
                logger.info(f"MIDI correspondente encontrado (recursivo): {midi_file}")
                return str(midi_file)
    
    logger.warning(f"Nenhum MIDI correspondente encontrado para: {audio_path}")
    logger.info(f"Tentou encontrar: {unique_names[:3]}... (e variações)")
    return None

def process_audio_file(audio_path, midi_dir, output_dir, topk=3, logger=None):
    """
    Processar um único arquivo de áudio
    
    Args:
        audio_path: Caminho para arquivo de áudio
        midi_dir: Diretório contendo MIDIs de referência
        output_dir: Diretório de saída
        topk: Número máximo de notas por frame
        logger: Logger
        
    Returns:
        bool: True se processado com sucesso
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        audio_name = Path(audio_path).stem
        logger.info(f"\n{'='*60}")
        logger.info(f"Processando: {audio_name}")
        logger.info(f"{'='*60}")
        
        # 1. Encontrar MIDI correspondente
        midi_path = find_matching_midi(audio_path, midi_dir, logger)
        if midi_path is None:
            logger.warning(f"Pulando {audio_name} - MIDI não encontrado")
            return False
        
        # 2. Criar filtro global MIDI
        midi_filter = create_global_midi_filter(midi_path, logger)
        if midi_filter is None:
            logger.error(f"Erro ao criar filtro MIDI para {audio_name}")
            return False
        
        # 3. Extrair logits Basic Pitch
        logger.info(f"Extraindo logits do Basic Pitch...")
        model_output, midi_data, note_events = predict(audio_path)
        logger.info("Logits extraídos com sucesso!")
        
        # 4. Aplicar filtro global
        filtered_output = apply_global_filter_with_topk(
            model_output, midi_filter, topk, logger
        )
        
        if filtered_output is None:
            logger.error(f"Erro ao aplicar filtro para {audio_name}")
            return False
        
        # 5. Converter logits filtrados em MIDI
        output_midi_path = Path(output_dir) / f"{audio_name}_filtered.mid"
        success = convert_filtered_logits_to_midi(
            filtered_output, str(output_midi_path), threshold=0.3, logger=logger
        )
        
        if success:
            logger.info(f"✅ Processamento concluído: {audio_name}")
            return True
        else:
            logger.error(f"❌ Erro ao salvar MIDI: {audio_name}")
            return False
            
    except Exception as e:
        logger.error(f"Erro ao processar {audio_path}: {e}")
        import traceback
        traceback.print_exc()
        return False

def print_statistics(original_logits, filtered_logits, midi_filter, logger):
    """Imprimir estatísticas comparativas"""
    original_notes = original_logits['note']
    filtered_notes = filtered_logits['note']
    
    # Contar notas ativas (threshold > 0.5)
    original_active = np.sum(original_notes > 0.5)
    filtered_active = np.sum(filtered_notes > 0.5)
    
    # Contar notas únicas ativas
    original_unique_pitches = len(np.where(np.max(original_notes, axis=0) > 0.5)[0])
    filtered_unique_pitches = len(np.where(np.max(filtered_notes, axis=0) > 0.5)[0])
    
    logger.info("\n" + "="*60)
    logger.info("ESTATÍSTICAS COMPARATIVAS")
    logger.info("="*60)
    logger.info(f"Filtro MIDI contém: {len(midi_filter)} notas únicas")
    logger.info(f"Notas ativas originais (>0.5): {original_active}")
    logger.info(f"Notas ativas filtradas (>0.5): {filtered_active}")
    logger.info(f"Redução total: {((original_active - filtered_active) / original_active * 100):.1f}%")
    logger.info("")
    logger.info(f"Pitches únicos originais: {original_unique_pitches}/88")
    logger.info(f"Pitches únicos filtrados: {filtered_unique_pitches}/88")
    logger.info(f"Redução de pitches: {((original_unique_pitches - filtered_unique_pitches) / original_unique_pitches * 100):.1f}%")
    
    # Atividade por frame
    original_per_frame = np.sum(original_notes > 0.5, axis=1)
    filtered_per_frame = np.sum(filtered_notes > 0.5, axis=1)
    
    logger.info("")
    logger.info(f"Média notas/frame original: {np.mean(original_per_frame):.2f}")
    logger.info(f"Média notas/frame filtrado: {np.mean(filtered_per_frame):.2f}")
    logger.info(f"Frames com atividade original: {np.sum(original_per_frame > 0)}")
    logger.info(f"Frames com atividade filtrada: {np.sum(filtered_per_frame > 0)}")
    logger.info("="*60)

def get_all_audio_files(base_dir, logger=None):
    """
    Obter todos os arquivos de áudio recursivamente
    
    Args:
        base_dir: Diretório base para buscar arquivos
        logger: Logger
        
    Returns:
        list: Lista de caminhos para arquivos de áudio
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
    audio_files = []
    
    base_path = Path(base_dir)
    if not base_path.exists():
        logger.warning(f"Diretório não existe: {base_dir}")
        return audio_files
    
    for ext in audio_extensions:
        audio_files.extend(base_path.rglob(f"*{ext}"))
        audio_files.extend(base_path.rglob(f"*{ext.upper()}"))
    
    logger.info(f"Encontrados {len(audio_files)} arquivos de áudio em {base_dir}")
    return [str(f) for f in audio_files]

def process_batch(audio_files, midi_dir, output_base_dir, topk=3, logger=None):
    """
    Processar múltiplos arquivos de áudio em lote
    
    Args:
        audio_files: Lista de caminhos para arquivos de áudio
        midi_dir: Diretório contendo MIDIs de referência
        output_base_dir: Diretório base de saída
        topk: Número máximo de notas por frame
        logger: Logger
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    total_files = len(audio_files)
    successful = 0
    failed = 0
    
    logger.info(f"\n{'='*60}")
    logger.info(f"PROCESSAMENTO EM LOTE")
    logger.info(f"{'='*60}")
    logger.info(f"Total de arquivos: {total_files}")
    logger.info(f"Top-k: {topk}")
    logger.info(f"Diretório MIDI: {midi_dir}")
    logger.info(f"Diretório de saída: {output_base_dir}")
    
    for i, audio_path in enumerate(audio_files, 1):
        logger.info(f"\n[{i}/{total_files}] Processando arquivo {i} de {total_files}")
        
        # Determinar diretório de saída baseado na estrutura de entrada
        audio_path_str = str(audio_path)
        parts = Path(audio_path).parts
        
        # Se está em audio_original, salvar em transcribed/audio_original
        if 'audio_original' in audio_path_str:
            output_dir = Path(output_base_dir) / 'audio_original'
        # Se está em sintetizados, manter estrutura
        elif 'sintetizados' in audio_path_str:
            # Extrair caminho relativo a partir de sintetizados
            sintetizados_idx = None
            for idx, part in enumerate(parts):
                if part == 'sintetizados':
                    sintetizados_idx = idx
                    break
            
            if sintetizados_idx is not None:
                # Pegar partes após 'sintetizados' até o nome do arquivo
                relative_parts = parts[sintetizados_idx+1:-1]  # Pular 'sintetizados' e nome do arquivo
                if relative_parts:
                    output_dir = Path(output_base_dir) / 'sintetizados' / Path(*relative_parts)
                else:
                    output_dir = Path(output_base_dir) / 'sintetizados'
            else:
                output_dir = Path(output_base_dir) / 'sintetizados'
        else:
            output_dir = Path(output_base_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Processar arquivo
        success = process_audio_file(audio_path, midi_dir, str(output_dir), topk, logger)
        
        if success:
            successful += 1
        else:
            failed += 1
        
        # Progresso
        progress = (i / total_files) * 100
        logger.info(f"Progresso: {progress:.1f}% ({successful} sucessos, {failed} falhas)")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"PROCESSAMENTO CONCLUÍDO")
    logger.info(f"{'='*60}")
    logger.info(f"Total processado: {total_files}")
    logger.info(f"Sucessos: {successful}")
    logger.info(f"Falhas: {failed}")
    logger.info(f"Taxa de sucesso: {(successful/total_files*100):.1f}%")

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description="Filtrar logits Basic Pitch com filtro global MIDI e gerar MIDIs transcritos")
    parser.add_argument("--audio_dir", type=str, help="Diretório contendo arquivos de áudio (processa todos recursivamente)")
    parser.add_argument("--audio_path", type=str, help="Caminho para um único arquivo de áudio (modo single)")
    parser.add_argument("--midi_dir", type=str, default="/Users/moises/Documents/force_align/f0_midi_audio_alignment/data/midi", 
                       help="Diretório contendo arquivos MIDI de referência")
    parser.add_argument("--midi_path", type=str, help="Caminho para um único arquivo MIDI (modo single)")
    parser.add_argument("--topk", type=int, default=3, help="Número máximo de notas por frame (default: 3)")
    parser.add_argument("--output_dir", type=str, default="/Users/moises/Documents/force_align/f0_midi_audio_alignment/data/transcribed",
                       help="Diretório base de saída")
    parser.add_argument("--show_examples", action="store_true", help="Mostrar exemplos detalhados de filtragem (apenas modo single)")
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Modo batch: processar múltiplos arquivos
    if args.audio_dir:
        logger.info("🎯 MODO BATCH: Processamento em lote")
        logger.info("="*60)
        logger.info(f"Diretório de áudio: {args.audio_dir}")
        logger.info(f"Diretório MIDI: {args.midi_dir}")
        logger.info(f"Diretório de saída: {args.output_dir}")
        logger.info(f"Top-k: {args.topk}")
        
        # Obter todos os arquivos de áudio
        audio_files = get_all_audio_files(args.audio_dir, logger)
        
        if not audio_files:
            logger.error("Nenhum arquivo de áudio encontrado!")
            return
        
        # Processar em lote
        process_batch(audio_files, args.midi_dir, args.output_dir, args.topk, logger)
        
    # Modo single: processar um único arquivo
    elif args.audio_path:
        if not args.midi_path:
            logger.error("Modo single requer --midi_path")
            return
        
        logger.info("🎯 MODO SINGLE: Processamento de arquivo único")
        logger.info("="*60)
        logger.info(f"Áudio: {args.audio_path}")
        logger.info(f"MIDI: {args.midi_path}")
        logger.info(f"Top-k: {args.topk}")
        
        # 1. Criar filtro global MIDI
        midi_filter = create_global_midi_filter(args.midi_path, logger)
        if midi_filter is None:
            return
        
        # 2. Extrair logits Basic Pitch
        logger.info("Extraindo logits do Basic Pitch...")
        model_output, midi_data, note_events = predict(args.audio_path)
        logger.info("Logits extraídos com sucesso!")
        
        # 3. Aplicar filtro global
        filtered_output = apply_global_filter_with_topk(
            model_output, midi_filter, args.topk, logger
        )
        
        if filtered_output is None:
            return
        
        # 4. Mostrar exemplos se solicitado
        if args.show_examples:
            logger.info("Iniciando demonstração de exemplos...")
            demonstrate_filtering_example(model_output, midi_filter, args.topk, logger)
            logger.info("Demonstração de exemplos concluída!")
        
        # 5. Converter e salvar MIDI
        base_name = Path(args.audio_path).stem
        output_midi_path = Path(args.output_dir) / f"{base_name}_filtered.mid"
        success = convert_filtered_logits_to_midi(
            filtered_output, str(output_midi_path), threshold=0.3, logger=logger
        )
        
        if success:
            logger.info(f"✅ MIDI salvo: {output_midi_path}")
        else:
            logger.error(f"❌ Erro ao salvar MIDI")
        
        # 6. Estatísticas
        logger.info("Calculando estatísticas...")
        print_statistics(model_output, filtered_output, midi_filter, logger)
        
        logger.info("✅ Processamento concluído!")
        
    else:
        logger.error("Forneça --audio_dir (modo batch) ou --audio_path (modo single)")
        parser.print_help()

if __name__ == "__main__":
    main()
