#!/usr/bin/env python3
"""
Script para plotar piano rolls comparativos usando Plotly
Compara dois arquivos MIDI lado a lado
"""

import argparse
import sys
from pathlib import Path

try:
    import pretty_midi
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    print("✅ Bibliotecas importadas com sucesso!")
except ImportError as e:
    print(f"❌ Erro ao importar bibliotecas: {e}")
    print("Execute: pip install pretty_midi plotly")
    sys.exit(1)

def extract_notes_from_midi(midi_path):
    """
    Extrair notas de um arquivo MIDI
    
    Args:
        midi_path: Caminho para o arquivo MIDI
        
    Returns:
        tuple: (notes_list, min_pitch, max_pitch, duration)
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        
        notes = []
        pitches = []
        
        for instrument in midi_data.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    notes.append({
                        'pitch': note.pitch,
                        'start': note.start,
                        'end': note.end,
                        'velocity': note.velocity
                    })
                    pitches.append(note.pitch)
        
        if not pitches:
            return [], 60, 72, midi_data.get_end_time()
        
        min_pitch = min(pitches)
        max_pitch = max(pitches)
        duration = midi_data.get_end_time()
        
        return notes, min_pitch, max_pitch, duration
        
    except Exception as e:
        print(f"❌ Erro ao carregar MIDI {midi_path}: {e}")
        return None, None, None, None

def create_piano_roll_plot(notes, min_pitch, max_pitch, duration, title, color='blue'):
    """
    Criar plot de piano roll usando Plotly
    
    Args:
        notes: Lista de notas
        min_pitch: Pitch mínimo
        max_pitch: Pitch máximo
        duration: Duração total
        title: Título do plot
        color: Cor das notas
        
    Returns:
        plotly.graph_objects.Figure
    """
    fig = go.Figure()
    
    if not notes:
        # Se não há notas, criar plot vazio
        fig.add_trace(go.Scatter(
            x=[0, duration] if duration > 0 else [0, 1],
            y=[min_pitch, max_pitch] if min_pitch != max_pitch else [60, 72],
            mode='markers',
            marker=dict(size=0.1, opacity=0),
            showlegend=False
        ))
    else:
        # Criar retângulos para cada nota
        for note in notes:
            fig.add_trace(go.Scatter(
                x=[note['start'], note['start'], note['end'], note['end'], note['start']],
                y=[note['pitch'] - 0.4, note['pitch'] + 0.4, note['pitch'] + 0.4, note['pitch'] - 0.4, note['pitch'] - 0.4],
                fill='toself',
                fillcolor=color,
                line=dict(color=color, width=1),
                mode='lines',
                showlegend=False,
                hoverinfo='text',
                text=f"Pitch: {pretty_midi.note_number_to_name(note['pitch'])} ({note['pitch']})<br>"
                     f"Start: {note['start']:.3f}s<br>"
                     f"End: {note['end']:.3f}s<br>"
                     f"Duration: {note['end'] - note['start']:.3f}s<br>"
                     f"Velocity: {note['velocity']}"
            ))
    
    # Configurar layout
    fig.update_layout(
        title=title,
        xaxis_title='Tempo (segundos)',
        yaxis_title='MIDI Pitch',
        height=600,
        hovermode='closest',
        yaxis=dict(
            tickmode='linear',
            tick0=min_pitch if notes else 60,
            dtick=12,  # Uma oitava
            range=[min_pitch - 1, max_pitch + 1] if notes else [55, 77]
        ),
        xaxis=dict(
            range=[0, duration] if duration > 0 else [0, 1]
        )
    )
    
    return fig

def create_comparison_plot(midi1_path, midi2_path, title1="MIDI 1", title2="MIDI 2"):
    """
    Criar plot comparativo com dois piano rolls lado a lado
    
    Args:
        midi1_path: Caminho para primeiro MIDI
        midi2_path: Caminho para segundo MIDI
        title1: Título do primeiro plot
        title2: Título do segundo plot
    """
    # Extrair notas dos dois MIDIs
    notes1, min_p1, max_p1, dur1 = extract_notes_from_midi(midi1_path)
    notes2, min_p2, max_p2, dur2 = extract_notes_from_midi(midi2_path)
    
    if notes1 is None or notes2 is None:
        print("❌ Erro ao processar arquivos MIDI")
        return None
    
    # Determinar range comum de pitch e duração
    min_pitch = min(min_p1, min_p2) if notes1 and notes2 else (min_p1 if notes1 else min_p2)
    max_pitch = max(max_p1, max_p2) if notes1 and notes2 else (max_p1 if notes1 else max_p2)
    max_duration = max(dur1, dur2)
    
    # Criar subplots lado a lado
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(title1, title2),
        horizontal_spacing=0.1
    )
    
    # Adicionar notas do primeiro MIDI
    if notes1:
        for note in notes1:
            fig.add_trace(
                go.Scatter(
                    x=[note['start'], note['start'], note['end'], note['end'], note['start']],
                    y=[note['pitch'] - 0.4, note['pitch'] + 0.4, note['pitch'] + 0.4, note['pitch'] - 0.4, note['pitch'] - 0.4],
                    fill='toself',
                    fillcolor='blue',
                    line=dict(color='blue', width=1),
                    mode='lines',
                    showlegend=False,
                    hoverinfo='text',
                    text=f"Pitch: {pretty_midi.note_number_to_name(note['pitch'])} ({note['pitch']})<br>"
                         f"Start: {note['start']:.3f}s<br>"
                         f"End: {note['end']:.3f}s<br>"
                         f"Duration: {note['end'] - note['start']:.3f}s<br>"
                         f"Velocity: {note['velocity']}",
                    name=''
                ),
                row=1, col=1
            )
    else:
        # Plot vazio
        fig.add_trace(
            go.Scatter(x=[0, max_duration], y=[min_pitch, max_pitch], mode='markers', 
                      marker=dict(size=0.1, opacity=0), showlegend=False),
            row=1, col=1
        )
    
    # Adicionar notas do segundo MIDI
    if notes2:
        for note in notes2:
            fig.add_trace(
                go.Scatter(
                    x=[note['start'], note['start'], note['end'], note['end'], note['start']],
                    y=[note['pitch'] - 0.4, note['pitch'] + 0.4, note['pitch'] + 0.4, note['pitch'] - 0.4, note['pitch'] - 0.4],
                    fill='toself',
                    fillcolor='red',
                    line=dict(color='red', width=1),
                    mode='lines',
                    showlegend=False,
                    hoverinfo='text',
                    text=f"Pitch: {pretty_midi.note_number_to_name(note['pitch'])} ({note['pitch']})<br>"
                         f"Start: {note['start']:.3f}s<br>"
                         f"End: {note['end']:.3f}s<br>"
                         f"Duration: {note['end'] - note['start']:.3f}s<br>"
                         f"Velocity: {note['velocity']}",
                    name=''
                ),
                row=1, col=2
            )
    else:
        # Plot vazio
        fig.add_trace(
            go.Scatter(x=[0, max_duration], y=[min_pitch, max_pitch], mode='markers', 
                      marker=dict(size=0.1, opacity=0), showlegend=False),
            row=1, col=2
        )
    
    # Configurar layout
    fig.update_layout(
        title_text="Comparação de Piano Rolls",
        height=600,
        hovermode='closest'
    )
    
    # Configurar eixos
    for col in [1, 2]:
        fig.update_xaxes(title_text="Tempo (segundos)", range=[0, max_duration], row=1, col=col)
        fig.update_yaxes(
            title_text="MIDI Pitch",
            tickmode='linear',
            tick0=min_pitch,
            dtick=12,
            range=[min_pitch - 1, max_pitch + 1],
            row=1, col=col
        )
    
    return fig

def main():
    parser = argparse.ArgumentParser(
        description="Plotar piano rolls comparativos usando Plotly"
    )
    parser.add_argument(
        '--midi',
        type=str,
        required=True,
        help='Caminho para o primeiro arquivo MIDI (transcrito)'
    )
    parser.add_argument(
        '--midi_ref',
        type=str,
        required=True,
        help='Caminho para o segundo arquivo MIDI (referência)'
    )
    parser.add_argument(
        '--title1',
        type=str,
        default='MIDI Transcrito',
        help='Título do primeiro plot (padrão: MIDI Transcrito)'
    )
    parser.add_argument(
        '--title2',
        type=str,
        default='MIDI Referência',
        help='Título do segundo plot (padrão: MIDI Referência)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Caminho para salvar HTML (opcional, se não fornecido mostra no navegador)'
    )
    
    args = parser.parse_args()
    
    # Verificar se os arquivos existem
    midi1_path = Path(args.midi)
    midi2_path = Path(args.midi_ref)
    
    if not midi1_path.exists():
        print(f"❌ Arquivo não encontrado: {midi1_path}")
        return
    
    if not midi2_path.exists():
        print(f"❌ Arquivo não encontrado: {midi2_path}")
        return
    
    print("=" * 60)
    print("PLOTANDO PIANO ROLLS COMPARATIVOS")
    print("=" * 60)
    print(f"MIDI 1: {midi1_path}")
    print(f"MIDI 2: {midi2_path}")
    print("=" * 60)
    print()
    
    # Criar plot comparativo
    fig = create_comparison_plot(
        str(midi1_path),
        str(midi2_path),
        title1=args.title1,
        title2=args.title2
    )
    
    if fig is None:
        print("❌ Erro ao criar plot")
        return
    
    # Salvar ou mostrar
    if args.output:
        output_path = Path(args.output)
        fig.write_html(str(output_path))
        print(f"✅ Plot salvo em: {output_path}")
    else:
        fig.show()
        print("✅ Plot exibido no navegador")

if __name__ == "__main__":
    main()


