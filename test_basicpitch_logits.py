#!/usr/bin/env python3
"""
Script simples para testar a extração de logits do Basic Pitch
"""

import sys
import numpy as np

try:
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH
    print("✅ Basic Pitch importado com sucesso!")
except ImportError as e:
    print(f"❌ Erro ao importar Basic Pitch: {e}")
    print("Execute: conda activate staffspy")
    sys.exit(1)

def test_logits_extraction(audio_path):
    """Testar extração de logits de um arquivo de áudio"""
    
    print(f"\n🎵 Testando extração de logits de: {audio_path}")
    print("-" * 60)
    
    try:
        # Fazer predição - model_output contém os logits brutos
        print("🔄 Executando predição...")
        model_output, midi_data, note_events = predict(audio_path)
        
        print("✅ Predição concluída!")
        print(f"📊 Tipo do model_output: {type(model_output)}")
        
        # Analisar model_output
        if isinstance(model_output, dict):
            print(f"📋 Chaves disponíveis: {list(model_output.keys())}")
            for key, value in model_output.items():
                if isinstance(value, np.ndarray):
                    print(f"  🔢 {key}: shape={value.shape}, dtype={value.dtype}")
                    print(f"      Min: {np.min(value):.4f}, Max: {np.max(value):.4f}, Mean: {np.mean(value):.4f}")
                    print(f"      Primeiros valores: {value.flatten()[:5]}")
                else:
                    print(f"  📝 {key}: {type(value)} = {value}")
                    
        elif isinstance(model_output, (list, tuple)):
            print(f"📦 Lista/Tupla com {len(model_output)} elementos:")
            for i, item in enumerate(model_output):
                if isinstance(item, np.ndarray):
                    print(f"  🔢 [{i}]: shape={item.shape}, dtype={item.dtype}")
                    print(f"       Min: {np.min(item):.4f}, Max: {np.max(item):.4f}")
                else:
                    print(f"  📝 [{i}]: {type(item)}")
                    
        elif isinstance(model_output, np.ndarray):
            print(f"🔢 Array único: shape={model_output.shape}, dtype={model_output.dtype}")
            print(f"   Min: {np.min(model_output):.4f}, Max: {np.max(model_output):.4f}")
            print(f"   Mean: {np.mean(model_output):.4f}, Std: {np.std(model_output):.4f}")
            print(f"   Primeiros 10 valores: {model_output.flatten()[:10]}")
        
        # Informações sobre notas detectadas
        print(f"\n🎼 Notas detectadas: {len(note_events) if note_events else 0}")
        if note_events and len(note_events) > 0:
            print("   Primeiras 5 notas:")
            print(f"   Tipo de note_events: {type(note_events)}")
            
            # Verificar se é uma lista de objetos ou uma estrutura diferente
            if hasattr(note_events[0], 'pitch'):
                for i, note in enumerate(note_events[:5]):
                    print(f"     {i+1}. Pitch: {note.pitch}, Start: {note.start_time:.3f}s, End: {note.end_time:.3f}s")
            else:
                # Se for uma estrutura diferente, mostrar o tipo e conteúdo
                for i, note in enumerate(note_events[:5]):
                    print(f"     {i+1}. Tipo: {type(note)}, Conteúdo: {note}")
                    if hasattr(note, '__dict__'):
                        print(f"         Atributos: {list(note.__dict__.keys())}")
                    elif isinstance(note, (list, tuple)) and len(note) >= 3:
                        print(f"         Valores: pitch={note[0]}, start={note[1]:.3f}s, end={note[2]:.3f}s")
        
        print("\n" + "="*60)
        print("🎯 LOGITS EXTRAÍDOS COM SUCESSO!")
        print("   Os logits estão em 'model_output'")
        print("   Use este objeto para análises mais detalhadas")
        print("="*60)
        
        return model_output, midi_data, note_events
        
    except Exception as e:
        print(f"❌ Erro durante extração: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python test_basicpitch_logits.py <arquivo_audio>")
        print("Exemplo: python test_basicpitch_logits.py audio.wav")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    print("🚀 TESTE DE EXTRAÇÃO DE LOGITS DO BASIC PITCH")
    print("=" * 60)
    
    model_output, midi_data, note_events = test_logits_extraction(audio_file)
    
    if model_output is not None:
        print("\n💡 DICA: Para usar os logits em seu código:")
        print("   model_output, midi_data, note_events = predict('audio.wav')")
        print("   # model_output contém os logits brutos")
        print("   # midi_data contém o MIDI processado")
        print("   # note_events contém as notas detectadas")
