import os
import sys
import numpy as np
print('importando')
import torch
print('importei coisas normais')
import mido
import torchaudio
import soundfile as sf
from dotenv import load_dotenv
# Import CLaMP 3 specific architectures from their repository
sys.path.append(os.path.abspath("/home/ldsbrandao/exp/clamp3/code/"))  
from config import *
from utils import *
from samplings import *
from transformers import BertConfig, Wav2Vec2FeatureExtractor, AutoModel
print("importei")


#----------------------------AUDIO TO MERT FEATURES--------------------------------------#
def load_mert_model(device):
    """Loads the MERT acoustic feature extractor."""
    print("Loading MERT audio processor...")
    processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
    mert_model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True).to(device)
    mert_model.eval()
    return processor, mert_model

def get_mert_from_audio(file_path, processor, mert_model, device):
    """
    Safely loads audio using soundfile, applies the 5-second chunking, 
    and perfectly replicates the original CLaMP 3 'mean' reduction logic.
    """
    # 1. Safe Load (Bypassing torchaudio.load / torchcodec error)
    audio_data, sample_rate = sf.read(file_path)
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    waveform = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0)
    
    # 2. Resample
    target_sr = 24000
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
        waveform = resampler(waveform)

    # 3. Process to input values
    wav = processor(waveform.squeeze().numpy(), sampling_rate=target_sr, return_tensors="pt").input_values
    wav = wav.to(device)

    # 4. EXACT Sliding Window from original code
    sliding_window_size_in_sec = 5
    overlap_in_sec = 0
    wavs = []
    
    for i in range(0, wav.shape[-1], int(target_sr * (sliding_window_size_in_sec - overlap_in_sec))):
        wavs.append(wav[:, i : i + int(target_sr * sliding_window_size_in_sec)])
        
    if wavs[-1].shape[-1] < target_sr * 1:
        wavs = wavs[:-1]

    # 5. Extract and Reduce exactly like hf_pretrains.py
    features = []
    with torch.no_grad():
        for wav_chunk in wavs:
            outputs = mert_model(wav_chunk, output_hidden_states=True).hidden_states
            
            # Stack to [Layers(13), Batch(1), Time(T), Hidden(768)]
            stacked_layers = torch.stack(outputs) 
            
            # This is the original reduction='mean' logic! Averages across Time (-2)
            # Resulting shape per chunk: [13, 1, 768]
            reduced_chunk = stacked_layers.mean(-2) 
            features.append(reduced_chunk)
            
    # 6. Concatenate chunks along the Batch dimension (dim=1) 
    # Shape becomes: [13, num_chunks, 768]
    features = torch.cat(features, dim=1) 
    
    # 7. Apply the --mean_features flag from original script (averages across the 13 layers)
    # Shape becomes: [1, num_chunks, 768]
    final_features = features.mean(dim=0, keepdim=True) 
    
    return final_features.cpu().numpy()
#--------------------------------TO CONVERT MIDI TO MTF------------------------------------#
def msg_to_str(msg):
    str_msg = ""
    for key, value in msg.dict().items():
        str_msg += " " + str(value)
    return str_msg.strip().encode('unicode_escape').decode('utf-8')

def get_mtf_from_midi(midi_path, m3_compatible=True):
    """
    Loads a MIDI file and converts it to an MTF (Music Text Format) string in-memory.
    Based on the official CLaMP 3 preprocessing logic.
    """
    # Load the MIDI file
    mid = mido.MidiFile(midi_path)
    
    # Initialize with ticks_per_beat
    msg_list = ["ticks_per_beat " + str(mid.ticks_per_beat)]

    # Traverse the merged MIDI track
    for msg in mid.merged_track:
        # Strip metadata if M3 compatible mode is enabled
        if m3_compatible and msg.is_meta:
            if msg.type in ["text", "copyright", "track_name", "instrument_name", 
                            "lyrics", "marker", "cue_marker", "device_name"]:
                continue        

        # Convert message dictionary to string
        str_msg = msg_to_str(msg)
        msg_list.append(str_msg)
            
    # MUST be joined by newline (\n) for the tokenizer
    return "\n".join(msg_list)

#--------------------------------LOAD THE CLAMP3 MODEL AND EXTRACT EMBEDDINGS-----------------------------#
def load_clamp_model(device):
    """Initializes the CLaMP 3 model and loads the pre-trained weights."""
    print("Initializing CLaMP 3 architecture...")
    audio_config = BertConfig(vocab_size=1, hidden_size=AUDIO_HIDDEN_SIZE, 
                              num_hidden_layers=AUDIO_NUM_LAYERS, num_attention_heads=AUDIO_HIDDEN_SIZE//64, 
                              intermediate_size=AUDIO_HIDDEN_SIZE*4, max_position_embeddings=MAX_AUDIO_LENGTH)
    
    symbolic_config = BertConfig(vocab_size=1, hidden_size=M3_HIDDEN_SIZE, 
                                 num_hidden_layers=PATCH_NUM_LAYERS, num_attention_heads=M3_HIDDEN_SIZE//64, 
                                 intermediate_size=M3_HIDDEN_SIZE*4, max_position_embeddings=PATCH_LENGTH)
    
    model = CLaMP3Model(audio_config=audio_config, symbolic_config=symbolic_config, 
                        text_model_name=TEXT_MODEL_NAME, hidden_size=CLAMP3_HIDDEN_SIZE, load_m3=CLAMP3_LOAD_M3)
    
    # Load weights
    checkpoint_path = "/home/ldsbrandao/exp/hf_models/clamp3/weights_clamp3_saas_h_size_768_t_model_FacebookAI_xlm-roberta-base_t_length_128_a_size_768_a_layers_12_a_length_128_s_size_768_s_layers_12_p_size_64_p_length_512.pth"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Cannot find weights at {checkpoint_path}")
        
    print("Loading CLaMP 3 weights into memory...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint['model'])
    
    model.eval()
    return model.to(device)

def get_embedding(model, file_name, original_dir, save_dir, modality, device, processor, mert_model, patchilizer):
    clean_file_name = os.path.splitext(file_name)[0]  # Remove any additional extensions
    save_path = os.path.join(save_dir, clean_file_name + ".npy")

    # Resume capability: skip if already processed
    if os.path.exists(save_path):
        return "SKIPPED (Already exists): " + file_name

    file_path = os.path.join(original_dir, file_name)

    """Extracts a global 768-dimensional embedding from a raw .wav or .mid file."""
    with torch.no_grad():
        if modality == "midi":
            # Process raw MIDI file into MTF directly in memory
            item = get_mtf_from_midi(file_path)
            input_data = torch.tensor(patchilizer.encode(item, add_special_patches=True))
            max_input_length = PATCH_LENGTH
            
        elif modality == "audio":
            # Process raw Audio file into MERT features directly in memory
            mert_array = get_mert_from_audio(file_path, processor, mert_model, device)
            input_data = torch.tensor(mert_array)

            input_data = input_data.reshape(-1, input_data.size(-1))
            zero_vec = torch.zeros((1, input_data.size(-1)))
            input_data = torch.cat((zero_vec, input_data, zero_vec), 0)
            max_input_length = MAX_AUDIO_LENGTH

        # Chunk and Pad the data
        segment_list = [input_data[i:i+max_input_length] for i in range(0, len(input_data), max_input_length)]
        segment_list[-1] = input_data[-max_input_length:]
        
        last_hidden_states_list = []
        
        for input_segment in segment_list:
            input_masks = torch.tensor([1] * input_segment.size(0))
            
            if modality == "midi":
                pad_indices = torch.ones((PATCH_LENGTH - input_segment.size(0), PATCH_SIZE)).long() * patchilizer.pad_token_id
            else:
                pad_indices = torch.ones((MAX_AUDIO_LENGTH - input_segment.size(0), AUDIO_HIDDEN_SIZE)).float() * 0.
                
            input_masks = torch.cat((input_masks, torch.zeros(max_input_length - input_segment.size(0))), 0)
            input_segment = torch.cat((input_segment, pad_indices), 0)

            # Forward pass through the correct modality encoder
            if modality == "midi":
                states = model.get_symbolic_features(symbolic_inputs=input_segment.unsqueeze(0).to(device),
                                                     symbolic_masks=input_masks.unsqueeze(0).to(device),
                                                     get_global=True)
            else:
                states = model.get_audio_features(audio_inputs=input_segment.unsqueeze(0).to(device),
                                                  audio_masks=input_masks.unsqueeze(0).to(device),
                                                  get_global=True)
            last_hidden_states_list.append(states)

        # Average the chunks globally
        full_chunk_cnt = len(input_data) // max_input_length
        remain_chunk_len = len(input_data) % max_input_length
        feature_weights = [max_input_length] * full_chunk_cnt
        if remain_chunk_len != 0: feature_weights.append(remain_chunk_len)
        feature_weights = torch.tensor(feature_weights, device=device).view(-1, 1)

        states_tensor = torch.concat(last_hidden_states_list, 0) * feature_weights
        final_embedding = states_tensor.sum(dim=0) / feature_weights.sum()
        
        np.save(save_path, final_embedding.squeeze().cpu().numpy())

        return "OK: " + file_name

def main():
    load_dotenv()
    DATASETS_FOLDER = os.environ["DATASETS_FOLDER"]
    datasets = ['maestro', 'maps/ENSTDkAm', 'maps/ENSTDkCl', 'musicnet', 'smd']
    
    print("Running in strictly sequential mode (0 background workers).")

    #----------------------Load the models once--------------------------#
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    
    processor, mert_model = load_mert_model(device)
    clamp_model = load_clamp_model(device)
    patchilizer = M3Patchilizer() 

    #main loop
    for dataset in datasets:
        print(f"\nProcessing: {dataset}")

        root_dir = os.path.join(DATASETS_FOLDER, dataset)
        corrupt_midi_dir = os.path.join(root_dir, "corrupt_midi")
        original_audio_dir = os.path.join(root_dir, "audios")

        corrupt_midi_clamp_dir = os.path.join(root_dir, "corrupt_midi_clamp")
        original_audio_clamp_dir = os.path.join(root_dir, "audios_clamp")
        
        # Ensure save directories exist
        os.makedirs(corrupt_midi_clamp_dir, exist_ok=True)
        os.makedirs(original_audio_clamp_dir, exist_ok=True)

        #----------------------Extract CLAMP from original audios----------------#
        if os.path.exists(original_audio_dir):
            audio_files = sorted(f for f in os.listdir(original_audio_dir) if f.lower().endswith((".wav", ".mp3", ".flac")))
            print(f"Found {len(audio_files)} audio files.")
            for i, file in enumerate(audio_files):
                msg = get_embedding(clamp_model, file, original_audio_dir, original_audio_clamp_dir, "audio", 
                                    device, processor, mert_model, patchilizer) 
                print(f"{i+1}/{len(audio_files)} | Audio | {msg}")
        
        #----------------------Extract CLAMP from corrupt MIDIs----------------#
        if os.path.exists(corrupt_midi_dir):
            midi_files = sorted(f for f in os.listdir(corrupt_midi_dir) if f.lower().endswith((".mid", ".midi")))
            print(f"Found {len(midi_files)} MIDI files.")
            for i, file in enumerate(midi_files):
                msg = get_embedding(clamp_model, file, corrupt_midi_dir, corrupt_midi_clamp_dir, "midi", 
                                    device, processor, mert_model, patchilizer) 
                print(f"{i+1}/{len(midi_files)} | MIDI | {msg}")

if __name__ == "__main__":
    main()