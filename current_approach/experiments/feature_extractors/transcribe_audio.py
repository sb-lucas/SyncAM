import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from basic_pitch.inference import predict, Model
from basic_pitch import ICASSP_2022_MODEL_PATH

#-----------------------------Load Paths----------------------------------#
root_dir = "/home/ldsbrandao/exp/smd"
synthesized_midi_dir = os.path.join(root_dir, "synthesized_midi")
synthesized_midi_transcribed_dir = os.path.join(root_dir, "synthesized_midi_transcribed")
os.makedirs(synthesized_midi_transcribed_dir, exist_ok=True)

basic_pitch_model = Model(ICASSP_2022_MODEL_PATH)

#----------------------Transcribe synthesized audio from original MIDIs----------------#
tam_original = len(os.listdir(synthesized_midi_dir))
audio_files = os.listdir(synthesized_midi_dir)
for i, file in enumerate(audio_files):
    model_output, midi_data, note_events = predict(os.path.join(synthesized_midi_dir, file), basic_pitch_model)
    midi_data.write(os.path.join(synthesized_midi_transcribed_dir, file.replace(".wav", ".mid")))