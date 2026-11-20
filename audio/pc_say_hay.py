import torch
import sounddevice

class Speaker:
    RATE = 22050

    def __init__(self):
        self._waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp32')

        self._waveglow = self._waveglow.remove_weightnorm(self._waveglow)
        self._waveglow = self._waveglow.to('cuda')
        self._waveglow.eval()

        self._tacotron2 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2', model_math='fp32')
        self._tacotron2 = self._tacotron2.to('cuda')
        self._tacotron2.eval()

        self._utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tts_utils')

    def convert(self, text):
        sequences, lengths = self._utils.prepare_input_sequence([text])

        with torch.no_grad():
            mel, _, _ = self._tacotron2.infer(sequences, lengths)
            audio = self._waveglow.infer(mel)
        audio_numpy = audio[0].data.cpu().numpy()

        return audio_numpy

    def speak(self, text: str):
        if not text:
            return
        audio_numpy = self.convert(text)
        sounddevice.play(audio_numpy, samplerate=self.RATE)
        sounddevice.wait()
