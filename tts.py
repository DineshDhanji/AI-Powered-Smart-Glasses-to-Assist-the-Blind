import pyttsx3
from constants import VOICE_RATE, VOICE_VOLUME, VOICE_ID


class TTS_Engine:
    def __init__(self, voice_id=VOICE_ID, rate=VOICE_RATE, volume=VOICE_VOLUME):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)  # setting up new voice rate
        self.engine.setProperty(
            "volume", volume
        )  # setting up volume level  between 0 and 1

        voices = self.engine.getProperty("voices")  # getting details of current voice
        # engine.setProperty('voice', voices[0].id)  #changing index, changes voices. o for male
        self.engine.setProperty(
            "voice", voices[voice_id].id
        )  # changing index, changes voices. 1 for female

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()
        self.engine.stop()
