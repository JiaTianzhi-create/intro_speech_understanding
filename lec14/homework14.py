import gtts, speech_recognition, librosa, soundfile

def synthesize(text, lang, filename):
    '''
    Use gtts.gTTS(text=text, lang=lang) to synthesize speech, then write it to filename.
    
    @params:
    text (str) - the text you want to synthesize
    lang (str) - the language in which you want to synthesize it
    filename (str) - the filename in which it should be saved
    '''
    tts = gtts.gTTS(text=text, lang=lang)
    tts.save(filename)


def make_a_corpus(texts, languages, filenames):
    '''
    Create many speech files, and check their content using SpeechRecognition.
    The output files should be created as MP3, then converted to WAV, then recognized.

    @param:
    texts - a list of the texts you want to synthesize
    languages - a list of their languages
    filenames - a list of their root filenames, without the ".mp3" ending

    @return:
    recognized_texts - list of the strings that were recognized from each file
    '''
    recognized_texts = []
    
    for text, lang, root in zip(texts, languages, filenames):
        mp3_file = root + ".mp3"
        wav_file = root + ".wav"
        

        synthesize(text, lang, mp3_file)
        
 
        y, sr = librosa.load(mp3_file)
        soundfile.write(wav_file, y, sr)
        

        r = speech_recognition.Recognizer()
        with speech_recognition.AudioFile(wav_file) as source:
            audio = r.record(source)
        recognized = r.recognize_google(audio, language=lang)
        
        recognized_texts.append(recognized)
    
    return recognized_texts
