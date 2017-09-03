import webrtcvad, os, wave, contextlib, collections, argparse

def read_wave(path):
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def write_wave(path, audio, sample_rate):
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


class Frame(object):
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    output = []
    while offset + n < len(audio):
        output.append(Frame(audio[offset:offset + n], timestamp, duration))
        timestamp += duration
        offset += n
    return output


def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms,
                  aggressiveness, buffer_ratio, frames, name):
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False
    voiced_frames = []
    vad = webrtcvad.Vad(aggressiveness)
    count_until_triggered = 0
    count_unvoiced_tail = 0
    for frame in frames:
        if not triggered:
            ring_buffer.append(frame)
            num_voiced = 0
            for f in ring_buffer:
                if vad.is_speech(f.bytes, sample_rate):
                    num_voiced += 1
            if num_voiced > buffer_ratio * ring_buffer.maxlen:
                triggered = True
                voiced_frames.extend(ring_buffer)
                ring_buffer.clear()
            count_until_triggered += 1
        else:
            voiced_frames.append(frame)
            if vad.is_speech(frame.bytes, sample_rate):
                count_unvoiced_tail = 0
            else:
                count_unvoiced_tail -= 1
    if voiced_frames:
        if count_unvoiced_tail != 0:
            voiced_frames = voiced_frames[:count_unvoiced_tail]
        vad_list = []
        vad = webrtcvad.Vad(aggressiveness)
        for f in voiced_frames[:ring_buffer.maxlen]:
            vad_list.append(vad.is_speech(f.bytes, sample_rate))
    else:
        print('Maybe unvoiced file.', name)
        vad_list = []
        vad = webrtcvad.Vad(aggressiveness)
        for f in frames:
            vad_list.append(vad.is_speech(f.bytes, sample_rate))
        voiced_frames = frames

    first_voice = vad_list.index(1)
    voiced_frames = voiced_frames[first_voice:]
    return b''.join([f.bytes for f in voiced_frames])


if __name__ == '__main__':
    """
    Codes from: https://github.com/wiseman/py-webrtcvad/blob/master/example.py
    """
    parser = argparse.ArgumentParser(description='training script')
    parser.add_argument('--agressiveness', type=int, default='2', help='integer from 0 to 3. aggressiveness about filtering out non-speech, 3 is the most aggressive.')
    parser.add_argument('--buffer_ratio', type=float, default='0.3', help='ratio of speeched frames to trigger buffer.')
    args = parser.parse_args()

    aggressiveness = args.aggressiveness
    buffer_ratio = args.buffer_ratio

    rDirectory = '/data2/lyg0722/TTS_corpus/etri/wav_old/'
    wDirectory = '/data2/lyg0722/TTS_corpus/etri/wav/'

    for root, dirnames, filenames in os.walk(rDirectory):
        for filename in sorted(filenames):
            if filename[-4:] == '.wav':
                rf = os.path.join(root, filename)
                audio, sample_rate = read_wave(rf)
                frames = frame_generator(30, audio, sample_rate)
                segment = vad_collector(sample_rate, 30, 300, aggressiveness, buffer_ratio, frames, filename)
                wPath = str(wDirectory + filename)
                write_wave(wPath, segment, sample_rate)

    # TODO: use multiprocess to speed up.