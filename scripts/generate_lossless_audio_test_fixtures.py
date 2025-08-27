import sys
from pathlib import Path
from pydub import AudioSegment

def main():
    """
    Takes a .flac file and generates .m4a (ALAC) and .wav versions in the same directory using pydub.
    Usage: (poetry run) python scripts/generate_lossless_audio_test_fixtures.py <path_to_flac_file>
    """
    if len(sys.argv) != 2:
        print("Usage: (poetry run) python scripts/generate_lossless_audio_test_fixtures.py <path_to_flac_file>")
        sys.exit(1)

    input_path = Path(sys.argv[1])

    if not input_path.is_file() or input_path.suffix.lower() != '.flac':
        print(f"Error: Input path '{input_path}' is not a valid .flac file.")
        sys.exit(1)

    audio = AudioSegment.from_file(input_path, format="flac")
    m4a_path = input_path.with_suffix('.m4a')
    audio.export(m4a_path, format="mp4", codec="alac")
    wav_path = input_path.with_suffix('.wav')
    audio.export(wav_path, format="wav")

if __name__ == "__main__":
    main()
