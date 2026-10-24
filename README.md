# hackathon_lipreading

Real-time lip reading: turn a silent video of someone speaking (from a webcam or an `.mp4` file) into English text.

## Description

The app looks at the speaker's mouth, predicts what is being said with a visual speech recognition (VSR) model, and improves the result in two steps: a language model first, then an LLM.

It comes with a Streamlit web interface that has two tabs:

- **Camera**: streams your webcam from the browser and shows the transcription live.
- **Video**: upload an `.mp4` file and run lip reading on it.

The VSR model comes from [Auto-AVSR](https://github.com/mpc001/auto_avsr) (see [src/vsr/README.md](src/vsr/README.md)).

## Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/), **or** Docker
- [Git LFS](https://git-lfs.com/): the model weights (`*.pth`, `*.bin`) and sample videos are stored with LFS
- An **Ollama API key** (see below)

### Getting an Ollama API key

The last correction step calls the `gemma4` model on [Ollama Cloud](https://ollama.com), so you need an API key:

1. Create an account (or sign in) at [ollama.com](https://ollama.com).
2. Go to **Settings → API keys** ([ollama.com/settings/keys](https://ollama.com/settings/keys)).
3. Click **Create new key** and copy it.
4. Copy the example env file and paste your key into it (without quotes):

   ```bash
   cp .env.example .env
   ```

   ```env
   OLLAMA_API=your_api_key_here
   ```

Without a valid key the app still runs, but you only get the raw output of the language model step, with no LLM correction.

## Setup

Clone the repository and pull the LFS files:

```bash
git lfs install
git clone https://github.com/ffavre13/hackathon_lipreading.git
cd hackathon_lipreading
git lfs pull
```

Then pick one of the following ways to run it. Run every command from the repository root.

### 1. uv (command line, local webcam)

```bash
uv sync
uv run src/main.py
```

This runs the pipeline without the interface. It reads from your default webcam (`0`) and prints the transcription in the terminal. Press **Enter** to stop recording; the final LLM correction then runs and the full text is printed.

### 2. Streamlit (web interface)

```bash
uv sync
uv run streamlit run src/interface.py --server.port=8080
```

Then open <http://localhost:8080>.

### 3. Docker

```bash
docker build -t lipreading .
docker run --rm --env-file .env -p 8080:8080 lipreading
```

Then open <http://localhost:8080>. The webcam is captured by the browser so allow camera access when prompted. 

### 4. Docker Compose

```bash
docker compose up --build
```

This builds the image, reads `.env`, publishes port `8080`, and mounts `./inputs` so that uploaded videos are kept on your machine. Then open <http://localhost:8080>.

## Pipeline

```
 webcam / .mp4
      │
      ▼
 ┌────────────┐   ┌──────────────────┐   ┌───────────────┐   ┌─────────────────┐   ┌──────────────┐
 │ 1. Capture │──▶│ 2. Mouth extract │──▶│ 3. VSR model  │──▶│ 4. KenLM rerank │──▶│ 5. LLM fixup │──▶ text
 └────────────┘   └──────────────────┘   └───────────────┘   └─────────────────┘   └──────────────┘
   batches of        MediaPipe face        Auto-AVSR,           5-gram English       gemma4 on
   75 frames         landmarks, 88×88      top-5 hypotheses     language model       Ollama Cloud
   (15 overlap)      mouth crops           per batch
```

Each stage runs in its own thread, and the stages pass batches to each other through queues ([src/main.py](src/main.py)), so frames keep being captured while earlier batches are still being processed.

1. **Capture** ([src/main.py](src/main.py), [src/browser_capture.py](src/browser_capture.py))
   Frames come from OpenCV (local webcam or video file) or from the browser through `streamlit-webrtc`. They are grouped into batches of 75 frames, and each batch overlaps the previous one by 15 frames so that words cut at a batch boundary are not lost.

2. **Mouth extraction** ([src/mouth_extraction/](src/mouth_extraction/))
   A MediaPipe face landmarker finds the face in every frame. Frames where no face is found get interpolated landmarks, and the landmarks are smoothed over time. Each face is aligned to a mean face with an affine transform, then the mouth is cropped to 88×88 grayscale. Long stretches where the mouth stays closed (silence) are dropped. The frames are resampled to 25 fps and normalised.

3. **Visual speech recognition** ([src/vsr/](src/vsr/))
   The Auto-AVSR model (`vsr_trlrs2lrs3vox2avsp_base.pth`, a conformer encoder with a transformer decoder) decodes the mouth crops with a CTC/attention beam search. It returns the **top 5** hypotheses for each batch, each with a score.

4. **Language model stabilisation** ([src/stabilizer/stabilizer.py](src/stabilizer/stabilizer.py))
   Over the last 3 batches, every combination of hypotheses is scored as the sum of the VSR scores plus a length-normalised score from an English **KenLM** 5-gram model (`wiki_en_dep.arpa.bin`). The best combination replaces the recent phrases, so earlier words can still change as more context arrives.

5. **LLM correction** ([src/stabilizer/stabilizer.py](src/stabilizer/stabilizer.py))
   Every 10 stabilised segments, and once more at the end, the segments that are no longer being changed are sent to `gemma4` on Ollama Cloud. The model removes words duplicated by the batch overlap, fixes spelling and grammar, and adds punctuation and capital letters. The output must follow a JSON schema that keeps each segment's timestamp.

## Project structure

```
src/
├── interface.py          # Streamlit UI (Camera / Video tabs)
├── main.py               # Threaded pipeline + Ollama client
├── browser_capture.py    # Adapts WebRTC frames to a cv2.VideoCapture-like object
├── mouth_extraction/     # MediaPipe landmarks, alignment, mouth cropping
├── vsr/                  # Auto-AVSR model, beam search decoding
└── stabilizer/           # KenLM reranking + LLM correction
inputs/                   # Sample videos, uploaded videos are saved here
```

## License

See [LICENSE](LICENSE). The VSR module under `src/vsr` is under the [Apache License](src/vsr/LICENSE).
