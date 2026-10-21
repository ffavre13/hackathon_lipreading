# Inference model
`src/vsr`folder contains everything related to the visual speech recognition.


## Usage example
``` py
video_with_the_expected_format = ...
model = AutoVSR(model_path=MODEL_PATH, device=<"cpu"/"gpu">)
results = model.predict(video_with_the_expected_format)
```

### `results` format : 
``` json
results = {
  "rank": int,
  "text": String,
  "score": float
}
```


## Source material
``` BibTeX
@inproceedings{ma2023auto,
  author={Ma, Pingchuan and Haliassos, Alexandros and Fernandez-Lopez, Adriana and Chen, Honglie and Petridis, Stavros and Pantic, Maja},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  title={Auto-AVSR: Audio-Visual Speech Recognition with Automatic Labels},
  year={2023},
  pages={1-5},
  doi={10.1109/ICASSP49357.2023.10096889}
}
```

> The source repo can be found at : https://github.com/mpc001/auto_avsr

> This VSR module follows the [APACHE LICENSE](./LICENSE)