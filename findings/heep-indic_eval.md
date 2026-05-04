
## Datasets Covered

* Kathbath
* Kathbath (Hard / Noisy)
* CommonVoice
* FLEURS
* IndicTTS
* Gramvaani
* RESPIN

---

## Heep Results Across Languages (WER %)

| Language      | Kathbath | Kathbath Hard | CommonVoice | FLEURS | IndicTTS | Gramvaani | RESPIN | Avg  |
| ------------- | -------- | ------------- | ----------- | ------ | -------- | --------- | ------ | ---- |
| Bengali       | 14.6     | 15.7          | 21.0        | 22.4   | 15.8     | –         | 32.5   | 20.4 |
| Bhojpuri      | –        | –             | –           | –      | –        | –         | 21.3   | 21.3 |
| Chhattisgarhi | –        | –             | –           | –      | –        | –         | 21.6   | 21.6 |
| Gujarati      | 17.4     | 18.5          | –           | 23.3   | 16.9     | –         | –      | 19.0 |
| Hindi         | 8.5      | 9.0           | 9.96        | 11.0   | 6.6      | 26.0      | 12.1   | 11.9 |
| Kannada       | 23.0     | 25.1          | –           | 23.1   | 19.6     | –         | 45.6   | 27.3 |
| Magahi        | –        | –             | –           | –      | –        | –         | 27.7   | 27.7 |
| Maithili      | –        | –             | –           | –      | –        | –         | 41.1   | 41.1 |
| Malayalam     | 39.3     | 41.2          | 46.0        | 34.4   | 26.4     | –         | –      | 37.5 |
| Marathi       | 19.2     | 20.4          | 21.5        | 25.5   | 14.5     | –         | 32.7   | 22.3 |
| Odia          | 25.4     | 27.7          | 34.6        | 33.3   | 14.8     | –         | –      | 27.2 |
| Punjabi       | 15.8     | 16.6          | 17.5        | 25.0   | –        | –         | –      | 18.7 |
| Sanskrit      | 41.4     | 43.6          | –           | –      | –        | –         | –      | 42.5 |
| Tamil         | 30.3     | 32.6          | 34.0        | 35.1   | 22.6     | –         | –      | 30.9 |
| Telugu        | 29.0     | 30.3          | –           | 31.9   | 31.3     | –         | 37.5   | 32.0 |
| Urdu          | 12.1     | 11.9          | 20.6        | 22.4   | –        | –         | –      | 16.7 |

---

## 🇮🇳 Hindi Benchmark Comparison (WER %)

| Model                     | Kathbath | Noisy    | CommonVoice | FLEURS    | IndicTTS | RESPIN    | Gramvaani | Avg      |
| ------------------------- | -------- | -------- | ----------- | --------- | -------- | --------- | --------- | -------- |
| Google STT                | 14.3     | 16.7     | 20.8        | 19.4      | 18.3     | –         | 59.9      | 24.9     |
| IndicWav2Vec              | 12.2     | 16.2     | 20.2        | 18.3      | 15.0     | –         | 42.1      | 20.7     |
| Azure STT                 | 13.6     | 15.1     | 14.6        | 24.3      | 15.2     | –         | 42.3      | 20.8     |
| Nvidia Conformer (Medium) | 14.0     | 15.6     | 20.4        | 19.4      | 12.3     | –         | 41.3      | 20.5     |
| Nvidia Conformer (Large)  | 12.7     | 14.2     | 21.2        | 15.7      | 12.2     | –         | 42.6      | 19.8     |
| IndicWhisper              | 10.3     | 12.0     | 15.0        | 11.4      | 7.6      | –         | 26.8      | 13.8     |
| **HEEP-Indic STT**        | **8.53** | **8.97** | **9.96**    | **11.04** | **6.59** | **12.05** | **25.98** | **11.9** |

---
