# Cascading a neural speech-separation front-end with an ASR back-end for single-channel overlapped speech

Legend for every figure: **[P]** = read by me from the primary source (arXiv HTML/PDF or ISCA PDF, fetched 2026-11-10); **[S]** = seen only in a search-result snippet, not verified in the paper. "Not Reported" = looked for and absent from the source. No number below is estimated.

Caveat on comparability that applies throughout: "Libri2Mix" is not one test set. Papers use (a) the original fully-overlapped mixtures, (b) the ESPnet `sot_asr1` variant with an onset offset on one speaker, in (c) clean or (d) noisy (WHAM!) form. A 2026 paper shows the same LLM-SOT model scores 11.0 % (offset) vs 17.6 % (original, no offset) on noisy Libri2Mix test and 38.9 % vs 61.1 % on noisy Libri3Mix test [P] — [Beyond Acoustic Prefixes, arXiv 2603.27205, Table VII](https://arxiv.org/html/2603.27205). Cross-paper rows must therefore be matched on variant before being compared.

---

## 1. WER of separation → strong pretrained ASR vs end-to-end multi-talker models (and gap to oracle clean sources)

### Takeaway
With a current TF-domain separator (TF-GridNet / TF-CrossNet), a plain cascade into an *unmodified* strong ASR (Whisper large, WavLM-Conformer) is competitive with or better than published end-to-end/SOT/LLM-SOT numbers on Libri2Mix and single-channel LibriCSS, but it still sits roughly 2× above the oracle-clean-source WER (5.0 vs 2.5 on Libri2Mix; 6.8 vs 3.5 on LibriCSS with Whisper). No published cascade result exists for Qwen3-ASR, Parakeet/Canary, or any Speech-LLM recognizer fed separated audio, and no 3-speaker (Libri3Mix) cascade-with-Whisper number was found.

### Cited Findings

**Libri2Mix (noisy, ESPnet offset variant; Yang et al. 2025, TF-CrossNet separator, 1-ch)** — all [P] from [Elevating Robust Multi-Talker ASR by Decoupling Speaker Separation and Speech Recognition, arXiv 2503.17886, Table 1](https://arxiv.org/html/2503.17886v1):

| System | Front-end? | Dev WER | Test WER | WER on ground-truth clean sources |
|---|---|---|---|---|
| SOT-Conformer (ESPnet baseline) | no | 24.7 | 23.3 | – |
| SOT-WavLM-Conformer (ESPnet) | no | 19.4 | 17.1 | – |
| Conditional-Conformer (Guo 2021) | no | 24.5 | 24.9 | – |
| TS-HuBERT (Zhang & Qian 2023) | no | – | 24.8 | – |
| TSE-Whisper (Zhang 2023) | yes (target-speaker extraction) | – | 12.0 | – |
| GEncSep (Shi 2024a) | no | 17.2 | 15.0 | – |
| WavLM + LLM (Shi 2024b) | no | 11.4 | 10.2 | – |
| WavLM + LLM, fine-tuned (Shi 2024b) | no | 10.3 | 9.0 | – |
| PIT-SOT (Shi 2024c) | no | – | 7.0 | – |
| TF-CrossNet → WRConformer+WavLM trained on Libri2Mix clean | yes | 5.1 | 5.1 | 2.3 |
| TF-CrossNet → same, noise-dependent training | yes | 5.0 | 4.9 | 2.9 |
| TF-CrossNet → same, noise-independent training | yes | 5.2 | 5.0 | 2.8 |
| TF-CrossNet → same, LibriSpeech-clean training | yes | 5.2 | 5.2 | 4.1 |
| **TF-CrossNet → Whisper large-v3 (no fine-tuning)** | yes | **5.6** | **5.0** | **2.5** |

- The authors state the cascade "significantly outperform[s] the current state-of-the-art result of 7.0% WER by over 25% relatively" [P] — [arXiv 2503.17886](https://arxiv.org/html/2503.17886v1).
- Their test set follows the ESPnet `librimix/sot_asr1` recipe in which "an offset is added to one speaker in each speech mixture"; 3000 utterances each in dev/test [P] — [arXiv 2503.17886 §3.1.1](https://arxiv.org/html/2503.17886v1). So the 5.0 % Whisper number is for the *offset* variant with WHAM! noise, not the fully-overlapped original.
- Residual gap with zero-shot Whisper large-v3: 5.0 % on separated test audio vs 2.5 % on ground-truth clean sources [P] — [arXiv 2503.17886 Table 1](https://arxiv.org/html/2503.17886v1).

**Libri2Mix / Libri3Mix, end-to-end and LLM-based reference points** (to position the cascade; all [P] from [arXiv 2603.27205 Table VII](https://arxiv.org/html/2603.27205); ✗ = offset variant, ✓ = original no-offset mixtures):

| System | Variant | L2M noisy test | L3M noisy test | L2M clean test | L3M clean test |
|---|---|---|---|---|---|
| Conformer SOT | ✗ | 17.1 | 28.2 | 7.0 | 14.7 |
| GEncSep | ✗ | 15.0 | 25.9 | 6.6 | 13.1 |
| TSE-V-Whisper | ✓ | 12.0 | – | – | – |
| UME (unified diar+sep+ASR encoder) | ✓ | 19.6 | 27.1 | 6.4 | 15.9 |
| SoloSpeech (generative TSE cascade) | ✓ | 18.0 | – | 15.0 | – |
| Whisper-Sidecar | ✓ | – | – | 4.7 | 16.8 |
| W2V-Sidecar-ft | ✓ | – | – | 8.1 | – |
| SOT-LLaMA-1B | ✗ / ✓ | 11.0 / 17.6 | 38.9 / 61.1 | 4.6 / 8.6 | 20.4 / 48.4 |
| SOP-LLaMA-3B | ✗ | 9.2 | 28.1 | 3.6 | 16.5 |
| Proposed (separator + gated cross-attn into LLM) ID-20.D | ✗ / ✓ | 7.5 / 13.1 | 18.0 / 45.3 | 3.9 / 8.2 | 10.8 / 31.6 |
| Proposed ID-30.D | ✗ / ✓ | 7.7 / 13.5 | 17.1 / 42.6 | 3.1 / 6.9 | 8.7 / 30.4 |

- LLM-SOT degrades sharply from 2 to 3 talkers: 1B-Instruct LLM-SOT goes 9.1 % → 32.3 % (noisy) and 3.7 % → 19.1 % (clean) from Libri2Mix to Libri3Mix [P] — [arXiv 2603.27205](https://arxiv.org/html/2603.27205).
- Whisper-Sidecar (Whisper-large + Sidecar separator + target-talker identifier, frozen Whisper) reports 4.66 % (LibriMix 2spk), 16.79 % (LibriMix 3spk), 3.43 % (LibriSpeechMix 2spk), 6.80 % (LibriSpeechMix 3spk) [P] — [Empowering Whisper as a Joint Multi-Talker and Target-Talker Speech Recognition System, arXiv 2407.09817, Table 2](https://arxiv.org/html/2407.09817v2). (Integrated, not a cascade; included as the same-family comparison point.)

**WSJ0-2mix (monaural, cascade without joint fine-tuning)** — [P] from [Exploring the Integration of Speech Separation and Recognition with SSL Representation, arXiv 2307.12231 (ar5iv), Table 1](https://ar5iv.labs.arxiv.org/html/2307.12231):
- Monaural TF-GridNet → WavLM-feature ASR: SDR 19.40 dB, WER 4.8 %; earlier time-domain-separator cascade: SDR 13.8 dB, WER 22.9 %. Same ASR on clean single-speaker WSJ eval: 1.3 %.

**WHAMR! (noisy + reverberant 2-speaker), Whisper large-v2 as unseen evaluator** — [P] from [Transcription-Free Fine-Tuning of Speech Separation Models for Noisy and Reverberant Multi-Speaker ASR, arXiv 2406.08914, Table 2](https://ar5iv.labs.arxiv.org/html/2406.08914):
- Oracle sources 10.9 % cpWER; unprocessed mixture 63.3 %; TD-Conformer-XL separator (SI-SDR-trained) 29.5 %; after fine-tuning the separator with an ASR-encoder embedding loss 26.8 %. I.e. in noisy-reverberant conditions the cascade is ~2.5–2.7× the oracle.

**SMS-WSJ 1-ch (reverberant 2-speaker)** — [P] [arXiv 2503.17886 §4.2](https://arxiv.org/html/2503.17886v1): 1-ch TF-CrossNet output gives 8.13 % WER with a reverb-noisy-trained AM and 7.60 % with a direct-path (clean)-trained AM; an AM retrained on TF-CrossNet-separated training data reaches 6.26 %.

**LibriCSS single-channel (continuous, 0–40 % overlap)** — see §3 for per-condition numbers. Headline: TF-GridNet CSS → Whisper large-v2 (no fine-tuning) 6.8 % ORC-WER vs Whisper on clean signals 3.5 % vs Whisper with VAD only (no separation) 26.5 %; t-SOT TT cited at 7.6 % (asclite WER) and MT-RNN-T at 23.6 % in the same table [P] — [Meeting Recognition with CSS and Transcription-Supported Diarization, arXiv 2309.16482, Table 1](https://arxiv.org/html/2309.16482).

### Inferences
- On the two benchmarks where a like-for-like table exists (noisy offset Libri2Mix; 1-ch LibriCSS), the separation → off-the-shelf Whisper cascade beats or matches the published SOT / LLM-SOT numbers listed beside it. The comparison is confounded: the cascades use far stronger separators (TF-GridNet/TF-CrossNet) than the E2E papers' training budgets, and the E2E numbers were often trained only on LibriMix.
- The remaining ~2× gap to oracle sources (5.0 vs 2.5; 6.8 vs 3.5) is the price of separation error + artifact mismatch; it is the headroom an ASR back-end fine-tuned on separated audio, or a hybrid design, can attack.
- For 3 simultaneous talkers the evidence favours explicit separation-style inductive bias: every LLM-SOT system in the 2026 table collapses on Libri3Mix (28–61 %), while separator-assisted designs hold 17–18 % (offset noisy). No pure cascade number exists for Libri3Mix with a strong ASR, so this is suggestive, not proven.
- The earlier survey's t-SOT LibriCSS average (9.0–10.1) differs from the 7.6 % t-SOT figure quoted in the TF-GridNet papers; the latter is labelled "t-SOT TT [13], LS960, asclite WER" — a different t-SOT configuration/scoring. Treat them as two published figures, not a contradiction to resolve by averaging.

### Gaps
- No published result feeds separated audio into Qwen3-ASR, Qwen-Audio, SALMONN, Parakeet or Canary. Searched; not found.
- No Libri3Mix / 3-speaker cascade → Whisper WER found (only integrated systems).
- No same-paper comparison of cascade vs SOT fine-tuning *of the same ASR backbone* on the same test set was found; all comparisons are cross-system rows in literature tables.
- SepALM (audio-LM as error corrector after SepFormer, WHAM!/WHAMR!/Libri2Mix) exists [S] — [arXiv 2505.03273](https://arxiv.org/pdf/2505.03273); its WER tables were not extracted (Not Reported here).

---

## 2. The artifact / mismatch problem and its remedies

### Takeaway
Separation output can raise SI-SDR/SDR while leaving WER flat or worse; the dominant culprit is the *artifact* component (energy that is neither speech nor interference), and separation actively hurts when the target is already dominant (high SIR) or noise is heavy. Published remedies with measured effect: observation adding / switching back to the mixture, ASR-aware separator losses, joint fine-tuning (WER improves while SDR falls), robust SSL front-end features in the ASR, and matched training of the ASR on separated audio.

### Cited Findings

**Artifacts vs noise (Iwamoto et al., Interspeech 2022)** — [P] [How Bad Are Artifacts?, arXiv 2201.06685 (ar5iv)](https://ar5iv.labs.arxiv.org/html/2201.06685); also [ISCA archive](https://www.isca-archive.org/interspeech_2022/iwamoto22_interspeech.html):
- Enhancement error is decomposed by orthogonal projection into a noise component and an artifact component ("cannot be represented as a linear combination of speech and noise sources").
- "The original enhanced signals do not improve the ASR performance even compared to the observed signals"; scaling down the artifact component "greatly improve[s] ASR performance", whereas "ASR performance is affected much less by scaling the noise error component."
- Observation adding (OA: add a scaled copy of the input mixture to the enhanced signal): as the OA weight rises, SDR and SNR decrease while SAR monotonically increases, and WER improves with SAR; with OA "we can outperform both the baseline observed signals and original enhanced signals." Too large a weight degrades back toward the observed-signal WER.
- Numeric WERs are given only in figures in the HTML: **Not Reported** as table values. Task was single-channel noise reduction (WSJ0 + CHiME-3 noise), not speaker separation — the paper notes multi-speaker extension as future work.

**Separation can hurt at high SIR (Sato et al., Interspeech 2021)** — [P] [Should We Always Separate?, arXiv 2106.00949 PDF](https://arxiv.org/pdf/2106.00949.pdf); [ISCA](https://www.isca-archive.org/interspeech_2021/sato21_interspeech.html). Setup: time-domain SpeakerBeam target-speaker extraction, Japanese CSJ + CHiME-4 noise, ~94 % overlap, CER:

| SIR (dB) | Enhanced CER @SNR 20/10/0 | Observed-mixture CER @SNR 20/10/0 |
|---|---|---|
| 0 | 8.4 / 10.9 / 31.2 | 80.7 / 80.6 / 83.9 |
| 5 | 6.7 / 9.0 / 28.2 | 40.9 / 37.5 / 40.1 |
| 10 | 5.8 / 7.8 / 24.9 | 16.5 / 13.8 / 20.4 |
| 15 | 5.4 / 7.6 / 23.6 | 7.9 / 7.5 / 15.9 |
| 20 | 6.3 / 8.9 / 23.2 | 5.3 / 6.0 / 14.8 |

- At SIR 0 dB extraction cuts CER ~80 % on average; at SIR 20 dB it *degrades* recognition (relative CER change −6 % to −75 % depending on noise type/SNR, Table 1).
- A rule-based switch (use the mixture when estimated SIR − SNR ≥ 10 dB) gave an average 22 % CER reduction in those conditions and "up to 27 % relative CER reduction", with a remaining gap to oracle switching.
- Follow-up "Learning to Enhance or Not" replaces the rule with a network that predicts which signal ASR will handle better and adds soft switching (blending); abstract claims performance "comparable to that of rule-based oracle switching" [S] — [deepai summary of arXiv 2201.00256](https://deepai.org/publication/learning-to-enhance-or-not-neural-network-based-switching-of-enhanced-and-observed-signals-for-overlapping-speech-recognition). Numeric tables Not Reported here.

**Signal metrics and WER decouple under joint training (Masuyama et al. 2023)** — [P] [arXiv 2307.12231 Table 1](https://ar5iv.labs.arxiv.org/html/2307.12231):
- 8-ch anechoic WSJ0-2mix, TF-GridNet front-end + WavLM ASR: without joint fine-tuning SDR 26.43 dB / WER 3.2 %; after joint fine-tuning with the ASR loss SDR 15.28 dB / WER 1.7 %. Reverberant: SDR 18.81 → 12.32 dB while WER 2.4 → 1.8 %. "Joint fine-tuning further reduced the WERs … while degrading the separation performance."
- SSL features as the mismatch absorber: replacing WavLM features with filterbanks in the ASR (no joint fine-tuning) raised WER from 3.2 → 6.3 % (anechoic) and 2.4 → 28.2 % (reverberant) on the same separated audio.
- Best jointly trained system reaches 2.5 % WER on reverberant WHAMR! (multi-channel) vs 28.9 % for a prior MVDR+filterbank integration.
- Note: the *monaural* models in this paper were not jointly fine-tuned (4.8 % cascade only).

**ASR-encoder (transcription-free) loss for separator fine-tuning (2024)** — [P] [arXiv 2406.08914 Tables 1–2](https://ar5iv.labs.arxiv.org/html/2406.08914), WHAMR!, TD-Conformer-XL:
- With the ASR model used for the loss: SI-SDR-trained separator 44.2 % cpWER (SI-SDR 5.5 dB) → +30 more epochs SI-SDR loss 43.5 % (5.6 dB) → ASR-embedding loss with guided PIT 37.1 % (SI-SDR *lower*, 5.1 dB; PESQ 1.54 vs 1.44; STOI 75.0 vs 71.9). Oracle 7.7 %, mixture 85.7 %.
- Gains transfer to an unseen recognizer (Whisper large-v2): 29.3 → 26.8 % cpWER.
- Training with the embedding loss without the guided-PIT modification diverged ("100% WER").

**Matched vs clean vs multi-condition ASR training on separated output (Yang et al. 2025)** — [P] [arXiv 2503.17886](https://arxiv.org/html/2503.17886v1):
- Libri2Mix test: back-end trained on clean only 5.1 %, on noisy single-speaker 4.9–5.0 %; but on clean ground truth the clean-trained back-end is better (2.3 vs 2.8–2.9). The authors argue that with a strong separator, a clean-trained back-end is preferable.
- SMS-WSJ 1-ch: reverb-noisy AM 8.13 % → direct-path AM 7.60 % → AM retrained on the separator's own output 6.26 % (best, but tied to that separator; "when a better frontend is available, retraining these backends is required").
- Condition dependence (SMS-WSJ-Large, 1-ch): the clean-trained back-end wins at T60 0.2–0.5 s and SNR 0–40 dB; "the mainstream [distortion-tolerant] approach achieves lower WER at higher T60 and lower SNR values."

**ASR fine-tuning on separator output in CSS (Vieting et al. 2023)** — [P] [Combining TF-GridNet and Mixture Encoder for CSS, arXiv 2309.08454, Table 2](https://arxiv.org/pdf/2309.08454), LibriCSS average ORC-WER, hybrid AM:
- TF-GridNet separator: no fine-tuning 7.9 (4-gram) / 6.4 (Transformer LM) → AM fine-tuned on separated simulated meetings (f-CE) 7.2 / 5.8 → + sMBR 7.0 / 5.6.
- BLSTM separator: 21.6 / 19.4 → fine-tuned 19.9 / 17.9 → + mixture encoder 19.3 / 17.6.

**Other evidence that "cleaner-sounding" front-ends hurt zero-shot Whisper** — [P] [When Audio Separation Hurts Zero-Shot ASR (SAM-Audio + Whisper), arXiv 2603.04710](https://arxiv.org/html/2603.04710): on a noisy Bengali set Whisper large-v3 WER rose from 65.83 % to 77.35 % and CER from 24.13 % to 34.74 % after SAM-Audio preprocessing; the degradation held across all Whisper sizes. (Denoising use-case, not talker separation.)

**NOTSOFAR-1 observation** — the NAIST team found residual reverberation in baseline CSS output; adding WPE dereverberation cut tcpWER 30.3 → 28.9 % on dev-set-2 [P] — [Summary of the NOTSOFAR-1 Challenge, arXiv 2501.17304 §4.5](https://arxiv.org/html/2501.17304v1). The STCON team reported that a standalone-trained neural TSE "didn't yield better ASR accuracy than GSS+beamforming … despite the good quality of target speaker's speech and high SNR" [P] — same source.

### Inferences
- Ranking of remedies by published evidence for *single-channel talker separation*: (1) adapt the ASR to separator output (LibriCSS 6.4→5.6; SMS-WSJ 7.60→6.26); (2) make the separator ASR-aware (WHAMR! 43.5→37.1; Wu 2021 from the earlier survey); (3) ensure the ASR front-end is an SSL/robust encoder (2.4 vs 28.2); (4) blend/switch with the mixture for high-SIR or noisy regions. Items 1 and 3 are directly available with a fully fine-tunable Qwen3-ASR.
- Observation adding has clean evidence only for noise reduction and target-speaker extraction. For 2-stream separation it has an obvious cost: adding the mixture back re-injects the competing talker into each stream. No paper quantifying OA for blind 2-talker separation → strong ASR was found.
- SI-SDR is not a safe model-selection metric for the separator; select on downstream WER.

### Gaps
- No published measurement of artifact sensitivity for LLM-based ASR (AuT/Qwen3-ASR, Whisper-LLM hybrids) versus conventional AED/CTC models.
- Iwamoto WER values: figure-only (Not Reported as numbers).
- Sato 2022 follow-up numbers not read (PDF text extraction yielded no tables).

---

## 3. Continuous Speech Separation (CSS) for long recordings; single-channel LibriCSS

### Takeaway
CSS = sliding fixed-length windows (2–3 s typical) through a PIT-trained separator with a fixed number of output streams (2 for LibriCSS, 3 in NOTSOFAR), stitched by matching overlapped frames between adjacent windows, then VAD + per-stream ASR. With TF-GridNet as the separator, single-channel LibriCSS reaches 5.6–6.8 % ORC-WER (6.8 % with off-the-shelf Whisper large-v2), and the residual is dominated by segmentation as much as by separation.

### Cited Findings

**Mechanics**
- NOTSOFAR baseline: separation network trained with PIT, applied "block-wise … to overlapping fixed-length segments"; N = 3 speech streams + 1 noise mask; adjacent segments are aligned by trying all permutations and choosing the lowest MSE between masked magnitude spectrograms on the shared frames; then Whisper large-v2 "applied independently to each audio stream produced by CSS", then per-stream diarization [P] — [NOTSOFAR-1 Challenge paper, arXiv 2401.08887](https://arxiv.org/html/2401.08887).
- Traditional CSS window lengths: 2.4 s and 3 s; K typically 3 outputs [P] — [DCF-DS, arXiv 2411.06667 §II](https://arxiv.org/html/2411.06667v3).
- TF-GridNet CSS on LibriCSS uses two overlap-free output streams, "assuming that overlap is limited to two speakers at any time"; energy-based VAD with smoothing that deliberately over-estimates activity [P] — [arXiv 2309.08454 §4.4](https://arxiv.org/pdf/2309.08454).

**Single-channel LibriCSS, per overlap condition** — [P] [arXiv 2309.08454 Table 3](https://arxiv.org/pdf/2309.08454) (metrics differ by row: asclite WER for t-SOT, cpWER w/ oracle speaker assignment for TS-SEP, ORC-WER otherwise):

| System | 0S | 0L | OV10 | OV20 | OV30 | OV40 | Avg |
|---|---|---|---|---|---|---|---|
| t-SOT (LS960) | 5.3 | 5.4 | 6.5 | 7.3 | 9.5 | 11.3 | 7.6 |
| TS-SEP (WavLM, Mix94k+LS960) | 3.6 | 3.0 | 4.0 | 5.9 | 8.0 | 8.5 | 5.8 |
| TF-GridNet CSS + mixture encoder, 4-gram | 5.5 | 4.3 | 5.0 | 6.5 | 8.5 | 10.0 | 6.9 |
| same, Transformer LM | 4.5 | 3.5 | 3.6 | 5.2 | 7.1 | 8.4 | 5.6 |
| same + oracle segmentation, Transformer LM | 2.9 | 2.7 | 2.7 | 3.5 | 5.4 | 6.6 | 4.1 |
| Oracle signals + oracle segmentation, Transformer LM | 2.2 | 2.4 | 1.9 | 2.0 | 2.1 | 2.0 | 2.1 |

- So with the same separator, replacing estimated by oracle segmentation moves the average from 5.6 to 4.1; the oracle-signal floor is 2.1 [P] — same source.
- Separator quality dominates: BLSTM separator 19.4 vs TF-GridNet 6.4 average ORC-WER with the identical un-fine-tuned AM and Transformer LM [P] — [arXiv 2309.08454 Table 2](https://arxiv.org/pdf/2309.08454).
- Separated primary channel vs oracle clean audio, 1-best WER: 4.4 % vs 2.1 % (Transformer LM), 5.9 % vs 3.1 % (4-gram) [P] — [arXiv 2309.08454 Table 4](https://arxiv.org/pdf/2309.08454).

**"CSS with Whisper" (von Neumann et al. 2023, CSS-AD pipeline: TF-GridNet CSS → VAD → Whisper large-v2 → diarization)** — [P] [arXiv 2309.16482 Tables 1–3](https://arxiv.org/html/2309.16482):
- ORC-WER on LibriCSS: no separation (VAD only) + Whisper 26.5; TF-GridNet + Whisper 6.8; TF-GridNet + ESPnet WavLM-Conformer 6.4; clean signals + Whisper 3.5. Literature rows in the same table: MT-RNN-T 23.6, t-SOT TT 7.6 (asclite), TS-SEP + ESPnet 5.8 (cpWER w/ oracle assignment).
- "We use the off-the-shelf ASR systems without fine-tuning them to the artifacts produced by the separation."
- Whisper large-v2 is weaker than in-domain models on LibriSpeech (5.2 % on dev-other vs 3.7 % for the ESPnet WavLM-Conformer), which explains part of the Whisper deficit.
- Speaker-attributed (cpWER): VAD segments + k-means 14.8; uniform 4 s sub-segments 9.8; sentence + word-level sub-segmentation 7.2 (Whisper) and 6.2 with a second ESPnet pass; DER 9.4 %.

**Single-channel LibriCSS cpWER literature table** — [P] [DCF-DS, arXiv 2411.06667 Table V](https://arxiv.org/html/2411.06667v3): clustering diarization → ASR without separation: AHC 36.7, VBx 33.4, SC 31.0; Transcribe-to-Diarize (E2E) 11.6; ADEnet → E2E 18.8; TS-SEP → E2E 11.61; TS-SEP → E2E-SSL 7.81; CSS-AD → Whisper 7.2; CSS-AD → E2E-SSL 6.2; DCF-DS → E2E 6.30; DCF-DS → E2E-SSL 4.43 (model not adapted to LibriCSS).

**Context-window evidence** — DCF-DS on NOTSOFAR-1: lengthening the window from 3 s to 12.8 s reduced tcpWER 37.31 → 34.93 % [P] — [arXiv 2411.06667 Table II discussion](https://arxiv.org/html/2411.06667v3). The challenge organizers hypothesize that 3-s CSS windows vs 5–120 s windows in Dia-Sep-ASR systems partly explain CSS's deficit [P] — [arXiv 2501.17304 §6.2](https://arxiv.org/html/2501.17304v1).

### Inferences
- On read speech with ≤2 concurrent talkers, a single-channel CSS cascade is already near the E2E state of the art; the easy remaining gains are in segmentation/VAD and ASR adaptation rather than the separator.
- A fixed 2-output CSS covers "2, sometimes 3" talkers only if three never overlap at once; otherwise 3 outputs (NOTSOFAR-style) are needed, and published LibriCSS evidence says nothing about that regime.
- Because the user needs no diarization, the ORC-WER rows (speaker-agnostic) are the relevant ones; the cpWER rows carry a diarization penalty they would not pay.

### Gaps
- USS ("universal"/unsupervised CSS variants) single-channel LibriCSS numbers: not located (Not Reported).
- Per-condition (0S…OV40) breakdown for TF-GridNet → Whisper specifically: Not Reported in arXiv 2309.16482 (average only).
- Wu et al. 2021 continuous-evaluation table (arXiv 2107.01922 Table 4) already covered by the earlier survey; not re-extracted.

---

## 4. Real-meeting evidence (NOTSOFAR-1 single-channel, AMI, AliMeeting, PixIT/ToTaToNet)

### Takeaway
On real far-field meetings the cascade works but far less well than on simulated data: most CHiME-8 NOTSOFAR-1 single-channel entries were CSS → Whisper → diarization and landed at 30–43 % tcpWER (speaker-agnostic tcorcWER 25.7–40.0 %); the winner (22.2 % tcpWER / 17.7 % tcorcWER) instead used diarization-guided neural separation with long windows plus a heavily fine-tuned "Enhanced Whisper". On AMI-SDM/AliMeeting, PixIT separation → off-the-shelf Whisper improves over diarization-only by 14–26 % relative, with absolute error still 32–48 %.

### Cited Findings

**CHiME-8 NOTSOFAR-1 single-channel track, eval set** — [P] [arXiv 2501.17304 Table 2](https://arxiv.org/html/2501.17304v1):

| Rank | Team | tcpWER | tcorcWER | Pipeline notes (from the papers cited below) |
|---|---|---|---|---|
| 1 | USTC-NERCSLIP | 22.2 | 17.7 | Dia → neural JDS/DCF-DS mask separation → Enhanced Whisper |
| 2 | NPU-TEA | 30.0 | 25.7 | CSS-based |
| 3 | NJU-AALab | 33.5 | 30.4 | TF-GridNet CSS → Whisper large-v2 → NME-SC |
| 4 | NAIST | 36.6 | 33.3 | baseline Conformer CSS + WPE |
| 5 | BUT/JHU | 40.1 | 27.2 | no separation; diarization-conditioned Whisper (target-speaker ASR) |
| 6 | ToTaTo | 41.2 | 29.2 | PixIT CSS → fine-tuned Whisper large-v3 |
| 7 | Baseline | 41.4 | 35.5 | Conformer CSS → Whisper large-v2 → NeMo diarization |
| 8 | Fano Labs | 43.1 | 40.0 | – |
| 9 | UWB | 45.8 | 39.1 | – |
| 10 | Blue Sky Wave Riders | 74.1 | 37.0 | – |

- "Without GSS, most single-channel submitted systems adopted CSS-ASR-Dia pipeline"; CSS architectures explored: Conformer, TF-GridNet, PixIT [P] — [arXiv 2501.17304 §5](https://arxiv.org/html/2501.17304v1).
- BUT/JHU's separation-free target-speaker Whisper had tcorcWER that was "surpassed by only one CSS-based system in the single-channel track (NPU-TEA)" [P] — same source.
- Multi-channel winner 10.8 % tcpWER vs single-channel winner 22.2 % (51 % relative gap); "debate-style overlapping speech is particularly challenging for SC systems" [P] — [arXiv 2501.17304 §4.5, §7](https://arxiv.org/html/2501.17304v1).
- All submitted CSS models were trained only on simulated data; "fine-tuning of CSS with real meetings data [was left] unexplored" [P] — [arXiv 2501.17304 §4.5](https://arxiv.org/html/2501.17304v1).
- "In the single-channel track the top three systems all utilizing fine-tuned versions of Whisper"; the organizers "hypothesize ASR improvements offered by USTC played a key role" [P] — [arXiv 2501.17304 §8.1, §4](https://arxiv.org/html/2501.17304v1).

**Winner's ablations (DCF-DS)** — [P] [arXiv 2411.06667](https://arxiv.org/html/2411.06667v3):
- Oracle diarization segmentation of the *unseparated* mixture → ASR: 39.73 % tcpWER; NOTSOFAR baseline (CSS) 41.4 %; DCF-DS with spectral-clustering prior 38.02 %; with 12.8-s window 34.93 %; + more simulated data 34.51 %; + MIMO speech-enhancement post-module 33.81 %; + re-clustering 31.72 % — all with baseline Whisper large-v3. Swapping in Enhanced Whisper: 22.2 %.
- So front-end work bought 41.4 → 31.7 (−9.7 abs) and back-end ASR fine-tuning/architecture bought 31.7 → 22.2 (−9.5 abs).
- USTC dev-set-2 single-channel: 22.989 % tcpWER [P] — [USTC-NERCSLIP system report, arXiv 2409.02041](https://arxiv.org/html/2409.02041v2).
- USTC separation ablation (multi-channel, train-set-1 plaza_0, original Whisper large-v3): CSS(3 s)+MVDR 26.68 → +overlap detection 25.14 → JDS(3 s)+MVDR 20.62 → +diarization mask 20.29 → +CSD RTTM 19.95 [P] — [arXiv 2409.02041 Table 6](https://arxiv.org/html/2409.02041v2).

**NJU-AALab (TF-GridNet CSS → Whisper large-v2, "without any fine-tuning")** — [P] [hu24_chime, ISCA](https://www.isca-archive.org/chime_2024/hu24_chime.pdf): 33.5 % tcpWER eval, 36.4 % dev. ASR-side ablation on dev plaza_0 (tcorcWER): raw 37.6 → +VAD on CSS output 36.1 → +merging short segments 33.0 → +filler-word prompt 31.0 → +LM rescoring 31.1.

**ToTaTo (PixIT)** — [P] [kalda24_chime, ISCA](https://www.isca-archive.org/chime_2024/kalda24_chime.pdf): PixIT used as CSS (6 output sources, WavLM-large features, trained on AMI-SDM + NOTSOFAR train + voice-converted data) → fine-tuned Whisper large-v3: 41.2 % tcpWER eval; PixIT used for diarization only: 43.9 %.

**PixIT / ToTaToNet on AMI-SDM and AliMeeting** — [P] [PixIT, arXiv 2403.02288 Tables 2–5](https://arxiv.org/html/2403.02288v2). ASR applied "off-the-shelf with no fine-tuning", independently per separated source, via WhisperX.

AMI-SDM cpWER (%), speaker attribution via pyannote 3.1 diarization → via PixIT separation:
- Whisper small.en 39.6 → 33.9 (−14.4 %); medium.en 38.8 → 32.8 (−15.4 %); large-v2 38.3 → 31.7 (−17.2 %); NeMo Conformer-CTC large 48.9 → 39.4 (−19.4 %).

AMI-SDM speaker-agnostic uWER (%), original audio → separated sources:
- Whisper small.en 37.6 → 36.3; medium.en 37.1 → 35.1; large-v2 35.4 → 33.8 (−4.5 %); NeMo Conformer 49.3 → 41.6 (−15.6 %). Gains come from deletions (e.g. large-v2 del 28.9 → 26.9) while substitutions rise slightly (5.2 → 5.5).

AliMeeting channel 1 (Mandarin) cpCER (%), diarization → PixIT separation:
- Whisper small 68.6 → 54.0 (−21.3 %); medium 65.9 → 50.3 (−23.7 %); large-v2 65.1 → 48.3 (−25.8 %).

- ToTaToNet model card: 5-s mono 16 kHz chunks, up to 3 speakers, trained on AMI-SDM [S] — [pyannote/separation-ami-1.0](https://huggingface.co/pyannote/separation-ami-1.0).
- PixIT inference: 5-s windows, 500-ms step, Kmax = 3; "less than 1% chance that a 5-second window contains more than three active speakers" on AMI/AliMeeting [P] — [arXiv 2403.02288](https://arxiv.org/html/2403.02288v2).

### Inferences
- For the user's speaker-agnostic objective, the most relevant real-data number is how little separation helps Whisper when attribution is not scored: AMI-SDM uWER 35.4 → 33.8 (large-v2). Most of the cpWER gain of separation on real meetings is attribution, not recognition.
- On real meetings a diarization-conditioned ASR without any separator (BUT/JHU, 27.2 % tcorcWER) beat every CSS cascade but one on speaker-agnostic error — consistent with the earlier survey's E2E findings.
- The winner's two halves were equally important; a team that can fully fine-tune the ASR should expect the back-end adaptation to be worth as much as the separator.
- Simulation-to-real mismatch of the separator is the recurring explanation for the LibriCSS (≈6 %) vs NOTSOFAR (≥25 % tcorcWER) contrast; nobody in the challenge tested real-data CSS fine-tuning (PixIT/MixIT-style training is the published route).

### Gaps
- NPU-TEA and Fano Labs system details not read (Not Reported here).
- No AMI/AliMeeting cascade result with an LLM-based ASR found.
- The NOTSOFAR summary's Table 6 has track labels that appear swapped in the HTML rendering (shows single-channel 10.8 / multi-channel 22.2); Table 2 and the prose make clear that 22.2/17.7 is single-channel and 10.8 is multi-channel.

---

## 5. Hybrid designs between cascade and end-to-end

### Takeaway
The best-documented hybrids keep a separator but give the recognizer access to the mixture or to separator *features* rather than only waveforms: mixture-encoder acoustic models, encoder-level separators (Sidecar, EncSep/GEncSep), and — newest — a separator producing talker-specific streams that an LLM decoder reads via cross-attention. Evidence for overlap-detector routing exists only as one multi-channel ablation; published work on separation-derived pseudo-labels or separator-as-teacher distillation for multi-talker ASR was searched for and not found.

### Cited Findings

**Separated stream + mixture into one acoustic model ("mixture encoder")**
- SMS-WSJ, BLSTM separator: modular baseline 20.4 / 14.6 % WER (dev93/eval92) → best mixture-aware model 19.1 / 13.6 %; merely enlarging the modular baseline hurt (20.8 / 14.9) [P] — [Mixture Encoder for Joint Speech Separation and Recognition, arXiv 2306.12173, Table 1](https://arxiv.org/html/2306.12173). The paper notes TF-GridNet alone gives 7.9 % on eval92 in the literature table — separator quality outweighs the hybrid trick.
- LibriCSS: mixture encoder helps a weak (BLSTM) separator (17.9 → 17.6) but gives no gain on TF-GridNet (5.8 vs 5.8; 5.6 vs 5.6 with sMBR): "TF-GridNet largely closes the gap between previous methods and oracle separation independent of mixture encoding" [P]/[S for the quoted sentence] — [arXiv 2309.08454 Table 2](https://arxiv.org/pdf/2309.08454).

**Separator inside the ASR encoder**
- Whisper-Sidecar (frozen Whisper + Conv-TasNet-style Sidecar + soft prompt): LibriMix 2spk 4.66 / 3spk 16.79; LibriSpeechMix 2spk 3.43 / 3spk 6.80 (large) [P] — [arXiv 2407.09817 Table 2](https://arxiv.org/html/2407.09817v2).
- GEncSep (encoder-output separation + separated embeddings guiding an SOT decoder): noisy Libri2Mix test 15.0 vs SOT 17.1; noisy Libri3Mix test 25.9 vs 28.2; clean 6.6 vs 7.0 and 13.1 vs 14.7 [P via Table VII of arXiv 2603.27205](https://arxiv.org/html/2603.27205); method description [P] — [arXiv 2409.00815](https://arxiv.org/html/2409.00815).

**Separator streams as a second branch for an LLM decoder (2026)** — [P] [Beyond Acoustic Prefixes, arXiv 2603.27205](https://arxiv.org/html/2603.27205): WavLM-Large → separator → talker-specific streams (serialized CTC) exposed to a LLaMA decoder through gated cross-attention + LoRA refinement.
- Simply enriching the static prefix helped little: "HybridPrompt reduced noisy Libri3Mix WER from 38.9% to 37.4%" (Base) and 32.3 → 31.3 % (Instruct).
- Persistent gated access to the separated acoustic memory: noisy Libri3Mix 38.9 → 20.5 % and clean 20.4 → 9.2 % (1B Base); Instruct 32.3 → 17.4 % noisy; 3B-Instruct stage-2: 27.9 → 17.1 % noisy, 15.2 → 8.7 % clean; 8B: 43.8 → 24.3 % / 34.9 → 21.7 % noisy.
- Two-talker gains are smaller: e.g. offset noisy Libri2Mix test 11.0 (SOT-LLaMA-1B) → 7.5–7.7; clean 4.6 → 3.1–3.9 (Table VII).
- CTC-only inference from the separator heads (no LLM) gives 14.6 / 23.6 % noisy and 6.1 / 12.5 % clean (L2M/L3M test), i.e. on 3 talkers the LLM-free separator branch beats plain LLM-SOT (Table VII).
- A companion encoder-only variant distils an LLM teacher into the separator/CTC model, removing the LLM at inference; described as comparable to LLM systems on 2 talkers and better on 3 [S] — [arXiv 2603.10587 summary](https://papers.cool/arxiv/2603.10587). Numbers Not Reported here.

**Unified encoders** — UME (shared OWSM-v3.1 encoder for diarization + separation + multi-speaker ASR): 19.6 % noisy / 6.4 % clean Libri2Mix test, 27.1 / 15.9 % Libri3Mix (original no-offset mixtures) [P via arXiv 2603.27205 Table VII](https://arxiv.org/html/2603.27205); architecture [P] — [arXiv 2508.20474](https://arxiv.org/html/2508.20474).

**Overlap-detector routing** — only measured instance found: adding a Conformer overlap-segment detector in front of CSS (so separation is applied to detected-overlap regions) improved tcpWER 26.68 → 25.14 % (multi-channel, plaza_0, Whisper large-v3) [P] — [arXiv 2409.02041 Table 6](https://arxiv.org/html/2409.02041v2). The signal-level analogue is Sato et al.'s SIR/SNR-based switch between mixture and enhanced input (22 % average / up to 27 % relative CER reduction where it applies) [P] — [arXiv 2106.00949](https://arxiv.org/pdf/2106.00949.pdf).

**Joint diarization + separation instead of blind CSS** — JDS/DCF-DS vs CSS with the same ASR: 25.14 → 20.62 % (multi-channel ablation) [P] — [arXiv 2409.02041 Table 6](https://arxiv.org/html/2409.02041v2); single-channel eval 41.4 (CSS baseline) → 31.7 % [P] — [arXiv 2411.06667](https://arxiv.org/html/2411.06667v3). TS-SEP → E2E-SSL on 1-ch LibriCSS 7.81 % cpWER [P] — [arXiv 2411.06667 Table V](https://arxiv.org/html/2411.06667v3).

**Separator trained with ASR-encoder supervision (ASR as teacher of the separator)** — WHAMR! 43.5 → 37.1 % cpWER, transferring to Whisper (29.3 → 26.8) [P] — [arXiv 2406.08914](https://ar5iv.labs.arxiv.org/html/2406.08914).

### Inferences
- The pattern across hybrids: giving the recognizer mixture context only helps when the separator is weak; with a strong separator the gain vanishes (LibriCSS), whereas giving an *LLM decoder* persistent access to separated streams helps most exactly where SOT fails (3 talkers).
- For Qwen3-ASR this suggests two plausible hybrid shapes with precedent: (a) waveform cascade + fine-tune the whole ASR on separator output; (b) keep AuT on the mixture and add talker-specific streams as extra decoder-visible memory. Only (a) has direct WER evidence with an off-the-shelf recognizer.
- Routing by overlap detection has thin evidence (one MC ablation, −1.5 abs); the much larger NOTSOFAR gains came from replacing blind CSS with diarization-guided separation.

### Gaps
- Searched and found nothing published on: using separator + single-talker ASR outputs as pseudo-labels to train SOT/multi-talker ASR on unlabeled overlapped speech; distilling a separator into an ASR encoder; iterative ASR-guided re-separation with measured WER on Libri2Mix/LibriCSS. (Related but different: pseudo-labels for *separator* training on real far-field data — PuLSS [S] — [arXiv 2605.19695](https://arxiv.org/html/2605.19695).)
- No paper feeds both separated waveforms and the mixture as dual audio inputs to a Speech-LLM.
- No single-channel measurement of overlap-routing benefit.

---

## 6. Permutation, continuity, leakage/duplicates, hallucination on empty streams, single-speaker input

### Takeaway
These are known, named failure modes with engineering fixes rather than solved problems: cross-chunk permutation is handled by overlap-region matching or by diarization-anchored outputs; leakage is handled by energy/SPP masking, diarization-gated zeroing, and transcript de-duplication; hallucination on near-silent streams is handled by running VAD on the separator output before Whisper. Quantified effects are large — un-gated leakage turned a −15 % improvement into a +29 % degradation in PixIT.

### Cited Findings

**Stream permutation across chunks**
- Standard stitching: pick the inter-chunk permutation minimizing MSE between masked spectrograms on the overlapped frames [P] — [arXiv 2401.08887](https://arxiv.org/html/2401.08887); same in NJU-AALab ("align adjacent segments in the stream dimension by minimizing the mean square error (MSE) in the overlapping regions") [P] — [hu24_chime](https://www.isca-archive.org/chime_2024/hu24_chime.pdf).
- CSS streams "can combine multiple speakers – CSS aims to 'unmix' overlapping speech rather than isolate individual speakers", which is why diarization follows ASR in CSS-ASR-Dia [P] — [arXiv 2501.17304 §3](https://arxiv.org/html/2501.17304v1). For a speaker-agnostic transcript this property is harmless.
- cpWER penalizes stream switches, ORC-WER does not [P] — [arXiv 2406.08914 §3](https://ar5iv.labs.arxiv.org/html/2406.08914). On WHAMR! the two were nearly identical (29.5 vs 29.4 with Whisper), i.e. utterance-level separation had few speaker swaps there.
- PixIT avoids blind stitching: separated sources are aligned with diarization branch outputs and stitched via speaker-embedding clustering, so each file-level source corresponds to one speaker [P] — [arXiv 2403.02288](https://arxiv.org/html/2403.02288v2); [kalda24_chime §4](https://www.isca-archive.org/chime_2024/kalda24_chime.pdf).
- DCF-DS/TS-SEP style: output channels are tied to diarization-identified speakers, removing the permutation problem and the fixed-N limit ("although the number of output nodes in DCF-DS is limited to 3 or 4, the method still performs well" with 4–8 speakers) [P] — [arXiv 2411.06667](https://arxiv.org/html/2411.06667v3).

**Leakage / duplicate words**
- Taxonomy: "channel leakage … can be categorized into under-separation and over-separation, occurring in overlapping and single-speaker regions, respectively"; over-separation (one talker smeared into several streams) is the common case and under-separation "is challenging to eliminate using post-processing" [P] — [hu24_chime](https://www.isca-archive.org/chime_2024/hu24_chime.pdf).
- NJU fix: two-step energy-based speech-presence-probability binary masking of the streams (relative energy across streams, then absolute energy), plus deleting recognized words attributable to leakage across streams [P] — same source.
- PixIT quantification (AMI-SDM, Whisper medium.en, cpWER; baseline pyannote diarization 38.8): separation without WavLM and without leakage removal 50.1 (ins 15.6, sub 19.2) = +29.1 %; + leakage removal (zeroing sources where the diarization branch says inactive) 36.2; with WavLM, no leakage removal 34.1 (ins 3.8); with both 32.8 (ins 1.2). "A lot of [leakage] pass[es] through the VAD component of WhisperX" [P] — [arXiv 2403.02288 Table 5](https://arxiv.org/html/2403.02288v2).
- With a strong separator leakage is small in transcripts: on LibriCSS with TF-GridNet, the frame-level coincidence rate between the primary-channel hypothesis and the cross-channel reference is 0.3 % in two-speaker regions and ≤0.1 % on words-only average [P] — [arXiv 2309.08454 Table 5](https://arxiv.org/pdf/2309.08454).

**Hallucination on silent / residual streams**
- "Whisper suffers from hallucination, meaning it will create transcripts even during silence frames. Therefore, it is necessary to perform voice activity detection (VAD) on the CSS output first" — VAD on CSS output: tcorcWER 37.6 → 36.1; merging short segments into ≤26-s inputs: → 33.0 [P] — [hu24_chime Table 1](https://www.isca-archive.org/chime_2024/hu24_chime.pdf).
- Scale of the problem on non-speech audio: of 301,317 Whisper inferences on non-speech clips, 121,378 (40.3 %) produced hallucinated text, 9.1 % of those with looping; over half came from the top-10 phrases [P] — [Investigation of Whisper ASR Hallucinations Induced by Non-Speech Audio, arXiv 2501.11378](https://arxiv.org/html/2501.11378v1). Decoder parameter tweaks "offer a limited solution"; beam size 1 had the lowest hallucination rate [S] — same paper via search summary.
- Activation steering reduces Whisper non-speech hallucination rate without fine-tuning (e.g. large-v3 to 19.88 % on UrbanSound8K, 27.05 % on WHAM!) [P] — [arXiv 2606.07473](https://arxiv.org/pdf/2606.07473v1).

**Single-speaker input / empty stream**
- LibriCSS 0S/0L (no overlap) conditions measure exactly this: TF-GridNet CSS + ASR gives 4.5 / 3.5 (Transformer LM) vs oracle signals 2.2 / 2.4; with oracle segmentation 2.9 / 2.7 [P] — [arXiv 2309.08454 Table 3](https://arxiv.org/pdf/2309.08454). So even with a strong separator, non-overlapped audio pays roughly a 1–2 point absolute tax, mostly from segmentation (0S).
- VAD thresholds are intentionally biased to over-estimate activity to avoid losing speech [P] — [arXiv 2309.08454 §4.4](https://arxiv.org/pdf/2309.08454); [arXiv 2309.16482 §4.5](https://arxiv.org/html/2309.16482).
- Sato et al.: at SIR 20 dB the mixture is recognized better than the extracted signal (5.3 vs 6.3 % CER at SNR 20) [P] — [arXiv 2106.00949 Table 2](https://arxiv.org/pdf/2106.00949.pdf).

### Inferences
- A minimal robust recipe consistent with the published systems: chunked separation with overlap stitching → per-stream energy/SPP gating (or a diarization/activity head on the separator) → VAD → merge into long segments → ASR → cross-stream duplicate-word removal. Each step has a measured benefit in at least one system above.
- The speaker-agnostic use-case removes the need for cross-chunk speaker consistency, but not for *within-utterance* continuity: a stream swap mid-sentence splits one utterance across two ASR passes. No paper quantifies this separately from cpWER/ORC-WER differences.
- LLM-decoder ASR is plausibly at least as prone as Whisper to generating fluent text on residual-noise streams, but this is an inference — no measurement on Speech-LLMs with separated-stream inputs was found.

### Gaps
- No study measures hallucination rate specifically on the "empty" output channel of a separator for Whisper or any LLM-based ASR.
- No published duplicate-word rate across streams for real meetings (only PixIT insertion rates and the LibriCSS coincidence analysis).
- No published stream-count/"how many talkers are active" gating accuracy tied to WER for single-channel CSS.

---

## 7. Mandarin / multilingual / cross-lingual cascade results

### Takeaway
Cross-lingual cascade evidence is thin: the one solid datapoint is PixIT separation → multilingual Whisper on Mandarin AliMeeting (cpCER 65.1 → 48.3 with large-v2, a larger relative gain than on English AMI). Nothing was found for Taiwanese Hokkien, code-switching, or any separator → Mandarin-centric ASR (Paraformer, FireRedASR, Qwen3-ASR) cascade.

### Cited Findings
- AliMeeting channel 1 (real Mandarin meetings, single far-field channel), multilingual Whisper off-the-shelf, cpCER: small 68.6 → 54.0; medium 65.9 → 50.3; large-v2 65.1 → 48.3 (diarization-based attribution → PixIT separated sources). Substitutions fall sharply (large-v2 17.6 → 10.6) and insertions 9.5 → 4.0, deletions stay ~34 [P] — [arXiv 2403.02288 Table 4](https://arxiv.org/html/2403.02288v2).
- The PixIT authors note the relative improvement on Mandarin exceeds that on AMI "even though WavLM has been pre-trained on English data", attributing it to AliMeeting's higher overlap [P] — same source. The separator itself was trained on the target-domain real recordings (PixIT needs only diarization labels, not clean sources).
- Aishell1Mix (Mandarin, LibriMix protocol, 2-speaker clean), Whisper-Sidecar-TTI *trained on English*: zero-shot CER 55.87 (small) / 36.28 (medium) / 28.94 (large); after "one-batch tuning" 28.95 / 19.83 / 17.81 [P] — [arXiv 2407.09817 Table 4](https://arxiv.org/html/2407.09817v2). Integrated, not cascade; shows an encoder-level separator transfers across languages only partially without adaptation.
- Japanese (CSJ) target-speaker extraction → ASR CER by SIR/SNR: see §2 table [P] — [arXiv 2106.00949](https://arxiv.org/pdf/2106.00949.pdf).
- Low-resource caution: audio "cleaning" before zero-shot Whisper worsened Bengali WER 65.83 → 77.35 % [P] — [arXiv 2603.04710](https://arxiv.org/html/2603.04710).

### Inferences
- A waveform separator is language-agnostic in principle, and the AliMeeting result shows a separator trained on in-domain Mandarin audio helps an unadapted multilingual ASR; whether an *English-trained* separator transfers to Mandarin/Hokkien speech with comparable WER benefit is not established by any source found.
- Tonal-language sensitivity to separation artifacts (pitch-contour distortion) is a reasonable concern but unmeasured in the literature located.

### Gaps
- No Taiwanese Hokkien, no Mandarin–English code-switch overlapped-speech cascade results. Searched; not found.
- No AISHELL-based separator → Whisper/Paraformer/Qwen cascade CER table found (M2MeT system descriptions were not reachable/located).
- No study of separator language-mismatch (train English, test Mandarin) with WER/CER.
