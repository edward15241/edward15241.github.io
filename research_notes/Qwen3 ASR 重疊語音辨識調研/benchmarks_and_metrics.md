# Benchmarks and Evaluation Metrics for Overlapped (Simultaneous) Speech Recognition in Single-Channel Audio

> **Access note (read first).** In the first pass only github.com was reachable. In a second verification pass (2026-11-10) arxiv.org and www.isca-archive.org became reachable and the following primary sources were downloaded and read as full text: MS-Bench (arXiv 2610.25824v3), Xiaomi-CocktailASR-1 (arXiv 2609.11274v1), MeetEval paper (arXiv 2307.11394), SOT (Kanda et al., Interspeech 2020), M2MeT/AliMeeting (arXiv 2110.07393), t-SOT (arXiv 2202.00842), H-SAGE (arXiv 2607.01566), Empowering Whisper (arXiv 2407.09817v1). The MS-Bench project page (aslp-lab.github.io/MS-Bench/) returns only a 743-byte JavaScript stub ("Opening the MS-Bench demo page…") with no readable content. Hosts still blocked: aclanthology.org, semanticscholar, openreview, alphaxiv, audioshake.ai etc. Each finding is tagged:
> - **[D]** = read directly from the primary source (GitHub README or paper full text).
> - **[S]** = taken from a search-engine summary/snippet of the cited URL; NOT verified against the full text. Re-check before publication.
> - **[D, corrected]** = the first-pass [S] claim was wrong or misleading and has been corrected from the primary source.
>
> **Still unverified after the second pass:** everything still tagged [S] — notably the CSPB and SDBench overlap figures, AISHELL-4 statistics, NOTSOFAR-1, WSJ0-2mix, LibriMix licence/hours, all Parakeet and MOSS-Transcribe-Diarize numbers, the MLC-SLM Qwen3-ASR adaptation numbers, the AudioShake and parakeet-rs figures, and the DI-cpWER / tcORC-WER / SAgWER descriptions.
>
> Items in "Inferences" marked *(prior knowledge)* come from the researcher's background knowledge and were NOT verified against a source in this session.

---

## Q1. MS-Bench (arXiv 2610.25824): composition, stratification, metrics, models, findings

### Takeaway
MS-Bench is a 32.72 h / 99-recording, real-recording, long-form multi-speaker benchmark (mainly Chinese + English) that stratifies by five condition axes and scores with DER, cpWER, tcpWER and speaker-agnostic WER; it is a *diarized conversational* benchmark, not a fully-overlapped-speech benchmark, and it reports overlap effects as paired deltas rather than a WER-by-overlap-bin table. Qwen3-ASR appears only inside a `pyannote+Qwen3-ASR` cascade — there is no standalone Qwen3-ASR row.

### Cited Findings
- **Second pass: the arXiv HTML full text (v3, 26 Oct 2026) was read directly and confirms every figure below taken from the GitHub README.** The project page aslp-lab.github.io/MS-Bench/ loads only a JavaScript stub and yielded no content. — [arXiv HTML 2610.25824](https://arxiv.org/html/2610.25824)
- [D, corrected] Paper identity: "MS-Bench: A Condition-Stratified Multi-Speaker ASR Benchmark", arXiv 2610.25824v3; 18 authors (Chunjiang He*, Zheng Zi*, Bingshen Mu, …, Lei Xie; NPU ASLP lab with Nanjing University, AISHELL, Nexdata, Magic Data and others) — the first-pass "five co-authors" was wrong. The paper frames MSASR as "who spoke what and when", i.e. speaker-attributed transcription. — [arXiv HTML 2610.25824](https://arxiv.org/html/2610.25824)
- [D] Sources: public benchmarks named are AISHELL-4, AliMeeting and AISHELL-5 ("such as"), plus internal benchmarks; no per-source hours table is given. Selection maximises a utility U = 0.40·C (condition coverage) + 0.25·M (co-occurrence of challenging conditions) + 0.20·D (diversity) + 0.15·R (under-represented sources). All recordings in the ≥9-speaker group come from one dataset. — [arXiv HTML 2610.25824](https://arxiv.org/html/2610.25824)
- [D] **Exact definition of the overlap analysis:** for each reference utterance u, r_ov(u) is the fraction of its duration covered by other speakers' speech; the comparison is only between utterances with r_ov = 0 and r_ov ≥ 0.5, restricted to inference units containing both conditions (within-unit pairing). Δx is the error-rate difference in percentage points between the two conditions, with 95% CIs from a recording-level cluster bootstrap. — [arXiv HTML 2610.25824](https://arxiv.org/html/2610.25824)
- [D] **Error decomposition:** E_lex = (S + I_lex + D_lex)/N_ref; E_seg = (I_seg + D_seg)/N_ref (content introduced or lost through falsely detected, missed or discarded speech regions); E_lex + E_seg = WER. E_attr = fraction of reference utterances attributed to an incorrect speaker. Overall, E_lex spans 19.27–38.17% and E_seg only 1.32–6.02% across systems (lexical errors dominate); SoulX-Transcriber is dominated by insertions (I_lex 15.5%). — [arXiv HTML 2610.25824](https://arxiv.org/html/2610.25824)
- [D] **Overlap-region behaviour (directly relevant to "is the second speaker dropped?"):** only 39.9–85.3% of reference overlap boundaries are localised (hypothesis boundary within ±0.5 s); 71–93% of overlapped speech is covered by at least one hypothesised speaker, but only 31–63% is covered by the correct speaker. The authors conclude overlap failures "are not simply caused by missing speech altogether". — [arXiv HTML 2610.25824](https://arxiv.org/html/2610.25824)
- [D] The paper contains **no absolute error-rate table per overlap bin O0–O4**; the bins are used for benchmark construction/coverage, and the only overlap result is the Δ_lex / Δ_attr contrast in Table 2. — [arXiv HTML 2610.25824](https://arxiv.org/html/2610.25824)
- [D] Speaker-agnostic WER is described only as "disregarding speaker labels during scoring"; the algorithm/tool (ORC-WER, concatenation, etc.) is not specified. cpWER/tcpWER cite MeetEval. Chunks are scored independently against their reference windows; long recordings are chunked at 90 min (MOSS), 60 min (VibeVoice-ASR), 5 min (SoulX). — [arXiv HTML 2610.25824](https://arxiv.org/html/2610.25824)
- [D] The cascade is pyannote speaker-diarization-community-1 followed by Qwen3-ASR; **the Qwen3-ASR model size (0.6B vs 1.7B) is not stated.** Data licence and download terms are not stated in the paper (the "License" line in the HTML is arXiv's own distribution licence for the manuscript). No CER-vs-WER or text-normalisation details are given. — [arXiv HTML 2610.25824](https://arxiv.org/html/2610.25824)
- [D] Scale: 32.72 hours, 99 recordings; recording duration 1.38–159.07 min (median 16.50 min); 2–14 speakers per recording; duration-weighted average overlap 21.87%. — [GitHub ASLP-lab/MS-Bench](https://github.com/ASLP-lab/MS-Bench)
- [D] Scenarios: meetings; spontaneous conversation (dinner party, in-vehicle, everyday dialogue); media (podcasts, film/TV, live streams); classroom. Capture devices: mobile phones, smart glasses, in-vehicle microphones, microphone arrays. — [GitHub ASLP-lab/MS-Bench](https://github.com/ASLP-lab/MS-Bench)
- [D] Languages: primarily Chinese and English, plus Portuguese, Japanese, Italian, Thai, Spanish. (Taiwanese Hokkien is not listed.) — [GitHub ASLP-lab/MS-Bench](https://github.com/ASLP-lab/MS-Bench)
- [D] Sources: public and internal benchmarks; audio integrity, transcripts, speaker annotations and timestamps manually checked. Per-source breakdown of the 32.72 h: Not Reported on the README. — [GitHub ASLP-lab/MS-Bench](https://github.com/ASLP-lab/MS-Bench)
- [D] Stratification bins (recording counts out of 99):
  - Speaker number P: P0 = 2 (27); P1 = 3–4 (29); P2 = 5–8 (33); P3 = ≥9 (10).
  - Overlap ratio O: O0 = 0 (23); O1 = (0, 0.10) (22); O2 = [0.10, 0.20) (10); O3 = [0.20, 0.40) (26); O4 = [0.40, 1.00] (18).
  - Speaker similarity S (max pairwise cosine similarity): S0 = [0.00, 0.32) (9); S1 = [0.32, 0.49) (12); S2 = [0.49, 0.65) (31); S3 = [0.65, 1.00] (47).
  - Speaker turn interval T (25th percentile): T0 ≥ 0.45 s (10); T1 = [0.15, 0.45) s (14); T2 = [0.06, 0.15) s (42); T3 = [0.00, 0.06) s (33).
  - Acoustic quality N: d_N = 1 − (p_D + p_N)/2, higher = poorer; N0 = [0.00, 0.27) (7); N1 = [0.27, 0.51) (5); N2 = [0.51, 0.74) (25); N3 = [0.74, 1.00] (62).
  — [GitHub ASLP-lab/MS-Bench](https://github.com/ASLP-lab/MS-Bench)
- [D] Metrics: DER (0.5 s collar); cpWER; tcpWER (5 s collar); WER that ignores speaker labels; condition analysis via paired error-rate differences Δlex and Δattr (percentage points) and Spearman ρ. — [GitHub ASLP-lab/MS-Bench](https://github.com/ASLP-lab/MS-Bench)
- [D] Seven systems evaluated: MOSS-Transcribe-Diarize-Pro, MOSS-Transcribe-Diarize, Qwen3.8-Omni-Flash, VibeVoice-ASR, pyannote+Qwen3-ASR, Doubao-ASR 2.0, SoulX-Transcriber (5-min segments). **No standalone Qwen3-ASR; no Whisper, Parakeet, Gemini or GPT-4o rows.** — [GitHub ASLP-lab/MS-Bench](https://github.com/ASLP-lab/MS-Bench)
- [D] Overall results (%, lower is better):

  | System | DER | cpWER | tcpWER | WER (speaker-agnostic) | Excluded units | Soft-degradation units |
  |---|---|---|---|---|---|---|
  | MOSS-Transcribe-Diarize-Pro | 10.96 | 21.95 | 22.81 | 22.49 | 15 | 2 |
  | MOSS-Transcribe-Diarize | 12.74 | 24.94 | 26.22 | 25.02 | 0 | 2 |
  | Qwen3.8-Omni-Flash | 17.97 | 29.43 | 32.59 | 27.86 | 5 | 2 |
  | VibeVoice-ASR | 20.19 | 43.41 | 44.76 | 33.11 | 10 | 12 |
  | pyannote+Qwen3-ASR | 17.35 | 43.25 | 46.54 | 34.96 | 0 | 1 |
  | Doubao-ASR 2.0 | 39.52 | 55.37 | 62.07 | 38.20 | 4 | 26 |
  | SoulX-Transcriber (5-min) | 19.42 | 49.16 | 52.60 | 39.96 | 8 | 30 |

  — [GitHub ASLP-lab/MS-Bench](https://github.com/ASLP-lab/MS-Bench)
- [D] Overlap effect, reported as Δlex / Δattr in percentage points (README has **no absolute WER-per-overlap-bin table**): MOSS-TD-Pro +27.3 / +10.6; MOSS-TD +27.3 / +10.9; Qwen3.8-Omni-Flash +28.7 / +10.5; VibeVoice-ASR +34.3 / +4.2; pyannote+Qwen3-ASR +38.6 / +19.1; Doubao-ASR 2.0 +27.2 / +6.3; SoulX-Transcriber +35.2 / +8.7. — [GitHub ASLP-lab/MS-Bench](https://github.com/ASLP-lab/MS-Bench)
- [D] Stated findings: overlap degrades both lexical recognition and speaker attribution for all seven systems; higher speaker similarity raises speaker confusion even without overlap; poor acoustic quality mainly hurts lexical recognition and is associated with shorter predicted speech timelines in six of seven systems; turn interval mainly affects attribution with no significant lexical/segmentation penalty; after controlling for dataset and overlap, more speakers make attribution harder (significant in four of six systems); similar DER can hide large cpWER differences (Qwen3.8-Omni-Flash vs. pyannote+Qwen3-ASR). — [GitHub ASLP-lab/MS-Bench](https://github.com/ASLP-lab/MS-Bench)
- [D] License: not stated on the README. — [GitHub ASLP-lab/MS-Bench](https://github.com/ASLP-lab/MS-Bench)

### Inferences
- The "+27 to +39 pp" figure is an utterance-level contrast (utterances ≥50% overlapped vs. 0% overlapped within the same inference unit), not a contrast between recording-level overlap bins; quote it that way.
- The 71–93% "covered by some speaker" statistic is a time-coverage measure, not a word-recovery measure; it suggests these diarizing systems usually emit something during overlap, but does not show both talkers' words are recovered.
- The pyannote+Qwen3-ASR cascade has the largest lexical overlap penalty (+38.6 pp) and the largest attribution penalty (+19.1 pp) of the seven systems. This is consistent with (but does not prove) a single-talker recogniser losing content in overlapped regions; the cascade confounds diarization errors with Qwen3-ASR's own behaviour.
- MS-Bench's "WER ignoring speaker labels" column is the closest fit to this team's no-attribution requirement, but which algorithm it uses (ORC-WER, single-stream concatenation, time-ordered merge) is not stated on the README.
- MS-Bench does not cover the team's hardest case: its top overlap bin is a recording-level ratio of [0.40, 1.00] on long conversational audio, which is not the same as utterance-level, fully-overlapped two-talker mixtures; and it has no Taiwanese Hokkien.

### Gaps
- Confirmed absent from the paper (v3) after a full read: per-overlap-bin absolute WERs, the speaker-agnostic WER algorithm, per-language breakdown, per-source hours, data licence, Qwen3-ASR model size. These are Not Reported by the authors, not merely unfetched.
- Project page content could not be read (JavaScript stub only); it may carry download links or licence terms.
- Whether the audio is released as mono, and download terms: Not Reported on the README or in the paper.

---

## Q2. Datasets and benchmarks: language, hours, real vs. synthetic, overlap statistics, mono availability, license

### Takeaway
Fully-overlapped evaluation exists almost only as English synthetic mixtures (WSJ0-2mix, LibriMix, LibriSpeechMix); real corpora (AMI, AliMeeting, AISHELL-4, NOTSOFAR-1, CHiME) are partial-overlap conversational data with recording-level overlap in the roughly 10–40% range and inconsistent overlap definitions across papers. No Taiwanese Hokkien overlapped corpus and no cross-lingual overlap test set was found — the team will have to build its own.

### Cited Findings

**Synthetic, utterance-level mixtures (true overlap evaluation)**
- **LibriSpeechMix** [D]: English; built from LibriSpeech dev-clean and test-clean; six JSONL lists (dev/test × 1mix/2mix/3mix); each original evaluation utterance appears exactly N times in the N-speaker set so WERs are comparable across speaker counts; no mixture contains two utterances from the same speaker; "partially overlapped" — each utterance has a `delays` value in seconds applied before mixing (README example: 0.0 s and ~4.47 s). Hours: Not Reported on the README. License: a LICENSE file exists, type not visible. — [GitHub NaoyukiKanda/LibriSpeechMix](https://github.com/NaoyukiKanda/LibriSpeechMix)
- **LibriSpeechMix-style mixing** [D]: in the SOT paper, (S−1) utterances are added with random delays; "the original volume of each utterance was kept unchanged, resulting in an average signal-to-interference ratio of about 0 dB"; delays are constrained so that (1) start times differ by ≥0.5 s and (2) every utterance has at least one speaker-overlapped region with another utterance. — [Kanda et al., Interspeech 2020](https://www.isca-archive.org/interspeech_2020/kanda20b_interspeech.pdf)
- **Aishell1Mix** [D]: a Mandarin multi-talker mixture set sourced from AISHELL-1, simulated "with a same protocol of the LibriMix"; its two-speaker clean subset ("AishellMix") is used for zero-shot / one-batch-tuning CER evaluation of Whisper-based multi-talker models. — [Empowering Whisper](https://arxiv.org/html/2407.09817v1)
- **A non-standard "LibriMix" variant** [D]: the Xiaomi report builds its LibriMix test set from LibriSpeech test-clean "with random signal-to-noise ratios (SNRs) ranging from 0 to 15 dB", explicitly so "the model cannot rely on shortcuts such as simply transcribing the loudest speaker". — [Xiaomi-CocktailASR-1 report](https://arxiv.org/html/2609.11274v1)
- **LibriMix (Libri2Mix / Libri3Mix)** [S]: English; mixtures pair utterances of different speakers; noisy variant adds WHAM! noise so that the loudest-speaker-to-noise SNR is −6 to +3 dB; a later paper reports Libri2Mix mixture SNR with mean 0 dB and standard deviation 3.6 dB; `min` mode truncates the mixture to the shortest utterance; train-100 subset reported as 58 h / 251 speakers and test as 11 h / 40 speakers. — [LibriMix paper](https://arxiv.org/pdf/2005.11262); [Libri2Vox paper](https://arxiv.org/pdf/2412.12512)
  - Conflict: the search summary attributes "SNR drawn uniformly from 0 to 5 dB" to the LibriMix paper, while another paper reports a 0 dB-mean / 3.6 dB-std distribution. The 0–5 dB uniform scheme is the WSJ0-2mix recipe; treat the LibriMix level distribution as unverified. — [LibriMix paper](https://arxiv.org/pdf/2005.11262); [Libri2Vox paper](https://arxiv.org/pdf/2412.12512)
- **LibriMix overlap geometry** [D for the first clause, from Empowering Whisper; [S] for the DiCoW statement]: mixtures are simulated "in a left-aligned manner… the shorter source speech is entirely overlapped with the longer one from the start"; each speaker has one utterance, and there are no speaker-turn transitions. — [Empowering Whisper as a Joint Multi-Talker and Target-Talker ASR](https://arxiv.org/html/2407.09817v1); [Adapting Diarization-Conditioned Whisper](https://arxiv.org/pdf/2510.03723)
- **LibriMix license** [S]: scripts MIT on GitHub; LibriSpeech speech is CC BY 4.0 per its own terms (summary said "public domain"); WHAM! noise is CC BY-NC 4.0, which is the binding restriction for the noisy variants. — [GitHub JorisCos/LibriMix](https://github.com/JorisCos/LibriMix)
- **WSJ0-2mix** [S]: English; 30 h train, 10 h validation, ~5 h test; random speaker pairs from WSJ0 si_tr_s (train/val) and si_dt_05/si_et_05 (test); mixed at SNR 0–5 dB; utterances are fully overlapping (both speakers talk for nearly the whole utterance). Mixing scripts Apache 2.0; underlying WSJ0 must be purchased from LDC. — [Deep CASA paper](https://arxiv.org/pdf/1904.11148); [Analysis of Deep Clustering as Preprocessing for ASR of Sparsely Overlapping Speech](https://arxiv.org/pdf/1905.03500); [GitHub mborsdorf/UniversalSpeakerExtraction](https://github.com/mborsdorf/UniversalSpeakerExtraction)

**Replayed / semi-real with controlled overlap**
- **LibriCSS** [D]: English; concatenated LibriSpeech utterances played back from loudspeakers in an office room and re-recorded with distant microphones (real acoustics, scripted overlap); organised into 10-minute mini-sessions; six conditions 0S, 0L, OV10, OV20, OV30, OV40; both 7-channel and 1-channel (monaural) test data are generated; two evaluation modes (utterance-wise with oracle boundaries; continuous-input with chunked segments and CTM output scored by `eval_continuous.sh` using NIST SCTK). License file exists, type not visible. — [GitHub chenzhuo1011/libri_css](https://github.com/chenzhuo1011/libri_css)
- **LibriCSS** [D]: "a set of 8-speaker recordings made by playing back 'test clean' of LibriSpeech in a real meeting room", 7-channel array (first channel used for monaural evaluation); 10 hours in total; categorised by speaker overlap ratio from 0% to 40%; each category has 10 mini-sessions of 10 min; sessions 1–9 are the official evaluation split (session 0 excluded); 0L and 0S are 0% overlap with long and short inter-utterance silences. — [t-SOT paper](https://arxiv.org/pdf/2202.00842)
- LibriCSS has no condition above 40% overlap. [S] — [AudioShake benchmarks](https://audioshake.ai/multi-speaker-benchmarks)

**Real conversational / meeting corpora (partial overlap, turn-taking dominated)**
- **AMI** [S]: English; 98 h, 168 sessions, 3–5 speakers, 13.6% overlap (CSPB table; overlap = duration with ≥2 active speakers ÷ duration with ≥1 active speaker). — [CSPB, EACL 2026](https://aclanthology.org/2026.eacl-long.275.pdf)
- **AliMeeting (M2MeT)** [D, from the dataset paper]: Mandarin real meetings; 118.75 h total = 104.75 h Train (212 sessions) + 4 h Eval (8 sessions) + 10 h Test; 15–30 min sessions; 2–4 participants; 13 venues; 8-channel far-field array plus per-participant near-field headset recordings (only the wearer's own speech transcribed). **Average overlap ratio: Train 42.27%; Eval 34.76% in the running text but 34.20% in the paper's own Table 1 and Table 2 (internal inconsistency in the source).** By number of speakers (Train / Eval): 2 speakers 14.65% / 15.80%; 3 speakers 32.43% / 27.21%; 4 speakers 59.00% / 49.75%. The paper's exact overlap-ratio formula was not checked. Licence: not found in the paper. — [M2MeT paper](https://arxiv.org/pdf/2110.07393)
  - Conflict resolution: the 42.27% / 34.2–34.76% figures are the authoritative dataset-paper values. The lower figures — 27.8% (CSPB, defined as ≥2-active ÷ ≥1-active duration; 126 h / 237 sessions) and 0.19 (SDBench) — remain [S] and unverified (aclanthology blocked; SDBench not opened); they most likely reflect different overlap definitions and/or subsets. The Xiaomi report also describes AliMeeting overlap as "merely 30% to 40%" [D]. — [CSPB](https://aclanthology.org/2026.eacl-long.275.pdf); [SDBench](https://www.isca-archive.org/interspeech_2025/durmus25_interspeech.pdf); [Xiaomi-CocktailASR-1 report](https://arxiv.org/html/2609.11274v1)
  - AliMeeting overlap depends strongly on speaker count (≈15% for 2-speaker sessions vs. 50–59% for 4-speaker sessions), so a single corpus-level number is a poor descriptor.
- **AISHELL-4** [S]: Mandarin; 120 h total (107.50 h train, 12.72 h eval); average overlap ratio 19.04% (train) and 9.31% (eval); lower overlap than AliMeeting. — [AISHELL-4 paper](https://arxiv.org/pdf/2104.03603)
- **NOTSOFAR-1** [S]: English; real meetings recorded in Microsoft conference rooms; 315 meetings averaging 6 min with 4–8 attendees per one source, "approximately 280 unique meetings" per the CHiME-8 page (conflict); single-channel (SC) dev set overlap ratio ≈ 20% per one participant's system paper; recorded meeting data stated as CC BY 4.0 on GitHub/Hugging Face; a separate 1000-hour simulated training set was licensed for challenge use only at the time of the CHiME-8 page. — [NOTSOFAR-1 paper](https://arxiv.org/html/2401.08887v1); [CHiME-8 Task 2 data page](https://www.chimechallenge.org/challenges/chime8/task2/data); [Fano Labs CHiME-8 system](https://www.isca-archive.org/chime_2024/broughton24_chime.pdf); [GitHub microsoft/NOTSOFAR1-Challenge](https://github.com/microsoft/NOTSOFAR1-Challenge)
- **CHiME-6** [S]: English dinner-party recordings in homes, about 50 h, 20 sessions (secondary practitioner guide, low authority). Overlap percentage: Not Reported in retrieved sources. — [Spirelight meeting datasets guide](https://www.spirelight.ai/guides/meeting-speech-datasets)
- **MS-Bench**: see Q1 (32.72 h, real, mainly Chinese/English, 21.87% duration-weighted overlap). — [GitHub ASLP-lab/MS-Bench](https://github.com/ASLP-lab/MS-Bench)

**Taiwanese Hokkien / cross-lingual overlap**
- [S] No dataset or test set built for overlapped speech, multi-speaker ASR, or cross-lingual overlap in Taiwanese Hokkien/Mandarin was found. Nearest resources are non-overlap: YT-THDC (~30 h YouTube drama, Hokkien speech with loosely aligned Mandarin subtitles, multiple speakers plus music/noise); MediaTek Breeze Taigi benchmark (30 Mandarin–Taigi audio pairs from public service announcements); TAT / FSR-2020 (read or single-speaker Hokkien). — [CLiFT-ASR](https://arxiv.org/pdf/2511.06860); [Breeze-ASR-26 model card](https://huggingface.co/MediaTek-Research/Breeze-ASR-26); [NTNU FSR-2020 system](https://aclanthology.org/2021.ijclclp-1.1.pdf)
- [S] A multilingual mixture precedent exists: a Parakeet-TDT multitalker port reports cpWER on two-overlapping-speaker mixtures built from MLS for "six other languages" (third-party documentation). — [GitHub altunenes/parakeet-rs](https://github.com/altunenes/parakeet-rs)
- [D] Aishell1Mix (see above) is the nearest existing Mandarin fully-overlapped analogue of LibriMix; the paper reports that fine-tuned Whisper medium/large multi-talker models show "acceptable" zero-shot CER on it. The specific 28.94 zero-shot figure from the first pass was not re-verified cell-by-cell and stays [S]. — [Empowering Whisper](https://arxiv.org/html/2407.09817v1)

### Inferences
- Overlap-ratio figures are not comparable across papers: AliMeeting alone is quoted at 19%, 27.8% and 34.76–42.27%. Any protocol should recompute overlap from the annotations with one stated definition (CSPB's ≥2-active ÷ ≥1-active is a clean choice).
- *(prior knowledge)* LibriMix `max` mode keeps the longest utterance so the tail is single-speaker; `min` mode yields ~100% overlap. For "all words recovered" ASR scoring `max` is the appropriate mode because `min` cuts words off the longer reference. Not verified this session.
- For the team's languages, the practical route is a LibriSpeechMix/LibriMix-style simulation over single-speaker Hokkien, Mandarin and English test sets, with same-language and cross-language pairings, controlled SIR and controlled overlap ratio — because no public equivalent exists.
- Real Mandarin partial-overlap evaluation can use AliMeeting and AISHELL-4 (speaker-agnostic scoring); real English can use AMI and NOTSOFAR-1 SC. These measure conversational robustness, not simultaneous-speech recovery.

### Gaps
- Not retrieved (hosts blocked or searches returned nothing), therefore **Not Reported** here: ICSI (hours, overlap, license); Fisher / CallHome (overlap %, license); DiPCo (hours, overlap); MISP (hours, overlap); CHiME-7/8 DASR composition; AMI IHM-mix vs. SDM specifics and license; Libri3Mix hours; LibriSpeechMix total hours; LibriCSS license; AliMeeting/AISHELL-4 licenses; mono-channel availability for AliMeeting, AISHELL-4, CHiME-6.
- No public cross-lingual (e.g. Mandarin-over-English) overlap test set was found; absence of evidence from a few searches, not proof of non-existence.

---

## Q3. Metrics: definitions, applicability, tools, and which to use without speaker identity

### Takeaway
Because the team does not need speaker attribution, the right family is the speaker-agnostic one: ORC-WER (or MIMO-WER) from MeetEval for multi-stream or single-stream output, and Kanda-style minimum-permutation WER for short SOT mixtures; cpWER/tcpWER/SA-WER penalise attribution and should only be secondary. No standard "secondary-speaker deletion rate" metric was found in the literature searched — it would have to be defined by the team.

### Cited Findings
- **MeetEval** [D] supports: `wer` (SISO), `cpwer`, `orcwer`, `greedy_orcwer`, `mimower`, `tcmimower`, `tcpwer`, `tcorcwer`, `greedy_tcorcwer`, `greedy_dicpwer`, plus DER wrappers (`meeteval-der dscore`, `md_eval_22`, collar 0.25). Input formats: STM, CTM (one file per output channel), SegLST (JSON; `session_id` and `words` required, `speaker`/`start_time`/`end_time` only where the metric needs them), RTTM (DER only). Time-constrained metrics take `--collar` (README examples use 5). MIT license. No README notes on Chinese/character-level scoring. — [GitHub fgnt/meeteval](https://github.com/fgnt/meeteval)
- [D] In MeetEval STM, the speaker-ID field denotes the speaker *or system output stream*, not a microphone channel. — [GitHub fgnt/meeteval](https://github.com/fgnt/meeteval)
- The following four definitions were verified against the MeetEval paper full text in the second pass. — [MeetEval paper, arXiv 2307.11394](https://arxiv.org/pdf/2307.11394)
- **cpWER** [D]: reference and hypothesis streams are grouped by speaker with segment transcriptions concatenated; the standard WER is computed between the hypothesis and all permutations of the reference streams, and the permutation minimising the WER is reported (found in polynomial time via the Hungarian algorithm). Recommended for diarization-style recognisers whose output is grouped by speaker. — [MeetEval paper](https://arxiv.org/pdf/2307.11394)
- **tcpWER** [D]: cpWER with the Levenshtein distance replaced by a time-constrained variant — reference and hypothesis words may only match (correct or substitution) if they overlap in time or the gap is smaller than a collar c. Segment-level timestamps are converted to pseudo-word-level timings. The paper finds tcpWER close to the desired WER for collars of 3–5 s and concludes "a collar of 5 s is thus a reasonable choice for the majority of systems". Still speaker-attributed. MS-Bench uses 5 s. — [MeetEval paper](https://arxiv.org/pdf/2307.11394); [arXiv HTML 2610.25824](https://arxiv.org/html/2610.25824)
- **MIMO-WER** [D]: "measures the WER for multi-speaker scenarios without considering segment-level speaker assignment errors"; solves the assignment problem at segment level. References must be sorted by begin time within a speaker, but "the order of utterances is unconstrained across speakers, which is necessary when evaluating SOT-style systems that can output utterance of different speakers in an arbitrary order". It "penalizes splitting utterances over multiple streams since a reference utterance is always assigned to one output stream continuously". Computed with a multi-dimensional Levenshtein dynamic programme. — [MeetEval paper](https://arxiv.org/pdf/2307.11394)
- **ORC-WER** [D]: "a special case of the MIMO-WER which keeps the temporal order across speakers intact", computed by ignoring the reference speaker labels and optimally assigning reference utterances to the J hypothesis streams. "The complexity does not depend on the number of reference speakers and is polynomial in the number of utterances." ORC-WER and MIMO-WER differ only "when the system modifies the order of utterances (e.g., certain SOT systems) or where the utterance order is ambiguous in the reference"; ORC-WER overestimates the WER in those cases. — [MeetEval paper](https://arxiv.org/pdf/2307.11394)
- **MeetEval's own recommendation table (Fig. 1)** [D]: diarization-style recogniser (output grouped by speaker) → cpWER / tcpWER; CSS-style recogniser (≤J overlap-free channels) → ORC-WER, MIMO-WER or asclite; **SOT-style recogniser → MIMO-WER, with ORC-WER in parentheses** (two SOT sub-cases shown: "serialized, speaker ordered" and "serialized, temporal ordered"). Text: "For CSS-style and SOT-style recognizers, where the output is not grouped by speakers, ORC-WER or MIMO-WER have to be used." — [MeetEval paper](https://arxiv.org/pdf/2307.11394)
- **Single-stream convention** [D]: "An SOT-style recognizer is one of the few exceptions where the recognizer can jump backward in time so that the MIMO-WER is the only applicable metric. Since the SOT-style output consists of a single stream (J = 1), the execution time of MIMO-WER is reasonable." So the paper explicitly treats SOT output as one hypothesis stream (J = 1) scored with MIMO-WER. — [MeetEval paper](https://arxiv.org/pdf/2307.11394)
- **asclite** [D]: per the MeetEval authors, asclite "is, according to its documentation, able to handle all recognizer styles while performing a word-level matching ignoring speaker labels. But in practice, it is computationally demanding and not all promoted features work as expected"; it uses a temporal constraint "mainly for speedup reasons", with "no clear notion of a collar". — [MeetEval paper](https://arxiv.org/pdf/2307.11394)
- **Other MIMO/ORC sources** [S]: multi-stream background and efficient algorithms. — [WER Definitions and Algorithms for Long-Form Multi-talker ASR](https://arxiv.org/html/2508.02112v1); [On WER Definitions and Their Efficient Computation](https://www.researchgate.net/publication/371287360_On_Word_Error_Rate_Definitions_and_Their_Efficient_Computation_for_Multi-Speaker_Speech_Recognition_Systems)
- **tcORC-WER** [S]: time-constrained ORC-WER; one paper used a 5 s collar "since hypotheses and references can be aligned across implausibly long temporal distances". — [Scaling Multi-Talker ASR with Speaker-Agnostic Activity Streams](https://arxiv.org/pdf/2510.03630)
- **DI-cpWER** (Diarization-Invariant cpWER) [S]: corrects speaker-assignment errors so that the error rate is minimised, to judge the impact of attribution errors on cpWER; greedy algorithms approximate ORC-WER and DI-cpWER with <0.1% deviation at polynomial instead of exponential runtime. — [WER Definitions and Algorithms for Long-Form Multi-talker ASR](https://arxiv.org/html/2508.02112v1)
- A secondary source defines attribution cost as cpWER − DI-cpWER. [S] — [Emergent Mind WDER topic page](https://www.emergentmind.com/topics/word-diarization-error-rate-wder)
- **Δcp** [S]: MOSS-Transcribe-Diarize reports Δcp = cpCER − CER to isolate speaker-attribution error; **cpCER** is the character-level analogue of cpWER used for Chinese (AISHELL-4, AliMeeting). — [MOSS-Transcribe-Diarize report](https://arxiv.org/pdf/2601.01554)
- **Minimum-permutation WER for SOT (Kanda's scoring)** [D]: "In multi-talker ASR, a system may produce a different number of hypotheses than references (i.e., speakers). To cope with this, all possible permutations of the hypothesis order were examined, and the one that yielded the lowest WER was picked." The number of speakers is read off the output "just by counting the occurrences of ⟨sc⟩ and ⟨eos⟩". — [Kanda et al., SOT, Interspeech 2020](https://www.isca-archive.org/interspeech_2020/kanda20b_interspeech.pdf)
- [S] MT-LLM states "permutations with minimum errors are used to compute WER" on LibriSpeechMix 2/3-speaker test sets. — [MT-LLM](https://arxiv.org/html/2409.08596)
- **TS-WER / TS-CER** (target-speaker error rate) [D]: error rate for one designated target speaker, the reference being only that speaker's transcript; on synthetic sets "every speaker in a mixed utterance is alternately treated as the target speaker". A model that transcribes *all* speakers is therefore penalised with insertions under this metric. — [Xiaomi-CocktailASR-1 report](https://arxiv.org/html/2609.11274v1)
- **CER for overlapped Mandarin (M2MeT)** [D]: CER = (N_Ins + N_Subs + N_Del)/N_Total × 100%, with a permutation consideration for overlapping speech (references sorted, per the challenge description). — [M2MeT paper](https://arxiv.org/pdf/2110.07393)
- **SAgWER** (Speaker-Agnostic WER) [S]: used in an Interspeech 2025 paper for speaker-agnostic evaluation; formula not retrieved. — [Improving Practical Aspects of End-to-End Multi-Talker Speech Recognition](https://www.isca-archive.org/interspeech_2025/subramanian25b_interspeech.pdf)
- **Other names seen**: "OCR-WER" (sic, presumably ORC-WER) in TellWhisper for Libri2Mix; **tcpMER** (mixed error rate, time-constrained) in the MLC-SLM challenge for multilingual two-speaker data; **UD-CER** (utterance-dependent CER) in BA-SOT to measure speaker-change prediction precision. — [TellWhisper](https://arxiv.org/html/2601.03712v1); [Diarization-Guided Qwen-ASR Adaptation](https://arxiv.org/pdf/2607.08208); [BA-SOT](https://arxiv.org/pdf/2305.13716)
- **LibriCSS official scoring** [D]: NIST SCTK; in utterance-wise mode, when several separation outputs exist for a mixture, the lowest-WER transcription is kept. — [GitHub chenzhuo1011/libri_css](https://github.com/chenzhuo1011/libri_css)
- **Per-speaker / dominant-vs-weaker reporting**: three targeted searches plus full-text reads of SOT, t-SOT, H-SAGE, MS-Bench and the Xiaomi report found **no** table reporting errors separately for the louder vs. quieter speaker or WER binned by SIR. [D] The SOT paper reports only an average SIR (~0 dB); the Xiaomi report randomises SNR over 0–15 dB to defeat a "transcribe the loudest speaker" shortcut but reports no SIR-binned results; the monaural multi-speaker ASR survey (arXiv 2505.10975) contains no keyword hits for SIR, same-gender, dominant or weaker speaker. — [Kanda et al., SOT](https://www.isca-archive.org/interspeech_2020/kanda20b_interspeech.pdf); [Xiaomi-CocktailASR-1 report](https://arxiv.org/html/2609.11274v1); [Survey 2505.10975](https://arxiv.org/html/2505.10975)
- **Speaker-counting accuracy as a companion metric** [D]: the SOT paper reports estimated vs. actual speaker count (1024-dim SOT): 1 speaker → 99.8% correct; 2 speakers → 97.0% correct (1.9% under-count); 3 speakers → 74.2% correct, 24.0% estimated as 2 (i.e. a whole speaker dropped in about a quarter of 3-speaker mixtures). — [Kanda et al., SOT](https://www.isca-archive.org/interspeech_2020/kanda20b_interspeech.pdf)

### Inferences
- **SOT output with separator tokens, no speaker identity (short mixtures, ≤3 streams):** split the hypothesis on the separator, then score with minimum-permutation WER against the per-speaker references — in MeetEval terms, `cpwer` with the split index used as a pseudo-speaker label. This is equivalent to Kanda's SOT scoring and yields per-stream insertion/deletion/substitution counts. Because the stream labels are arbitrary, cpWER here measures no attribution; it is purely a recovery metric. (Inference from the definitions above.)
- **SOT output where one stream may contain several utterances, or long-form audio:** use `orcwer` (or `tcorcwer` if timestamps exist), since ORC-WER assigns reference utterances to streams and does not require stream = speaker.
- **Single unsegmented text stream:** the MeetEval paper confirms the convention — treat the output as one hypothesis stream (J = 1) and use MIMO-WER, which is the only applicable metric when the model may emit speakers' utterances out of temporal order (speaker-ordered SOT); ORC-WER is equivalent only when output follows temporal order. The exact MeetEval CLI encoding of J = 1 (one stream ID for all hypothesis segments) is still worth a toy-example check. Caveat by inference from the definition ("a reference utterance is always assigned to one output stream continuously"): word-level interleaving of two simultaneous talkers inside one stream (t-SOT-style output without channel tokens) cannot be matched well by either metric, so the model should emit utterance-serialised text.
- With SOT separators on short two/three-talker mixtures, Kanda's min-permutation WER and MIMO-WER (J = 1) should give the same or very similar numbers; the practical difference is that min-permutation scoring exposes per-stream error counts, which the team needs for a secondary-speaker deletion rate.
- *(prior knowledge, unverified this session)* SA-WER (Kanda, speaker-attributed WER) requires speaker profiles/identities and is not applicable when no identity is output. asclite is covered under Cited Findings; given the MeetEval authors' practical warnings, prefer MeetEval over asclite.
- Do not use TS-WER for this team's goal: it rewards suppressing the other talkers, the opposite of "recover all speakers' words".
- **Chinese / Hokkien / mixed-language scoring:** use character-level tokenisation for Mandarin and Han-script Hokkien and word-level for English, i.e. a mixed error rate (MER), as MLC-SLM does with tcpMER. MeetEval's README does not document a CJK mode, so tokenise before scoring (insert spaces between Han characters) and apply identical text normalisation to reference and hypothesis.
- **Proposed "secondary-speaker deletion rate" (team-defined, no published precedent found):** on simulated mixtures with known SIR, take the optimal permutation/ORC alignment, then report deletions ÷ reference words separately for the higher-energy and lower-energy reference stream; additionally report a "stream-drop rate" = fraction of mixtures where one reference stream has ≥ X% deletions (e.g. 90%), and a speaker-count accuracy (number of separator-delimited streams vs. truth). State the definition explicitly since it is not standard.
- Always co-report the insertion rate: fully-overlapped WER above 100% (as in the Qwen3-ASR numbers in Q5) signals hallucination/repetition, which a deletion-only view would miss.

### Gaps
- DI-cpWER, tcORC-WER, tcMIMO-WER and SAgWER descriptions remain [S] (the long-form WER paper, arXiv 2508.02112, was downloaded but only keyword-checked, not read in full).
- SA-WER formal definition not retrieved from a primary source.
- Whether MeetEval documents character-level (CER) scoring anywhere other than the README: unknown.

---

## Q4. Published results conditioned on overlap ratio, SIR, speaker count, gender, language

### Takeaway
Overlap-ratio-conditioned results are well established on LibriCSS (WER roughly triples to quadruples from 0% to 40% overlap for a single-talker recogniser without separation) and exist as paired deltas in MS-Bench; conditioning on SIR, louder/quieter speaker, gender pairing, backchannels or same- vs. cross-language overlap was not found in the sources reachable this session.

### Cited Findings
- **LibriCSS baseline, single-talker ASR, no separation** [D] (WER %, official repo):

  | Mode | 0S | 0L | OV10 | OV20 | OV30 | OV40 |
  |---|---|---|---|---|---|---|
  | Utterance-wise, no separation | 11.5 | 11.3 | 18.3 | 26.4 | 34.6 | 43.2 |
  | Utterance-wise, 7-ch separation | 8.3 | 8.4 | 11.6 | 16.0 | 18.4 | 21.6 |
  | Continuous-input, no separation | 15.4 | 11.4 | 21.7 | 27.2 | 34.3 | 40.4 |
  | Continuous-input, 7-ch separation | 11.9 | 9.7 | 13.4 | 15.1 | 19.7 | 22.0 |

  — [GitHub chenzhuo1011/libri_css](https://github.com/chenzhuo1011/libri_css)
- **LibriCSS, ESPnet recogniser baseline** [S]: 17.6 / 16.3 / 20.9 / 26.1 / 32.6 / 36.1 (0S / 0L / OV10 / OV20 / OV30 / OV40). — [MMS-MSG paper](https://arxiv.org/pdf/2209.11494)
- **LibriCSS, vendor separation + Whisper large-v3, cpWER via MeetEval** [S]: 5.81 (0S), 7.04 (10%), 7.18 (20%), 13.66 (30%), 10.03 (40%); vendor self-report, baseline labelling unclear in the summary. — [AudioShake benchmarks](https://audioshake.ai/multi-speaker-benchmarks)
- **t-SOT (streaming, single-channel) on LibriCSS** [S]: reports WER per overlap ratio 0–40% and claims better WER for all conditions than prior streaming models; secondary summary gives *average* WER 10.1 (TT-18, 160 ms), 9.0 (TT-36, 160 ms), 7.6 (TT-36, 2560 ms). Per-condition cells: Not Reported in retrieved text. — [t-SOT paper](https://arxiv.org/pdf/2202.00842); [Emergent Mind LibriCSS page](https://www.emergentmind.com/topics/libricss)
- **By speaker count, fully-overlapped vs. partially-overlapped synthetic sets** [S] (WER %, Xiaomi-CocktailASR-1 vs. off-the-shelf Qwen3-ASR-1.7B; note Xiaomi's model is a target-speaker system, so the task setup for the baseline may differ):

  | Test set | Xiaomi-CocktailASR-1 | Qwen3-ASR-1.7B |
  |---|---|---|
  | LibriMix 2-speaker | 4.11 | 68.75 |
  | LibriMix 3-speaker | 12.29 | 106.04 |
  | LibriSpeechMix 2-speaker | 2.90 | 92.17 |
  | LibriSpeechMix 3-speaker | 4.91 | 160.82 |
  | AMI SDM | 21.81 | 38.18 |
  | AliMeeting Far | 20.63 | 39.64 |
  | AMI IHM (single-speaker check) | 8.89 | 10.56 |
  | AliMeeting Near (single-speaker check) | 6.57 | 6.39 |

  — [Xiaomi-CocktailASR-1 Technical Report](https://arxiv.org/html/2609.11274v1); [GitHub xiaomi-research/xiaomi-cocktailasr-1](https://github.com/xiaomi-research/xiaomi-cocktailasr-1)
- **MS-Bench overlap deltas** [D]: Δlex +27.2 to +38.6 pp and Δattr +4.2 to +19.1 pp across seven systems (see Q1). — [GitHub ASLP-lab/MS-Bench](https://github.com/ASLP-lab/MS-Bench)
- **H-SAGE** [S]: stratifies LibriSpeechMix into low / mid / high overlap levels, with overlap rate defined as overlapping-segment duration ÷ total mixture duration. Numbers: Not Reported in retrieved text. — [H-SAGE](https://arxiv.org/pdf/2607.01566)
- **Two overlapping speakers, plain single-talker TDT + diarization vs. multitalker TDT** [S] (third-party port documentation, cpWER): English LibriSpeechMix 36.7% → 4.4%; six other languages (MLS mixtures) 39.0% → 10.7%. — [GitHub altunenes/parakeet-rs](https://github.com/altunenes/parakeet-rs)
- **SOT-family reference points on mixtures** [S]: SA-SOT cpWER 3.41% on LibriSpeechMix; SQ-Whisper (target-speaker) 14.6% WER on Libri2Mix test; separation (TIGER) + Whisper 37.96 / 37.97 on Libri2Mix dev/test; Whisper-D baseline 14.39 "OCR-WER" on Libri2Mix; a decoupled separation + ASR system claims 5.1% on Libri2Mix dev/test. — [SA-SOT](https://arxiv.org/pdf/2403.02010); [SQ-Whisper](https://arxiv.org/abs/2412.05589); [TellWhisper](https://arxiv.org/html/2601.03712v1); [Decoupling Speaker Separation and Speech Recognition](https://arxiv.org/pdf/2503.17886)
- **Backchannel-related failure** [S]: a 2026 paper reports that an LLM-based system trained on overlapped speech repeats backchannel words such as "yes" in overlapped regions, attributed to the LLM detecting that overlap exists without reliably inferring its content. — [Balancing ASR and diarization in end-to-end LLMs for multi-talker speech recognition](https://arxiv.org/pdf/2606.13095)
- **MT-LLM** [S]: evaluated on LibriSpeechMix 2- and 3-speaker sets; roughly on par with SOT-Conformer for 2 speakers and behind for 3; supports instructions by talker sex, order, language, keyword; no breakdown by overlap ratio, SIR or gender found. — [MT-LLM](https://arxiv.org/html/2409.08596)

### Inferences
- The LibriCSS no-separation rows are the cleanest public dose-response curve for a *single-talker* recogniser: 11.5 → 43.2% WER from 0% to 40% overlap (utterance-wise), i.e. roughly +0.8 pp WER per +1 pp overlap. The recogniser there is the 2020 LibriCSS baseline, not a modern foundation model.
- The gap between Qwen3-ASR-1.7B on fully/heavily-overlapped synthetic sets (69–161% WER) and on real far-field meetings (38–40%) quantifies how much real-meeting benchmarks understate the overlap problem: they are dominated by single-speaker segments (the Xiaomi report's own explanation).
- A protocol for this team should report a grid: overlap ratio (0 / 10 / 20 / 40 / 100%), SIR (e.g. 0, ±5, ±10 dB), speaker count (1 / 2 / 3), gender pairing, and language pairing (same vs. cross) — because, apart from overlap ratio and speaker count, none of these axes has published reference numbers that could be found.

### Gaps
- No retrieved paper reports WER by SIR/energy ratio, per louder/quieter speaker, same- vs. different-gender, short interjections/backchannels as a stratum, or same- vs. cross-language overlap. Searches for these returned nothing specific; the full texts of SOT, t-SOT, H-SAGE and the monaural multi-speaker ASR survey (arXiv 2505.10975) could not be opened and may contain such tables.
- Per-condition t-SOT and H-SAGE numbers: Not Reported (full text not accessible).

---

## Q5. Behaviour of existing models on overlapped speech

### Takeaway
The only quantitative evidence found for an off-the-shelf single-talker foundation model on fully-overlapped speech is Qwen3-ASR-1.7B at 69–161% WER on LibriMix/LibriSpeechMix (competitor-reported) — a figure above 100% implies heavy insertions, not merely a dropped second speaker. No source measured secondary-speaker deletion rates for Whisper, Parakeet, Canary, SenseVoice, Gemini or GPT-4o, and no number at all was found for Qwen3-ASR-0.6B.

### Cited Findings
- **Qwen3-ASR** [S]: the technical report's stress tests include multi-speaker Chinese conversational speech but no overlap-specific breakdown was visible. — [Qwen3-ASR Technical Report](https://arxiv.org/html/2601.21337v1)
- **Qwen3-ASR-1.7B on overlap** [S]: Xiaomi states that "existing top-tier ASR models such as Qwen3-ASR-1.7b and StepAudio2 lack the ability to recognize multiple speakers"; reported WERs 68.75 (LibriMix-2), 106.04 (LibriMix-3), 92.17 (LibriSpeechMix-2), 160.82 (LibriSpeechMix-3); better on real data (AMI SDM 38.18, AliMeeting Far 39.64), which the authors attribute to single-speaker segments mitigating degradation. Competitor-reported, not independent. — [Xiaomi-CocktailASR-1 Technical Report](https://arxiv.org/html/2609.11274v1)
- **Qwen3-ASR adaptation** [S]: an MLC-SLM 2026 entry adapts Qwen3-ASR-1.7B with diarization guidance and LoRA for multilingual two-speaker conversations, reporting average tcpMER 17.97 on the final evaluation set and an adaptation gain 30.53 → 24.3. — [Diarization-Guided Qwen-ASR Adaptation](https://arxiv.org/pdf/2607.08208)
- **Qwen3-ASR in a cascade** [D]: pyannote+Qwen3-ASR on MS-Bench: DER 17.35, cpWER 43.25, tcpWER 46.54, WER 34.96; largest overlap penalty of seven systems (Δlex +38.6, Δattr +19.1). — [GitHub ASLP-lab/MS-Bench](https://github.com/ASLP-lab/MS-Bench)
- **Qwen3-ASR toolkit** [S]: an open GitHub issue requests speaker diarization and multi-speaker transcript segmentation, i.e. not a built-in capability. — [QwenLM/Qwen3-ASR-Toolkit issue #13](https://github.com/QwenLM/Qwen3-ASR-Toolkit/issues/13)
- **Whisper** [S]: "not explicitly trained for multi-talker ASR and lacks the ability to perform speaker attribution"; "performance deteriorates in overlapping speech scenarios" (qualitative statements by DiCoW authors). Diarization-conditioned Whisper (DiCoW) itself degrades on fully-overlapped LibriMix because its silence/target/non-target/overlap masks become ambiguous. — [Adapting Diarization-Conditioned Whisper](https://arxiv.org/pdf/2510.03723)
- **Whisper with front-end separation** [S]: TIGER + Whisper reaches only 37.96 / 37.97 WER on Libri2Mix and "still falls behind direct multi-speaker recognition". — [TellWhisper](https://arxiv.org/html/2601.03712v1)
- **Parakeet multitalker (NVIDIA `multitalker-parakeet-streaming-0.6b-v1`)** [S]: self-speaker adaptation — speaker kernels generated from diarization (Sortformer) activations injected into selected Fast-Conformer encoder layers; needs only diarization output, no enrollment; one model instance per speaker. Model-card cpWER at 1.12 s latency: AMI IHM 21.26, AMI SDM 37.44, CH109 15.81, Mixer 6 23.81. Conflict: the arXiv paper reports CH109 cpWER 26.21% at 1120 ms. Third-party GGUF port: 19.35% cpWER on AMI-IHM test (16 meetings, MeetEval). Streaming Sortformer v2.1 handles up to 4 speakers. — [NVIDIA model card](https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1); [Speaker Targeting via Self-Speaker Adaptation](https://arxiv.org/pdf/2506.22646); [myflow-ai GGUF card](https://huggingface.co/myflow-ai/multitalker-parakeet-streaming-0.6b-v1-gguf)
- **Plain Parakeet-TDT (single-talker) + diarization on two overlapping speakers** [S]: cpWER 36.7% (LibriSpeechMix) and 39.0% (MLS mixtures, six languages) before multitalker fine-tuning; 4.4% / 10.7% after (third-party port docs). — [GitHub altunenes/parakeet-rs](https://github.com/altunenes/parakeet-rs)
- **MOSS-Transcribe-Diarize** [S]: 0.9B model, 50+ languages, long-form transcription with speaker labels and timestamps. GitHub card: AISHELL-4 CER 14.84 / cpCER 15.83 / Δcp 0.99; AliMeeting CER 24.86 / cpCER 22.17; Pro variant cpCER 14.02 (AISHELL-4) and 13.94 (AliMeeting). Conflict: the arXiv report's table gives MOSS cpCER 20.04 on AISHELL-4 (Doubao 27.86), likely a different model version. All vendor-reported. — [GitHub OpenMOSS/MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize); [MOSS-Transcribe-Diarize report](https://arxiv.org/pdf/2601.01554)
- **Gemini / GPT-4o** [S]: in the MOSS report, Gemini 2.5 Pro scores cpCER 27.43% with Δcp 4.68% on AISHELL-4 (secondary summary); GPT-4o and Gemini 3 Pro are described as failing on long-form inputs due to length constraints or output-format issues. These are long-form diarized-transcription results, not overlap-specific. — [MOSS-Transcribe-Diarize report](https://arxiv.org/pdf/2601.01554)
- **Speech-LLM multi-talker systems** [S]: Qwen3.8-Omni-Flash on MS-Bench: cpWER 29.43, WER 27.86; VibeVoice-ASR cpWER 43.41, WER 33.11; Doubao-ASR 2.0 cpWER 55.37, WER 38.20. — [GitHub ASLP-lab/MS-Bench](https://github.com/ASLP-lab/MS-Bench)
- **Scoring practice hides the problem** [S]: a practitioner guide notes overlapping speech has historically been scored out in many corpora. Low-authority source. — [Kili diarization guide](https://kili-technology.com/blog/speaker-diarization-models-guide-benchmarks-and-failure-modes-2026)

### Inferences
- WER > 100% for Qwen3-ASR-1.7B on 3-speaker mixtures cannot be produced by deletions alone (dropping two of three speakers caps deletions near 67%); insertions/substitutions from garbled or hallucinated output must be substantial. So the failure mode is "garble + drop", not a clean "transcribe the dominant speaker only". How the Xiaomi evaluation scored the baseline (which reference it compared against, given their model is target-speaker) could change this reading — the setup was not visible.
- Qwen3-ASR-0.6B is the smaller sibling of the 1.7B model; it is reasonable to expect it to be no better on overlap, but no measurement was found — the team should measure it as the first baseline row.
- The "single-talker model drops the secondary speaker" claim is widely asserted qualitatively but was not found quantified (per-speaker deletion) for any model. This is a publishable gap the team's protocol can fill.
- Systems in MS-Bench and the MOSS/Parakeet results solve *speaker-attributed turn-taking* transcription; good cpWER there does not demonstrate simultaneous-speech recovery.

### Gaps
- No zero-shot Whisper large-v2/v3 WER on Libri2Mix/LibriSpeechMix, and no per-speaker deletion analysis, was found despite three targeted searches.
- No overlap-specific results found for Canary, SenseVoice, Gemini audio, or GPT-4o(-transcribe).
- No result of any kind for Qwen3-ASR-0.6B on overlapped speech.
- No result for any model on Taiwanese Hokkien overlap or cross-lingual overlap.
- Hugging Face model cards and arXiv full texts could not be opened; all model numbers above are from search summaries except MS-Bench.

---

## Q6. Taxonomy: diarization vs. turn-taking multi-speaker ASR vs. partial-overlap vs. fully-overlapped ASR

### Takeaway
Most "multi-speaker ASR" benchmarks — including MS-Bench, AMI, AliMeeting, AISHELL-4, NOTSOFAR-1 and CHiME — measure who-spoke-when plus turn-taking transcription with incidental partial overlap; only LibriCSS (controlled partial overlap) and the synthetic mixtures (LibriSpeechMix partial, LibriMix/WSJ0-2mix full) isolate simultaneous-speech recovery.

### Cited Findings
- Fully overlapped by construction: WSJ0-2mix ("fully overlapping speech") and LibriMix (left-aligned; shorter utterance entirely overlapped; no speaker turns). [S] — [Analysis of Deep Clustering as Preprocessing](https://arxiv.org/pdf/1905.03500); [Empowering Whisper](https://arxiv.org/html/2407.09817v1)
- Partially overlapped by construction, utterance-level: LibriSpeechMix (random delays; 1/2/3-speaker lists). [D] — [GitHub NaoyukiKanda/LibriSpeechMix](https://github.com/NaoyukiKanda/LibriSpeechMix)
- Controlled overlap ratio in continuous audio: LibriCSS (0S, 0L, OV10–OV40; 10-minute mini-sessions; mono available). [D] — [GitHub chenzhuo1011/libri_css](https://github.com/chenzhuo1011/libri_css)
- Real conversational, diarization + speaker-attributed ASR, overlap incidental: MS-Bench (DER/cpWER/tcpWER; 21.87% overlap), AMI (13.6%), AliMeeting (19–42% depending on source), AISHELL-4 (9.31–19.04%), NOTSOFAR-1 (≈20% on SC dev). [D]/[S] — [GitHub ASLP-lab/MS-Bench](https://github.com/ASLP-lab/MS-Bench); [CSPB](https://aclanthology.org/2026.eacl-long.275.pdf); [AISHELL-4 paper](https://arxiv.org/pdf/2104.03603); [Fano Labs CHiME-8 system](https://www.isca-archive.org/chime_2024/broughton24_chime.pdf)
- Metric side of the taxonomy: DER measures diarization only; cpWER / tcpWER / cpCER measure speaker-attributed transcription; ORC-WER / MIMO-WER / speaker-agnostic WER measure word recovery irrespective of attribution. [D]/[S] — [GitHub fgnt/meeteval](https://github.com/fgnt/meeteval); [WER Definitions for Long-Form Multi-talker ASR](https://arxiv.org/html/2508.02112v1)

### Inferences
- Classification table (synthesised from the findings above):

  | Category | What is tested | Benchmarks | Natural metric |
  |---|---|---|---|
  | A. Speaker diarization | who spoke when; no words | any of the meeting corpora scored with DER | DER |
  | B. Turn-taking multi-speaker ASR (speaker-attributed) | words + speaker labels on long conversations; overlap incidental (~10–40%) | MS-Bench, AMI, ICSI, AliMeeting, AISHELL-4, NOTSOFAR-1, CHiME-6/7/8, Fisher/CallHome, DiPCo, MISP | cpWER / tcpWER / cpCER |
  | C. Partial-overlap ASR with controlled overlap | word recovery as a function of overlap ratio | LibriCSS (OV10–OV40), LibriSpeechMix, H-SAGE's stratified LibriSpeechMix, MS-Bench O-bins (observational) | ORC-WER / min-permutation WER, per overlap bin |
  | D. Fully-overlapped ASR | recover all talkers when they speak simultaneously for (nearly) the whole utterance | WSJ0-2mix, Libri2Mix / Libri3Mix (min mode ≈ 100% overlap) | min-permutation WER / ORC-WER; per-speaker deletions |

  (ICSI, CHiME-6/7/8, Fisher/CallHome, DiPCo and MISP are placed in B from prior knowledge of their design; their statistics were not retrieved this session.)
- The team's requirement ("all speakers' words recovered, no attribution, including cross-lingual overlap") sits in C and D, scored speaker-agnostically. Category B benchmarks should be used only as a regression check (single-speaker and conversational performance must not degrade), scored with ORC-WER or speaker-agnostic WER/CER rather than cpWER.
- Suggested protocol skeleton (inference): (1) single-talker regression sets per language; (2) simulated 2- and 3-talker mixtures per language pair (Hokkien–Hokkien, Mandarin–Mandarin, English–English, and the three cross pairs) on a grid of overlap ratio × SIR, scored by min-permutation MER with per-stream deletion/insertion and stream-drop rate; (3) LibriCSS mono OV0–OV40 for a public comparable curve; (4) real partial-overlap sets (AliMeeting, AISHELL-4, AMI, NOTSOFAR-1 SC, MS-Bench) with ORC-WER / speaker-agnostic CER.

### Gaps
- No published taxonomy using exactly these four categories was retrieved; the classification is the researcher's synthesis. The monaural multi-speaker ASR survey (arXiv 2505.10975), which may contain an authoritative taxonomy, could not be opened.
