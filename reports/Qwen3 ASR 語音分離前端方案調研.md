# 解除單模型限制後，分離器串接與 SOT 並列首選

把「單一模型」這條限制拿掉之後，「外掛語音分離器＋Qwen3-ASR-0.6B 逐串流辨識」不再是對照組，而是與 SOT 微調**並列第一順位**的路線，兩者該由一個不需訓練的第 0 階段實驗來分勝負，而不是由文獻來決定。理由是：在英文合成與重播資料上，強分離器加上完全未微調的 ASR 已是公開數字中最好的一群——noisy Libri2Mix（ESPnet 偏移版）上 TF-CrossNet → Whisper large-v3 為 5.0%，同表的單次前傳系統為 7.0–23.3%（[Yang et al. 2025](https://arxiv.org/html/2503.17886v1)）；單通道 LibriCSS 上 TF-GridNet → Whisper large-v2 為 6.8%，不做分離是 26.5%（[von Neumann et al. 2023](https://arxiv.org/html/2309.16482)）。但這些證據有四個同等重要的但書：**沒有任何一個數字是把 Qwen3-ASR（或任何 Speech-LLM）接在分離器後面量到的**；台語與跨語言混音完全沒有任何分離或串接結果；英文訓練的分離器換到華語會大幅退化（Conv-TasNet 由 11.98 dB 掉到 2.08 dB）；而在真實會議上，分離對「不計說話者歸屬」的錯誤率幫助很小（AMI-SDM uWER 35.4→33.8）。串接的結構性優點是基礎 ASR 權重完全不動，單人與三語能力在「不經分離器」的路徑上原封保留；結構性缺點是多一顆模型、成本由分離器主導、CPU 上只有輕量分離器跑得動，以及一整組已有名字的失敗模式（假的第二串流、洩漏、殘留串流上的幻覺）。SOT 的優缺點正好相反。因此本報告的建議是：先用現成 16 kHz 分離器接未改動的 Qwen3-ASR-0.6B，在自建的台語／華語／英語合成重疊測試集上量一次，同時跑前一份報告的 baseline 與 Encoder 診斷；再依結果決定主攻串接、SOT，或兩者的混合。

本報告是技術評估，不是定案。筆記中的數字帶有來源等級：〔P〕為調研時直接讀自原始來源（論文全文、官方 README、模型卡）；〔P2〕為讀自「另一篇論文的比較表」，量測條件可能與原論文不同；〔S〕只在搜尋摘要看到，本報告一律標示「未驗證」或省略。論文未公布者寫 Not Reported，調研未讀到者寫「未取得」。內文未特別標示的數字皆為〔P〕。由筆記自行推算、非論文公布的算術，一律標示「推算」。

## 先前的結論哪些還成立，哪些要改寫

[前一份報告](2026-11-10-qwen3-asr-overlapped-speech.html)把「單一模型、推論架構簡單」當成硬限制，因此把分離前端排在最後一名，只當外部 baseline。該報告關於端到端路線的發現仍然有效，這裡只摘要，細節請回原報告。

端到端路線的重點有五項。其一，utterance-level SOT 只改訓練標籤、不加推論模組，是 2 人重疊最成熟的單模型解法，但 LLM decoder 上的純 SOT 到 3 人就明顯退化。其二，把可訓練參數放在哪裡是成敗關鍵：未調適的 WavLM-Large 凍結、只訓練 projector 與 LLM LoRA 時 Libri2Mix 為 46.7%，解凍 Encoder 為 10.2%，而凍結的 Whisper-large-v2 卻可做到 6.6%；AuT Encoder 屬於哪一類沒有人量過（[Shi et al. 2024](https://arxiv.org/html/2408.17431)、[MT-LLM](https://arxiv.org/html/2409.08596)）。其三，多語遺忘是實測風險：Whisper 以英文多人資料全參數微調後中文多人能力受損，adapter 才保得住（[Li et al. 2023](https://arxiv.org/html/2305.18747)）；同一顆 0.6B 基底做 word-level SOT 微調後 Open ASR 平均 WER 由 7.16 升到 16.95（[NVIDIA 2026](https://arxiv.org/abs/2609.10265)）。其四，per-speaker CTC／SOP、persistent memory、第二顆 Encoder 的增益主要在 3 人；t-SOT 在 Transducer 上證據充分，在 LLM decoder 上沒有任何結果。其五，公開文獻沒有 Qwen3-ASR 的 all-speaker 重疊 baseline，所以第 0 階段必須先診斷。

需要改寫的是關於串接的判斷。前一份報告依據 2021 年的 Conformer CSS（mono LibriCSS 平均 11.8，ASR 導向微調後 9.6）寫下「單一串流的 t-SOT（9.0–10.1）已與串接持平或更好」。這句話對 2021 年的分離器成立，對 2023 年以後的 TF 域分離器不成立：同樣是單通道 LibriCSS，TF-GridNet CSS 接未微調的 ASR 已到 6.4–6.8，微調聲學模型後為 5.6（[Vieting et al. 2023](https://arxiv.org/pdf/2309.08454)、[von Neumann et al. 2023](https://arxiv.org/html/2309.16482)）。同一批論文引用的 t-SOT 數字是 **7.6**（標示為「t-SOT TT、LS960、asclite WER」），與前一份報告引用的 9.0–10.1 不同；兩者來自不同的 t-SOT 組態與計分方式，是兩個各自成立的已發表數字，本報告並列而不取平均，也不擇一。不論採哪一個，強分離器串接在這個基準上都不輸 t-SOT。

還有一點沒有變：團隊的需求（還原所有重疊者文字、不需 diarization 與 enrollment、保住單人與台語／華語／英語能力、沒有重疊語料、在意 RTF／VRAM／參數量與 CPU 部署）中，前三項串接天生容易滿足，最後一項則是串接最吃虧的地方。

## 分離器候選比較表：16 kHz、可商用、有 WER 實證很難三者兼得

幾乎所有 22–25.5 dB 的 SI-SDRi 頭條數字都是 **8 kHz、WSJ0-2mix、2 人、完全重疊** 的結果。下表除註明者外，分離品質皆為 8 kHz 的 SI-SNRi／SI-SDRi（兩者同義）。「下游 WER」只填論文有報告的數字，且各列的 ASR、測試集與計分方式不同，**不可跨列比較**。

| 模型 | 年份 | 參數量 | 取樣率 | 論文報告的分離品質（dB） | 下游 WER（若有報告） | 論文報告的 CPU／GPU 成本 | Checkpoint 與授權 | 本案適用性（本報告判斷） |
|---|---|---|---|---|---|---|---|---|
| Conv-TasNet | 2019 | 5.1M〔P2〕；16 kHz 版 5.62M | 原始 8 kHz；Asteroid 有 16 kHz 版 | WSJ0-2mix 15.3〔P2〕；Asteroid 16 kHz Libri2Mix 15.24、Libri3Mix 12.30 | Not Reported | 每 1 秒 16 kHz 音訊：CPU 64.21 ms、GPU 13.17 ms、GPU 記憶體 28.78 MB | `JorisCos/ConvTasNet_Libri2Mix/Libri3Mix_sepclean_16k`，CC-BY-SA-4.0（卡片內文寫 3.0）；Asteroid 程式碼 MIT | CPU 可行的下限基準；share-alike 授權需法務確認 |
| SuDoRM-RF | 2020/21 | 6.4M〔P2〕；1.0x 版 2.72M | 8 kHz（效率表以 16 kHz 量測） | WSJ0-2mix 18.9〔P2〕；因果版 8.4–10.3 | Not Reported | 每 1 秒 16 kHz：CPU 104.32 ms、GPU 20.66 ms、24.42 MB；因果版在 i7-8750H 上每 1 秒 8 kHz 為 50.4–165.9 ms | 程式碼 MIT；16 kHz checkpoint 未取得 | CPU／串流備案，需自行訓練 |
| SepFormer | 2021 | 25.7–26.0M | 多數 checkpoint 8 kHz；僅 `sepformer-whamr16k` 為 16 kHz | WSJ0-2mix 22.3（+DM）；Libri2Mix checkpoint 20.6；whamr16k 13.5（含雜訊殘響） | Libri2Mix（16 kHz、max 模式）＋Whisper-base：17.77（讀自 SLM-SS 比較表，比較系統的訓練資料未查核） | 86.9／69.6 G MACs/s〔P2〕；CPU 數字 Not Reported | SpeechBrain，Apache-2.0；WSJ0 衍生者牽涉 LDC 條款 | 16 kHz 版分離分數低；WER 實證不利 |
| RE-SepFormer | 2022 | 8.0M | 8 kHz | WSJ0-2mix 18.6（因果 14.2） | Not Reported | 6.3 GMACs/s；256 秒輸入比 SepFormer-Light 快 7 倍 | `speechbrain/resepformer-wsj02mix`，Apache-2.0 | 輕量但只有 8 kHz |
| TF-GridNet | 2022/23 | 14.5M（16 kHz 效率表 14.43M） | 論文 8 kHz；有 16 kHz STFT 組態 | WSJ0-2mix 23.5（無 DM）；Libri2Mix 19.8〔P2〕 | WSJ0-2mix＋WavLM ASR 4.8；單通道 LibriCSS ORC-WER 6.8（Whisper large-v2，未微調）、5.6（微調聲學模型）；NOTSOFAR 單通道 tcpWER 33.5 | 每 1 秒 16 kHz：CPU 2019.60 ms、GPU 94.30 ms、491.73 MB；MACs 各論文 231–461 G/s 不一 | ESPnet checkpoint 與授權未驗證 | WER 實證最強；CPU 不可行；需自行以 16 kHz 訓練 |
| TF-CrossNet | 2024 | 未取得 | noisy Libri2Mix 實驗的取樣率未取得；SMS-WSJ 為 8 kHz | 未取得（結果表未讀） | noisy Libri2Mix（偏移版）5.0（Whisper large-v3，未微調）；單通道 SMS-WSJ 8.13／7.60／6.26 | 未取得 | 未取得 | 目前串接最佳數字的來源，但公開資源不明 |
| MossFormer2 | 2023/24 | 55.7M（S：37.8M） | 論文 8 kHz；`MossFormer2_SS_16K` 為 16 kHz | WSJ0-2mix 24.1、Libri2Mix 21.7、WHAM! 18.1、WHAMR! 17.0；16K 統一模型：LRS2-2Mix 15.5、WSJ0-2Mix 22.0、Libri2Mix 16.7、WHAM! 17.4 | Libri2Mix（16 kHz）＋Whisper-base：8.78（SLM-SS 比較表） | GPU RTF 0.053（V100、8 kHz）；MACs、CPU、16 kHz RTF 皆 Not Reported | ClearerVoice-Studio，模型與程式碼皆 Apache-2.0，未設門檻；訓練資料為「開源與私有資料」，語言 Not Reported | **第 0 階段首選**：16 kHz、可商用、附微調腳本；輸出串流數與 3 人行為 Not Reported |
| TF-Locoformer | 2024 | S 5.0M／M 15.0M／L 22.5M | 分離實驗皆 8 kHz | WSJ0-2mix M 23.6（24.6 +DM）、L 24.2（25.1）；Libri2Mix M 22.1；WHAMR! M 18.5 | Libri2Mix（16 kHz）＋Whisper-base：5.88（SLM-SS 表中最低） | L：204.3 G MACs〔P2〕；CPU Not Reported；訓練用 4 張 GPU | MERL，Apache-2.0；權重隨 ESPnet 配方放在 repo | 自行訓練 16 kHz 分離器的主要候選 |
| SepReformer | 2024 | T 3.5M 至 L 59.4M | 8 kHz | WSJ0-2mix 22.4–25.1；Libri2Mix 19.7–22.0 | Not Reported | 10.4–155.5 G MACs/s；CPU Not Reported | Apache-2.0；只釋出 B 版 WSJ0-2mix 權重 | T 版 MACs 與 Conv-TasNet 同級，值得當輕量訓練候選 |
| SPMamba | 2024 | 6.14M | 公開基準 8 kHz；Echo2Mix 為 16 kHz | WSJ0-2mix 22.5；Libri2Mix 19.9；Echo2Mix 15.3 | Not Reported | 238.69 G MACs/s | Apache-2.0；Libri2Mix 與 Echo2Mix checkpoint | 中等；無 WER 實證 |
| SepTDA | 2024 | 21.2M | 8 kHz | WSJ0-2mix／3mix 24.0／23.7；未知人數時計數正確率 99.97%／97.93% | Not Reported | Not Reported | 僅展示頁，無官方 checkpoint | 人數可變的設計參考 |
| TIGER | 2025 | 0.82M | **16 kHz** | Libri2Mix（16 kHz、train-100）16.67／17.97；EchoSet 12.58／13.73 | Not Reported | 每 1 秒 16 kHz：CPU 351.15／765.47 ms、GPU 42.38／74.51 ms、122.23 MB（Xeon Gold 6326、RTX 4090；執行緒數未取得） | Apache-2.0；`JusperLee/TIGER-speech` 及 small／tiny | **第 0 階段 CPU 候選**；殘響資料訓練 |
| SR-CorrNet | 2026 | S 7.0M／B 13.6M／L 38.1M | 公開單通道 checkpoint 為 8 kHz 訓練資料 | WSJ0-2mix 23.6／24.1／25.5；WHAMR! B 19.7 | 單通道 LibriCSS 逐句 WER 6.0–7.7（0S 至 40% 重疊，動態 0–3 串流，LibriCSS 原始 ASR） | 88.9–467.6 G MACs；RTF 0.315 是 7 通道組態在 RTX 4090 上的數字 | 程式碼 MIT；HF 模型卡無授權標籤 | 唯一內建「0–3 條串流」動態輸出的公開模型；需 16 kHz 重訓 |
| PixIT／ToTaToNet | 2024 | Not Reported（含 WavLM-large） | **16 kHz** | 未取得 | AMI-SDM cpWER 38.3→31.7、uWER 35.4→33.8（Whisper large-v2）；AliMeeting cpCER 65.1→48.3 | RTF／VRAM Not Reported | `pyannote/speech-separation-ami-1.0`，MIT，需登記同意條款 | 唯一以真實會議訓練、附活動偵測的現成管線；輸出含不需要的說話者分群 |
| Conformer CSS（NOTSOFAR baseline） | 2024 | Not Reported | 未取得 | Not Reported | 單通道 tcpWER／tcORC-WER：dev-set-1 46.8／38.5；eval 41.4／35.5 | Not Reported | 開源 baseline；資料集 CC BY 4.0 | 3 串流 CSS 的參考實作 |

表中數字出自 [SepReformer](https://arxiv.org/abs/2406.05983)、[SepFormer](https://arxiv.org/abs/2010.13154)、[RE-SepFormer](https://arxiv.org/abs/2206.09507)、[TF-GridNet](https://arxiv.org/abs/2211.12433)、[MossFormer2](https://arxiv.org/abs/2312.11825)、[TF-Locoformer](https://arxiv.org/abs/2408.03440)、[SPMamba](https://arxiv.org/abs/2404.02063)、[SepTDA](https://arxiv.org/abs/2401.12473)、[TIGER](https://arxiv.org/abs/2410.01469)、[SR-CorrNet](https://arxiv.org/abs/2603.29097)、[SuDoRM-RF](https://arxiv.org/abs/2103.02644)、[SLM-SS](https://arxiv.org/abs/2610.27785)、[PixIT](https://arxiv.org/abs/2403.02288)、[NOTSOFAR-1](https://arxiv.org/html/2401.08887) 與各模型卡（[MossFormer2_SS_16K](https://huggingface.co/alibabasglab/MossFormer2_SS_16K)、[sepformer-whamr16k](https://huggingface.co/speechbrain/sepformer-whamr16k)、[ConvTasNet 16k](https://huggingface.co/JorisCos/ConvTasNet_Libri2Mix_sepclean_16k)、[pyannote](https://huggingface.co/pyannote/speech-separation-ami-1.0)）。

這張表有三個讀法。第一，**SI-SDRi 的排名不等於 WER 的排名**：在同一個 Libri2Mix／Whisper-base 測試上，24 dB 級的 MossFormer2 是 8.78%，TF-Locoformer 是 5.88%，22 dB 級的 SepFormer 是 17.77%（[SLM-SS](https://arxiv.org/abs/2610.27785)；該表比較系統的訓練條件未查核，只能當參考）。有下游 WER 實證且數字最低的都是 TF 域模型（TF-GridNet、TF-CrossNet、TF-Locoformer、SR-CorrNet），選分離器應以下游 WER 而非 SI-SDRi 為準。第二，**能直接下載、16 kHz、授權寬鬆**的選項只有 MossFormer2_SS_16K、TIGER、sepformer-whamr16k、Asteroid ConvTasNet 16k 與 pyannote PixIT，而 WER 實證最強的那幾個模型公開 checkpoint 都是 8 kHz 或狀態不明。第三，**沒有任何一個模型同時具備強 WER 紀錄與已發表的 CPU RTF 小於 1**；TF-GridNet 在該效率表中每秒音訊要 2 秒 CPU 時間。

人數可變與單人輸入方面，attractor 式模型在合成全重疊資料上計數正確率約 98–100%（SepTDA；EEND-SS 在 Libri2Mix＋Libri3Mix 上 97.9%，[EEND-SS](https://arxiv.org/abs/2203.17068)），但筆記讀到的來源沒有任何一篇報告真實部分重疊錄音上的計數正確率。固定 2 輸出的現成 checkpoint（SepFormer、MossFormer2_SS_16K、TIGER）在讀到的文件中都沒有提供「現在只有一個人」的訊號。

## 串接對端到端的實證比較：英文合成資料上串接領先，但沒有一列是同骨幹對照

### 比較前必須先對齊的四件事

「Libri2Mix」不是單一測試集。有原始的完全重疊版與 ESPnet `sot_asr1` 的偏移版（其中一位說話者加上起始偏移），又各有 clean 與 noisy。同一個 LLM-SOT 模型在 noisy Libri2Mix test 上，偏移版是 11.0%，原始無偏移版是 17.6%；noisy Libri3Mix 是 38.9% 對 61.1%（[Shi et al. 2603.27205 表 VII](https://arxiv.org/html/2603.27205)）。跨論文比較必須先確認是哪一版。其次，下列所有表格都是**文獻表中的跨系統並列**：沒有任何一篇論文在同一個測試集上比較「同一顆 ASR 骨幹的串接」與「同一顆 ASR 骨幹的 SOT 微調」。串接列用的是遠強於端到端論文訓練預算的分離器，端到端列又多半只用 LibriMix 訓練，兩邊的資源並不對等。第三，各列的計分方式不同（asclite WER、ORC-WER、帶 oracle 指派的 cpWER），原論文自己也提出警告。第四，沒有任何一列的辨識器是 Qwen3-ASR、Parakeet、Canary 或任何 Speech-LLM。

### Noisy Libri2Mix（ESPnet 偏移版）：未微調的 Whisper 接在分離器後面是 5.0%

| 系統 | 是否有分離前端 | Dev WER | Test WER | 同一 ASR 在乾淨單人來源上的 WER |
|---|---|---|---|---|
| SOT-Conformer（ESPnet） | 否 | 24.7 | 23.3 | – |
| SOT-WavLM-Conformer（ESPnet） | 否 | 19.4 | 17.1 | – |
| GEncSep | 否 | 17.2 | 15.0 | – |
| TSE-Whisper | 是（目標說話者抽取） | – | 12.0 | – |
| WavLM＋LLM | 否 | 11.4 | 10.2 | – |
| WavLM＋LLM（微調） | 否 | 10.3 | 9.0 | – |
| PIT-SOT | 否 | – | 7.0 | – |
| TF-CrossNet → WavLM-Conformer（乾淨語音訓練） | 是 | 5.1 | 5.1 | 2.3 |
| **TF-CrossNet → Whisper large-v3（未微調）** | 是 | **5.6** | **5.0** | **2.5** |

全表出自 [Yang et al. 2025 表 1](https://arxiv.org/html/2503.17886v1)。作者的結論是串接比先前最佳的 7.0% 相對改善 25% 以上。要同時記住兩件事：5.0% 是偏移版加 WHAM! 雜訊的數字，不是完全重疊的原始版；而串接結果仍是乾淨來源的**約兩倍**（5.0 對 2.5），這個差距就是分離誤差與失真不匹配的代價。

### LibriMix 2 人與 3 人：純 SOT 在 3 人崩潰，但 3 人沒有純串接數字

| 系統 | 版本 | L2M noisy | L3M noisy | L2M clean | L3M clean |
|---|---|---|---|---|---|
| Conformer SOT | 偏移 | 17.1 | 28.2 | 7.0 | 14.7 |
| GEncSep | 偏移 | 15.0 | 25.9 | 6.6 | 13.1 |
| UME（統一 Encoder） | 無偏移 | 19.6 | 27.1 | 6.4 | 15.9 |
| Whisper-Sidecar | 無偏移 | – | – | 4.7 | 16.8 |
| SOT-LLaMA-1B | 偏移／無偏移 | 11.0／17.6 | 38.9／61.1 | 4.6／8.6 | 20.4／48.4 |
| SOP-LLaMA-3B | 偏移 | 9.2 | 28.1 | 3.6 | 16.5 |
| 分離器串流＋gated cross-attention 進 LLM（ID-20.D） | 偏移／無偏移 | 7.5／13.1 | 18.0／45.3 | 3.9／8.2 | 10.8／31.6 |
| 同上（ID-30.D） | 偏移／無偏移 | 7.7／13.5 | 17.1／42.6 | 3.1／6.9 | 8.7／30.4 |

全表為 test 集，出自 [Shi et al. 2603.27205 表 VII](https://arxiv.org/html/2603.27205)（其中他人系統的數字為該表轉引）。這張表支持一個方向性的判斷：3 位說話者同時發聲時，帶有「顯式分離」歸納偏置的設計（17–18%）遠比純 LLM-SOT（28–61%）穩。同一篇論文中，不經 LLM、只用分離分支的 CTC 解碼在 3 人 noisy 上是 23.6%，也勝過純 LLM-SOT。但這仍是整合式設計而非「波形分離器＋現成 ASR」；**筆記找不到任何 Libri3Mix 上的純串接→強 ASR 數字**，所以「3 人時串接較好」是有旁證的推測，不是已證實的結論。

### 單通道 LibriCSS（重播錄音，0–40% 重疊）：強分離器把平均壓到 5.6–6.8

| 系統 | 計分 | 0S | 0L | 10% | 20% | 30% | 40% | 平均 |
|---|---|---|---|---|---|---|---|---|
| t-SOT（LS960） | asclite WER | 5.3 | 5.4 | 6.5 | 7.3 | 9.5 | 11.3 | 7.6 |
| TS-SEP（WavLM） | cpWER，oracle 說話者指派 | 3.6 | 3.0 | 4.0 | 5.9 | 8.0 | 8.5 | 5.8 |
| TF-GridNet CSS＋mixture encoder，Transformer LM | ORC-WER | 4.5 | 3.5 | 3.6 | 5.2 | 7.1 | 8.4 | 5.6 |
| 同上＋oracle 分段 | ORC-WER | 2.9 | 2.7 | 2.7 | 3.5 | 5.4 | 6.6 | 4.1 |
| Oracle 訊號＋oracle 分段 | ORC-WER | 2.2 | 2.4 | 1.9 | 2.0 | 2.1 | 2.0 | 2.1 |
| TF-GridNet CSS → Whisper large-v2（未微調） | ORC-WER | Not Reported | Not Reported | Not Reported | Not Reported | Not Reported | Not Reported | 6.8 |
| 只做 VAD、不分離 → Whisper large-v2 | ORC-WER | – | – | – | – | – | – | 26.5 |
| 乾淨訊號 → Whisper large-v2 | ORC-WER | – | – | – | – | – | – | 3.5 |

前五列出自 [Vieting et al. 2023 表 3](https://arxiv.org/pdf/2309.08454)，後三列出自 [von Neumann et al. 2023 表 1](https://arxiv.org/html/2309.16482)。前一份報告引用的 t-SOT 9.0（TT-36、mono LibriCSS 平均）與此表的 7.6 並存，原因見上一節。三個值得帶走的觀察：分離器品質主導結果（同一顆未微調聲學模型，BLSTM 分離器 19.4 對 TF-GridNet 6.4）；換成 oracle 分段可再由 5.6 降到 4.1，表示剩餘誤差中分段／VAD 的份量不亞於分離本身；而未微調 Whisper 的 6.8 仍是乾淨訊號 3.5 的約兩倍，與 Libri2Mix 的比例一致。另有一篇把單通道 CSS 接在 SOT 模型前面的研究：串流 CT-tSOT 平均 11.52→11.38，0% 重疊變差（0S 7.33→8.76），30–40% 重疊相對改善 8–9%（[Microsoft 2025](https://arxiv.org/html/2506.14204v1)）。

### 真實會議：串接有效，但絕對錯誤率仍高，且「不計歸屬」時增益很小

CHiME-8 NOTSOFAR-1 單通道賽道的 eval 結果（[挑戰賽總結表 2](https://arxiv.org/html/2501.17304v1)）顯示，多數隊伍採「CSS → Whisper → diarization」，落在 tcpWER 30–43%、speaker-agnostic 的 tcorcWER 25.7–40.0%；冠軍 USTC（22.2／17.7）用的是 diarization 引導的神經分離、長視窗，加上大幅微調的 Whisper。**不使用任何分離器**的 BUT/JHU（diarization 條件化 Whisper）tcorcWER 為 27.2%，只輸給一個 CSS 系統。冠軍的消融把功勞拆成兩半：前端由 41.4 改善到 31.7，換上微調過的 Whisper 再到 22.2（[DCF-DS](https://arxiv.org/html/2411.06667v3)）。所有參賽的 CSS 模型都只用模擬資料訓練，以真實會議微調 CSS「未被探索」。

NOTSOFAR baseline 的數字在不同來源不同，因為量的是不同資料集：官方 README 的單通道 dev-set-1 是 tcpWER 46.8／tcORC-WER 38.5（[README](https://github.com/microsoft/NOTSOFAR1-Challenge)），挑戰賽總結的 eval 集是 41.4／35.5。另外，總結論文 HTML 版的表 6 賽道標籤疑似對調（顯示單通道 10.8、多通道 22.2），依表 2 與內文，22.2／17.7 是單通道、10.8 是多通道。

對本案最切題的真實資料數字來自 PixIT：分離後接未微調 Whisper large-v2，AMI-SDM 的 cpWER 由 38.3 降到 31.7（相對 −17.2%），但 speaker-agnostic 的 uWER 只由 35.4 降到 33.8（−4.5%），增益幾乎都來自 deletion 減少，substitution 還微升；華語 AliMeeting 的 cpCER 由 65.1 降到 48.3（[PixIT](https://arxiv.org/html/2403.02288v2)）。換句話說，在真實會議上分離帶來的 cpWER 改善大半是「歸屬」而非「辨識」，而團隊不需要歸屬。唯一與 Qwen3-ASR 有關的串接數字是 MS-Bench 的「pyannote diarization → Qwen3-ASR」，cpWER 43.25%，那是 diarization 串接而非分離串接，模型大小也未註明（[MS-Bench](https://arxiv.org/abs/2610.25824)）。

綜合四組證據：在有同表對照的兩個基準（noisy 偏移版 Libri2Mix、單通道 LibriCSS）上，強分離器接現成 ASR 不輸或勝過同表的 SOT／LLM-SOT；在 LibriCSS 約 6% 與 NOTSOFAR 25% 以上之間的落差，各論文一再歸因於分離器的模擬—真實不匹配。這些都是英文、2 人為主、朗讀語音或重播錄音的結果。

## 五種串接管線設計，各有一組已發表的數字

**一律分離（always-separate → N 次 ASR）** 是最簡單的拓樸：每段音訊都過分離器，每條串流各跑一次 ASR。它的代表數字就是上一節的 5.0%（Libri2Mix）與單通道 SMS-WSJ 的 8.13%（任務標準聲學模型）／7.60%（乾淨訓練聲學模型），未處理的混音是 79.11%（[Yang et al. 2025](https://arxiv.org/html/2503.17886v1)）。弱點是單人音訊也要付分離的代價：2020 年的單通道 BLSTM 分離器在 LibriCSS 0% 重疊反而變差（0S 11.8→12.7、0L 11.7→12.1），作者直言單通道分離「在沒有重疊時使 WER 變差」（[LibriCSS](https://arxiv.org/html/2001.11482)）。ESPnet-SE++ 提供 `enh_asr.sh` 可組合前端與 ASR，但論文中的組合配方不是 LibriMix 分離＋Whisper（[ESPnet-SE++](https://arxiv.org/html/2207.09514)）。

**重疊偵測閘控（OSD → 只分離重疊區段）** 的直接實證很薄：唯一量到的是 USTC 系統在分離器前加上重疊區段偵測後，tcpWER 由 26.68 降到 25.14，而且是**多通道**消融（[USTC 系統報告表 6](https://arxiv.org/html/2409.02041v2)）；單通道沒有任何量測。訊號層級的類比是 Sato 等人的「該不該分離」規則切換，見下一節。OSD 元件本身的準確率不高：pyannote 2.0 在 AMI 上 precision 80.7／recall 70.5／F1 75.3，作者自己說重疊偵測「仍未解決」（[Bredin & Laurent 2021](https://arxiv.org/html/2104.04045)）；較新的 WavLM 模型為 F1 82.76（[Sun et al. 2025](https://arxiv.org/html/2505.23207v1)）。後者的比較表把 pyannote 列為 recall 80.7／precision 70.5，**與 pyannote 原論文的 precision／recall 順序相反**（F1 相同）；本報告以原論文為準，但標明此衝突。`pyannote/segmentation-3.0` 每幀最多表示兩位同時說話者，3 人重疊只會被標成 2 人重疊，無法用來切換 2 輸出與 3 輸出分離器（[模型卡](https://huggingface.co/pyannote/segmentation-3.0)）。依 AMI 的 recall 約 70% 推算，閘控設計會讓約三成的重疊時間未被分離（推算，且僅限該資料域）；門檻應偏向高 recall，因為誤報只多跑一次分離，漏報則直接丟掉次要說話者。

**PixIT 聯合 diarization＋分離** 是唯一現成、以真實會議訓練、可直接跑長音檔的單通道管線：5 秒滑動視窗、500 ms 步進、每視窗最多 3 位說話者，以說話者嵌入分群拼接，並用 diarization 分支把「該說話者不活動」的時段歸零（洩漏移除）。成績即上一節的 AMI／AliMeeting 數字。它多做了團隊不需要的說話者分群，但那個活動偵測分支正是洩漏移除的來源：不含 WavLM 且不做洩漏移除時 cpWER 是 50.1（比不分離的 38.8 還差 29%），加上洩漏移除後 36.2；含 WavLM 時 34.1→32.8（[PixIT 表 5](https://arxiv.org/html/2403.02288v2)）。在 NOTSOFAR 上以 PixIT 當 CSS 的 ToTaTo 隊為 tcpWER 41.2（[Kalda et al.](https://www.isca-archive.org/chime_2024/kalda24_chime.pdf)）。

**CSS 切塊拼接** 是長錄音的標準作法：固定長度視窗（已發表系統為 2.4 秒視窗／0.8 秒步進、3 秒、4 秒視窗／3 秒位移）滑過 PIT 訓練的分離器，輸出固定條數的串流（LibriCSS 用 2 條，NOTSOFAR 用 3 條語音加 1 條雜訊遮罩），相鄰視窗以重疊幀上的最小 MSE 選排列，再做逐串流 VAD 與 ASR（[NOTSOFAR-1](https://arxiv.org/html/2401.08887)、[Vieting et al.](https://arxiv.org/pdf/2309.08454)）。成績即 LibriCSS 的 5.6–6.8 與 NOTSOFAR 的 30–43。視窗長度有實證影響：DCF-DS 把視窗由 3 秒拉到 12.8 秒，tcpWER 由 37.31 降到 34.93；主辦單位推測 3 秒視窗是 CSS 輸給長視窗系統的原因之一（[DCF-DS](https://arxiv.org/html/2411.06667v3)、[總結](https://arxiv.org/html/2501.17304v1)）。CSS 串流可能在同一條上輪流出現不同說話者，這對不需歸屬的本案無害。

**混合設計（分離器的輸出不只以波形交給 ASR）** 有三種已發表的形態。把分離串流與混音一起餵進聲學模型的 mixture encoder，只在分離器弱時有幫助（BLSTM：17.9→17.6），遇到 TF-GridNet 就沒有增益（5.8 對 5.8）（[Vieting et al.](https://arxiv.org/pdf/2309.08454)）。把分離器放進 ASR Encoder 內的 Whisper-Sidecar，LibriMix 2／3 人為 4.66／16.79（[Whisper-Sidecar](https://arxiv.org/html/2407.09817v2)）。最新的是讓 LLM decoder 以 gated cross-attention 持續讀取分離器產生的逐說話者串流：只把分離結果塞進靜態前綴幾乎沒用（noisy Libri3Mix 38.9→37.4），持續存取則是 38.9→20.5；2 人的增益較小（11.0→7.5–7.7）（[Shi et al. 2603.27205](https://arxiv.org/html/2603.27205)）。整體型態是：給辨識器混音脈絡只在分離器弱時有用；給 LLM decoder 分離後的串流，則正好在 SOT 失效的 3 人情境最有用。另外兩個有量測的混合手法是「以 ASR Encoder 嵌入當 loss 微調分離器」（WHAMR! cpWER 43.5→37.1，換到沒看過的 Whisper 上 29.3→26.8，而 SI-SDR 反而下降；[Transcription-free fine-tuning](https://ar5iv.labs.arxiv.org/html/2406.08914)）與「在分離器輸出上微調 ASR」（見失敗模式一節）。筆記搜尋過但**找不到**下列作法的已發表結果：用「分離器＋單人 ASR」的輸出當偽標籤訓練 SOT、把分離器蒸餾進 ASR Encoder、以及把分離波形與混音當雙音訊輸入餵給 Speech-LLM。

## 已知失敗模式都有名字，多數有量測過的對策

下表整理每個失敗模式的實證與對策。「未知」欄是文獻的空白，也就是第 0 階段該量的東西。

| 失敗模式 | 已發表的證據 | 有量測的對策 | 對 Qwen3-ASR／台語華語仍未知者 |
|---|---|---|---|
| 單人音訊退化、假的第二串流 | 弱分離器在 0% 重疊變差（11.8→12.7）；CSS 接 SOT 時 0S 7.33→8.76；固定 2 輸出「在單人區段產生人造的第二串流，頻譜洩漏在長錄音中累積」；強分離器在 LibriCSS 0S／0L 仍比 oracle 多 1–2 個點（4.5／3.5 對 2.2／2.4） | 以「全重疊＋單人語句」訓練分離器；attractor 動態輸出 0–3 串流（0S 6.2→6.0、0L 6.2→5.8）；重疊閘控或直通路徑 | 沒有任何來源量過**乾淨、近講、單人**音訊通過 SepFormer／MossFormer2／TIGER 後的 WER；LibriCSS 是遠場，分離器的去殘響效果混在裡面（SR-CorrNet 固定 2 輸出反而把 0S 由 11.8 改善到 6.2） |
| 高 SIR 時分離反而有害 | 日語 TSE→ASR：SIR 0 dB 時 CER 80.7→8.4；SIR 20 dB 時 5.3→6.3（SNR 20 dB） | 規則切換（估計 SIR−SNR ≥ 10 dB 時改用混音）平均 CER 相對降 22%、最多 27%；神經網路切換的數字未驗證 | 該實驗是目標說話者抽取，不是盲分離；盲分離下「加回混音」會把另一位說話者灌回每條串流，沒有量測 |
| 失真（artifact）與 ASR 不匹配 | SDR 上升但 WER 不降；artifact 成分比殘留雜訊更傷 ASR；聯合微調後 SDR 26.43→15.28 而 WER 3.2→1.7 | 在分離器輸出上微調 ASR（LibriCSS 6.4→5.8；SMS-WSJ 7.60→6.26）；ASR 感知的分離器 loss（43.5→37.1）；使用 SSL／強健特徵的 ASR（同一分離音訊 2.4 對 filterbank 的 28.2） | LLM 式 ASR 對 artifact 的敏感度沒有任何量測；也有研究主張分離器夠強時乾淨訓練的後端較好 |
| 洩漏與重複字 | 過度分離（單人被抹到多條串流）最常見，欠分離難以後處理；PixIT 不做洩漏移除時 insertion 15.6% | 兩步驟能量式語音存在機率遮罩＋刪除跨串流洩漏字；diarization 活動歸零（insertion 15.6→1.7；含 WavLM 3.8→1.2）；強分離器下洩漏很小（LibriCSS 兩人區段跨通道重合率 0.3%） | 不計說話者情境下的跨串流去重，除了「依起始時間排序串接」外沒有已發表演算法；真實會議的重複字比例未見報告 |
| 殘留／空串流上的 ASR 幻覺 | Whisper 在非語音片段上 40.3% 產生幻覺文字；NOTSOFAR 隊伍指出必須先對 CSS 輸出做 VAD | 對 CSS 輸出做 VAD（tcorcWER 37.6→36.1）、合併短片段（→33.0）；活動閘控 | Qwen3-ASR 在分離器殘留串流上的幻覺／insertion 率沒有任何量測（詳下） |
| 8 kHz 對 16 kHz | SpeechBrain 的 SepFormer 系列除 whamr16k 外皆要求 8 kHz 輸入；多數 SOTA 釋出為 8 kHz | 使用 16 kHz 原生分離器或以 16 kHz 重訓 | 「8 kHz 分離後再給 16 kHz ASR」的 WER 代價沒有任何來源量過 |
| 英文訓練的分離器跨語言退化 | Libri2Mix 訓練的 Conv-TasNet：域內 11.98 → 華語 Aishell2Mix 2.08 dB；DPCCN 13.04→5.09；英文訓練模型在 CommonVoice zh-TW 上 5.27 dB（英文口音約 8–9 dB） | 域內訓練把 Aishell2Mix 拉回 8.32／8.14 dB；多語訓練改善未見語言（未驗證） | 2023 年後的 TF 域或大資料模型是否縮小落差未測；台語完全沒有數字；沒有以 WER／CER 量測的語言不匹配研究 |
| 超過 2 位說話者 | CSS 假設短視窗內同時說話者不超過輸出條數；純 LLM-SOT 在 3 人崩潰 | 3 條輸出（NOTSOFAR）；Kmax=3（PixIT，5 秒視窗內超過 3 人的機率小於 1%）；attractor 計數；diarization 綁定輸出 | 3 輸出單通道 CSS 在 LibriCSS 類基準上沒有證據；MossFormer2_SS_16K 的 3 人行為 Not Reported |
| 切塊排列錯誤 | 相鄰視窗輸出順序不一致 | 重疊幀最小 MSE 排列對齊；說話者嵌入分群（PixIT）；diarization 綁定輸出（DCF-DS／TS-SEP） | 句中串流互換把一句話切到兩次 ASR 的影響，沒有論文單獨量化（WHAMR! 上 cpWER 與 ORC-WER 幾乎相同，29.5 對 29.4，顯示逐句分離時互換很少） |

證據出處依序為 [LibriCSS](https://arxiv.org/html/2001.11482)、[Microsoft 2025](https://arxiv.org/html/2506.14204v1)、[SR-CorrNet](https://arxiv.org/abs/2603.29097)、[Vieting et al.](https://arxiv.org/pdf/2309.08454)、[Sato et al. 2021](https://arxiv.org/pdf/2106.00949.pdf)、[Iwamoto et al. 2022](https://ar5iv.labs.arxiv.org/html/2201.06685)、[Masuyama et al. 2023](https://ar5iv.labs.arxiv.org/html/2307.12231)、[Yang et al. 2025](https://arxiv.org/html/2503.17886v1)、[Hu et al.（NJU-AALab）](https://www.isca-archive.org/chime_2024/hu24_chime.pdf)、[PixIT](https://arxiv.org/html/2403.02288v2)、[Whisper 非語音幻覺](https://arxiv.org/html/2501.11378v1)、[Heterogeneous Separation Consistency Training](https://arxiv.org/abs/2204.11032)、[Huang et al. 2022](https://ar5iv.labs.arxiv.org/html/2203.05882) 與 [DCF-DS](https://arxiv.org/html/2411.06667v3)。

有三點需要展開。

關於 **Qwen3-ASR 的 `language None` 行為**：技術報告說明 SFT 階段「也使用非語音資料」，並定義「未偵測到語音」的輸出樣式為 `language None<asr_text>` 加空白文字；之後的 RL 階段有 30% 的「功能性資料」，目標是複雜環境下的轉寫穩定度（[Qwen3-ASR 技術報告](https://arxiv.org/html/2601.21337)）。這讓 Qwen3-ASR 在**真正的非語音**輸入上有一道內建防線，這一點可能優於 Whisper。但洩漏的分離器產生的不是非語音，而是**低音量的殘留語音**；模型在這種輸入上會輸出空白、輸出另一位說話者的片段，還是生成流暢的幻覺，文件沒有說明，也沒有任何量測。「LLM decoder 至少與 Whisper 一樣容易在殘留串流上生成流暢文字」是合理的擔憂，但只是推論。所以逐串流的活動閘控（能量 VAD 或分離器／分段模型的活動輸出）仍應是第一道防線，`language None` 只能當第二道，其可靠度要在第 0 階段直接量。

關於 **「加回混音」（observation adding）**：它在降噪上有乾淨的證據（提高 SAR、改善 WER，但該論文的 WER 只以圖呈現，沒有表格數字），在目標說話者抽取上有切換規則的證據，但在盲 2 串流分離上沒有任何量測，而且有明顯的副作用。不應未經實驗就列入管線。

關於**跨語言**：上表兩組數字是目前僅有的受控證據，方向一致地不利於「分離與語言無關」的直覺。作者的原話是「語言不匹配造成的分離退化遠比聲學殘響嚴重」，不過該測試同時也換了說話者與錄音條件，語言因素並未被單獨隔離。另一篇 JASA 2021 的研究據稱顯示英文訓練的網路可泛化到華語（未驗證，原文無法開啟），與此在強調點上相左。正面的旁證是 PixIT：分離器在華語 AliMeeting 域內資料上訓練時，即使 WavLM 是英文預訓練，仍讓未調適的多語 Whisper 相對改善 21–26%。聲調語言對分離失真（音高輪廓扭曲）是否特別敏感，是合理的疑問，文獻沒有量測。

## 用自有單人語料訓練分離器：配方成熟，但沒有人為華語或台語寫過微調紀錄

團隊沒有重疊語料並不構成障礙，因為監督式分離訓練本來就是拿單人語音動態混音。標準配方的資料量在數十到兩百小時之間：wsj0-2mix 訓練集 30 小時；Libri2Mix train-360 為 212 小時混音（來源 LibriSpeech train-clean-360 為 364 小時、921 位說話者），train-100 為 58 小時；LibriCSS 的單通道分離器用 206 小時模擬資料；NOTSOFAR 的 CSS 用約 1,000 小時、以 15,000 個真實聲學轉移函數模擬的資料（[LibriMix](https://arxiv.org/html/2005.11262)、[LibriCSS](https://arxiv.org/html/2001.11482)、[NOTSOFAR-1](https://arxiv.org/html/2401.08887)）。TIGER 的 EchoSet 訓練集是 20,268 句 6 秒語句（約 33.8 小時，推算），而以它訓練的模型在真實錄音上優於以 Libri2Mix 或 LRS2-2Mix 訓練者（[TIGER](https://arxiv.org/html/2410.01469)）。動態混音（on-the-fly 產生新混音，可加 95–105% 速度擾動）是標準增強，SepFormer 與 TF-Locoformer 都報告明顯增益；TF-GridNet 則不靠它就達到 23.5 dB。

運算成本方面，論文給的是片段而非總量。SepFormer（26M）以 batch size 1 在單張 V100-32GB 上訓練，一天後才超過 17 dB；SpeechBrain 配方每個 epoch 約 2 小時（單張 V100，動態混音）；MossFormer2（55.7M）同樣單張 V100、最多 200 epochs；TF-GridNet 最大組態在 A100-40GB 上每 epoch 151.4 分鐘（20,000 段 4 秒片段），反向傳播時每段 4 秒片段占 13,663 MB；TF-Locoformer 用 4 張 GPU（[SepFormer](https://arxiv.org/html/2010.13154)、[SpeechBrain 配方](https://github.com/speechbrain/speechbrain/blob/develop/recipes/WSJ0Mix/separation/README.md)、[MossFormer2](https://arxiv.org/html/2312.11825)、[TF-GridNet](https://arxiv.org/html/2211.12433)、[TF-Locoformer repo](https://github.com/merlresearch/tf-locoformer)）。把「每 epoch 約 2 小時」乘上 200 epochs，從頭訓練一個 WSJ0-2mix 規模的模型約為 400 GPU 小時；這個乘積是筆記的**推算**，沒有論文公布總時數。TF-GridNet、MossFormer2、TF-Locoformer 的總訓練天數皆 Not Reported。

微調現成 checkpoint 在工具上是支援的：ClearerVoice-Studio 提供 8 kHz 與 16 kHz 分離的訓練與微調腳本，以 scp 清單指定混音與來源路徑，並可從既有 checkpoint 續訓（[ClearerVoice 訓練說明](https://github.com/modelscope/ClearerVoice-Studio/blob/main/train/speech_separation/README.md)）；為領域偏移而微調也有先例（TF-CrossNet 以較低學習率在 4 倍大、條件更廣的資料上續訓）。完整公開訓練程式碼且授權寬鬆的還有 TF-Locoformer（ESPnet 配方，Apache-2.0）、SR-CorrNet-SS（MIT）與 TIGER（Apache-2.0）。但**沒有任何論文或 repo 記錄過把 SepFormer、TF-GridNet、MossFormer2 或 TF-Locoformer 從英文微調到華語或台語**；需要多少小時、會不會遺忘英文、要不要混入英文資料，都沒有文件。域內訓練有效的唯一數字是 Aishell2Mix 上由 2.08 回到 8.32 dB。

若日後取得真實重疊錄音，半監督方法是有實證的出路。MixIT 只需要單通道混音，加入 10% 的同域無監督資料就在不匹配條件下多 2–3 dB（[MixIT](https://arxiv.org/html/2006.12701)）；在真實 AMI 會議上，「殘響 Libri2Mix 的 PIT＋AMI 的 MixIT」的半監督微調勝過用多幾個數量級資料訓練的通用模型，而 AMI 只有數十小時（[Sivaraman et al.](https://arxiv.org/html/2110.10739)）。PixIT 只需要帶 diarization 標註的單通道錄音，不需要乾淨來源，但混合的兩個片段必須來自**同一場錄音**，否則模型會利用錄音條件差異取巧。Self-Remixing 可在無監督下精煉域外預訓練的分離器（[Self-Remixing](https://arxiv.org/html/2211.10194)）。

把團隊的 ASR 語料拿來當混音來源時，文獻留下三個提醒（以下是筆記依文獻所做的設計推論）。來源語音本身的雜訊與殘響會成為分離「目標」的一部分，分離器會學著把它們重現出來，這對 ASR 可以接受。含背景人聲的語句會污染目標，NOTSOFAR 在模擬前先以 MOS 篩選 Librivox 是已發表的先例。把兩個錄音條件不同的語料互混，等於給模型一條「靠錄音條件分辨」的捷徑；台語＋華語的跨語料混音正好踩在這裡，所以訓練混音必須同時包含同語料配對與跨語料配對，並對錄音條件做增強。訓練混音的音量差也應比 WSJ0-2mix 的 0–5 dB 更寬，因為真實 AMI 重疊中兩個來源的平均輸入 SI-SNR 是 0.5 dB 與 −9.2 dB。跨語言配對對單通道分離器是較易還是較難，沒有任何來源報告；來源語料 SNR 對串接下游 WER 的影響，同樣沒有研究。

## 部署成本：辨識器不是瓶頸，分離器才是，而 CPU 是分水嶺

以下只使用已發表的數字。辨識器端，Qwen3-ASR-0.6B 在 vLLM v0.14.0、bfloat16、約 2 分鐘音訊、未具名 GPU 上的 offline RTF：concurrency 1 為 0.00923（每秒處理 108.34 秒音訊），concurrency 2 為 0.01124（177.94），concurrency 4 為 0.01284（311.53）（[Qwen3-ASR 技術報告表 2](https://arxiv.org/html/2601.21337)）。由此**推算**：concurrency 2 時每秒可處理約 89 秒的雙串流混音，concurrency 4 時約 104 秒的三串流混音；也就是說，多跑一兩條串流只讓 ASR 端的 RTF 由約 0.009 變成 0.011–0.013。官方沒有 VRAM 數字；唯一的一手資料是使用者貼出的 vLLM 啟動紀錄（模型載入 1.53 GiB），見前一份報告。官方也沒有 CPU RTF；社群 C 移植版在 DGX Spark 的 CUDA 版上，0.6B 轉寫 45.0 秒音訊需 0.626 秒（[antirez/qwen-asr](https://github.com/antirez/qwen-asr)），前一份報告另記有 Apple M3 Max 上 offline 約 7–13 倍即時。

分離器端的成本橫跨兩個數量級。MossFormer2 的 GPU RTF 為 0.053（55.7M、V100、8 kHz WSJ0-2mix 測試集），16 kHz 版的 RTF Not Reported。同一張效率表（每 1 秒 16 kHz 音訊；RTX 4090 與 Xeon Gold 6326；執行緒數未取得，因此是相對比較而非可部署的 RTF）中：Conv-TasNet 為 CPU 64.21 ms／GPU 13.17 ms／28.78 MB；SuDoRM-RF 104.32／20.66／24.42；TIGER small 351.15／42.38／122.23；TIGER large 765.47／74.51／122.23；BSRNN 897.27／78.27／130.24；TF-GridNet 2019.60／94.30／491.73（[TIGER 表 3](https://arxiv.org/html/2410.01469)）。TF-GridNet 最大組態對 4 秒片段的前傳占 4,918.8 MB。PixIT 管線（WavLM-large＋DPRNN、5 秒視窗 500 ms 步進，等於每個時間點被約 10 個視窗處理，再加嵌入抽取與分群）的 RTF 與 VRAM 皆 Not Reported。因果／串流分離器要付出數 dB 的代價：因果 SepMamba 為 19.2／21.4 dB，因果 SkiM 在老舊 1.9 GHz CPU 單執行緒下 RTF 0.23（[SepMamba](https://arxiv.org/abs/2410.20997)、[SkiM](https://arxiv.org/abs/2201.10800)）。

| 項目 | 串接（分離器＋N 次 ASR） | 單次 SOT 前傳 |
|---|---|---|
| ASR 端 RTF（已發表） | 0.01124（2 串流並行）；0.01284（4 並行） | 0.00923（基礎模型數字；沒有 SOT 版 Qwen3-ASR 的計時） |
| 額外模型成本（已發表） | GPU：MossFormer2 0.053（V100、8 kHz）；TF-GridNet 每秒音訊 94.30 ms（RTX 4090）。CPU：Conv-TasNet 64.21 ms、TIGER small 351.15 ms、TF-GridNet 2019.60 ms | 無 |
| 參數量 | 分離器 0.82M（TIGER）至 55.7M（MossFormer2）另計 | 不增加 |
| VRAM | 分離器 24–492 MB（1 秒輸入）至 4.9 GB（TF-GridNet、4 秒片段）；Qwen3-ASR 官方 Not Reported | 官方 Not Reported |
| 串流 | CSS 固有延遲 0.8–1.2 秒；Qwen3-ASR 串流不支援批次，N 條串流需 N 個並行工作階段 | 前一份報告已指出 utterance-level SOT 與 Qwen3-ASR 的串流機制不相容 |
| 其他系統的單次 SOT 計時 | – | LLaMA-1B SOT：RTF 0.1150（H100、batch 1；不同系統，見前一份報告） |

把不同硬體上的 RTF 相加並不嚴謹，但數量級已足以下判斷（以下為推論）：GPU 上串接的總成本約等於分離器的成本，即使用 MossFormer2 級的分離器，整體仍遠快於即時；CPU 上只有 Conv-TasNet／SuDoRM-RF／TIGER small 這一級跑得動，而它們沒有任何下游 WER 紀錄，分離分數也明顯較低；TF-GridNet 級的品質在 CPU 上不可行。重疊閘控可把分離器的工作週期降到重疊比例（AMI 約 19%、AliMeeting 約 42%），代價是 OSD 的漏報。串流方面，端到端延遲至少是 CSS 的前視（已發表 0.4–1.2 秒）加上 Qwen3-ASR 的 2 秒切塊，沒有任何已發表系統示範過這個組合。若 CPU 部署是硬需求，這一節是 SOT 路線最有力的論點。

## 修訂後的路線排序與分階段實驗計畫

### 排序：兩條主線並列，其餘依觸發條件啟動

下表各欄是依文獻證據所做的判斷，不是量測值。

| 順位 | 路線 | 工程難度 | 訓練成本 | 預期改善能力 | 單人／三語退化風險 | 推論成本 | 實證可靠度 |
|---|---|---|---|---|---|---|---|
| 0 | 第 0 階段：現成 16 kHz 分離器＋未改動的 Qwen3-ASR；baseline；Encoder 診斷 | 低–中 | 極低 | 不改善，決定後續投資 | 無 | – | 量測本身就是文獻缺口 |
| 1（並列） | **串接**：自有語料訓練／微調的 16 kHz 分離器 → Qwen3-ASR，含逐串流活動閘控 | 中（第二顆模型、拼接與合併邏輯） | 中（只訓練分離器） | 2 人高；3 人有旁證 | 低：ASR 權重不動；風險在單人音訊被分離器處理 | GPU 低；CPU 只容許輕量分離器 | 英文合成／重播強；真實會議弱；Qwen3-ASR 與台語無 |
| 1（並列） | **SOT＋LoRA（Encoder＋projector＋LLM）＋三語單人 replay** | 低（資料工程為主） | 中 | 2 人高；3 人有限 | 中：多語遺忘與單人退化有實測反例 | 與單人相同，最適合 CPU | LLM-SOT 多篇，但無 Qwen3-ASR |
| 2 | 串接的加強件：重疊閘控／直通路徑 | 中 | 低（可用現成 OSD） | 保護單人、降低成本 | 降低 | 降低 | 薄（一個多通道消融；OSD F1 75–83） |
| 3 | 串接的加強件：以 LoRA 在分離器輸出上做匹配微調（只在分離串流上啟用） | 中 | 中 | 相對 7–18%（傳統／Whisper 後端） | 低（若 adapter 可關閉；此為設計假設） | 約 0 | 中；Speech-LLM 上無 |
| 4 | 混合：分離器串流以 cross-attention 交給 LLM decoder | 高 | 高 | 3 人大（38.9→20.5）；2 人中 | 低–中 | 分離器＋額外注意力層（前一份報告記為 +64M、延遲約 +55%） | 一組論文（WavLM＋LLaMA、LibriMix） |
| 5 | PixIT 式聯合 diarization＋分離（有真實錄音時） | 中 | 中（需 diarization 標註） | 真實長錄音 | 低 | 未公布 | 真實會議上最直接，但 uWER 增益小 |
| 6 | 凍結 Encoder 上的 separator＋per-talker CTC／SOP | 中–高 | 中–高 | 2 人小、3 人大 | 低 | 小 | 中 |
| 7 | t-SOT、第二顆 Encoder、Sidecar／PIT 多分支 | 中–高 | 中 | 視需求 | 低–未知 | 各異 | 見前一份報告 |
| 8 | Diarization 條件化（Parakeet／DiCoW 類） | 高 | 高 | 外部 baseline | 低 | diarizer＋N 次解碼 | 強，但超出需求 |

兩條主線並列而非分先後，是因為它們的風險互補，而文獻無法替 Qwen3-ASR 與台語分出高下。串接把風險集中在分離器（語言不匹配、模擬—真實落差、成本），ASR 完全不動；SOT 把風險集中在 ASR 本身（Encoder 是否留著第二人、遺忘、3 人退化），部署最簡單。第 6–8 順位相對於前一份報告的位置沒有變，只是整體後移。

### 第 0 階段：不需訓練的量測，加上前一份報告的診斷

測試集沿用前一份報告的設計：以各語言**未被用作訓練混音來源**的單人測試語句合成，語言配對涵蓋台語＋台語、華語＋華語、英語＋英語與三種跨語言配對（台語＋華語為重點），格點為重疊比例（0／10／20／40／100%）、SIR（0、±5、±10 dB）與人數（1／2／3）。短語句的全重疊集對應 Libri2Mix，長錄音的部分重疊集對應 LibriCSS。

無訓練部分跑四組系統。第一，原始 Qwen3-ASR-0.6B 直接吃混音，這是所有後續實驗的 baseline，也回答它是「只轉寫主導者」還是混疊。第二，Qwen3-ASR-0.6B 吃混音前的乾淨來源，這是串接的 oracle 上限，對應文獻中的 2.5（Libri2Mix）與 3.5（LibriCSS）。第三，現成 16 kHz 分離器接未改動的 Qwen3-ASR-0.6B：MossFormer2_SS_16K、TIGER、Asteroid ConvTasNet 16k（2 人與 3 人版）、sepformer-whamr16k，以及 pyannote PixIT 管線；先不加任何後處理，再加上逐串流能量 VAD 重跑一次，兩者的差就是閘控的價值。第四，同一批分離器吃**單人**音訊，量單人 WER 的變化、第二串流輸出非空白文字的比例與其 insertion 字數，以及 Qwen3-ASR 回傳 `language None` 的比例。因為測試集是合成的，還可以順手算出每個分離器在各語言配對上的 SI-SDRi，直接補上「英文分離器遇到台語與跨語言混音」的文獻空白。若人力允許，把同一批分離音訊也餵給 Whisper large-v3，可以分辨「分離器不行」與「Qwen3-ASR 對失真特別敏感」。

診斷部分照前一份報告執行（基礎權重不動的短訓練）：凍結消融（只訓練 LLM LoRA、再加 projector、再加 Encoder LoRA）與分層 per-talker CTC read-out，回答 AuT 是否留著第二位說話者。

### 決策規則

以下規則是設計建議；門檻數值應由團隊依產品需求設定，文獻中可參考的比例是「串接約為 oracle 的兩倍」。

若現成分離器在台語／華語／跨語言混音上已把 baseline 與 oracle 之間的差距收回大半，且加上活動閘控後單人退化很小，則以串接為主線；第 1 階段做重疊閘控、合併與去重邏輯，以及以自有語料微調分離器，SOT 降為對照組。若串接在英語配對上有效、在台語或跨語言配對上明顯變差，而分離器的 SI-SDRi 也同步下降，問題是分離器的語言不匹配；第 1 階段改為以三語單人語料訓練 16 kHz 分離器（TF-Locoformer 或 MossFormer2 微調），訓練完重跑第 0 階段的表，同時啟動 SOT 主實驗作為並行賭注。若分離器的 SI-SDRi 不差、Whisper 在同一批音訊上也正常，唯獨 Qwen3-ASR 的 WER 高或在殘留串流上大量幻覺，問題是 Speech-LLM 對失真的敏感度；此時優先嘗試匹配微調（僅在分離串流上啟用的 LoRA），若無效則轉向 SOT。

若 Encoder 診斷顯示「LLM LoRA＋projector」已接近「含 Encoder LoRA」的成績，或頂層 read-out 良好，表示 SOT 的訓練成本與遺忘風險都低，SOT 升為主線，尤其在 CPU 部署為硬需求時。若診斷顯示必須大幅調適 Encoder，而串接已可用，則串接為主線，SOT 的 Encoder 調適降為研究項目。

混合路線有兩個觸發條件。當串接在重疊區有效、但單人退化無法靠活動閘控壓下來時，採重疊閘控（非重疊區走未改動的 Qwen3-ASR）。當 3 人重疊成為真實需求、SOT 在 2 人可用而 3 人崩潰時，才投資「分離器串流交給 LLM decoder」；這是工程最重的一項。兩條主線都可用時，以三項比較決勝：未見過的單人回歸測試、跨語言配對的次要說話者刪除率，以及在同一硬體上自行量測的 RTF 與 VRAM。

### 第 1、第 2 階段

第 1 階段是兩條主線各做一個最小可行版本，在同一份測試集上對打。串接線：以三語單人語料動態混音訓練或微調一個 16 kHz 分離器（含單人語句與寬音量差，並同時涵蓋同語料與跨語料配對），加上 CSS 切塊拼接、逐串流活動閘控，以及以 Qwen3-ForcedAligner 時間戳做「依起始時間排序」的合併。SOT 線：依前一份報告的配方（先驗證微調腳本的遮蔽問題、分隔符選擇、Encoder LoRA、三語單人 replay）。這個階段第一次能提供文獻缺席的「同一顆 ASR 骨幹、同一測試集」的串接對 SOT 比較。

第 2 階段依第 1 階段的結果擇一深化：串接線做匹配微調與重疊閘控；SOT 線做多任務輔助與可切換 adapter；兩者皆可延伸到 3 人。取得真實台語／華語重疊錄音後，串接線以 MixIT／PixIT 式半監督調適分離器，SOT 線以少量真實資料微調。「以串接輸出當偽標籤，為真實重疊錄音產生 SOT 訓練目標」是把兩條線接起來的自然作法，但文獻中沒有先例，屬研究假設。

### 評測協定

主指標是 speaker-agnostic 的錯誤率，讓串接與 SOT 能放在同一把尺上。以 MeetEval 計 ORC-WER（長錄音有時間戳時用 tcORC-WER）：串接的每條輸出串流、SOT 以分隔符切開的每一段，都當作一條假設串流；ORC-WER 不懲罰「哪一句落在哪一條串流」（[MeetEval](https://github.com/fgnt/meeteval)）。SOT 的輸出若當作單一串流，前一份報告已說明應以 MIMO-WER 交叉檢查；短的 2–3 人混音可另報最小排列 WER，以取得逐串流的 insertion／deletion／substitution。cpWER 只當拼接品質的診斷，因為它額外懲罰串流互換，而且新舊定義對多餘串流的懲罰不同，跨論文數字不可直接比。華語與漢字台語以字、英文以詞為單位，計分前統一正規化。

必報的分項有六個：insertion／deletion／substitution 分開（分離主要減少 deletion，洩漏表現為 insertion）；依重疊比例分層並**包含 0% 重疊**；依 SIR 分層的次要說話者刪除率；依語言配對分層；單人輸入時的假第二串流比例；以及各語言未見過資料上的單人回歸。每張表都附上 baseline 與 oracle 兩列。分離器另報各語言配對的 SI-SDRi，但只當診斷，選型以下游錯誤率為準。RTF、VRAM 與 CPU 計時一律在同一台機器上自行量測，因為已發表數字的硬體各不相同。最終驗收保留一份真實的台語／華語重疊錄音，因為合成訓練在同法合成的測試集上會顯得特別好看。

## 未解決的研究問題

### 已被論文實證的結論

- 在 noisy Libri2Mix（ESPnet 偏移版）上，強 TF 域分離器接未微調的 Whisper large-v3 為 5.0%，低於同表所有單次前傳系統（7.0–23.3%），但仍為乾淨來源（2.5%）的兩倍（[Yang et al. 2025](https://arxiv.org/html/2503.17886v1)）。
- 在單通道 LibriCSS 上，TF-GridNet CSS 接未微調 ASR 為 6.4–6.8% ORC-WER，不分離為 26.5%；分離器品質主導結果（BLSTM 19.4 對 TF-GridNet 6.4），oracle 分段可再降到 4.1（[Vieting et al.](https://arxiv.org/pdf/2309.08454)、[von Neumann et al.](https://arxiv.org/html/2309.16482)）。
- 同一模型在偏移版與無偏移版 Libri2Mix／Libri3Mix 上的 WER 差距很大（11.0 對 17.6；38.9 對 61.1），跨論文比較必須先對齊版本（[Shi et al. 2603.27205](https://arxiv.org/html/2603.27205)）。
- 純 LLM-SOT 在 3 人混音上大幅退化；讓 LLM decoder 持續讀取分離後串流可把 noisy Libri3Mix 由 38.9 降到 20.5。
- SI-SDR 與 WER 會脫鉤：聯合微調使 SDR 下降而 WER 改善；ASR 感知的分離器 loss 使 SI-SDR 略降而 cpWER 由 43.5 降到 37.1（[Masuyama et al.](https://ar5iv.labs.arxiv.org/html/2307.12231)、[2406.08914](https://ar5iv.labs.arxiv.org/html/2406.08914)）。
- 目標已佔優勢（高 SIR）時，抽取後的訊號辨識率不如原混音（[Sato et al. 2021](https://arxiv.org/pdf/2106.00949.pdf)）。
- 弱的單通道分離器在無重疊語音上使 WER 變差；固定 2 輸出的分離器在單人區段會產生人造的第二串流（[LibriCSS](https://arxiv.org/html/2001.11482)、[SR-CorrNet](https://arxiv.org/abs/2603.29097)）。
- 未加活動閘控的洩漏可把改善變成惡化（PixIT：cpWER 50.1 對不分離的 38.8；洩漏移除後 36.2）。
- 在分離器輸出上微調聲學模型有一致的增益（LibriCSS 6.4→5.8；SMS-WSJ 7.60→6.26），但後端會綁定該分離器。
- 英文訓練的分離器在華語上大幅退化，域內訓練可補回（Conv-TasNet：11.98→2.08→8.32 dB）（[HSCT](https://arxiv.org/abs/2204.11032)）。
- 真實會議上串接的絕對錯誤率仍高（NOTSOFAR 單通道 tcorcWER 17.7–40.0%），且對不計歸屬的錯誤率增益小（AMI-SDM uWER 35.4→33.8）；冠軍系統的前端與 ASR 微調各貢獻約一半。
- Qwen3-ASR-0.6B 多跑一兩條串流的成本很低（concurrency 1 與 2 的 RTF 為 0.00923 與 0.01124）；分離器成本橫跨兩個數量級，TF-GridNet 級在 CPU 上慢於即時。

### 有實驗支持但尚未確認能移植到 Qwen3-ASR 或台語華語的方法

- 「強分離器＋未微調 ASR」的串接：只在 Whisper、WavLM-Conformer 與混合式聲學模型上驗證，全為英文（PixIT 另有華語 AliMeeting）。
- 現成 16 kHz 分離器（MossFormer2_SS_16K、TIGER 等）當前端：有分離分數，但除 MossFormer2 的一筆 Whisper-base 數字外沒有下游 WER，也沒有任何華語／台語測試。
- 以自有語料域內訓練分離器：Aishell2Mix 上有效，台語與跨語言配對未測。
- 逐串流活動閘控、能量式語音存在機率遮罩、對 CSS 輸出做 VAD、合併短片段：在 Whisper 管線上各有量測增益。
- 重疊偵測閘控：一個多通道消融（26.68→25.14）；單通道與台語上的 OSD 準確率未知。
- 在分離器輸出上匹配微調 ASR：傳統聲學模型與 Whisper 上成立，Speech-LLM 與 LoRA 形式無先例。
- 以 ASR Encoder 嵌入微調分離器：WHAMR! 上成立並可轉移到未見過的 Whisper；以 AuT 當嵌入來源未試。
- Attractor 式動態串流數（0–3 條）：合成資料與 LibriCSS 上成立，公開的單通道 checkpoint 為 8 kHz。
- 分離器串流經 cross-attention 交給 LLM decoder：WavLM＋LLaMA、LibriMix 上成立。
- MixIT／PixIT／Self-Remixing 的半監督真實資料調適：AMI、AliMeeting 上成立。
- 前一份報告列出的端到端方法（SOT＋LoRA、Encoder LoRA、per-talker CTC、SOP、t-SOT 等）仍屬此類。

### 純粹的研究假設

- Qwen3-ASR-0.6B 接在分離器後面，能像 Whisper 一樣拿到接近 oracle 兩倍的錯誤率；沒有任何 Speech-LLM 的量測。
- Qwen3-ASR 的 `language None` 空白輸出能可靠地處理分離器的殘留串流；文件只涵蓋非語音輸入。
- 英文或來源不明的現成分離器在台語、華語與台語＋華語跨語言混音上仍然可用；跨語言配對比同語言配對容易（或困難）。
- 聲調語言對分離失真特別敏感。
- 串接在 3 人重疊上優於 SOT；目前只有整合式設計的旁證，沒有純串接的 Libri3Mix 數字。
- 8 kHz 分離器的輸出升頻後餵給 16 kHz 的 Qwen3-ASR，其損失可以接受。
- 只在分離串流上啟用的 LoRA 可以兼顧匹配增益與單人能力不變。
- 以起始時間排序加上時間對齊的重複字抑制，足以合併逐串流轉寫；沒有已發表的去重演算法。
- 以串接輸出為偽標籤訓練 SOT、或把分離器蒸餾進 AuT；搜尋後沒有找到任何已發表結果。
- 跨語料混音（台語語料＋華語語料）不會讓分離器學到「靠錄音條件分辨」的捷徑。
- 輕量分離器（Conv-TasNet／TIGER 級）接 Qwen3-ASR 能在 CPU 上達到可用的準確率與速度；這一級分離器沒有任何下游 WER 紀錄。

### 本次調研未能取得的資料

TF-CrossNet／CrossNet、DiffSep 與 EDSep 的結果表未開啟，其參數量與分離分數不在本報告中。ESPnet 的 TF-GridNet 預訓練 checkpoint（取樣率、授權）未能驗證，ESPnet `egs2/librimix/enh_asr1` 的說明頁回傳 404；NVIDIA NeMo 中沒有確認到任何通用單通道分離 checkpoint。pyannote 的分離模型卡設有門檻，每塊最多說話者數等細節只有搜尋摘要層級（管線卡的 16 kHz 單聲道輸入已直接讀到）。Healy et al.（JASA 2021）與 GlobalPhone 多語混音資料庫的原文無法開啟，只有摘要層級，本報告未採為依據。Iwamoto et al. 的 WER 只以圖呈現；Sato 等人的後續神經切換論文、SepALM、以及「LLM 老師蒸餾進純 Encoder」論文的數字未讀到。NPU-TEA 與 Fano Labs 的 NOTSOFAR 系統細節未讀。LDC 對 WSJ0 的條款只到摘要層級（非會員限非商業研究用途，未驗證），WSJ0 專屬授權文字、AMI 語料與 WHAM! 雜訊的授權條款皆未查核；`MossFormer2_SS_16K` 上游訓練資料的授權 Not Reported。因此授權面的結論只有三點可以確定：程式碼與多數模型卡為 Apache-2.0 或 MIT；pyannote 模型為 MIT 但需登記；Asteroid 的 LibriMix ConvTasNet 為 CC-BY-SA-4.0、Sortformer v1 為 CC-BY-NC-4.0（[HF API](https://huggingface.co/api/models/nvidia/diar_sortformer_4spk-v1)）。以自有語料自行訓練分離器，是避開 WSJ0／LDC 與 share-alike 疑慮最乾淨的作法。CHiME-5／6／7 上這些分離器的單通道結果、enrollment-free 的目標說話者抽取（如 TS-SEP 的專門檢索）、以及 MeetEval 論文對 ORC-WER 的正式定義，本次都沒有重新查閱。

## Conclusion

解除單模型限制之後，問題的形狀變了。前一份報告的核心問題是「可訓練參數該放在 Qwen3-ASR 的哪裡」；現在多了一個選項，就是把參數整個放到 Qwen3-ASR **外面**。這個選項之所以值得與 SOT 並列，不是因為它在英文基準上的數字比較好看，而是因為它把風險換了位置：SOT 賭的是一顆沒人診斷過的 Encoder 與多語遺忘，串接賭的是分離器能否跨到台語與真實錄音。兩個賭注都沒有文獻能代為下注，但串接的那一個可以在不訓練任何東西的情況下，花幾天就看到第一張牌。

第 0 階段因此比前一份報告設想的更有份量。它現在會產出四樣文獻中都不存在的東西：Qwen3-ASR-0.6B 的 all-speaker 重疊 baseline、Speech-LLM 接在分離器後面的第一組數字、現成分離器在台語與台語＋華語混音上的第一組分離分數，以及 `language None` 在殘留串流上是否可靠的答案。另一個值得放在心上的觀察是：真實會議的證據顯示，不論走哪條線，最後決定成敗的往往不是架構，而是有沒有用真實重疊資料調適過，以及後端 ASR 是否為這個情境微調過。兩條主線最終很可能不是二選一，而是由分離器替 SOT 準備資料、或由 SOT 接手分離器處理不好的區段；但那是第 2 階段才需要回答的問題。
