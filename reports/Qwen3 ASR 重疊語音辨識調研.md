# 先診斷 Encoder，再用 SOT 教 Qwen3-ASR-0.6B 聽懂重疊語音

對「只要還原所有重疊說話者文字、推論架構要簡單」這個需求，現有文獻最支持的路線是 **utterance-level SOT（Serialized Output Training）微調**：它只改訓練標籤（`說話者1文字 <分隔符> 說話者2文字`），不加任何推論模組，Qwen3-ASR 官方微調腳本的資料格式可以直接承載。但有三件事必須先講清楚。第一，**筆記中沒有任何一個數字是在 Qwen3-ASR-0.6B 的重疊語音上量到的**，公開文獻也查無 Qwen3-ASR／Qwen2-Audio／Qwen-Omni 的 SOT 微調先例，以下所有「移植到 Qwen3-ASR」的說法都是待驗證的假設。第二，最關鍵的已發表消融顯示「只在 LLM 掛 LoRA」很可能不夠：未經重疊資料調適的 WavLM-Large 若凍結、只訓練 projector 與 LLM LoRA，Libri2Mix WER 是 46.7%，解凍 Encoder 後是 10.2%，而 LLM LoRA 本身只值約 1 個 WER 點（[Shi et al. 2024](https://arxiv.org/html/2408.17431)）；但凍結的 Whisper-large-v2 加 LoRA-LLM 卻能在 2 人 LibriSpeechMix 做到 6.6%（[MT-LLM](https://arxiv.org/html/2409.08596)），所以 AuT Encoder 屬於哪一種，只能自己量。第三，多語保留是真實風險：Whisper 用英文多人資料做全參數微調後，中文多人能力明顯受損，改用 adapter 才保住（[Li et al. 2023](https://arxiv.org/html/2305.18747)）。因此建議的順序是：先做不改動基礎權重的診斷（baseline 失效型態、凍結消融、分層 per-talker CTC read-out），回答「Encoder 是否還留著第二位說話者」；再做 SOT + LoRA（含 Encoder LoRA）的主實驗；per-speaker CTC／SOP、t-SOT、第二顆 Encoder 等都排在後面，只有在診斷結果或 3 人重疊、串流需求出現時才啟動。

本報告是技術評估，不是定案。文中數字一律抄自筆記標示為「已讀原始來源」的表格；只在搜尋摘要看到的數字會標示「未驗證」或直接省略；論文未公布者寫 Not Reported，筆記未讀到或來源無法存取者寫「未取得」，兩者不混用。

## 問題其實是兩個：Encoder 丟了第二個人，還是 Decoder 不會寫兩句話

Qwen3-ASR-0.6B 由 AuT Encoder（18 層、d=896、約 180M 參數、128 維 Fbank、8 倍降採樣得 12.5 Hz）、兩層 MLP projector（896→896→1024）與 Qwen3-0.6B decoder（28 層、hidden 1024、vocab 151,936、輸入輸出 embedding 綁定）組成，輸出格式為 `language X<asr_text>文字<|im_end|>`（[Qwen3-ASR 技術報告](https://arxiv.org/html/2601.21337v2)、[config.json](https://huggingface.co/Qwen/Qwen3-ASR-0.6B/blob/main/config.json)）。AuT 以約 4,000 萬小時的偽標註單人 ASR 資料預訓練，SFT 階段還刻意把模型訓練成「不遵循 prompt 中自然語言指令」的純 ASR 模型（[技術報告](https://github.com/QwenLM/Qwen3-ASR/blob/main/assets/Qwen3_ASR.pdf)），所以不能期待用 prompt 寫一句「請轉寫兩位說話者」就 zero-shot 生效，行為必須靠微調教會。

公開文獻對 Qwen3-ASR 在重疊語音上的實際行為幾乎是空白。唯一接近的數字來自競品報告：小米的技術報告列出 **Qwen3-ASR-1.7B** 在合成混音上 68.75–160.82%，但那是 **target-speaker WER**（輪流指定一位說話者為目標，參考答案只有該說話者的文字），測試集還是 SNR 0–15 dB 的非標準 LibriMix 變體，而 Qwen3-ASR 根本沒有輸入 enrollment 的管道（[Xiaomi-CocktailASR-1](https://arxiv.org/html/2609.11274v1)）。在這種計分下，一個完美轉寫兩人的模型也會得到約 100%，所以這組數字只說明「它不會挑目標說話者」，**無法區分它是丟掉第二人、把兩人混在一起，還是亂碼**，更不是 all-speaker WER；0.6B 則完全沒有數字。MS-Bench 的 `pyannote+Qwen3-ASR` 串接系統在七個系統中重疊懲罰最大（Δlex +38.6、Δattr +19.1 個百分點），但那混入了 diarization 誤差、未註明模型大小，而且 MS-Bench 並沒有「各重疊區間的絕對 WER」表格，只有重疊≥50% 與 0% 語句的成對差值（[MS-Bench](https://arxiv.org/html/2610.25824)）。

至於「Encoder 是否保留第二人」，筆記沒有找到直接探測第二說話者詞彙內容的 probing 論文，只有四組間接消融，方向一致但結論不能直接套用。凍結的單人 wav2vec 2.0 CTC 模型中，把分離模組插在第 2 層得到 10.36% WER，第 6 層 30.32%，第 12 層 62.72%，作者的解讀是「Encoder 越深，說話者資訊越被消除」（[Sidecar](https://arxiv.org/html/2302.09908)）。相反地，經過 SOT 微調的 WavLM-Large 是越深越好（取第 12／18／24 層特徵，noisy Libri2Mix 為 31.0／15.2／7.9），而且凍結後接一個約 14M 的 LSTM separator 加 per-talker CTC，純 Encoder 讀出就能到 10.1%（noisy）／4.6%（clean），與 LLM-SOT 的 9.1／3.7 相當（[Shi et al. 2603.27205](https://arxiv.org/html/2603.27205)）。換句話說，**單人 ASR Encoder 的淺層通常還留著兩個人，上層會逐步丟掉，除非 Encoder 本身被重疊資料訓練過**。AuT 是監督式單人 ASR Encoder，文獻無法預測它落在「凍結即失敗」（WavLM 46.7%）還是「凍結仍可用」（Whisper 6.6%）哪一邊；兩個結果的資料量、重疊型態、Encoder 與 LLM 規模都不同，沒有任何研究把這些因素分離開來。這就是為什麼第 0 階段必須是診斷而不是直接訓練。

## A. 論文比較表：2 人重疊已有成熟解法，但沒有一篇用 Qwen3-ASR

下表的 WER 欄位皆為原論文表格數字（%），不同列的測試集、計分方式與重疊型態不同，**不可跨列直接比較**。「Training Hours」填的是論文明示的訓練資料量；「適用性」是本報告的評估而非論文結論。

| Paper | Year | Architecture | Training Hours | Overlap Type | WER | Code | Qwen3-ASR 適用性 |
|---|---|---|---|---|---|---|---|
| [SOT](https://arxiv.org/html/2003.12687)（Kanda et al.） | 2020 | BLSTM AED，單一輸出層＋`<sc>`，FIFO 排序 | LibriSpeech 960 h 來源，混音總時數 Not Reported | 部分重疊、1–3 人、平均 SIR≈0 dB | 1/2/3 人：4.6 / 11.2 / 24.0（1024-dim）；單人模型 5.7 / 68.5 / 92.7 | 訓練碼 Not Reported；[LibriSpeechMix 評測清單](https://github.com/NaoyukiKanda/LibriSpeechMix) | 高：只改標籤 |
| [大規模預訓練 SOT](https://arxiv.org/abs/2103.16776)（Kanda et al.） | 2021 | Conformer AED + SOT | 75K h 單人來源，約 900K h 模擬；AMI-SDM 微調 | 真實會議 AMI-SDM | AMI-SDM eval：純模擬 44.2；純 AMI 59.1；模擬→AMI 21.2 | Not Reported | 中：資料策略依據 |
| [t-SOT](https://arxiv.org/html/2202.00842)（Kanda et al.） | 2022 | Transformer-Transducer（82M／139M），token 依時間交錯＋`<cc>` | LibriSpeech 960 h，on-the-fly 混音 | 部分重疊、最多 2 人同時 | LibriSpeechMix 1/2 人（TT-36，160 ms）4.3 / 6.2；mono LibriCSS 平均 9.0 | Not Reported | 低–中：僅在需保留串流時 |
| [多語 Whisper + SOT](https://arxiv.org/html/2305.18747)（Li et al.） | 2023 | Whisper small/medium/large；全參數或 bottleneck adapter＋`<sc>` embedding | AMI-SDM 66 h（＋120／360 h 模擬） | 真實會議 | AMI-SDM：全參數 23.6、adapter 25.0；AliMeeting CER（zero-shot）：50.7 / 43.7 / +360 h 36.9 | Not Reported | 高：多語遺忘的直接證據 |
| [Sidecar](https://arxiv.org/html/2302.09908)（Meng et al.） | 2023 | 凍結 wav2vec 2.0 CTC＋8.7M Sidecar（mask）＋PIT-CTC | 未取得 | Libri2Mix 全重疊；LibriSpeechMix 部分重疊 | Libri2Mix test 10.36；LibriSpeechMix-2spk 7.56 | 未取得 | 低–中：需 N 次後段前傳 |
| [Whisper-Sidecar + TTI](https://arxiv.org/html/2407.09817)（Meng et al.） | 2024 | 凍結 Whisper，Sidecar 插在 encoder block 2 後；PIT；可訓練 18.41M（large） | LibriMix＋LibriSpeech 960 h 衍生混音，總時數 Not Reported | 全重疊與部分重疊 | large：LibriMix 2/3 人 4.66 / 16.79；LibriSpeechMix 3.43 / 6.80 | [有](https://github.com/LingweiMeng/Whisper-Sidecar) | 低–中：保留多語但需固定人數與多次前傳 |
| [LLM-SOT（Tencent）](https://arxiv.org/html/2408.17431)（Shi et al.） | 2024 | WavLM-Large＋projector＋Vicuna-7B（LoRA r=16） | 約 830 h LibriMix（ESPnet SOT 版）；AMI 76.9 h | 延遲 1–1.5 s 的 Libri2Mix；AMI-SDM | LibriMix test 10.2；Encoder 先調適再凍結 9.0；**未調適且凍結 46.7**；AMI-SDM cpWER 21.0 | Not Reported | 高：最接近的架構與凍結消融 |
| [MT-LLM](https://arxiv.org/html/2409.08596)（Meng et al.） | 2024 | 凍結 Whisper-large-v2＋WavLM-base＋adapter；Llama-2-7b LoRA r=32；可訓練 76.6M（1%） | 約 6.3K h（含單人原始語料；約 10% 含德語） | 模擬部分重疊；英德跨語言混音 | LibriSpeech 2.3；LibriSpeechMix 2/3 人 5.2 / 10.2；En-De-Mix 2 人 21.0 | [repo 存在](https://github.com/cuhealthybrains/MT-LLM)，內容未查核 | 高：多任務指令與跨語言混音的唯一先例 |
| [EncSep / GEncSep](https://arxiv.org/html/2409.00815)（H. Shi et al.） | 2024 | 凍結 WavLM-Large 特徵＋Conformer AED＋LSTM separator＋per-speaker CTC（固定順序，非 PIT） | Libri2Mix／Libri3Mix（時數見下列） | 全重疊＋起始偏移 | eval noisy 2/3mix、clean 2/3mix：SOT 17.1 / 28.2 / 7.0 / 14.7；GEncSep(Bi) 15.0 / 25.9 / 6.6 / 13.1 | [有](https://github.com/hshi-speech/Multi-talker-ASR-with-LLMs)（無授權檔） | 中：訓練期輔助 loss |
| [SOP](https://arxiv.org/html/2509.04488)（H. Shi et al.） | 2025 | WavLM-Large＋LLaMA-3.2-1B/3B、3.1-8B；separator＋serialized CTC；CTC 文字當 prompt | Libri2Mix 約 270 h；Libri3Mix 約 186 h | 同上 | 1B eval：SOT 11.3 / 39.1 / 4.6 / 21.6 → SOP 10.5 / 28.5 / 4.0 / 22.0 | 同上 repo | 中：2 人增益小，3 人增益大 |
| [Persistent serialized memory](https://arxiv.org/html/2603.27205)（H. Shi et al.） | 2026 | 同上＋每層 gated cross-attention 讀分離後特徵 | LibriMix | 同上 | 1B-Instruct test：LLM-SOT 9.1 / 32.3 / 3.7 / 19.1；純 serialized CTC 10.1 / 20.5 / 4.6 / 10.2；gated 7.6 / 17.4 / 3.2 / 14.1 | 同上 repo（含變體） | 低–中：＋64M 參數、延遲約＋55% |
| [Encoder-only + LLM 蒸餾](https://arxiv.org/html/2603.10587v1)（H. Shi et al.） | 2026 | WavLM＋separator＋serialized CTC＋Talker-Count Head；LLM 只當訓練期老師 | LibriMix | 同上 | noisy eval 2/3mix 9.7 / 24.5；RTF 0.0043（LLaMA-1B SOT 為 0.1150） | 未取得 | 低：會放棄 LLM decoder |
| [Speaker-Aware CTC](https://arxiv.org/html/2409.12388)（Kang et al.） | 2024 | SOT AED＋CTC／SACTC 輔助 loss | 未取得 | LibriSpeechMix，依重疊低／中／高分層 | 2mix：SOT 9.2 → +CTC 8.8 → +SACTC 8.0；單人 4.6 → 4.1 | [有](https://github.com/kjw11/Speaker-Aware-CTC) | 中：推論零成本輔助 |
| [GLAD](https://arxiv.org/html/2509.13093v1)／[H-SAGE](https://arxiv.org/html/2607.01566) | 2025／2026 | Conformer AED＋Encoder 內 Mixture-of-LoRA-Experts；**整個模型約 35M**，不是小 adapter | 1,351.6 h／1,770.2 h | LibriSpeechMix 依重疊分層 | H-SAGE：LibriSpeech 3.8；2mix 5.7（低／中／高 4.7 / 5.7 / 8.2） | GLAD 未找到；[H-SAGE 有](https://github.com/NKU-HLT/H-SAGE) | 低：未在預訓練 Speech-LLM 上驗證 |
| SOT-Whisper（Chan & Lian） | 2026 | Whisper medium.en／large-v3 全參數微調＋新增分隔 token | 「>8,000 h」合成（摘要層級，未驗證） | LibriSpeech 合成 1–2 人、固定延遲 | 論文本文付費牆未取得；摘要層級（未驗證）單人 2.09、雙人 2.48 | [有](https://github.com/Greg0124/SOT-Whisper-MultiSpeaker-ASR) | 高：資料與標籤格式的參考實作 |
| [Self-Speaker Adaptation](https://arxiv.org/html/2506.22646)／[multitalker Parakeet](https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1)（NVIDIA） | 2025 | FastConformer-Transducer＋Sortformer diarizer＋1.1M 注入模組；每位說話者一個模型實例 | Not Reported | LibriSpeechMix；真實對話 | 論文 offline cpWER 1/2/3mix 2.2 / 2.8 / 5.0；模型卡（1.12 s）AMI-SDM 37.44、CH109 15.81（論文研究系統為 26.21，模型不同） | 模型公開（NVIDIA Open Model License） | 低：需 diarizer、N 個實例、英文基底 |
| [SA-DiCoW](https://arxiv.org/html/2510.03723)（BUT） | 2025 | Whisper＋diarization 條件化＋SOT decoder | AMI 80.67 h＋NOTSOFAR-1 54.27 h＋Libri2Mix（DiCoW 系列） | LibriMix-clean；NOTSOFAR | cpWER 2/3 人：DiCoW 4.8 / 32.1；SA-DiCoW 3.9 / 18.0 | [有](https://github.com/BUTSpeechFIT/SOT-DiCoW) | 低：需要 diarization |
| [CSS 串接](https://www.isca-archive.org/interspeech_2021/wu21f_interspeech.pdf)（Wu et al.） | 2021 | Conformer 分離器（26.03M）＋AED ASR＋LM | 未取得 | mono LibriCSS 0–40% | 平均 11.8；ASR 導向微調後 9.6（數字見 [t-SOT 表 4](https://www.isca-archive.org/interspeech_2022/kanda22_interspeech.pdf)） | 未取得 | 低：第二顆模型、每串流各解碼一次 |
| [MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize) | 2026 | Whisper-Medium 組態 Encoder＋4× 時間合併＋MLP＋Qwen3-0.6B 風格 decoder（0.9B） | Not Reported | 長音檔多人對話；未說明如何處理同時說話 | AISHELL-4 CER 14.84／cpCER 15.83；AliMeeting 24.86／22.17（廠商自報） | 有（權重＋`finetune.py`，Apache-2.0） | 外部 baseline；輸出含時間戳與說話者標籤，超出需求 |
| [DM-ASR](https://arxiv.org/abs/2604.22467) | 2026 | Whisper-turbo 或 Qwen3-ASR-1.7B Encoder＋Qwen3 decoder；LoRA r=16、α=32 | 真實資料 212 h–7,500 h，不使用模擬 | 真實會議（含 diarization／時間） | Qwen3-0.6B decoder、CN 1,300 h：AliMeeting cpCER 21.60（表格重建，需複核） | 未取得 | 中：Qwen 家族 LoRA 設定先例 |
| [Speaker-Reasoner](https://arxiv.org/abs/2604.03074) | 2026 | Qwen3-Omni；LoRA r=8 掛在 **audio encoder、projector、LLM 所有線性層** | AliMeeting 約 120 h＋AISHELL-4 約 120 h（其餘未取得） | 真實會議 | 未取得 | 未取得 | 中：Encoder LoRA 在 Qwen 家族的先例 |
| [Xiaomi-CocktailASR-1](https://arxiv.org/html/2609.11274v1) | 2026 | 0.6B Encoder＋Qwen3-8B；需 enrollment 的 target-speaker ASR | 多人約 400,000 h＋單人約 600,000 h | 合成（SNR 0–15 dB）與真實 | TS-WER：LibriMix 2mix 4.11；Qwen3-ASR-1.7B 同表 68.75（**TS-WER，非 all-speaker**） | 論文未說明 | 低：違反「不需 enrollment」 |
| [CoT + RL TS-ASR](https://arxiv.org/html/2509.15612v2) | 2025 | Data2Vec2（0.3B）＋Qwen2.5-0.5B；需 enrollment | 未取得 | Libri2Mix／Libri3Mix | TS-WER 4.84 / 12.23 | 未取得 | 低；但證明 <1B Speech-LLM 容量足夠 |
| [Cascaded encoders](https://arxiv.org/html/2306.16398)（Google） | 2023 | 在預訓練單人 Encoder 後串接多人 Encoder | 未取得 | 內部單人／重疊測試集 | 單人／重疊：單人模型 16.4 / 48.1；直接多人訓練 20.4 / 23.1；cascade 16.5 / 21.3 | 未取得 | 中：支持「可切換多人路徑」的設計 |
| [Streaming 比較](https://arxiv.org/abs/2609.10265)（NVIDIA） | 2026 | 同一 0.6B 基底：speaker-conditioned 多實例 vs word-level SOT 微調 | Not Reported | 真實＋模擬 | Open ASR 平均：基底 7.16；SSA 7.44；**WL-SOT 16.95** | 未取得 | 高：SOT 微調傷害單人能力的警訊 |
| [Mind the Gap](https://arxiv.org/abs/2605.15442)（Polok et al.） | 2026 | DiCoW＋FastMSS 合成對話 | 每來源 500 h，五來源 2,500 h | NOTSOFAR-1、AMI、LibriSpeechMix、Mixer6 | macro tcpWER：純合成 9.8；純真實 10.9；合成→真實 8.7 | [FastMSS](https://github.com/popcornell/FastMSS) | 中：合成—真實落差的量化 |

把這張表濃縮成三個判斷。其一，2 人重疊在 LLM decoder 上用純 SOT 就已能勝過非 LLM 系統（clean Libri2Mix 4.1 對 GEncSep 6.6），但 3 人時 LLM-SOT 反而輸（22.0 對 13.1），SOP 作者歸因於降採樣與 projector 把語音特徵映射到文字空間時流失資訊（[SOP](https://arxiv.org/html/2509.04488)）；團隊若以 2 人為主，這個證據是有利的。其二，所有報告良好重疊結果的 Speech-LLM，要嘛在某階段調適 Encoder，要嘛用了預訓練就看過重疊的 Encoder（WavLM），**沒有證據顯示單靠 LLM LoRA 能救一顆單人訓練的 Encoder**。其三，LLM 越大不一定越好：SOT baseline 在 noisy Libri2Mix 上 1B／3B／8B 為 11.3／9.8／14.3，作者推測是資料量有限（[SOP](https://arxiv.org/html/2509.04488)），這對 0.6B decoder 不是壞消息，但也不是保證。

## B. 技術原理：五種方法其實只差在「在哪裡把兩個人分開」

### Utterance-level SOT 把分離工作全部交給 decoder 的自回歸

SOT 的架構就是單人 ASR 的架構，只有一個輸出層。兩位說話者的參考文字依**起始時間先後（FIFO）**串接，中間插入一個分隔符：`Y = y¹₁…y¹_{N₁} <sc> y²₁…y²_{N₂} <eos>`。Loss 是對序列化目標做一般的 cross-entropy，`L_SOT = −Σ_n log P(y_n | y_{<n}, X)`，沒有排列搜尋；對比之下 PIT 需要 `L_PIT = min_{π} Σ_s CE(Y_{π(s)}, O_s)`，輸出分支數固定且排列成本為 O(S!)。原始論文發現 FIFO 比「最小 loss 排列」更好（test-clean 16.5 對 17.1），唯一前提是兩句話不在完全相同的時間開始；說話者人數由 `<sc>` 的個數隱式給出（[SOT](https://arxiv.org/html/2003.12687)）。

資料格式上，單人樣本的目標就是不含分隔符的普通轉寫，多人樣本則含分隔符，兩者混在同一訓練集。訓練混音時原論文要求起始時間至少相差 0.5 s、且每句至少與另一句重疊，並保持原始音量（平均 SIR 約 0 dB）。訓練流程在 LLM 版本中通常分段：Tencent 的做法是 projector → Encoder → LoRA，比單階段好 0.2–0.3 個點；最佳結果是先用 AED-SOT 把 Encoder 調適到混音，再凍結它訓練 projector 與 LoRA（9.0）（[Shi et al. 2024](https://arxiv.org/html/2408.17431)）。推論流程與單人完全相同，只是輸出以分隔符切開；計分時要對切開後的片段做排列不變的比對。SOT 的已知弱點有三：同時起始時 FIFO 失去排序線索、人數增加時會重複輸出同一句、以及 utterance-level 輸出不單調因而不可串流（[t-SOT 比較表](https://www.isca-archive.org/interspeech_2022/kanda22_interspeech.pdf)）。容量也是條件：原論文 512 維模型 2 人 WER 17.3、1024 維為 10.9，作者明言「模型夠大是 SOT 成功的必要條件」。

### t-SOT 以字級時間戳換取串流，但還沒有 LLM decoder 的實證

t-SOT 把所有說話者的 token 依**發射時間**排成單一序列，相鄰兩個 token 若來自不同「虛擬通道」就插入 `<cc>`；推論時再依 `<cc>` 把序列還原成兩條通道。Loss 仍是原模型的 loss（原文為 RNN-T），架構與單人模型完全相同。它需要每位說話者的**字級對齊**來產生標籤（原文用 Montreal Forced Aligner），基本形式假設最多兩句同時發生。訓練資料為 LibriSpeech on-the-fly 混音，第二句延遲取 Uniform(0, len(u₁))。成績上，TT-36 在 160 ms 延遲下 LibriSpeechMix 1／2 人為 4.3／6.2，mono LibriCSS 從 0% 到 40% 重疊只由 6.7 升到 12.9，而同架構單人模型由 6.5 升到 34.0（[t-SOT](https://arxiv.org/html/2202.00842)）。代價是失去「不限人數」與說話者計數。筆記找到的所有 LLM 多人 ASR 論文都採用 utterance-level SOT，沒有一篇把 t-SOT 用在 LLM decoder 上；字與字交錯的輸出是否破壞 LLM 的語言模型優勢，目前只是推論。

### Serialized CTC 與 SOP 在 Encoder 端先分一次，再把結果當提示

這條路線在 Encoder 後加一個 separator：`Ĥ = LayerNorm(LSTM(H))`，每位說話者一個分支 `H^s_sep = ReLU(Linear_s(Ĥ))`，各接一個 CTC head。關鍵設計是**固定順序指派**：第 s 個 head 對應 SOT 順序中的第 s 位說話者，不做 PIT（官方程式碼的註解明寫 PIT 分支已移除）。混合 loss 在程式碼中是 `L = α·L_CE + (1−α)·mean_s L_CTC(Y_s, H^s_sep)`，預設 `ctc_alpha = 0.7`；要注意這個 0.7 是 **attention CE 項的權重**，CTC 權重是 0.3（[losses.py](https://github.com/hshi-speech/Multi-talker-ASR-with-LLMs/blob/d955266085e130658007a747df58dff904a35c27/models/losses.py)）。SOP 再進一步：各分支 greedy CTC 解碼的文字依說話者順序串接，經 LLM 的 embedding 層得到 `E_sop`，decoder 輸入為 `[E_sop ; H_p ; E_t]`（CTC 提示、混合語音 embedding、文字 prompt）。

訓練流程為三階段：SOT 微調（Encoder＋LLM LoRA，之後合併 LoRA）→ 訓練 separator 與 serialized CTC → 凍結前述部分、只訓練新的一組 LoRA。作者直接單階段訓練時效果較差。推論時 separator 與 CTC head 都要執行。實測要點有四。一是 2 人增益小、3 人增益大（1B：11.3→10.5 對 39.1→28.5）。二是第二階段讓 CTC 梯度流進共用 Encoder 時，LLM 路徑的 2 人成績**每一列都變差**（1B noisy 11.3→13.4），作者歸因於 CTC blank 讓語音 embedding 變稀疏。三是拿掉混合語音 embedding、只留 CTC 文字提示會崩潰（34.4%）。四是 separator 接的是第二層 CNN 降採樣的輸出而不是最後一層，因為「最後一層降採樣過度」（[SOP](https://arxiv.org/html/2509.04488)）。後續論文改成「凍結 SOT 調適後的 Encoder，只訓練 separator＋CTC」，不再動到 SOT 路徑，CTC head 在部署時移除；但用 LLM 詞彙做 CTC head 每位說話者約 131M 參數（2 人 262.9M），訓練期成本不小（[2603.27205](https://arxiv.org/html/2603.27205)）。純 Encoder 只用 serialized CTC loss 從頭訓練「無法有效收斂」，保留 LLM-SOT loss 當老師才行（[2603.10587](https://arxiv.org/html/2603.10587v1)）。

### Sidecar 與 PIT 在 Encoder 淺層切開，之後每位說話者各跑一次

Sidecar 把一個 Conv-TasNet 風格的 TCN 插在凍結 Encoder 的淺層之間，對混合 embedding 預測每位說話者的 mask，分離後的 embedding 在 batch 維度並排，通過其餘凍結層；loss 為 `L = min_π Σ_s L_CTC(Y_{π(s)}, H_s)`（wav2vec2 版）或 PIT 的 ASR loss 加 `0.01·L_TTI`（Whisper 版）。可訓練參數只有 8.7M（wav2vec2）到 18.41M（Whisper-large，1.18%）；wav2vec2 版在 8 張 V100 上約 7 小時收斂。Whisper-large 版在全重疊的 Libri2Mix 上達 4.66（[Whisper-Sidecar](https://arxiv.org/html/2407.09817)、[Sidecar](https://arxiv.org/html/2302.09908)）。它保留基礎模型權重的特性對多語有利：筆記中列有僅以英文訓練後 zero-shot 中文 Aishell1Mix 的 CER（large 為 28.94），但這個數字在後續查核中未逐格複核，引用前需回原表確認。代價是人數上限必須事先固定，且 Encoder 後段與 decoder 要對每位說話者各跑一次；t-SOT 論文對多分支模型的描述是「decoder 推論成本為單人模型的 K 倍」。

### Diarization 條件化的 target-speaker ASR 準確但不符合單次前傳的需求

NVIDIA 的 Self-Speaker Adaptation 在 FastConformer 的 pre-encode 層注入說話者活動訊號：`X_inj = FeedForward(X ⊙ y_spk) + X`，其中 `y_spk ∈ (0,1)^{T×1}` 是某位說話者的逐幀活動機率，由 Sortformer diarizer 提供。訓練時每筆樣本隨機選一位目標說話者，標籤只有該說話者的文字，其餘與單人 ASR 訓練幾乎相同；推論時以 batch size＝說話者人數的方式跑 N 個實例。研究系統共 238M（ASR 114M＋diarizer 123M＋注入模組 1.1M），offline LibriSpeechMix cpWER 為 2.2／2.8／5.0（[arXiv 2506.22646](https://arxiv.org/html/2506.22646)）。它不需要 enrollment，但「不需 enrollment」不等於「不需 diarization」：每個系統都要吃逐幀說話者活動。DiCoW 的作者自己指出，完全重疊時各說話者的活動 mask 變得模糊，模型失去該轉寫誰的線索（3 人 LibriMix cpWER 32.1）（[SA-DiCoW](https://arxiv.org/html/2510.03723)）。這類方法在本案的價值是當外部比較基準，以及一個旁證：1.1M 的淺層條件訊號就足以把單人 Encoder 導向任一位說話者，與 Sidecar「淺層仍保有兩人」的觀察一致。

## C. Qwen3-ASR 修改建議：資料格式不用動程式，但 Encoder 與 tokenizer 有幾個坑

### 最小改動只需要一個分隔符，但三種選法各有代價

官方微調腳本 `finetuning/qwen3_asr_sft.py` 吃 JSONL 的 `audio` 與 `text` 兩欄，`text` 就是 assistant 端的字面目標，collator 以 chat template 組出前綴並遮蔽，loss 是標準 causal-LM CE（[finetuning/README.md](https://github.com/QwenLM/Qwen3-ASR/blob/main/finetuning/README.md)）。因此把目標寫成 `language X<asr_text>語句1<分隔符>語句2` 就是 SOT，訓練腳本、loss 與 `parse_asr_output`（以 `<asr_text>` 切分，其後原樣回傳）都不必改。這是推論而非實測，但依據是直接讀過的程式碼。

分隔符有三個候選。第一，用現有 BPE 能切出的純文字字串：不動 embedding、最適合 LoRA，但可能被切成多個 token，也要確保不會與真實轉寫內容衝突。第二，新增 special token：需要 `add_tokens` 與 `resize_token_embeddings`，而官方 repo 與 HF 模型程式碼完全沒有呼叫過這兩者，沒有官方範例。第三，**重用 tokenizer 已有的 27 個佔位 token `<blank1>`–`<blank27>`（id 151677–151703）**：詞表維持 151,936、不需 resize、單一 token 且不可能與轉寫文字衝突（[tokenizer_config.json](https://huggingface.co/Qwen/Qwen3-ASR-0.6B/blob/main/tokenizer_config.json)）。第三案最省事，但有兩個必須處理的問題。這些 token 被標為 `special: true`（`<asr_text>` 與 `<non_speech>` 則是 `special: false`），任何 `skip_special_tokens=True` 的解碼都會把分隔符吃掉，兩句話被無聲地接在一起；解碼路徑必須改用 `skip_special_tokens=False` 再手動清理。另外，這些 token 的列是否曾被訓練過、原本用途為何，官方沒有文件；由於輸入輸出 embedding 綁定而 LoRA 預設凍結 embedding，該列（連同綁定的 lm_head 列）必須設為可訓練並隨 adapter 一起出貨，這與 Li et al. 在 adapter 之外只額外訓練「`<sc>` 的 token embedding」的做法一致。官方 `qwen_asr` 解碼路徑實際如何呼叫 `batch_decode`，筆記未檢查，動手前要先確認。建議在第一輪實驗把「`<blankN>`」與「純文字分隔符」當作一個便宜的對照。

### 語言標籤沒有台語選項，跨語言重疊的標頭要先定規則

`SUPPORTED_LANGUAGES` 共 30 個值，沒有 Minnan／Hokkien／Taiwanese；閩南語只列在「22 種中文方言」之中，歸在 `Chinese` 之下，技術報告也沒有閩南語單獨的 CER（[utils.py](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/utils.py)、[技術報告](https://github.com/QwenLM/Qwen3-ASR/blob/main/assets/Qwen3_ASR.pdf)）。輸出格式每次只有一個 `language X` 標頭，無法表達「說話者 1 講台語、說話者 2 講華語」。可行的選擇是沿用 `language Chinese`、改用 `language None`（官方說明此時該樣本不學 LID），或以 prefill 強制語言；台語與華語都以漢字書寫，無法像英德混音那樣從文字反推每位說話者的語言，這是文獻中沒有人面對過的格式問題。建議第一階段不新增語言 token，台語＋華語一律用同一種標頭規則，並把「標頭選擇對單人台語 CER 的影響」列為回歸測試項目。

### Encoder、projector、decoder 各自該怎麼訓練

Decoder 端的任務主要是學會輸出格式。Tencent 的消融中 LLM LoRA 只值 0.5–1.1 個 WER 點，作者的說法是「LoRA 讓 LLM 輸出適應 SOT 風格」。可參考的設定有 Hao Shi 的 r=16、α=32、dropout 0.1 掛在 self-attention 的 q/k/v/o，另開 `lm_head`、`embed_tokens` 與 projector；MT-LLM 用 r=32；DM-ASR 在 Qwen3 decoder 上用 r=16、α=32、peak LR 1e-4。Projector 參數量小（官方未公布確切數字），建議全程可訓練。

Encoder 端是成敗所在，也是「優先 LoRA」這個偏好最需要被檢驗的地方。依 46.7 對 10.2 的消融，應預設 AuT **需要**調適，並把「LLM-only LoRA」當成一個預期偏弱的 baseline 來跑。調適方式由輕到重為：Encoder LoRA（Speaker-Reasoner 在 Qwen3-Omni 的 AuT 所有線性層掛 r=8 是家族內先例）、部分解凍、全參數微調（官方腳本預設即全參數，LR 2e-5；Whisper 多人微調文獻用的 LR 更小，1e-6 到 1e-5）。全參數在域內重疊準確率上略勝（Li et al.：23.6 對 adapter 25.0），但跨語言保留明顯較差（AliMeeting CER 50.7 對 43.7）；本案把保留三語能力列為硬需求，所以 LoRA 的主要理由是**保留**而非重疊準確率。文獻沒有任何 LoRA rank、LoRA 掛載位置（Encoder／projector／LLM）或 LoRA 對全參數的多人 SOT 對照實驗，這些都得自己做。

是否需要新增參數，取決於走到哪一階段。純 SOT 不需要任何新模組。若診斷顯示需要輔助監督，再加 separator（約 14M）與 per-talker CTC head，並依文獻教訓採取兩個保守做法：接在凍結的 Encoder 上、不讓 CTC 梯度回流到共用層；CTC 詞表若用小詞表（EncSep 用 32 個字元）成本遠低於 LLM 詞表，但就不能當 SOP 的文字提示。AuT 輸出只有 12.5 Hz，比 SOP 論文中被嫌「降採樣過度」的位置還低，separator 接在 AuT 最終輸出是否可行沒有任何實證；Tencent 以 5 Hz 的語音 embedding 餵 LLM 做 2 人 SOT 仍然可行，所以 12.5 Hz 對純 SOT 不見得是瓶頸。

### 相容性風險清單

官方微調腳本有一則未修復的回報（issue #216，2026-10-05 開立，無維護者回應，筆記未獨立重現）：其一，patch 過的 `forward` 接受 `**kwargs`，導致 HF `Trainer` 把 loss 乘上 GPU 數×梯度累積步數；其二，`labels[i, :pl] = -100` 假設右側 padding，但 batch 實際為左側 padding，只有每個 batch 中最長的那一列前綴被正確遮蔽。回報者稱同一批資料上原始遮蔽的 loss 為 11.37、修正後為 0.143（[issue #216](https://github.com/QwenLM/Qwen3-ASR/issues/216)）。該行程式碼確實仍在 main 上。SOT 訓練會把很短的單人樣本與很長的重疊樣本放進同一個 batch，正好踩在這個 bug 上，**開跑前必須先驗證或修掉**，否則結果不可信。

官方沒有 LoRA 腳本，只有社群版本（例如 [MeiLiMaoA/Qwen3-ASR-LoRA](https://github.com/MeiLiMaoA/Qwen3-ASR-LoRA)，r=16、α=32，掛在 q/k/v/o 與 MLP 投影層），除這一個之外都未經查核；另有使用者回報某社群 LoRA 在生產資料上比原始 1.7B 落後 2–5 個點（[issue #174](https://github.com/QwenLM/Qwen3-ASR/issues/174)）。服務端方面，搜尋摘要顯示 vLLM 當時尚不支援 `Qwen3ASRForConditionalGeneration` 的 LoRA adapter，後續 PR 修的是 audio tower 的 LoRA 被靜默略過的問題，合併狀態未查證（[vLLM #37223](https://github.com/vllm-project/vllm/issues/37223)，未驗證）；保守做法是**把 LoRA 合併回基礎權重再部署**，這同時讓社群 CPU 移植版可以直接沿用。

串流與 utterance-level SOT 不相容。Qwen3-ASR 的串流是每 2 秒把累積音訊重新解碼，並把先前輸出扣掉最後幾個 token 後當作前綴續寫（[qwen3_asr.py](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_asr.py)）；這假設轉寫只會在尾端成長，而 SOT 在第二位說話者開口後，第一位的文字還要在分隔符**之前**繼續長。SOT 微調後的模型應預設為 offline-only。最後還有兩個小地方要檢查：transformers 後端預設 `max_new_tokens=512`，重疊輸出較長；解析函式內建 `detect_and_fix_repetitions(threshold=20)`，兩位說話者講類似內容時可能被誤判為重複。

## D. 實驗優先順序：先花最少的錢回答最貴的問題

### 路線排序

下表依團隊指定的六個面向評比。「預期改善」與「退化風險」欄是依文獻證據所做的判斷，不是量測值。

| 順位 | 路線 | 工程實作難度 | 訓練成本 | 預期改善能力 | 單人 ASR 退化風險 | 推論成本 | 論文實證可靠度 |
|---|---|---|---|---|---|---|---|
| 0 | 診斷：baseline 量測、凍結消融、分層 per-talker CTC read-out | 低–中（需自建測試集與小型 read-out） | 極低（基礎權重不動） | 不改善，但決定後續投資方向 | 無 | 無 | 方法有先例；在未調適 Encoder 上的 read-out 無先例 |
| 1 | Utterance-level SOT＋LoRA（Encoder＋projector＋LLM）＋單人 replay | 低（資料工程為主；先修腳本 bug） | 中 | 高（2 人）；3 人有限 | 中，可用 PEFT 與單人比例控制 | 與單人相同架構，輸出 token 變多 | 強（多篇 LLM-SOT），但無 Qwen3-ASR |
| 1b | 同上但全參數微調（對照組） | 最低（官方腳本預設） | 中–高 | 域內可能略高 | 高（多語遺忘有實測） | 同上 | 強（Li et al.、NVIDIA WL-SOT 為反面證據） |
| 2 | 多任務指令（性別／語言／順序／關鍵詞）當輔助任務；人數 token；可切換的多人 adapter | 低–中 | 與 1 共用資料 | 小而穩（5.5→5.2） | 低；adapter 關閉時與原模型一致（推論） | 約 0 | 各元件分別有實證，組合無 |
| 3 | 凍結 Encoder 上的 separator＋per-talker CTC（訓練期輔助或 SOP 提示） | 中–高（新模組、多階段） | 中–高 | 2 人約 0.6–0.8 點；3 人大 | 低（若不動共用層） | SOP：＋LSTM＋較長 prompt；純輔助：0 | 中（僅 WavLM＋LLaMA、LibriMix） |
| 4 | t-SOT（需保留串流時） | 中（需字級對齊） | 中 | 未知 | 未知 | 與單人相同 | Transducer 上強；LLM decoder 上無 |
| 5 | 第二顆 mixture-aware Encoder（WavLM-base 級） | 中 | 中 | 3 人明顯（16.1→10.7） | 低 | 每筆音訊多跑約 95M Encoder | 中（凍結雙 Encoder 一例） |
| 6 | Sidecar／PIT 多分支 | 高（改推論圖） | 低（7 h×8 V100 級） | 中–高 | 低（基礎凍結） | 後段×N | 強（Whisper、wav2vec2），LLM decoder 無 |
| 7 | Diarization 條件化（Parakeet／DiCoW）、CSS 串接 | 高（第二顆模型） | 高 | 當外部 baseline | 低 | diarizer 或分離器＋N 次解碼 | 強，但不符單模型需求 |

### 第 0 階段：不改動 Qwen3-ASR 權重的三個診斷

第一個診斷完全不需訓練：把原始 Qwen3-ASR-0.6B 跑在自建的 2 人混音上，整段輸出視為單一串流，分別對兩位說話者的參考文字計分，同時計最小排列 WER。若一份參考的錯誤率接近 0–20%、另一份幾乎全是 deletion，就是「只轉寫主導說話者」；若兩份都很高，就是混疊或亂碼。這直接補上公開文獻的空白，因為「單人模型會丟掉次要說話者」雖被廣泛宣稱，筆記沒有找到任何模型的 per-speaker deletion 量化數字。這一列同時是之後所有實驗的 baseline。

第二個診斷是凍結消融，回答「Decoder 是否只是不會寫兩句話」。用同一批小規模混音資料跑三個短訓練：只訓練 LLM LoRA 與分隔符 embedding；再加 projector；再加 Encoder LoRA。若前兩者已接近第三者，問題在 decoder 端，LoRA-only 路線成立（對應 MT-LLM 的凍結 Whisper 6.6%）；若差距懸殊，問題在 Encoder 端（對應 WavLM 的 46.7 對 10.2）。

第三個診斷是 Encoder-only read-out，回答「Encoder 是否還留著第二人」。凍結 AuT，在數個深度（例如 18 層中的第 6、12、18 層，以及 projector 之後）各接一個小型 separator 加 per-talker CTC，固定以起始時間排序指派目標，比較各層讀出的兩人 WER。三種結果對應三種決策：頂層讀出良好，表示資訊還在，瓶頸在 projector／decoder，LLM 端訓練為主；淺層良好而頂層差，表示上層把第二人丟了，必須訓練 Encoder（或走 Sidecar 式淺層分離）；各層都差，則需大幅調適或考慮第二顆 Encoder。要誠實註明兩個限制。文獻中 per-talker CTC read-out 的成功案例是接在**已做過 SOT 調適**的 Encoder 上，接在未調適單人 Encoder 上的只有 Sidecar 的插入位置消融可類比；而且「只用 serialized CTC loss 訓練無法收斂」有過報告，所以 read-out 失敗不能單獨當作「資訊不存在」的證明，必須與第二個診斷交叉判讀。

### 評測協定

團隊不需要說話者歸屬，所以主指標應屬 speaker-agnostic 一族。MeetEval 論文的建議表明確寫道：SOT 式輸出應使用 MIMO-WER（ORC-WER 列為次選），因為 SOT 可以「往回跳時間」，把輸出視為單一串流（J=1）時 MIMO-WER 是唯一適用的指標；cpWER／tcpWER 是給「輸出已依說話者分組」的系統用的（[MeetEval](https://arxiv.org/pdf/2307.11394)）。對短的 2–3 人混音，另一個等價做法是沿用 Kanda 的最小排列 WER：以分隔符切開假設，窮舉排列取最低者；它的好處是能拿到每條串流各自的 insertion／deletion／substitution，這正是下面自訂指標需要的。TS-WER 不可用於本案，它獎勵的是壓抑其他說話者。華語與漢字台語以字為單位、英文以詞為單位計混合錯誤率；MeetEval 的 README 沒有 CJK 模式的說明，需先自行在漢字間插入空白，並對參考與假設套用相同的正規化。

次要指標建議四項。**次要說話者刪除率**是團隊自訂指標，文獻中沒有先例，定義必須寫明：在已知 SIR 的模擬混音上取最佳排列對齊後，分別對能量較高與較低的參考串流計算 deletion ÷ 參考字數；另報「整條串流遺失率」（某條參考串流 deletion 超過門檻，例如 90%，的混音比例）。其次是說話者計數正確率，分成單人音訊上誤出分隔符的比例與 2 人音訊上漏出分隔符的比例；參考值是模擬資料上 0.2%／1.9%（[SOT](https://arxiv.org/html/2003.12687)），真實 AMI 上 LLM-SOT 為 3.2%／11.7%（[Shi et al. 2024](https://arxiv.org/html/2408.17431)）。第三是 insertion、deletion、substitution 分開報，避免幻覺或重複被 deletion 視角蓋掉。第四是各語言的單人回歸測試，而且必須使用**沒有被拿來當混音來源**的資料，因為文獻中「SOT 不傷單人」的結論都是域內測試，兩個實測到退化的案例都是域外。

測試集必須自建，因為查無台語重疊語料，也查無跨語言重疊測試集。建議以 LibriSpeechMix 的方式從各語言的單人測試集合成：語言配對涵蓋台語＋台語、華語＋華語、英語＋英語與三種跨語言配對；條件格點為重疊比例（0／10／20／40／100%）× SIR（0、±5、±10 dB）× 人數（1／2／3）× 性別配對。除重疊比例與人數外，SIR、性別、語言配對這幾個軸在文獻中都沒有可比的參考數字，筆記讀過的全文也沒有任何一篇報告 SIR 分層的 WER。公開可比的曲線可用 mono LibriCSS（0–40% 重疊），目標形狀參考 t-SOT：重疊升高時 WER 只緩升，0% 重疊時不輸單人模型。真實部分重疊資料（AliMeeting、AISHELL-4、AMI、NOTSOFAR-1 單通道、MS-Bench）只當回歸檢查，用 ORC-WER 或 speaker-agnostic 錯誤率計分；它們量的是輪流發言的對話穩健度，不是同時說話的還原能力。務必保留一份真實的台語／華語重疊錄音當最終驗收，因為純合成訓練在以同法合成的測試集上會顯得特別好看。

### 訓練資料指引

模擬配方建議採 Kanda 式的 on-the-fly 部分重疊混音，而不是 LibriMix 式的完全對齊重疊：每筆隨機抽 1–2（之後 1–3）句，第二句加上隨機延遲，保證起始時間至少相差 0.5 s 且確實有重疊，讓 FIFO 標籤有明確定義。HEAT 論文也指出，開頭一小段未重疊的語音是模型追蹤第一位說話者的強線索（[arXiv 2011.13148](https://arxiv.org/html/2011.13148)）。音量方面，主流 ASR 配方保持原始音量、沒有處理 clipping；本案三種語言的語料錄製音量不一，借用 LibriMix 的響度正規化（每個來源取 −33 到 −25 LUFS，混音後檢查 clipping）是合理做法，但沒有 ASR 論文對此做過消融。相對音量差應當作明確的模擬變數並分桶回報；小米把 SNR 隨機取在 0–15 dB，理由就是不讓模型靠「轉寫最大聲的人」取巧。雜訊與殘響的優先度較低：對 Whisper 系的 DiCoW，加雜訊只把 macro tcpWER 從 14.0 改善到 13.7（[Mind the Gap](https://arxiv.org/abs/2605.15442)）。

單人與多人的比例，唯一有對照實驗的範圍是 50:50 到 67:33，來自從頭訓練的英文 Transducer：TT-18 在 67:33 時單人／雙人為 4.7／7.4，50:50 為 4.9／6.9，差別只在零點幾（[t-SOT](https://arxiv.org/html/2202.00842)）。完全不放單人資料則會崩潰（單人 WER 16.7 對 5.7）。對一個還要保住三種語言的預訓練模型，起點取較高的單人比例是站得住腳的選擇，但沒有消融直接支持，需自行掃描。資料量方面，270 h、830 h、960 h 的來源語音都訓練出了可用的 2 人 SOT 系統；on-the-fly 混音下「重疊資料時數」只是訓練步數的函數，真正重要的是來源語音的時數與多樣性。

合成與真實的落差很大也很一致。純模擬訓練在真實遠場會議上的錯誤率大約是加入真實資料後的兩倍（AMI-SDM：44.2 對 21.2；[SURT 的 SDM](https://github.com/k2-fsa/icefall/blob/master/egs/ami/SURT/README.md)：65.4 對 46.9）。拆解來看，最大的增益來自域內語音加隨機重疊（44.2→29.0），其次是真實的重疊樣態（→23.0），真實雜訊與殘響的貢獻相對小（→21.2）（[Kanda et al. 2021](https://arxiv.org/abs/2103.16776)）。合成比例也不是越高越好：真實對模擬 0.2:0.8 到 0.5:0.5 最佳，推到 0.1:0.9 時 AMI 與 ICSI 變差（[Yang et al. 2023](https://www.microsoft.com/en-us/research/wp-content/uploads/2023/11/ICASSP2023_Real_Simulation_tSOT.pdf)）。對本案的含意是：先用團隊已有的三語單人語料合成，就能拿到最大的一塊增益；之後再以少量真實重疊錄音微調或混訓。

遺忘的防護依證據強度排列為：凍結主幹加 adapter／LoRA；每種語言都保留單人 replay；重疊混音本身要涵蓋三種語言，而且**台語必須出現在重疊的兩個位置**，因為 Li et al. 的結果顯示多人能力的跨語言遷移並不完整（adapter 加 360 h 英文模擬後中文 CER 36.9，仍落後域內微調的 28.3）；調整單人對多人的取樣比例；必要時把多人能力做成可關閉的 adapter。原始 Qwen3-ASR 還可以當標註老師：對乾淨的單人台語／華語／英語音訊產生轉寫，混音後直接成為 SOT 目標，不需要人工轉寫重疊音訊。

## 原清單之外的替代方法：值得做的都是訓練配方，不是新架構

替代方法的整體結論是，沒有任何一項有證據在 2 人混音上大幅勝過調校良好的 LLM-SOT；它們改變的是「SOT 該怎麼訓練」，而不是取代它。以下排序是研究者的先驗評估，每一項都缺 Qwen3-ASR-0.6B、華語／台語混音與 Speech-LLM 上實測 RTF／VRAM 的直接證據。

| 排名 | 方法 | 新模組 | 推論成本增加 | 證據 | 誠實評語 |
|---|---|---|---|---|---|
| 1 | 把現有 Encoder 調適到混音（LoRA 或部分解凍），再對齊 projector＋LLM LoRA，並保留大量單人資料 | 無 | 0 | 強：凍結 46.7 對訓練 10.2；分段訓練最佳 | 這不是替代方案，是前提 |
| 2 | 指令多任務訓練（全體轉寫＋依性別／語言／順序／關鍵詞轉寫） | 無 | 推論只用全體轉寫 prompt 時為 0 | 一個 Speech-LLM 上驗證：主任務 5.5→5.2 | 免費的輔助監督；但 Qwen3-ASR 被訓練成不理會指令，可行性需實測 |
| 3 | 可切換的多人 adapter＋人數 token／重疊狀態輔助 head | 一個 token 或極小的 head | 約 0 | 元件分別驗證（Google cascade、CoT 人數消融 5.29 對 5.46、H-SAGE 重疊 loss） | 唯一能保證單人行為不變的設計；組合本身未經驗證 |
| 4 | 小型 mixture-aware 第二 Encoder（WavLM-base 級，約 95M） | 第二 Encoder＋adapter | 中，每筆都跑 | 凍結雙 Encoder 一例：3 人 16.1→10.7 | 沒有與「調適第一顆 Encoder」的對照；WavLM 為英文預訓練 |
| 5 | 單人老師蒸餾（老師看乾淨來源） | 無（僅訓練期） | 0 | 小：CER 15.8→15.0；非 LLM | 老師更大的價值是幫台語資料打偽標籤 |
| 6 | Encoder 內的 MoLE（GLAD／H-SAGE） | MoLE block＋gate | 小，未量測 | 僅在約 35M、從頭訓練的完整模型上 | 不是可外掛的小 adapter；移植到預訓練 Speech-LLM 無證據 |
| 7 | 以「第 N 位說話者」prompt 多次解碼當主要推論模式 | 無 | 每位說話者一次 prompt | 實測較差：順序 prompt 9.0，單次全體輸出 5.2 | 留作輔助任務，不當推論路徑 |
| 8 | Speaker-agnostic 雙串流（HEAT）＋diarizer | diarizer＋條件化 | diarizer＋2 次解碼 | AMI／ICSI 低重疊下驗證 | 適合超過 2 人的長會議，超出目前需求 |
| 9 | 需 enrollment 的 LLM TS-ASR、conditional chain、token 空間分離 | 參考音訊／新模組／新輸出模態 | 較高 | TS-ASR 強，其餘弱或離題 | 違反「不需 enrollment」，或不具競爭力，或等於換模型 |

兩個補充觀察值得帶進實驗設計。通用 audio-LLM 未經專門訓練時在重疊上普遍失敗：SALMONN-7B 的訓練任務清單中包含重疊語音辨識，2 人 LibriSpeechMix 仍是 32.9%，而專門訓練的 MT-LLM 是 5.2%（[MT-LLM](https://arxiv.org/html/2409.08596v2)）；光是把任務列進去不夠，資料量與 Encoder 調適才是關鍵。另一方面，「一個模型兼顧條件式目標說話者模式與無條件全體模式」並非免費：Whisper-small 規模的共享模型全體轉寫 cpWER 為 23.51，只做 SOT 微調則是 14.10（[arXiv 2609.30476](https://arxiv.org/html/2609.30476)），所以第 2 名的多任務訓練必須以「全體轉寫是否真的變好」來驗收。

## 部署成本比較：只有純 SOT 與 t-SOT 能沿用現有的單模型管線

基準線先講清楚。Qwen3-ASR-0.6B 在 vLLM v0.14.0、bfloat16、約 2 分鐘單人音訊、未具名 GPU 上，concurrency 1 時 offline RTF 0.00923、線上 TTFT 平均 92 ms；concurrency 128 時線上 RTF 0.064、吞吐 2,000 音訊秒／秒，但此時 TTFT 平均已是 3,210 ms，92 ms 是 concurrency 1 的數字（[技術報告表 2](https://arxiv.org/html/2601.21337v2)）。官方沒有任何 VRAM 指引；唯一的一手資料是使用者貼出的 vLLM 啟動 log：模型載入 1.53 GiB、CUDA graph 0.50 GiB（[issue #20](https://github.com/QwenLM/Qwen3-ASR/issues/20)）。CPU 路徑全是社群移植：純 C 實作在 Apple M3 Max 上 offline 約 7–13 倍即時、峰值 RSS 約 3.25 GiB，但串流模式在 135 秒音檔上掉到 0.96 倍即時（[antirez/qwen-asr](https://github.com/antirez/qwen-asr)）。官方沒有量化、GGUF、ONNX 或 CPU 版本。

| 路線 | 新增參數（論文數字） | 推論時額外模型 | 保持單模型管線 | 串流 | 已發表速度 | 已發表訓練成本 |
|---|---|---|---|---|---|---|
| SOT 微調 | 無（一個分隔符） | 否 | 是 | 不可 | LLaMA-1B SOT：RTF 0.1150／0.0981（Libri2Mix／3Mix，H100，batch 1） | Not Reported |
| t-SOT | 無 | 否 | 是 | 可（40–2560 ms，Transducer） | Not Reported | 16 GPU、225K iterations（GPU 型號未說明） |
| SOT＋separator／CTC | Separator 13.9M（2 人）；LLM 詞表 CTC head 262.9M（僅訓練期）；Sidecar 8.7M | 否，但模型內新增模組 | 修改過的單模型 | Not Reported | 純 serialized CTC：RTF 0.019；gated cross-attention：0.130（A100） | Sidecar：8 張 V100 約 7 小時 |
| PIT 多輸出 | 兩個分支，「多數參數載入並使用兩次」 | 否 | 多輸出分支 | 可（SURT 等） | decoder 成本為單人的 K 倍 | Not Reported |
| NVIDIA multitalker Parakeet | 注入模組 1.1M（研究系統） | 是：Sortformer（研究系統 123M） | 否，每位說話者一個 600M 實例 | 可（0.08–1.12 s） | Not Reported | 研究系統：8 張 A100、200k steps |
| 分離前端串接 | 分離器 9.97–26.03M（LibriCSS Conformer） | 是 | 否，每條串流各解碼一次 | CSS 增加 1.2 s 延遲 | Not Reported（一手來源） | Not Reported |
| MOSS-TD 0.9B | 獨立的 0.9B 模型 | 否 | 是 | Not Reported | RTF 0.0612（c=1，短音檔，H100） | Not Reported |

表中數字出自 [t-SOT](https://www.isca-archive.org/interspeech_2022/kanda22_interspeech.pdf)、[Shi et al. 2603.10587](https://www.isca-archive.org/interspeech_2026/shi26_interspeech.pdf)、[2603.27205](https://arxiv.org/html/2603.27205)、[Sidecar](https://arxiv.org/pdf/2302.09908)、[NVIDIA SSA](https://arxiv.org/html/2506.22646)、[Wu et al. 2021](https://www.isca-archive.org/interspeech_2021/wu21f_interspeech.pdf) 與 [MOSS repo](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize)。有三點要留意。**沒有任何一條路線公布 VRAM**，VRAM 比較只能靠團隊自己量。沒有論文在同一個 LLM decoder 上比較單人與多人的 RTF；t-SOT 論文說 SOT 的 decoder 成本「與單人模型相同」指的是每個 token，總解碼時間仍隨輸出 token 數成長，而 Libri3Mix 的 RTF 反而低於 Libri2Mix，所以「token 加倍則 RTF 加倍」不該未經量測就寫進規格。MOSS 的 0.061 與 Qwen3-ASR 的 0.009 在後端、硬體與任務上都不同，不是對等比較。就團隊的優先順序（單模型、無前端、沿用既有 CPU 移植與微調腳本）而言，純 SOT 的整合成本最低，因為只換權重；GPU 上餘裕很大，吃緊的是 CPU。LibriCSS 的同測試集比較還提供了準確率上的理由：單一串流模型 t-SOT（平均 9.0–10.1）已與帶分離器、AED 與 LM 的 offline 串接系統（9.6–11.8）持平或更好，而且串接系統的最佳數字還要靠針對特定 ASR 微調分離器才拿得到。

## E. 未解決的研究問題

### 已被論文實證的結論

- 單人 ASR 模型在重疊語音上會嚴重失效：2020 年的單人模型在 2／3 人混音上為 68.5%／92.7%（最小排列 WER），單人 Transducer 在 LibriCSS 40% 重疊時為 34.0%（[SOT](https://arxiv.org/html/2003.12687)、[t-SOT](https://arxiv.org/html/2202.00842)）。
- SOT 只需改標籤即可讓單輸出模型轉寫多人，FIFO 排序優於最小 loss 排列；混入單人樣本時單人 WER 不退化，不混則崩潰（5.7 對 16.7）。
- LLM decoder 上的純 SOT 對 2 人有效，對 3 人明顯退化；SOP 與 persistent memory 的主要增益都在 3 人（[SOP](https://arxiv.org/html/2509.04488)、[2603.27205](https://arxiv.org/html/2603.27205)）。
- 未調適的 Encoder 凍結時，只訓練 projector 與 LLM LoRA 會失敗（WavLM-Large：46.7 對 10.2）；LLM LoRA 本身只值約 0.5–1.1 點（[Shi et al. 2024](https://arxiv.org/html/2408.17431)）。
- 以單一語言的多人資料做全參數微調會損害其他語言；凍結主幹加 adapter 可保住並讓多人能力部分跨語言遷移（[Li et al. 2023](https://arxiv.org/html/2305.18747)）。
- SOT 式微調可能重創單人能力：同一 0.6B 基底的 word-level SOT 微調使 Open ASR 平均 WER 由 7.16 升到 16.95（[NVIDIA](https://arxiv.org/abs/2609.10265)）。
- CTC 輔助 loss 的梯度若流進與 LLM 共用的 Encoder／降採樣層，2 人 SOT 成績會變差；拿掉混合語音前綴、只給 CTC 文字提示會崩潰（[SOP](https://arxiv.org/html/2509.04488)）。
- 凍結的單人 Encoder 中，淺層仍可分離兩位說話者，深層不行（wav2vec2：第 2 層 10.36，第 12 層 62.72）；SOT 調適後則相反，越深越好。
- 純模擬訓練在真實會議上有約兩倍的錯誤率落差，少量真實資料或合成後接真實可補回大部分（[Kanda et al. 2021](https://arxiv.org/abs/2103.16776)、[Mind the Gap](https://arxiv.org/abs/2605.15442)）。
- SOT 式輸出的正確計分指標是 MIMO-WER／ORC-WER 或最小排列 WER，而非 cpWER 或 TS-WER（[MeetEval](https://arxiv.org/pdf/2307.11394)）。
- 一個 Speech-LLM 可以在同一個序列化輸出中寫出兩種語言（英德混音 21.0%），不需改架構（[MT-LLM](https://arxiv.org/html/2409.08596)）。

### 有實驗支持但尚未確認能移植 Qwen3-ASR 的方法

- Utterance-level SOT＋LoRA：在 WavLM＋Vicuna／LLaMA 與 Whisper＋Llama-2 上成立，Encoder、資料語言與 LLM 規模都與 Qwen3-ASR-0.6B 不同。
- Encoder LoRA：Speaker-Reasoner 在 Qwen3-Omni 的 AuT 上使用過，但任務含 diarization，筆記未取得其結果數字。
- 凍結 Encoder＋只訓練 adapter 與 LLM LoRA：對 Whisper-large-v2 可行（6.6%），對 WavLM 不可行；AuT 屬於哪一類未知。
- 凍結 Encoder 上的 separator＋per-talker CTC read-out：只在已做 SOT 調適的 WavLM-Large、50 Hz 特徵上驗證；AuT 為 12.5 Hz。
- SOP 式文字提示：2 人增益約 0.6–0.8 點，只有 LibriMix 英文。
- 指令多任務訓練、人數 token、重疊狀態輔助 loss：各自在其他系統上有小幅增益；Qwen3-ASR 在 SFT 階段被訓練成不理會自然語言指令。
- 可切換的多人路徑：Google 的串接 Encoder 證明保留單人 Encoder 可消除單人退化（16.5 對 20.4），但不是 LoRA 開關的形式。
- 單人對多人比例 50:50 至 67:33：來自從頭訓練的英文 Transducer。
- 以英文訓練的多人能力 zero-shot 遷移到中文：Whisper 上有部分效果（adapter＋360 h 模擬：CER 36.9），不完整。
- t-SOT 串流：Transducer 上證據充分，LLM decoder 上無任何結果。

### 純粹的研究假設

- 重用 `<blank1>`–`<blank27>` 當分隔符可行，且優於純文字分隔符或新增 token（這些 token 的原始用途與是否受過訓練皆無文件）。
- 把 SOT 目標直接寫進官方腳本的 `text` 欄即可訓練（依程式碼閱讀所做的推論，尚未實跑；issue #216 的兩個 bug 也未獨立重現）。
- AuT 的頂層仍保留第二位說話者的詞彙資訊，或其淺層保留而頂層丟失；兩者都沒有任何量測。
- Qwen3-ASR-0.6B 原始模型在重疊上的失效型態是「只轉寫主導說話者」；小米的 TS-WER 數字無法支持或否定這一點，而且量的是 1.7B。
- 12.5 Hz 的 AuT 輸出足以支撐 2 人 SOT，但可能不利於接 separator。
- 台語＋華語的跨語言重疊可由三語混音訓練學會；兩個以漢字書寫、聲學上接近的漢語系語言重疊，文獻中沒有任何實驗，以語言 prompt 選擇說話者是否可行同樣未知。
- 提高單人樣本比例與 LoRA 能同時保住台語、華語、英語的單人 CER；沒有論文在多語模型上報告多人微調前後各語言的單人錯誤率。
- 次要說話者刪除率會隨 SIR 惡化、並可靠音量差增強改善；沒有論文報告 SIR 分層的 ASR 結果。
- SOT 輸出的 token 數增加對 CPU RTF 的影響可接受；沒有同一 LLM decoder 上單人對多人的 RTF 比較。
- 重疊比例或 SIR 的由易到難 curriculum 有幫助；筆記未找到任何此類消融。

### 本次調研未能取得的資料

Chan & Lian（2026）的論文本文在付費牆後，放大曲線、合成參數、各模型大小的 WER 與單人 baseline 都未取得，只讀了摘要層級資訊與官方 repo，其訓練 CSV 為未下載的 Git-LFS 物件。MS-Bench 的專案頁面只回傳 JavaScript stub，無法閱讀（論文全文與 README 已讀）。aclanthology、Semantic Scholar、OpenReview 等站台無法連線，CSPB 與 SDBench 的重疊統計、AISHELL-4 與 NOTSOFAR-1 的資料集統計、多數分離模型的參數與 RTF 都只有搜尋摘要層級，本報告未採用為依據。單通道多人 ASR 的綜述（arXiv 2505.10975）只做了關鍵字掃描。Hugging Face Hub 的模型搜尋沒有執行，因此不能排除有只發在 HF 上的 Qwen3-ASR 多人衍生模型；arXiv 也只透過搜尋代理檢索，最近數週的新論文可能漏掉。MOSS-Transcribe-Diarize 的訓練時數官方未公布，次級網站上流傳的時數與其報告不符，本報告不採用。

## Conclusion

這次調研改變的認知是：對 Qwen3-ASR 而言，「要不要用 SOT」幾乎不是問題，真正的問題是**把可訓練參數放在哪裡**。團隊偏好的「LLM 掛 LoRA、其餘凍結」恰好是文獻中唯一有明確失敗紀錄的組態，而看似保守的「調適 Encoder」才是每個成功系統的共同點；但調適 Encoder 又正是多語遺忘風險最高的動作。Encoder LoRA、三語單人 replay、台語出現在重疊兩側的混音，是同時滿足這兩組證據的組合，而它在任何 LLM-ASR 模型上都還沒有人測過。

所以第 0 階段的診斷不是前置作業，它本身就是這個專案最有資訊量的實驗：它會告訴團隊 AuT 更像凍結後失敗的 WavLM 還是凍結後可用的 Whisper，並順帶產出文獻中缺席的三樣東西——Qwen3-ASR-0.6B 的 all-speaker 重疊 baseline、次要說話者刪除率的量化定義，以及台語／華語跨語言重疊的測試集。這三樣無論後續選哪條路線都用得到，也是這項工作對外最有發表價值的部分。
