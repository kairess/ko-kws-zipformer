# 한국어 스트리밍 키워드 검출 (Zipformer tiny)

브라우저나 모바일에서 돌릴 수 있는 작은 한국어 키워드 검출(KWS) 모델의 학습 코드입니다.
[icefall](https://github.com/k2-fsa/icefall)의 GigaSpeech KWS 레시피(스트리밍 Zipformer + pruned RNN-T)를 한국어에 맞게 옮겼고,
추론은 [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)의 `KeywordSpotter`로 합니다.

- 학습된 모델(ONNX, int8): https://huggingface.co/kairess/ko-kws-zipformer-tiny
- 키워드는 학습 없이 텍스트로 바꿔 넣을 수 있습니다(오픈 어휘).
- 데모 노트북: [`demo.ipynb`](demo.ipynb) — 키워드 등록, 검출과 전사, 실시간 스트림 지연, 잡음 속 검출, 문턱값, 처리 속도

## 결과

`exp/pt5k-tiny2`, epoch 12(평균 1). 스트리밍 청크 16, 왼쪽 문맥 64 프레임으로 디코딩했습니다.

| 평가셋 | CER | FRR@τ0.35 | FA/h@τ0.35 | FRR@1FA/h | FRR@0.1FA/h |
|---|---|---|---|---|---|
| KsponSpeech eval_clean | 13.8% | 25.6% | 0.81 | 23.1% | 31.4% |
| KsponSpeech eval_other | 14.7% | 15.9% | 2.00 | 35.2% | 100% |
| FLEURS ko test | 14.6% | 5.9% | 0.00 | 4.9% | 4.9% |
| AI Hub 명령어 (`aihub_cmd_test`) | 4.7% | 6.5% | 9.51 | 37.7% | 61.3% |
| AI Hub 소음 명령어 (`aihub_noisy_test`) | 7.7% | 8.7% | 32.48 | 83.0% | 92.6% |

- **CER**: 그리디 디코딩, 음절 단위, 공백 무시.
- **FRR / FA/h**: 키워드 20개에 대해 τ=0.01로 한 번 디코딩한 뒤 문턱값을 오프라인으로 바꿔 가며 계산했습니다.
  FRR은 놓친 양성 비율, FA/h는 키워드가 없는 발화 1시간당 오탐 횟수입니다.
- **FRR@1FA/h, FRR@0.1FA/h**: 오탐이 시간당 1회, 0.1회 이하가 되는 가장 낮은 문턱값에서의 FRR입니다.

### 결과를 읽을 때 주의할 점

- **오탐 평가 음성이 짧습니다.** 음성이 1~3.5시간뿐이라 FRR@1FA/h와 FRR@0.1FA/h는 오탐 한두 건에 크게 흔들립니다.
  예를 들어 eval_other의 FRR@0.1FA/h 100%는 확신도가 높은 오탐 하나 때문입니다. τ0.35 수치를 함께 보세요.
- **소음 명령어의 오탐이 많습니다.** τ0.35에서 시간당 32회입니다. 주된 원인은 세 가지입니다.
  - 대본 데이터에 예시 이름으로 자주 나오는 "홍길동"을 모델이 강한 사전 지식으로 배웠습니다.
  - "티비"와 "티브이"처럼 표기가 다른 경우가 오탐으로 집계됩니다.
  - "도서 반납"과 "도서관"처럼 앞부분이 같은 단어를 헷갈립니다.

## 모델

| 항목 | 값 |
|---|---|
| 구조 | 스트리밍(causal) Zipformer + stateless RNN-T 디코더 |
| 인코더 | 6개 스택 × 2층, dim 128, feedforward 192 |
| 디코더 / 조이너 | 192 / 192 |
| 파라미터 | 6.61M (학습 전용 층 포함) |
| 출력 단위 | 음절 기반 unigram BPE 2,000 (`data/lang_syl_k2000`) |
| int8 ONNX 크기 | encoder 8.5 MB + decoder 0.4 MB + joiner 0.4 MB = 9.3 MB |
| 스트리밍 | 청크 16 프레임(320 ms), 내보내기 왼쪽 문맥 128 |

icefall GigaSpeech KWS의 tiny(3.3M)는 영어 어휘 500개 기준입니다. 어휘가 2,000개가 되면 디코더·조이너 임베딩이
파라미터의 절반 가까이를 차지합니다. 그래서 디코더·조이너를 320에서 192로 줄이고, 아낀 몫을 인코더 층 수에 넣었습니다(`tiny2`).

## 학습 데이터

전체 5,559시간, 4,189,980개 발화(`kws_cuts_pt5k`).

| 데이터 | 시간 | 발화 수 | 비고 |
|---|---|---|---|
| KsponSpeech (AI Hub 123) | 929 h | 609,355 | 자유 대화 |
| Zeroth-Korean | 156 h | 66,786 | 낭독, 속도 변형 0.9/1.1 포함 |
| FLEURS ko_kr | 22 h | 6,657 | 낭독, 속도 변형 0.9/1.1 포함 |
| AI Hub 자유대화(일반남여, 109) | 3,201 h | 2,268,523 | 휴대폰·AI 스피커·스튜디오 |
| AI Hub 문학작품 낭송·낭독 (485) | 370 h | 227,939 | 문장 단위, 감정 스타일 |
| AI Hub 명령어(일반남여, 96) | 184 h | 209,159 | 문장당 최대 20회 |
| AI Hub 명령어 인식용 소음 환경 (71405) | 696 h | 801,561 | 문장당 최대 20회, 마이크 2종 |

데이터 처리에서 결과에 영향을 준 결정은 다음과 같습니다.

- **평가 문장 제거**: 다섯 평가셋에 나온 문장(공백 무시)과 같은 발화 124,597개를 학습에서 뺐습니다.
  명령어 코퍼스는 대본을 여러 사람이 읽은 데이터라, 빼지 않으면 명령어 평가 문장의 99%가 학습에 들어갑니다.
- **문장 반복 상한**: 명령어 두 코퍼스는 고유 문장 5만여 개를 수백만 번 읽은 구조라, 문장당 20개까지만 썼습니다.
- **잘린 음성 제외**: 명령어(96)의 일부 wav는 배포본에서 2.601초나 1.3초로 잘려 있어 전사 끝이 들리지 않습니다.
  라벨의 녹음 길이보다 0.25초 넘게 짧은 발화는 뺐습니다. 같은 문제가 더 심한 명령어(소아·유아, 95)는 쓰지 않았습니다.
- **텍스트**: `(SN:x)`, `(SP:x)` 같은 태그는 안의 말만 남겼습니다. 정규화 후 숫자나 영문이 남은 발화는 뺐습니다.
- **오디오 저장**: AI Hub 음성은 16 kHz 모노 Opus 32 kbps로 저장했습니다.
  FLAC 대비 KsponSpeech eval_clean CER 차이는 0.1%p였습니다(10.0% → 10.1%).
- **증강**: MUSAN 잡음·음악 CutMix(p=0.5, SNR 10~20 dB)와 SpecAugment만 썼습니다. 특징은 학습 중에 계산합니다.

## 평가셋

| 이름 | 출처 | 발화 | 시간 | KWS 양성 쌍 | 음성(오탐) 시간 |
|---|---|---|---|---|---|
| kspon_eval_clean | KsponSpeech eval_clean | 2,999 | 2.6 h | 121 | 2.47 h |
| kspon_eval_other | KsponSpeech eval_other | 3,000 | 3.8 h | 182 | 3.50 h |
| fleurs_test | FLEURS ko_kr test | 370 | 1.3 h | 102 | 0.96 h |
| aihub_cmd_test | AI Hub 96 검증, 학습과 겹치지 않는 화자 234명 | 3,000 | 2.75 h | 750 | 2.10 h |
| aihub_noisy_test | AI Hub 71405 검증, 화자 437명 | 3,000 | 2.60 h | 470 | 2.19 h |

- 두 AI Hub 평가셋은 검증 세트에서 무작위로 3,000개를 뽑았고, 같은 문장은 2개까지만 넣었습니다(`local/build_pt5k.py`, seed 0).
- **KWS 평가셋**(`data/kws-test`)은 icefall 방식으로 만들었습니다(`local/make_kws_testsets.py`).
  - 키워드는 각 평가셋에서 자주 나오는 3음절 이상 명사 20개입니다(Kiwi 형태소 분석).
  - 다른 키워드에 포함되는 키워드는 뺐습니다.
  - 양성은 어떤 어절이 키워드로 시작하는 발화, 음성은 키워드가 하나도 없는 발화입니다.
- 모델 선택(평균할 epoch 수)은 평가셋이 아니라 KsponSpeech dev + FLEURS dev로만 했습니다.

## 학습 설정

- 12 epoch, `--lr-epochs 3.5`, `--max-duration 1600`, fp16, RTX 5090 한 장, epoch당 약 1시간 45분.
- 평균할 epoch 수는 kspon_dev CER로 골랐습니다: 1개 13.1%, 3개 13.1%, 5개 13.4%, 8개 14.3% → 1개.
- `train.py` 메인 프로세스의 메모리가 처음 몇 시간 동안 약 25 GB까지 늘어납니다.
  메모리 상한 44 GB에서 한 번 종료되어, 52 GB로 올리고 `local/supervise_pt5k.sh`로 epoch 7부터 이어서 학습했습니다.

## 재현

### 1. 환경

```bash
bash setup.sh      # uv venv(.venv, Python 3.12) + torch 2.9.1 cu128 + k2 1.24.4 + icefall(고정 커밋)
. ./env.sh
```

### 2. 데이터 받기 (`data/raw`)

| 데이터 | 받는 곳 | 놓을 위치 |
|---|---|---|
| Zeroth-Korean | https://www.openslr.org/40/ | `data/raw/zeroth_korean.tar.gz` |
| MUSAN | https://www.openslr.org/17/ | `data/raw/musan.tar.gz` |
| FLEURS ko_kr | https://huggingface.co/datasets/google/fleurs | `data/raw/fleurs/data/ko_kr/` |
| KsponSpeech | AI Hub 123 (한국어 음성) | `data/raw/ksponspeech/` |
| AI Hub 485·109·96·71405 | AI Hub, 데이터셋별 이용 신청 | `local/aihub_fetch.py`가 받음 |

- AI Hub 데이터는 내국인만 신청할 수 있고, 재배포와 국외 반출이 금지되어 있습니다. 이 저장소에는 음성과 전사를 넣지 않았습니다.
- AI Hub API 키는 환경 변수 `AIHUB_API_KEY`나 `~/.config/ko-kws/aihub.key`(권한 600)에 둡니다.
  파일 목록 조회에 AI Hub의 `aihubshell`이 필요하며, 경로는 `AIHUBSHELL` 환경 변수로 지정합니다.
- `local/aihub_fetch.py`는 zip을 하나씩 받아 16 kHz Opus로 바꾸고 원본 zip을 지웁니다.
  디스크는 변환된 음성과 라벨 약 250 GB에, 작업용 약 100 GB가 더 필요합니다. 중간에 끊겨도 다시 실행하면 이어서 받습니다.

### 3. 준비 · 학습 · 평가 · 내보내기

```bash
bash prepare.sh                              # 매니페스트 → fbank → 음절 컷 → AI Hub → 학습/평가 세트
bash run.sh --stage 0 --stop-stage 0         # 학습 (기본값 = 이번 실험 설정)
bash local/finalize.sh exp/pt5k-tiny2 tiny2 12   # dev로 평균 선택 → 다섯 평가셋 보고서 → ONNX 내보내기
```

- 오래 걸리는 작업은 `local/job.sh <이름> <메모리 상한> "<명령>"`으로 systemd 사용자 유닛에서 돌리면 터미널이 닫혀도 계속됩니다.
- epoch마다 자동 평가하려면 `local/watch_bench.sh exp/pt5k-tiny2 tiny2`를 함께 돌립니다.

## 저장소 구성

| 경로 | 내용 |
|---|---|
| `zipformer/` | 수정한 icefall 파일: 데이터 모듈, 학습, 디코딩(점수 덤프), 빔 서치. 나머지는 `setup.sh`가 icefall에서 링크 |
| `local/` | 데이터 준비, AI Hub 수집·변환, KWS 평가셋 생성, 지표 계산 |
| `data/lang_syl_k2000/` | BPE 모델과 토큰 목록 |
| `data/kws-test/` | 다섯 평가셋의 키워드와 양성·음성 발화 ID |
| `demo.ipynb` | sherpa-onnx 데모 (Hugging Face 모델을 받아 실행) |

## 라이선스

- 코드와 학습된 모델은 Apache License 2.0입니다. 코드에는 icefall(Apache 2.0)의 파일을 수정해 포함합니다.
- 학습 데이터는 포함하지 않으며, 데이터의 이용 조건은 각 제공처를 따릅니다. KsponSpeech와 AI Hub 데이터는 AI Hub 이용 약관을 따릅니다.
