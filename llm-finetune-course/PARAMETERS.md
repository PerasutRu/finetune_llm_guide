# 📖 Fine-tuning Parameters Guide

คู่มืออธิบาย parameter สำคัญทั้งหมดที่ใช้ในการ Fine-tune LLM
แบ่งตามหมวดหมู่เพื่อให้ค้นหาและทำความเข้าใจได้ง่าย

---

## สารบัญ

1. [LoRA / QLoRA Parameters](#1-lora--qlora-parameters)
2. [Model Loading Parameters](#2-model-loading-parameters)
3. [Training Arguments](#3-training-arguments)
4. [SFTTrainer Parameters](#4-sfttrainer-parameters)
5. [DPO Parameters](#5-dpo-parameters)
6. [Generation Parameters](#6-generation-parameters)
7. [Export Parameters](#7-export-parameters)
8. [ตารางเปรียบเทียบ Config ตาม GPU](#8-ตารางเปรียบเทียบ-config-ตาม-gpu)

---

## 1. LoRA / QLoRA Parameters

ตั้งค่าผ่าน `FastLanguageModel.get_peft_model()`

---

### `r` — LoRA Rank

```python
r = 16
```

**คืออะไร:**
LoRA แทนที่จะแก้ไข weight matrix `W` ขนาด `(d × d)` โดยตรง
จะเพิ่ม matrix คู่ `A (d × r)` และ `B (r × d)` เข้าไปแทน

```
W_new = W_frozen + (alpha/r) × A × B

ตัวอย่าง Llama-3 8B (d = 4096):
  Full:      4096 × 4096 = 16,777,216 params ต่อ layer
  LoRA r=16: 2 × 4096 × 16 = 131,072 params ต่อ layer  (ลดลง 128×)
  LoRA r=64: 2 × 4096 × 64 = 524,288 params ต่อ layer  (ลดลง 32×)
```

**ผลกระทบ:**
| r | Trainable Params | VRAM | ความสามารถเรียนรู้ | เหมาะกับ |
|---|-----------------|------|-------------------|---------|
| 4 | น้อยมาก | ต่ำสุด | จำกัด | task ง่าย, VRAM น้อย |
| 8 | น้อย | ต่ำ | พอใช้ | VRAM 6-8GB |
| 16 | ปานกลาง | กลาง | ดี | **แนะนำสำหรับ T4** |
| 32 | มาก | สูง | ดีมาก | A100 40GB |
| 64 | มากมาก | สูงมาก | สูงสุด | Production |

**กฎง่ายๆ:** เริ่มที่ `r=16` ถ้า loss ไม่ลงให้เพิ่มเป็น `r=32`

---

### `lora_alpha` — Scaling Factor

```python
lora_alpha = 32  # แนะนำ = 2 × r
```

**คืออะไร:**
ตัวคูณที่ควบคุมว่า LoRA adapter มีอิทธิพลแค่ไหนต่อโมเดลเดิม

```
effective_weight = W + (lora_alpha / r) × A × B
                              ↑
                        scaling factor
```

**ผลกระทบ:**
- `alpha/r` สูง → LoRA มีอิทธิพลมากขึ้น (เรียนรู้เร็ว แต่อาจ unstable)
- `alpha/r` ต่ำ → LoRA มีอิทธิพลน้อย (stable แต่เรียนรู้ช้า)
- `alpha = r` → scaling = 1.0 (neutral)
- `alpha = 2r` → scaling = 2.0 (แนะนำ)

**แนวทาง:** ตั้ง `lora_alpha = 2 × r` เสมอ เช่น r=16 → alpha=32

---

### `target_modules` — Layer ที่ใส่ LoRA

```python
# ตัวเลือก 1: ทุก linear layer (แนะนำ)
target_modules = "all-linear"

# ตัวเลือก 2: เฉพาะ attention (ประหยัด VRAM)
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

# ตัวเลือก 3: attention + MLP (balance)
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
```

**โครงสร้าง Transformer layer:**
```
Attention:
  q_proj  → Query projection
  k_proj  → Key projection
  v_proj  → Value projection
  o_proj  → Output projection

MLP (Feed-Forward):
  gate_proj → Gating (SwiGLU)
  up_proj   → Up projection
  down_proj → Down projection
```

**เปรียบเทียบ:**
| target_modules | Trainable Params | VRAM | คุณภาพ |
|---------------|-----------------|------|--------|
| attention only | ~0.3% | ต่ำ | ดี |
| attention + MLP | ~0.5% | กลาง | ดีกว่า |
| all-linear | ~0.6% | สูงกว่าเล็กน้อย | ดีที่สุด |

---

### `lora_dropout`

```python
lora_dropout = 0  # แนะนำ
```

**คืออะไร:** Dropout rate สำหรับ LoRA layers (ปิด neuron แบบสุ่มระหว่างเทรน)

**แนวทาง:**
- `0` = ไม่ใช้ dropout → เร็วกว่า เหมาะกับ dataset ขนาดใหญ่
- `0.05-0.1` = ใช้ dropout เล็กน้อย → ช่วย regularize ถ้า dataset เล็ก
- Unsloth แนะนำ `0` เพราะ LoRA เองมี regularization effect อยู่แล้ว

---

### `bias`

```python
bias = "none"  # แนะนำ
```

**ตัวเลือก:**
- `"none"` → ไม่เทรน bias (ประหยัด memory, แนะนำ)
- `"all"` → เทรน bias ทุก layer
- `"lora_only"` → เทรน bias เฉพาะ LoRA layers

---

### `use_gradient_checkpointing`

```python
use_gradient_checkpointing = "unsloth"  # แนะนำ
```

**คืออะไร:** เทคนิคประหยัด VRAM โดยไม่เก็บ intermediate activations ทั้งหมด
แต่คำนวณใหม่ตอน backward pass แทน

```
ปกติ:    เก็บ activations ทุก layer → VRAM สูง, เร็ว
Checkpointing: เก็บแค่บางจุด → VRAM ลด 30-40%, ช้าลงเล็กน้อย
```

**ตัวเลือก:**
- `"unsloth"` → Unsloth's optimized version (ประหยัด VRAM สูงสุด)
- `True` → HuggingFace standard
- `False` → ปิด (ต้องการ VRAM มากขึ้น แต่เร็วกว่า)

---

## 2. Model Loading Parameters

ตั้งค่าผ่าน `FastLanguageModel.from_pretrained()`

---

### `load_in_4bit` — LoRA vs QLoRA

```python
load_in_4bit = True   # QLoRA (แนะนำสำหรับ T4)
load_in_4bit = False  # LoRA ปกติ (ต้องการ VRAM มากกว่า)
```

**ความแตกต่าง:**

```
LoRA  (load_in_4bit=False):
  Base Model  [bf16/fp16]  ← โหลดเต็ม 16-bit (~14GB)
  LoRA Adapter [bf16/fp16] ← trainable (~100MB)
  VRAM รวม: ~14-16GB

QLoRA (load_in_4bit=True):
  Base Model  [NF4 4-bit]  ← quantized (~5GB)
  LoRA Adapter [bf16/fp16] ← trainable (~100MB)
  VRAM รวม: ~5-6GB ✅
```

**NF4 (NormalFloat4)** คืออะไร:
- Quantization format ที่ออกแบบมาเฉพาะสำหรับ LLM weights
- ดีกว่า INT4 เพราะ LLM weights มีการกระจายแบบ normal distribution
- ใช้ double quantization เพิ่มเติม → ประหยัดอีก ~0.4 bits/param

---

### `max_seq_length` — ความยาว Context

```python
max_seq_length = 2048  # แนะนำสำหรับ T4
```

**คืออะไร:** จำนวน token สูงสุดที่โมเดลจะรับได้ต่อ 1 ตัวอย่าง

**ผลกระทบต่อ VRAM:**
```
VRAM ∝ max_seq_length²  (attention เป็น quadratic)

seq_len=512  → VRAM ต่ำสุด
seq_len=1024 → VRAM ~4× ของ 512
seq_len=2048 → VRAM ~16× ของ 512
seq_len=4096 → VRAM ~64× ของ 512
```

**แนวทางเลือก:**
| max_seq_length | เหมาะกับ | VRAM (T4) |
|---------------|---------|-----------|
| 512 | คำถาม-คำตอบสั้น | ~4GB |
| 1024 | บทสนทนาปกติ | ~5GB |
| 2048 | บทสนทนายาว, code | ~7GB ✅ |
| 4096 | document QA, long context | ~12GB |
| 8192 | ต้องการ A100+ | >16GB |

---

### `dtype`

```python
dtype = None          # ให้ Unsloth เลือกอัตโนมัติ (แนะนำ)
dtype = torch.float16 # T4, V100 (Turing/Volta)
dtype = torch.bfloat16 # A100, H100 (Ampere+)
```

**ความแตกต่าง fp16 vs bf16:**
```
fp16:  sign(1) | exponent(5) | mantissa(10) → range เล็ก, precision สูง
bf16:  sign(1) | exponent(8) | mantissa(7)  → range ใหญ่, precision ต่ำกว่า

bf16 ดีกว่าสำหรับ LLM เพราะ:
- range ใหญ่กว่า → ไม่ overflow ง่าย
- ตรงกับ float32 ในส่วน exponent → แปลงไปมาง่าย
```

---

## 3. Training Arguments

ตั้งค่าผ่าน `TrainingArguments()`

---

### `per_device_train_batch_size` — Batch Size

```python
per_device_train_batch_size = 2  # แนะนำสำหรับ T4
```

**คืออะไร:** จำนวน training samples ที่ประมวลผลพร้อมกันใน 1 forward pass

**ผลกระทบ:**
- สูงขึ้น → เทรนเร็วขึ้น, gradient estimate แม่นขึ้น, แต่กิน VRAM มากขึ้น
- ต่ำลง → ประหยัด VRAM แต่เทรนช้าลง, gradient noisy กว่า

**แนวทาง:**
| GPU VRAM | batch_size |
|----------|-----------|
| 6-8GB | 1 |
| 10-12GB | 1-2 |
| 16GB (T4) | 2 ✅ |
| 40GB (A100) | 4-8 |
| 80GB (A100) | 8-16 |

---

### `gradient_accumulation_steps` — Gradient Accumulation

```python
gradient_accumulation_steps = 4
# effective_batch_size = batch_size × gradient_accumulation_steps
# = 2 × 4 = 8
```

**คืออะไร:** สะสม gradient หลาย step ก่อน update weights ครั้งเดียว
ทำให้ได้ effective batch size ที่ใหญ่กว่า VRAM จะรองรับได้

```
ปกติ (batch=8):
  [sample 1-8] → forward → backward → update weights

Gradient Accumulation (batch=2, accum=4):
  [sample 1-2] → forward → backward → สะสม gradient
  [sample 3-4] → forward → backward → สะสม gradient
  [sample 5-6] → forward → backward → สะสม gradient
  [sample 7-8] → forward → backward → สะสม gradient → update weights
```

**ผลลัพธ์เหมือนกัน แต่ใช้ VRAM น้อยกว่า**

**แนวทาง:** ตั้ง `effective_batch_size = 8-16` เสมอ
```python
# ถ้า VRAM น้อย:
batch_size=1, gradient_accumulation_steps=8  # effective=8

# ถ้า VRAM พอ:
batch_size=4, gradient_accumulation_steps=2  # effective=8 (เร็วกว่า)
```

---

### `learning_rate` — อัตราการเรียนรู้

```python
learning_rate = 2e-4  # สำหรับ SFT + LoRA
learning_rate = 5e-5  # สำหรับ DPO (ต่ำกว่า)
```

**คืออะไร:** ขนาดของ step ที่ใช้ update weights ในแต่ละครั้ง

```
weights_new = weights_old - learning_rate × gradient
```

**ผลกระทบ:**
```
lr สูงเกินไป → loss oscillate หรือ diverge (NaN)
lr ต่ำเกินไป → เทรนช้ามาก, อาจติด local minimum
lr พอดี     → loss ลดลงสม่ำเสมอ ✅
```

**แนวทางตาม method:**
| Method | Learning Rate |
|--------|--------------|
| Full Fine-tune | 1e-5 ~ 5e-5 |
| LoRA / QLoRA (SFT) | 1e-4 ~ 3e-4 |
| DPO | 5e-6 ~ 1e-4 |
| RLHF | 1e-6 ~ 1e-5 |

---

### `lr_scheduler_type` — Learning Rate Schedule

```python
lr_scheduler_type = "linear"   # ลด lr เป็นเส้นตรง (แนะนำสำหรับ SFT)
lr_scheduler_type = "cosine"   # ลด lr แบบ cosine (แนะนำสำหรับ DPO)
lr_scheduler_type = "constant" # คงที่ตลอด
```

**Visualization:**
```
linear:   lr ████████░░░░░░░░  (ลดเป็นเส้นตรงจนถึง 0)
cosine:   lr ████████▓▓▒▒░░░░  (ลดแบบ smooth curve)
constant: lr ████████████████  (คงที่ตลอด)
```

**แนวทาม:** `linear` สำหรับ SFT, `cosine` สำหรับ DPO และ long training

---

### `warmup_steps` / `warmup_ratio`

```python
warmup_steps = 5      # จำนวน step แน่นอน
warmup_ratio = 0.1    # 10% ของ total steps
```

**คืออะไร:** ช่วงแรกของการเทรนที่ค่อยๆ เพิ่ม lr จาก 0 ขึ้นมา

```
ไม่มี warmup:  lr = 2e-4 ทันที → อาจ unstable ช่วงแรก
มี warmup:     lr: 0 → 2e-4 (ค่อยๆ เพิ่ม) → stable กว่า
```

**แนวทาง:**
- Short training (< 200 steps): `warmup_steps = 5`
- Long training: `warmup_ratio = 0.03` (3% ของ total steps)

---

### `max_steps` vs `num_train_epochs`

```python
max_steps = 100        # หยุดหลัง 100 steps (ไม่ว่า dataset จะใหญ่แค่ไหน)
max_steps = -1         # ไม่จำกัด steps (ใช้ num_train_epochs แทน)

num_train_epochs = 1   # เทรน 1 รอบทั้ง dataset
num_train_epochs = 3   # เทรน 3 รอบ
```

**เลือกใช้อะไร:**
| สถานการณ์ | แนะนำ |
|-----------|-------|
| ทดสอบ / debug | `max_steps=100` |
| Dataset เล็ก (< 1K) | `num_train_epochs=3-5` |
| Dataset กลาง (1K-10K) | `num_train_epochs=1-3` |
| Dataset ใหญ่ (> 10K) | `num_train_epochs=1` หรือ `max_steps` |

---

### `optim` — Optimizer

```python
optim = "adamw_8bit"   # แนะนำ (ประหยัด VRAM)
optim = "adamw_torch"  # AdamW มาตรฐาน (ต้องการ VRAM มากกว่า)
optim = "sgd"          # SGD (ไม่แนะนำสำหรับ LLM)
```

**AdamW 8-bit คืออะไร:**
AdamW ปกติเก็บ optimizer states (momentum, variance) แบบ 32-bit
AdamW 8-bit quantize optimizer states เป็น 8-bit → ประหยัด VRAM ~75%

```
AdamW 32-bit: model params × 2 (momentum + variance) × 4 bytes = ~64GB สำหรับ 8B model
AdamW 8-bit:  model params × 2 × 1 byte = ~16GB ✅
```

---

### `fp16` / `bf16` — Mixed Precision Training

```python
fp16 = not torch.cuda.is_bf16_supported()  # ใช้ fp16 ถ้าไม่รองรับ bf16
bf16 = torch.cuda.is_bf16_supported()      # ใช้ bf16 ถ้ารองรับ
```

**Mixed Precision คืออะไร:**
เก็บ weights เป็น 16-bit แต่คำนวณบางส่วนเป็น 32-bit เพื่อความแม่นยำ

```
ปกติ (fp32):  ทุกอย่างเป็น 32-bit → VRAM สูง, ช้า
Mixed (fp16): weights=16-bit, loss scale=32-bit → VRAM ลด 50%, เร็วขึ้น 2-3×
```

**เลือกใช้:**
| GPU | ใช้ |
|-----|-----|
| T4, V100, RTX 20xx | `fp16=True` |
| A100, H100, RTX 30xx+ | `bf16=True` |

---

### `save_strategy` / `save_steps`

```python
save_strategy = "steps"   # บันทึกทุกกี่ steps
save_steps = 50           # บันทึกทุก 50 steps

save_strategy = "epoch"   # บันทึกทุก epoch
save_strategy = "no"      # ไม่บันทึก checkpoint ระหว่างเทรน
```

**แนวทาม:** ใน Colab ตั้ง `save_steps` ให้บ่อยขึ้น เผื่อ session หลุด

---

## 4. SFTTrainer Parameters

ตั้งค่าผ่าน `SFTTrainer()`

---

### `dataset_text_field`

```python
dataset_text_field = "text"  # ชื่อ column ที่เก็บ text ใน dataset
```

Dataset ต้องมี column ชื่อนี้ที่เก็บ formatted text ทั้งหมด

---

### `packing`

```python
packing = False  # ง่ายกว่า แนะนำสำหรับผู้เริ่มต้น
packing = True   # เร็วกว่า แต่ซับซ้อนกว่า
```

**คืออะไร:** รวม samples สั้นๆ หลายอันเข้าด้วยกันให้เต็ม `max_seq_length`

```
packing=False:
  [sample_1 (200 tokens)] [padding (1848 tokens)]  ← เสีย VRAM
  [sample_2 (150 tokens)] [padding (1898 tokens)]

packing=True:
  [sample_1 (200)] [sample_2 (150)] [sample_3 (300)] ... [sample_N]  ← ไม่เสีย
```

**แนวทาง:**
- `False` → ง่าย, ปลอดภัย, เหมาะสำหรับ dataset ที่ samples ยาวพอสมควร
- `True` → เร็วกว่า 2-3× เหมาะสำหรับ dataset ที่ samples สั้นมาก

---

### `dataset_num_proc`

```python
dataset_num_proc = 2  # จำนวน CPU cores สำหรับ tokenize dataset
```

เพิ่มได้ถ้า CPU มีหลาย core แต่ใน Colab ใช้ 2 พอ

---

### `max_seq_length` (ใน SFTTrainer)

```python
max_seq_length = 2048
```

ตัดทอน samples ที่ยาวเกินกว่านี้ออก ควรตั้งให้ตรงกับที่ตั้งตอนโหลดโมเดล

---

### `data_collator` — DataCollatorForCompletionOnlyLM

```python
from trl import DataCollatorForCompletionOnlyLM

collator = DataCollatorForCompletionOnlyLM(
    response_template="<|start_header_id|>assistant<|end_header_id|>",
    tokenizer=tokenizer,
)
```

**คืออะไร:** Collator พิเศษที่ทำ Loss Masking อัตโนมัติ

```
Input tokens:  [system] [user_msg] [assistant_header] [answer] [eot]
Labels:        [-100]   [-100]     [-100]              [loss]   [loss]
                                                        ↑ คิด Loss เฉพาะส่วนนี้
```

**ทำไม -100:** PyTorch CrossEntropyLoss จะ ignore index ที่เป็น -100 โดยอัตโนมัติ

**response_template ตาม format:**
| Chat Template | response_template |
|--------------|-------------------|
| Llama-3 | `"<\|start_header_id\|>assistant<\|end_header_id\|>"` |
| ChatML | `"<\|im_start\|>assistant"` |
| Mistral | `"[/INST]"` |
| Alpaca | `"### Response:"` |

---

## 5. DPO Parameters

ตั้งค่าผ่าน `DPOConfig()`

---

### `beta` — DPO Temperature

```python
beta = 0.1  # ค่าแนะนำ
```

**คืออะไร:** ควบคุมว่าโมเดลจะ "เบี่ยงเบน" จาก reference model มากแค่ไหน

**DPO Loss formula:**
```
L_DPO = -log σ(β × (log π(chosen|x)/π_ref(chosen|x) - log π(rejected|x)/π_ref(rejected|x)))

β สูง → penalty สูงถ้าเบี่ยงจาก reference มาก → โมเดลเปลี่ยนแปลงน้อย (conservative)
β ต่ำ → penalty ต่ำ → โมเดลเปลี่ยนแปลงมาก (aggressive)
```

**ผลกระทบ:**
| beta | พฤติกรรม | เหมาะกับ |
|------|---------|---------|
| 0.01-0.05 | เปลี่ยนแปลงมาก | dataset คุณภาพสูง, SFT model ดีแล้ว |
| 0.1 | balance ✅ | ทั่วไป (แนะนำ) |
| 0.3-0.5 | เปลี่ยนแปลงน้อย | dataset เล็ก, ป้องกัน overfit |
| > 0.5 | แทบไม่เปลี่ยน | ไม่แนะนำ |

---

### `max_prompt_length`

```python
max_prompt_length = 512  # ความยาวสูงสุดของ prompt
```

ตัดทอน prompt ที่ยาวเกินนี้ออก ควรตั้งให้น้อยกว่า `max_length`

---

### `max_length`

```python
max_length = 1024  # ความยาวสูงสุดรวม (prompt + chosen/rejected)
```

**แนวทาง:** `max_length = max_prompt_length + max_response_length`
```python
max_prompt_length = 512
max_length = 1024  # เหลือ 512 tokens สำหรับ response
```

**ผลต่อ VRAM:** DPO ต้องการ VRAM มากกว่า SFT เพราะโหลด 2 โมเดล
ถ้า OOM ให้ลด `max_length` ก่อน

---

### `ref_model`

```python
ref_model = None  # ใช้ frozen copy ของ model เป็น reference (ประหยัด VRAM)
```

**คืออะไร:** Reference model ที่ใช้คำนวณ KL divergence ใน DPO loss

```
ref_model=None:
  → Unsloth สร้าง frozen copy ของ model อัตโนมัติ
  → ประหยัด VRAM เพราะ share weights กัน

ref_model=some_model:
  → โหลด reference model แยก
  → ต้องการ VRAM 2× (2 โมเดลเต็มๆ)
  → ใช้เมื่อต้องการ reference model ที่ต่างจาก policy model
```

---

### `loss_type` (DPO variant)

```python
loss_type = "sigmoid"  # DPO ดั้งเดิม (แนะนำ)
loss_type = "ipo"      # IPO (Identity Preference Optimization)
loss_type = "kto_pair" # KTO แบบ paired
```

**แนวทาง:** ใช้ `"sigmoid"` (DPO ดั้งเดิม) เป็นค่าเริ่มต้น

---

## 6. Generation Parameters

ใช้ตอน inference ใน `model.generate()`

---

### `max_new_tokens`

```python
max_new_tokens = 256  # จำนวน token สูงสุดที่จะ generate
```

ไม่รวม input tokens เช่น input=100 tokens, max_new_tokens=256 → output สูงสุด 356 tokens

---

### `temperature`

```python
temperature = 0.7  # แนะนำ
```

**คืออะไร:** ควบคุมความ "สร้างสรรค์" ของการ generate

```
softmax(logits / temperature)

temperature → 0:  distribution แหลมมาก → เลือก token ที่น่าจะเป็นที่สุดเสมอ (deterministic)
temperature = 1:  distribution ปกติ
temperature > 1:  distribution แบน → สุ่มมากขึ้น (creative แต่อาจ incoherent)
```

**แนวทาง:**
| temperature | เหมาะกับ |
|------------|---------|
| 0.0 - 0.3 | fact-based QA, code generation |
| 0.5 - 0.8 | general chat, instruction following ✅ |
| 0.8 - 1.2 | creative writing, brainstorming |
| > 1.2 | ไม่แนะนำ (incoherent) |

---

### `top_p` — Nucleus Sampling

```python
top_p = 0.9  # แนะนำ
```

**คืออะไร:** เลือกเฉพาะ tokens ที่รวมกันแล้วมี probability ≥ top_p

```
ตัวอย่าง top_p=0.9:
  token A: 0.5  ← เลือก (cumsum=0.5)
  token B: 0.3  ← เลือก (cumsum=0.8)
  token C: 0.1  ← เลือก (cumsum=0.9) ← หยุดที่นี่
  token D: 0.05 ← ไม่เลือก
  token E: 0.05 ← ไม่เลือก
```

**ผลกระทบ:**
- `top_p=1.0` → ไม่ filter (ใช้ทุก token)
- `top_p=0.9` → ใช้ top 90% probability mass ✅
- `top_p=0.5` → conservative มาก

---

### `top_k`

```python
top_k = 50  # เลือกจาก 50 tokens ที่น่าจะเป็นที่สุด
top_k = 0   # ปิด top_k filtering
```

**คืออะไร:** เลือกเฉพาะ k tokens ที่มี probability สูงสุด

**แนวทาง:** ใช้ `top_p` หรือ `top_k` อย่างใดอย่างหนึ่ง หรือใช้ทั้งคู่ร่วมกัน

---

### `do_sample`

```python
do_sample = True   # ใช้ sampling (temperature, top_p, top_k มีผล)
do_sample = False  # Greedy decoding (เลือก token ที่น่าจะเป็นที่สุดเสมอ)
```

**แนวทาง:**
- `True` + temperature=0.7 → สำหรับ chat, creative tasks
- `False` → สำหรับ benchmark, reproducibility

---

### `repetition_penalty`

```python
repetition_penalty = 1.1  # ลดโอกาสซ้ำคำ
repetition_penalty = 1.0  # ปิด (ค่าเริ่มต้น)
```

**คืออะไร:** ลด probability ของ tokens ที่เคย generate ไปแล้ว

```
logit_adjusted = logit / repetition_penalty  (ถ้า token เคยปรากฏแล้ว)

1.0 = ไม่มีผล
1.1 = ลดโอกาสซ้ำเล็กน้อย ✅
1.3 = ลดโอกาสซ้ำมาก (อาจทำให้ text แปลก)
```

---

### `pad_token_id`

```python
pad_token_id = tokenizer.eos_token_id
```

ต้องตั้งค่านี้เสมอเพื่อป้องกัน warning และให้ generation ทำงานถูกต้อง

---

## 7. Export Parameters

ใช้ตอน export โมเดลเป็น `.gguf`

---

### `quantization_method` — GGUF Quantization

```python
model.save_pretrained_gguf("output_dir", tokenizer, quantization_method="q4_k_m")
```

**ตัวเลือกทั้งหมด:**

| Method | ขนาด (8B) | VRAM/RAM | คุณภาพ | เหมาะกับ |
|--------|-----------|----------|--------|---------|
| `f32` | ~32GB | สูงมาก | สูงสุด | ไม่แนะนำ |
| `f16` | ~16GB | สูง | สูงมาก | GPU ที่มี VRAM เยอะ |
| `q8_0` | ~8.5GB | สูง | สูง | GPU/CPU RAM 16GB+ |
| `q6_k` | ~6.1GB | กลาง | ดีมาก | CPU RAM 12GB+ |
| `q5_k_m` | ~5.3GB | กลาง | ดี | CPU RAM 10GB+ |
| **`q4_k_m`** | **~4.8GB** | **กลาง** | **ดี** | **แนะนำ (CPU 8GB+)** |
| `q4_0` | ~4.3GB | ต่ำ | ดี | CPU RAM 8GB+ |
| `q3_k_m` | ~3.5GB | ต่ำ | พอใช้ | CPU RAM 6GB+ |
| `q2_k` | ~2.7GB | ต่ำมาก | ต่ำ | CPU RAM 4GB+ |

**ความหมายของชื่อ:**
```
q4_k_m
│ │ │
│ │ └── m = medium (balance ระหว่าง size และ quality)
│ │     s = small, l = large
│ └──── k = k-quant (algorithm ที่ดีกว่า)
└────── 4 = 4-bit quantization
```

**k-quant vs legacy:**
- `q4_k_m` ดีกว่า `q4_0` เพราะ k-quant quantize แต่ละ layer ต่างกัน
  (layers ที่สำคัญกว่าจะได้ precision สูงกว่า)

---

### Export หลาย format พร้อมกัน

```python
model.push_to_hub_gguf(
    "username/model-name",
    tokenizer,
    quantization_method=["q4_k_m", "q8_0", "f16"],  # export ทีเดียวหลาย format
)
```

---

## 8. ตารางเปรียบเทียบ Config ตาม GPU

### Google Colab Free (T4 16GB) — แนะนำ

```python
# โหลดโมเดล
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,   # QLoRA
)

# LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16, lora_alpha=32,
    target_modules="all-linear",
    use_gradient_checkpointing="unsloth",
)

# Training
TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,   # effective batch = 8
    learning_rate=2e-4,
    max_steps=100,                   # หรือ num_train_epochs=1
    optim="adamw_8bit",
    fp16=True,
)
# VRAM ที่ใช้: ~7-9GB
```

---

### Google Colab Pro (A100 40GB)

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b",
    max_seq_length=4096,
    load_in_4bit=False,  # LoRA เต็ม 16-bit
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32, lora_alpha=64,
    target_modules="all-linear",
    use_gradient_checkpointing="unsloth",
)

TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,   # effective batch = 8
    learning_rate=2e-4,
    num_train_epochs=3,
    optim="adamw_8bit",
    bf16=True,
)
# VRAM ที่ใช้: ~20-25GB
```

---

### Local GPU น้อย (RTX 3060 12GB / GTX 1080 Ti 11GB)

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=1024,   # ลด seq_length
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=8, lora_alpha=16,    # ลด rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # เฉพาะ attention
    use_gradient_checkpointing="unsloth",
)

TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,   # effective batch = 8
    learning_rate=2e-4,
    optim="adamw_8bit",
    fp16=True,
)
# VRAM ที่ใช้: ~5-6GB
```

---

### DPO Config (T4 16GB)

```python
DPOConfig(
    beta=0.1,
    max_prompt_length=512,
    max_length=1024,
    per_device_train_batch_size=1,   # DPO ต้องการ VRAM มากกว่า SFT
    gradient_accumulation_steps=8,
    learning_rate=5e-5,              # ต่ำกว่า SFT
    lr_scheduler_type="cosine",
    num_train_epochs=1,
    optim="adamw_8bit",
)
# VRAM ที่ใช้: ~10-12GB
```

---

## 9. สัญญาณเตือนและวิธีแก้

### Loss ไม่ลด (Loss Plateau)

```
สาเหตุที่เป็นไปได้:
1. learning_rate ต่ำเกินไป → เพิ่ม lr เป็น 2-3×
2. r ต่ำเกินไป → เพิ่ม r จาก 8 → 16 หรือ 32
3. Dataset มีปัญหา → ตรวจสอบ format และ EOS token
4. Warmup นานเกินไป → ลด warmup_steps
```

### Loss Spike / NaN

```
สาเหตุที่เป็นไปได้:
1. learning_rate สูงเกินไป → ลด lr เป็น 0.5×
2. ไม่มี warmup → เพิ่ม warmup_steps=5-10
3. Gradient exploding → เพิ่ม max_grad_norm=0.3
4. Dataset มี outlier → ตรวจสอบและ clean data
```

### Out of Memory (OOM)

```
แก้ตามลำดับ:
1. ลด per_device_train_batch_size → 1
2. เพิ่ม gradient_accumulation_steps (ให้ effective batch เท่าเดิม)
3. ลด max_seq_length → 1024
4. ลด r → 8
5. เปลี่ยน target_modules เป็นเฉพาะ attention
6. ตรวจสอบว่าใช้ load_in_4bit=True (QLoRA)
7. ใช้ use_gradient_checkpointing="unsloth"
```

### โมเดลตอบซ้ำๆ (Repetition)

```
แก้ได้ด้วย:
1. เพิ่ม repetition_penalty=1.1-1.3 ตอน generate
2. ลด temperature
3. ตรวจสอบว่าใส่ EOS_TOKEN ท้ายทุก training example
4. เพิ่ม dataset ที่หลากหลายขึ้น
```

---

## 10. Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│  Parameter          │ T4 (16GB)  │ A100 (40GB) │ Local 8GB │
├─────────────────────┼────────────┼─────────────┼───────────┤
│ load_in_4bit        │ True       │ False       │ True      │
│ max_seq_length      │ 2048       │ 4096        │ 1024      │
│ r                   │ 16         │ 32          │ 8         │
│ lora_alpha          │ 32         │ 64          │ 16        │
│ target_modules      │ all-linear │ all-linear  │ attn only │
│ batch_size          │ 2          │ 4           │ 1         │
│ grad_accum          │ 4          │ 2           │ 8         │
│ effective_batch     │ 8          │ 8           │ 8         │
│ learning_rate (SFT) │ 2e-4       │ 2e-4        │ 2e-4      │
│ learning_rate (DPO) │ 5e-5       │ 5e-5        │ 5e-5      │
│ precision           │ fp16       │ bf16        │ fp16      │
│ optimizer           │ adamw_8bit │ adamw_8bit  │ adamw_8bit│
│ VRAM used (SFT)     │ ~7-9GB     │ ~20-25GB    │ ~5-6GB    │
│ VRAM used (DPO)     │ ~10-12GB   │ ~25-30GB    │ OOM ⚠️    │
└─────────────────────┴────────────┴─────────────┴───────────┘
```

---

*อ้างอิง: [Unsloth Docs](https://github.com/unslothai/unsloth) · [TRL Docs](https://huggingface.co/docs/trl) · [QLoRA Paper](https://arxiv.org/abs/2305.14314) · [DPO Paper](https://arxiv.org/abs/2305.18290)*
