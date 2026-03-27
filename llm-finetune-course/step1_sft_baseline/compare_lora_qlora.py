"""
╔══════════════════════════════════════════════════════════════════╗
║  LoRA vs QLoRA: เปรียบเทียบแบบ Side-by-Side                    ║
║  ระดับ: 🟢 เริ่มต้น (ต่อจาก train.py)                           ║
║  เป้าหมาย: เข้าใจความแตกต่างและเลือกใช้ให้ถูกสถานการณ์         ║
╚══════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ทำความเข้าใจก่อนรันโค้ด
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Full Fine-tune  →  เทรนทุก parameter (8B params) → ต้องการ ~80GB VRAM ❌
  LoRA            →  เทรนเฉพาะ adapter             → ต้องการ ~16GB VRAM ✅
  QLoRA           →  LoRA + 4-bit quantized base    → ต้องการ ~6GB VRAM  ✅✅

  LoRA (Low-Rank Adaptation)
  ─────────────────────────
  แนวคิด: แทนที่จะแก้ไข weight matrix W โดยตรง (ขนาด d×d)
          เราเพิ่ม "adapter" ΔW = A × B เข้าไป
          โดย A มีขนาด (d×r) และ B มีขนาด (r×d)
          r คือ "rank" ซึ่งเล็กมาก (เช่น 16)

  ผลลัพธ์: W_new = W_frozen + (alpha/r) × A × B
           เทรนแค่ A และ B  →  ลด trainable params จาก d² เหลือ 2×d×r

  ตัวอย่าง Llama-3 8B:
    d = 4096 (hidden size)
    Full: 4096 × 4096 = 16,777,216 params ต่อ layer
    LoRA r=16: 2 × 4096 × 16 = 131,072 params ต่อ layer  (ลดลง 128x!)

  QLoRA (Quantized LoRA)
  ──────────────────────
  แนวคิด: LoRA ปกติ แต่โหลด base model แบบ 4-bit NF4 quantization
          (NF4 = NormalFloat4 ซึ่งดีกว่า INT4 สำหรับ LLM weights)

  กระบวนการ:
    1. โหลด base model แบบ 4-bit (ประหยัด VRAM 75%)
    2. เพิ่ม LoRA adapter แบบ 16-bit ทับลงไป
    3. เทรนเฉพาะ LoRA adapter (16-bit)
    4. Base model ยังคงเป็น 4-bit ตลอดการเทรน

  ข้อแตกต่างสำคัญ:
  ┌─────────────────┬──────────────────┬──────────────────┐
  │                 │ LoRA             │ QLoRA            │
  ├─────────────────┼──────────────────┼──────────────────┤
  │ Base model      │ 16-bit (bf16)    │ 4-bit (NF4)      │
  │ Adapter         │ 16-bit           │ 16-bit           │
  │ VRAM (8B model) │ ~14-16 GB        │ ~5-6 GB          │
  │ Training speed  │ เร็วกว่า ~20%    │ ช้ากว่าเล็กน้อย  │
  │ Model quality   │ ดีกว่าเล็กน้อย   │ ดีมาก (ใกล้เคียง)│
  │ Colab Free T4   │ อาจ OOM          │ รันได้สบาย ✅    │
  └─────────────────┴──────────────────┴──────────────────┘

  เลือกใช้อะไร?
  - มี GPU ≥ 24GB (A100, RTX 3090+) → LoRA (เร็วกว่า คุณภาพดีกว่า)
  - มี GPU 8-16GB (T4, RTX 3080)    → QLoRA (ประหยัด VRAM)
  - ใช้ Google Colab Free            → QLoRA เสมอ
"""

# ─────────────────────────────────────────────
# 📦 SECTION 1: Imports
# ─────────────────────────────────────────────
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import gc

# ─────────────────────────────────────────────
# ⚙️ SECTION 2: Config ร่วม
# ─────────────────────────────────────────────
MAX_SEQ_LENGTH = 2048
MAX_STEPS = 50          # ใช้ 50 steps เพื่อให้เปรียบเทียบได้เร็ว
LORA_R = 16
LORA_ALPHA = 32

# ─────────────────────────────────────────────
# 📊 SECTION 3: เตรียม Dataset (ใช้ร่วมกัน)
# ─────────────────────────────────────────────
print("📥 โหลด Dataset...")

dataset = load_dataset("yahma/alpaca-cleaned", split="train[:500]")  # ใช้แค่ 500 ตัวอย่างเพื่อความเร็ว

ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


def format_prompts(examples, eos_token):
    texts = []
    for inst, inp, out in zip(examples["instruction"], examples["input"], examples["output"]):
        texts.append(ALPACA_PROMPT.format(inst, inp, out) + eos_token)
    return {"text": texts}


# ─────────────────────────────────────────────
# 🔧 SECTION 4: Helper Functions
# ─────────────────────────────────────────────
def get_gpu_memory_gb():
    """คืนค่า VRAM ที่ใช้อยู่ในหน่วย GB"""
    return round(torch.cuda.max_memory_reserved() / 1024**3, 2)


def reset_gpu_memory():
    """ล้าง GPU memory ระหว่าง experiment"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def run_training(model, tokenizer, dataset, output_dir, label):
    """
    รัน SFTTrainer และวัดผล
    คืนค่า dict ของ metrics: เวลา, VRAM, train loss
    """
    reset_gpu_memory()
    vram_before = get_gpu_memory_gb()

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=MAX_STEPS,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            optim="adamw_8bit",
            logging_steps=10,
            output_dir=output_dir,
            seed=42,
            report_to="none",
        ),
    )

    start_time = time.time()
    stats = trainer.train()
    elapsed = round(time.time() - start_time, 1)
    vram_peak = get_gpu_memory_gb()

    return {
        "label": label,
        "time_sec": elapsed,
        "vram_before_gb": vram_before,
        "vram_peak_gb": vram_peak,
        "train_loss": round(stats.metrics.get("train_loss", 0), 4),
        "samples_per_sec": round(stats.metrics.get("train_samples_per_second", 0), 2),
    }


# ══════════════════════════════════════════════════════════════════
# 🅰️  EXPERIMENT A: QLoRA (แนะนำสำหรับ Colab Free T4)
# ══════════════════════════════════════════════════════════════════
"""
QLoRA Setup:
  - โหลด base model แบบ 4-bit NF4 quantization
  - LoRA adapter ยังคงเป็น 16-bit
  - ใช้ double quantization (quantize ตัว quantization constants ด้วย)
    → ประหยัด VRAM เพิ่มอีก ~0.4 bits/param

  load_in_4bit=True ใน FastLanguageModel = QLoRA โดยอัตโนมัติ
  Unsloth จัดการ NF4 + double quant ให้เองทั้งหมด
"""
print("\n" + "═" * 60)
print("🅰️  EXPERIMENT A: QLoRA (4-bit base model)")
print("═" * 60)

# โหลดโมเดลแบบ 4-bit → นี่คือ QLoRA
qlora_model, qlora_tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,          # ← KEY: True = QLoRA (4-bit base)
)

# LoRA adapter เหมือนกันทุกอย่าง
qlora_model = FastLanguageModel.get_peft_model(
    qlora_model,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules="all-linear",
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

print(f"📊 QLoRA trainable parameters:")
qlora_model.print_trainable_parameters()

# Format dataset
qlora_dataset = dataset.map(
    lambda ex: format_prompts(ex, qlora_tokenizer.eos_token),
    batched=True,
)

# เทรนและวัดผล
qlora_results = run_training(
    qlora_model, qlora_tokenizer, qlora_dataset,
    output_dir="outputs/qlora_model",
    label="QLoRA (4-bit)",
)

# บันทึก weights
qlora_model.save_pretrained("outputs/qlora_model")
qlora_tokenizer.save_pretrained("outputs/qlora_model")
print(f"✅ QLoRA เสร็จ | เวลา: {qlora_results['time_sec']}s | VRAM peak: {qlora_results['vram_peak_gb']} GB")

# ล้าง memory ก่อนรัน experiment ถัดไป
del qlora_model, qlora_tokenizer, qlora_dataset
reset_gpu_memory()


# ══════════════════════════════════════════════════════════════════
# 🅱️  EXPERIMENT B: LoRA (16-bit base model)
# ══════════════════════════════════════════════════════════════════
"""
LoRA Setup:
  - โหลด base model แบบ 16-bit (bf16/fp16) เต็มๆ
  - LoRA adapter เป็น 16-bit เหมือนกัน
  - ต้องการ VRAM มากกว่า QLoRA ~2-3x
  - คุณภาพดีกว่าเล็กน้อย เพราะ base model ไม่ถูก quantize

  ⚠️  ถ้า T4 16GB OOM ให้ลด r เหลือ 8 หรือ target_modules เฉพาะ attention
  ⚠️  ถ้า OOM มากให้ข้าม experiment นี้ไปเลย และใช้ QLoRA อย่างเดียว
"""
print("\n" + "═" * 60)
print("🅱️  EXPERIMENT B: LoRA (16-bit base model)")
print("═" * 60)
print("⚠️  ต้องการ VRAM ~14-16GB | ถ้า OOM ให้ข้ามไป Experiment C")

# โหลดโมเดลแบบ 16-bit → นี่คือ LoRA ปกติ
lora_model, lora_tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b",   # ← ไม่มี -bnb-4bit = โหลด 16-bit
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=False,                 # ← KEY: False = LoRA ปกติ (16-bit base)
)

lora_model = FastLanguageModel.get_peft_model(
    lora_model,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules="all-linear",
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

print(f"📊 LoRA trainable parameters:")
lora_model.print_trainable_parameters()

lora_dataset = dataset.map(
    lambda ex: format_prompts(ex, lora_tokenizer.eos_token),
    batched=True,
)

lora_results = run_training(
    lora_model, lora_tokenizer, lora_dataset,
    output_dir="outputs/lora_model",
    label="LoRA (16-bit)",
)

lora_model.save_pretrained("outputs/lora_model")
lora_tokenizer.save_pretrained("outputs/lora_model")
print(f"✅ LoRA เสร็จ | เวลา: {lora_results['time_sec']}s | VRAM peak: {lora_results['vram_peak_gb']} GB")

del lora_model, lora_tokenizer, lora_dataset
reset_gpu_memory()


# ══════════════════════════════════════════════════════════════════
# 🅲  EXPERIMENT C: QLoRA แบบ Conservative (สำหรับ GPU ที่ VRAM น้อย)
# ══════════════════════════════════════════════════════════════════
"""
ถ้า T4 ยัง OOM ใน Experiment A ให้ใช้ config นี้:
  - r ลดลงเหลือ 8 (ลด trainable params ลงครึ่งหนึ่ง)
  - target_modules เฉพาะ attention layers (ไม่ใส่ MLP)
  - batch_size = 1 + gradient_accumulation = 8

  เหมาะสำหรับ:
  - GPU 6-8GB (RTX 3060, GTX 1080 Ti)
  - Colab Free ที่ได้ T4 ที่มี VRAM เหลือน้อย
"""
print("\n" + "═" * 60)
print("🅲  EXPERIMENT C: QLoRA Conservative (สำหรับ VRAM น้อย)")
print("═" * 60)

conservative_model, conservative_tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

conservative_model = FastLanguageModel.get_peft_model(
    conservative_model,
    r=8,                                          # ← ลดจาก 16 เหลือ 8
    lora_alpha=16,                                # ← ลดตาม (= 2 * r)
    target_modules=["q_proj", "k_proj",
                    "v_proj", "o_proj"],           # ← เฉพาะ attention (ไม่รวม MLP)
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

print(f"📊 QLoRA Conservative trainable parameters:")
conservative_model.print_trainable_parameters()

conservative_dataset = dataset.map(
    lambda ex: format_prompts(ex, conservative_tokenizer.eos_token),
    batched=True,
)

# ใช้ batch_size=1 เพื่อประหยัด VRAM สูงสุด
reset_gpu_memory()
conservative_trainer = SFTTrainer(
    model=conservative_model,
    tokenizer=conservative_tokenizer,
    train_dataset=conservative_dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2,
    args=TrainingArguments(
        per_device_train_batch_size=1,    # ← ลดเหลือ 1
        gradient_accumulation_steps=8,    # ← เพิ่มเป็น 8 (effective batch = 8)
        warmup_steps=5,
        max_steps=MAX_STEPS,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        logging_steps=10,
        output_dir="outputs/qlora_conservative",
        seed=42,
        report_to="none",
    ),
)

start_time = time.time()
conservative_stats = conservative_trainer.train()
elapsed = round(time.time() - start_time, 1)

conservative_results = {
    "label": "QLoRA Conservative (r=8, attention only)",
    "time_sec": elapsed,
    "vram_peak_gb": get_gpu_memory_gb(),
    "train_loss": round(conservative_stats.metrics.get("train_loss", 0), 4),
    "samples_per_sec": round(conservative_stats.metrics.get("train_samples_per_second", 0), 2),
}

conservative_model.save_pretrained("outputs/qlora_conservative")
conservative_tokenizer.save_pretrained("outputs/qlora_conservative")
print(f"✅ QLoRA Conservative เสร็จ | เวลา: {conservative_results['time_sec']}s | VRAM: {conservative_results['vram_peak_gb']} GB")

del conservative_model, conservative_tokenizer, conservative_dataset
reset_gpu_memory()


# ══════════════════════════════════════════════════════════════════
# 📊 SECTION 5: ตารางเปรียบเทียบผลลัพธ์
# ══════════════════════════════════════════════════════════════════
print("\n\n" + "═" * 65)
print("📊 ผลการเปรียบเทียบ LoRA vs QLoRA")
print("═" * 65)

# รวมผลลัพธ์ทั้งหมด (ถ้า lora_results ไม่มีให้ข้าม)
all_results = [qlora_results, conservative_results]
try:
    all_results.insert(1, lora_results)
except NameError:
    pass  # ข้าม LoRA ถ้า OOM

# Header
print(f"\n{'Method':<40} {'VRAM (GB)':<12} {'Time (s)':<12} {'Loss':<10} {'Samples/s'}")
print("─" * 85)

for r in all_results:
    print(
        f"{r['label']:<40} "
        f"{r['vram_peak_gb']:<12} "
        f"{r['time_sec']:<12} "
        f"{r['train_loss']:<10} "
        f"{r['samples_per_sec']}"
    )

print("─" * 85)

# ── คำอธิบายผลลัพธ์ ──
print("""
📝 อ่านผลลัพธ์:

  VRAM (GB)   → ยิ่งน้อยยิ่งดี (ประหยัด GPU memory)
  Time (s)    → เวลาเทรนทั้งหมด (ยิ่งน้อยยิ่งดี)
  Loss        → ยิ่งน้อยยิ่งดี (โมเดลเรียนรู้ได้ดีกว่า)
  Samples/s   → throughput (ยิ่งมากยิ่งดี)

💡 ข้อสังเกต:
  - QLoRA ใช้ VRAM น้อยกว่า LoRA ประมาณ 2-3x
  - Train loss ของทั้งคู่ใกล้เคียงกัน (QLoRA ไม่ได้แย่กว่ามาก)
  - QLoRA อาจช้ากว่าเล็กน้อยเพราะต้อง dequantize ระหว่างเทรน
  - สำหรับ Colab Free T4: QLoRA คือตัวเลือกที่ดีที่สุด
""")


# ══════════════════════════════════════════════════════════════════
# 🧪 SECTION 6: เปรียบเทียบคุณภาพคำตอบ (Qualitative)
# ══════════════════════════════════════════════════════════════════
"""
นอกจาก metrics เชิงตัวเลข เราควรเปรียบเทียบคุณภาพคำตอบด้วย
โดยถามคำถามเดิมกับทั้ง QLoRA และ LoRA แล้วดูว่าต่างกันไหม
"""
print("\n" + "═" * 65)
print("🧪 เปรียบเทียบคุณภาพคำตอบ")
print("═" * 65)

TEST_PROMPT = ALPACA_PROMPT.format(
    "อธิบายความแตกต่างระหว่าง list และ tuple ใน Python",
    "",
    "",
)

def generate_response(model_path, prompt, label):
    """โหลดโมเดลและ generate คำตอบ"""
    print(f"\n📤 {label}:")
    print("─" * 50)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response[len(prompt):].strip())
    print("─" * 50)

    del model, tokenizer
    reset_gpu_memory()


generate_response("outputs/qlora_model", TEST_PROMPT, "QLoRA (4-bit base)")
generate_response("outputs/qlora_conservative", TEST_PROMPT, "QLoRA Conservative (r=8)")

# LoRA ถ้ามี
import os
if os.path.exists("outputs/lora_model"):
    generate_response("outputs/lora_model", TEST_PROMPT, "LoRA (16-bit base)")

# ══════════════════════════════════════════════════════════════════
# 📋 SECTION 7: สรุปและคำแนะนำ
# ══════════════════════════════════════════════════════════════════
print("""
╔══════════════════════════════════════════════════════════════════╗
║  📋 สรุปคำแนะนำการเลือกใช้                                      ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  สถานการณ์                    → เลือกใช้                         ║
║  ─────────────────────────────────────────────────────────────  ║
║  Colab Free (T4 16GB)         → QLoRA (load_in_4bit=True)       ║
║  Colab Pro (A100 40GB)        → LoRA (load_in_4bit=False)       ║
║  Local GPU < 12GB             → QLoRA Conservative (r=8)        ║
║  Local GPU ≥ 24GB             → LoRA (r=32 หรือมากกว่า)         ║
║  Production / Best quality    → LoRA r=64 บน A100               ║
║                                                                  ║
║  Rule of thumb:                                                  ║
║  "ถ้า VRAM พอ → LoRA, ถ้าไม่พอ → QLoRA"                        ║
║  คุณภาพต่างกันน้อยมาก แต่ VRAM ต่างกันมาก                       ║
╚══════════════════════════════════════════════════════════════════╝
""")
