"""
╔══════════════════════════════════════════════════════════════════╗
║  STEP 1: SFT Baseline ด้วย Unsloth                              ║
║  ระดับ: 🟢 เริ่มต้น                                              ║
║  เป้าหมาย: Fine-tune Llama-3 8B แบบพื้นฐานบน T4 GPU 16GB       ║
╚══════════════════════════════════════════════════════════════════╝

สิ่งที่จะได้เรียนใน Step นี้:
1. โหลดโมเดลแบบ 4-bit quantization ด้วย Unsloth
2. ตั้งค่า LoRA Adapter
3. โหลด Dataset และ Format ข้อมูล
4. รัน SFTTrainer
"""

# ─────────────────────────────────────────────
# 📦 SECTION 1: Import Libraries
# ─────────────────────────────────────────────
from unsloth import FastLanguageModel  # หัวใจหลัก: โหลดโมเดลแบบ 4-bit อย่างรวดเร็ว
from trl import SFTTrainer             # Trainer สำหรับ Supervised Fine-Tuning
from transformers import TrainingArguments
from datasets import load_dataset
import torch

# ─────────────────────────────────────────────
# ⚙️ SECTION 2: ตั้งค่าพารามิเตอร์หลัก
# ─────────────────────────────────────────────

# ชื่อโมเดลบน HuggingFace Hub
# เปลี่ยนได้เป็น: "unsloth/llama-3-8b-bnb-4bit", "unsloth/mistral-7b-bnb-4bit"
MODEL_NAME = "unsloth/llama-3-8b-bnb-4bit"

# max_seq_length: ความยาวสูงสุดของ token ที่โมเดลจะรับได้ต่อ 1 ตัวอย่าง
# - ยิ่งสูง = จำบริบทได้มากขึ้น แต่กินหน่วยความจำมากขึ้น
# - สำหรับ T4 16GB แนะนำ 2048
# - ถ้า Out of Memory ให้ลดเหลือ 1024
MAX_SEQ_LENGTH = 2048

# dtype: ชนิดข้อมูลของ weights
# - None = ให้ Unsloth เลือกอัตโนมัติ (แนะนำ)
# - torch.float16 สำหรับ T4, torch.bfloat16 สำหรับ A100
DTYPE = None

# load_in_4bit: โหลดโมเดลแบบ 4-bit quantization
# - True = ประหยัด VRAM ~75% (8B model ใช้แค่ ~5GB แทน ~16GB)
# - ต้องใช้ bitsandbytes library
LOAD_IN_4BIT = True

# ─────────────────────────────────────────────
# 🚀 SECTION 3: โหลดโมเดลและ Tokenizer
# ─────────────────────────────────────────────
print("📥 กำลังโหลดโมเดล... (อาจใช้เวลา 2-5 นาทีครั้งแรก)")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
    # token="hf_...",  # ใส่ HuggingFace token ถ้าโมเดลต้องการ login (เช่น Llama-3)
)

print(f"✅ โหลดโมเดลสำเร็จ: {MODEL_NAME}")

# ─────────────────────────────────────────────
# 🔧 SECTION 4: ตั้งค่า LoRA Adapter
# ─────────────────────────────────────────────
"""
LoRA (Low-Rank Adaptation) คือเทคนิคที่เพิ่ม "adapter" ขนาดเล็กเข้าไปในโมเดล
แทนที่จะ Fine-tune พารามิเตอร์ทั้งหมด (8B params) เราเทรนแค่ adapter เล็กๆ

พารามิเตอร์สำคัญ:
- r (rank): ขนาดของ LoRA matrix
  * ยิ่งสูง = โมเดลเรียนรู้ได้มากขึ้น แต่ใช้ VRAM มากขึ้น
  * ค่าแนะนำ: 8, 16, 32, 64
  * สำหรับ T4: ใช้ 16

- lora_alpha: scaling factor สำหรับ LoRA weights
  * มักตั้งเป็น 2x ของ r (เช่น r=16 → alpha=32)
  * ควบคุมว่า LoRA มีอิทธิพลแค่ไหนต่อโมเดลเดิม

- target_modules: layer ไหนบ้างที่จะใส่ LoRA
  * "all-linear" = ใส่ทุก linear layer (แนะนำ)
  * หรือระบุเฉพาะ: ["q_proj", "v_proj"] (ประหยัด VRAM กว่า)

- lora_dropout: dropout rate สำหรับ LoRA (0 = ไม่ใช้ dropout)

- bias: "none" = ไม่เทรน bias (แนะนำ ประหยัด memory)

- use_gradient_checkpointing: "unsloth" = ประหยัด VRAM เพิ่มอีก 30%
"""

model = FastLanguageModel.get_peft_model(
    model,
    r=16,                          # rank ของ LoRA
    lora_alpha=32,                 # scaling factor (แนะนำ = 2 * r)
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],,   # ใส่ LoRA ทุก linear layer
    lora_dropout=0,                # 0 = ไม่ใช้ dropout (เร็วกว่า)
    bias="none",                   # ไม่เทรน bias
    use_gradient_checkpointing="unsloth",  # ประหยัด VRAM
    random_state=42,               # seed สำหรับ reproducibility
)

# แสดงจำนวน parameters ที่จะเทรน
model.print_trainable_parameters()
# ตัวอย่างผลลัพธ์: trainable params: 41,943,040 || all params: 8,072,204,288 || trainable%: 0.52%

# ─────────────────────────────────────────────
# 📊 SECTION 5: โหลดและ Format Dataset
# ─────────────────────────────────────────────
"""
ใช้ Dataset รูปแบบ Alpaca (instruction-following)
โครงสร้าง:
{
    "instruction": "คำสั่งหรือคำถาม",
    "input": "ข้อมูลเพิ่มเติม (ถ้ามี)",
    "output": "คำตอบที่ต้องการ"
}
"""

# โหลด dataset จาก HuggingFace Hub
# "yahma/alpaca-cleaned" = Alpaca dataset ที่ clean แล้ว ~52K ตัวอย่าง
dataset = load_dataset("yahma/alpaca-cleaned", split="train")

# Alpaca Prompt Template
# EOS_TOKEN สำคัญมาก! ต้องใส่ท้ายทุก example ไม่งั้นโมเดลจะไม่รู้จักจุดสิ้นสุด
EOS_TOKEN = tokenizer.eos_token  # </s> หรือ <|end_of_text|>

ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


def format_prompts(examples):
    """
    แปลง dataset แต่ละแถวให้เป็น text ตาม Alpaca format
    
    examples: dict ที่มี key "instruction", "input", "output"
    return: dict ที่มี key "text" สำหรับ SFTTrainer
    """
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]

    texts = []
    for instruction, inp, output in zip(instructions, inputs, outputs):
        # รวม prompt + คำตอบ + EOS token
        text = ALPACA_PROMPT.format(instruction, inp, output) + EOS_TOKEN
        texts.append(text)

    return {"text": texts}


# Apply formatting ให้ทั้ง dataset (batched=True = เร็วกว่า)
dataset = dataset.map(format_prompts, batched=True)

print(f"✅ Dataset โหลดสำเร็จ: {len(dataset)} ตัวอย่าง")
print("\n📝 ตัวอย่างข้อมูล:")
print(dataset[0]["text"][:500])

# ─────────────────────────────────────────────
# 🏋️ SECTION 6: ตั้งค่าและรัน SFTTrainer
# ─────────────────────────────────────────────
"""
TrainingArguments: ตั้งค่าการเทรนทั้งหมด

พารามิเตอร์สำคัญ:
- per_device_train_batch_size: จำนวน sample ต่อ batch ต่อ GPU
  * ยิ่งสูง = เทรนเร็วขึ้น แต่กิน VRAM มากขึ้น
  * T4 16GB: ใช้ 2 (ถ้า OOM ลดเหลือ 1)

- gradient_accumulation_steps: สะสม gradient กี่ step ก่อน update
  * ใช้แทน batch_size ที่ใหญ่กว่า
  * effective_batch_size = batch_size × gradient_accumulation_steps
  * เช่น batch=2, accum=4 → effective batch = 8

- warmup_steps: จำนวน step แรกที่ค่อยๆ เพิ่ม learning rate
  * ช่วยให้การเทรนเสถียรขึ้นในช่วงแรก

- max_steps: จำนวน step สูงสุด (-1 = เทรนจนครบ epoch)
  * สำหรับทดสอบ: ใช้ 100-200 steps
  * สำหรับเทรนจริง: ใช้ -1 หรือ num_train_epochs

- learning_rate: อัตราการเรียนรู้
  * LoRA แนะนำ: 2e-4 (สูงกว่า full fine-tune ได้)

- fp16/bf16: ใช้ mixed precision training
  * fp16 สำหรับ T4 (Turing architecture)
  * bf16 สำหรับ A100/H100 (Ampere+)

- optim: optimizer ที่ใช้
  * "adamw_8bit" = AdamW แบบ 8-bit (ประหยัด VRAM)

- logging_steps: log ทุกกี่ step
"""

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",   # ชื่อ column ที่เก็บ text
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2,          # จำนวน CPU process สำหรับ tokenize
    packing=False,               # False = ง่ายกว่า, True = เร็วกว่าแต่ซับซ้อน
    args=TrainingArguments(
        # ── Batch & Gradient ──
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,   # effective batch = 2×4 = 8

        # ── Learning Rate Schedule ──
        warmup_steps=5,
        max_steps=100,           # เปลี่ยนเป็น -1 เพื่อเทรนจนครบ epoch
        learning_rate=2e-4,
        lr_scheduler_type="linear",

        # ── Precision ──
        fp16=not torch.cuda.is_bf16_supported(),  # ใช้ fp16 ถ้าไม่รองรับ bf16
        bf16=torch.cuda.is_bf16_supported(),

        # ── Optimizer ──
        optim="adamw_8bit",      # ประหยัด VRAM กว่า adamw ปกติ

        # ── Logging & Saving ──
        logging_steps=10,
        output_dir="outputs",
        save_strategy="steps",
        save_steps=50,

        # ── Misc ──
        seed=42,
        report_to="none",        # เปลี่ยนเป็น "wandb" ถ้าต้องการ tracking
    ),
)

# แสดงสถิติ GPU ก่อนเทรน
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"\n🖥️  GPU: {gpu_stats.name}")
print(f"💾 VRAM ที่ใช้อยู่: {start_gpu_memory} GB / {max_memory} GB")

# ── เริ่มเทรน! ──
print("\n🏋️  เริ่มการเทรน...")
trainer_stats = trainer.train()

# แสดงสถิติหลังเทรน
end_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
print(f"\n✅ เทรนเสร็จสิ้น!")
print(f"⏱️  เวลาที่ใช้: {trainer_stats.metrics['train_runtime']:.2f} วินาที")
print(f"💾 VRAM สูงสุดที่ใช้: {end_gpu_memory} GB")

# ─────────────────────────────────────────────
# 💾 SECTION 7: บันทึก LoRA Weights
# ─────────────────────────────────────────────
print("\n💾 กำลังบันทึก LoRA weights...")

# บันทึกเฉพาะ LoRA adapter (ขนาดเล็ก ~100MB)
model.save_pretrained("outputs/lora_model")
tokenizer.save_pretrained("outputs/lora_model")

print("✅ บันทึกสำเร็จที่ outputs/lora_model/")

# ─────────────────────────────────────────────
# 🧪 SECTION 8: ทดสอบโมเดลที่เทรนแล้ว
# ─────────────────────────────────────────────
print("\n🧪 ทดสอบโมเดล...")

# เปิด inference mode (เร็วกว่า training mode)
FastLanguageModel.for_inference(model)

# สร้าง prompt ทดสอบ
test_prompt = ALPACA_PROMPT.format(
    "อธิบายความแตกต่างระหว่าง Machine Learning และ Deep Learning",  # instruction
    "",  # input (ว่างเปล่า)
    "",  # output (ว่างเปล่า - ให้โมเดลเติม)
)

inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")

# Generate คำตอบ
outputs = model.generate(
    **inputs,
    max_new_tokens=256,      # จำนวน token สูงสุดที่จะ generate
    temperature=0.7,         # ความ "สร้างสรรค์" (0=deterministic, 1=random)
    do_sample=True,          # True = sampling, False = greedy
    pad_token_id=tokenizer.eos_token_id,
)

# Decode และแสดงผล
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n📤 คำตอบจากโมเดล:")
print("─" * 50)
print(response[len(test_prompt):])  # แสดงเฉพาะส่วนที่โมเดล generate
print("─" * 50)
print("\n🎉 Step 1 เสร็จสมบูรณ์! ไปต่อที่ Step 2 ได้เลย")
