"""
╔══════════════════════════════════════════════════════════════════╗
║  STEP 3: Preference Alignment ด้วย DPO                          ║
║  ระดับ: 🔴 ขั้นสูง                                               ║
║  เป้าหมาย: สอนโมเดลให้ตอบตรงใจมนุษย์ด้วย DPO                   ║
╚══════════════════════════════════════════════════════════════════╝

DPO (Direct Preference Optimization) คือวิธีที่ง่ายกว่า RLHF แบบดั้งเดิม
แทนที่จะต้องเทรน Reward Model แยก DPO เทรนโดยตรงจาก preference data

Pipeline:
1. โหลด SFT model จาก Step 2 (หรือเทรนใหม่)
2. เตรียม DPO dataset (prompt, chosen, rejected)
3. รัน DPOTrainer
"""

# ─────────────────────────────────────────────
# 📦 SECTION 1: Import Libraries
# ─────────────────────────────────────────────
from unsloth import FastLanguageModel, PatchDPOTrainer
from unsloth.chat_templates import get_chat_template
from trl import DPOTrainer, DPOConfig
from datasets import Dataset, load_dataset
import torch

# Patch DPOTrainer ให้ทำงานกับ Unsloth ได้ (สำคัญ!)
PatchDPOTrainer()

# ─────────────────────────────────────────────
# ⚙️ SECTION 2: Config
# ─────────────────────────────────────────────
MODEL_NAME = "unsloth/llama-3-8b-bnb-4bit"
MAX_SEQ_LENGTH = 2048
DTYPE = None
LOAD_IN_4BIT = True

# ─────────────────────────────────────────────
# 🚀 SECTION 3: โหลดโมเดล
# ─────────────────────────────────────────────
"""
ตัวเลือกการโหลดโมเดลสำหรับ DPO:

Option A: โหลดจาก SFT checkpoint (แนะนำ)
    model_name = "outputs/lora_model_chat"  # จาก Step 2

Option B: โหลด base model แล้วเทรน DPO โดยตรง
    model_name = "unsloth/llama-3-8b-bnb-4bit"
    (ไม่แนะนำ แต่ใช้ได้ถ้าไม่มี SFT checkpoint)
"""
print("📥 กำลังโหลดโมเดล...")

model, tokenizer = FastLanguageModel.from_pretrained(
    # เปลี่ยนเป็น "outputs/lora_model_chat" ถ้าทำ Step 2 แล้ว
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
)

tokenizer = get_chat_template(tokenizer, chat_template="llama-3")

# ─────────────────────────────────────────────
# 🔧 SECTION 4: ตั้งค่า LoRA สำหรับ DPO
# ─────────────────────────────────────────────
"""
สำหรับ DPO ใช้ r ที่เล็กกว่า SFT ได้ เพราะเราแค่ "ปรับ" พฤติกรรม
ไม่ได้สอนทักษะใหม่ทั้งหมด
"""
model = FastLanguageModel.get_peft_model(
    model,
    r=8,                           # เล็กกว่า SFT (16 → 8)
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # เฉพาะ attention
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# ─────────────────────────────────────────────
# 📊 SECTION 5: เตรียม DPO Dataset
# ─────────────────────────────────────────────
"""
DPO Dataset ต้องมี 3 columns:
1. prompt:   คำถาม/บริบท (string หรือ list of messages)
2. chosen:   คำตอบที่ดี (มนุษย์ prefer)
3. rejected: คำตอบที่แย่ (มนุษย์ไม่ prefer)

ตัวอย่าง Dataset จริงที่ใช้ได้:
- "Anthropic/hh-rlhf" (English)
- "Intel/orca_dpo_pairs" (English)
- "argilla/ultrafeedback-binarized-preferences" (English)
"""

# ── ตัวเลือก A: โหลดจาก HuggingFace ──
# raw_dataset = load_dataset("Intel/orca_dpo_pairs", split="train[:1000]")
# จากนั้น format ให้ตรงกับ prompt/chosen/rejected

# ── ตัวเลือก B: สร้าง Dataset เองสำหรับสาธิต ──
"""
ตัวอย่างนี้แสดงให้เห็นว่า DPO เรียนรู้ความแตกต่างระหว่าง:
- chosen: คำตอบที่ละเอียด มีตัวอย่าง เป็นประโยชน์
- rejected: คำตอบที่สั้นเกินไป ไม่มีรายละเอียด หรือผิด
"""
dpo_examples = [
    {
        "prompt": "อธิบาย recursion ใน Python พร้อมตัวอย่าง",
        "chosen": (
            "Recursion คือการที่ฟังก์ชันเรียกตัวเองซ้ำๆ จนกว่าจะถึง base case\n\n"
            "ตัวอย่าง factorial:\n"
            "```python\n"
            "def factorial(n):\n"
            "    if n <= 1:  # base case\n"
            "        return 1\n"
            "    return n * factorial(n - 1)  # recursive call\n\n"
            "print(factorial(5))  # 120\n"
            "```\n\n"
            "สำคัญ: ต้องมี base case เสมอ ไม่งั้นจะเกิด RecursionError"
        ),
        "rejected": "recursion คือฟังก์ชันที่เรียกตัวเอง",
    },
    {
        "prompt": "วิธีจัดการ exception ใน Python",
        "chosen": (
            "ใช้ try-except block:\n\n"
            "```python\n"
            "try:\n"
            "    result = 10 / 0\n"
            "except ZeroDivisionError as e:\n"
            "    print(f'Error: {e}')\n"
            "except Exception as e:\n"
            "    print(f'Unexpected error: {e}')\n"
            "else:\n"
            "    print('สำเร็จ!')  # รันถ้าไม่มี exception\n"
            "finally:\n"
            "    print('รันเสมอ')  # cleanup code\n"
            "```\n\n"
            "Best practice: ระบุ exception type ให้ชัดเจน อย่าใช้ bare except"
        ),
        "rejected": "ใช้ try except แล้วก็จับ error",
    },
    {
        "prompt": "ความแตกต่างระหว่าง list และ tuple ใน Python",
        "chosen": (
            "ความแตกต่างหลัก:\n\n"
            "List []: mutable (แก้ไขได้)\n"
            "- my_list = [1, 2, 3]\n"
            "- my_list.append(4)  # ได้\n\n"
            "Tuple (): immutable (แก้ไขไม่ได้)\n"
            "- my_tuple = (1, 2, 3)\n"
            "- my_tuple[0] = 4  # TypeError!\n\n"
            "เมื่อไหร่ใช้อะไร:\n"
            "- List: ข้อมูลที่ต้องเปลี่ยนแปลง\n"
            "- Tuple: ข้อมูลคงที่ เช่น coordinates, config\n"
            "- Tuple เร็วกว่าและประหยัด memory กว่า"
        ),
        "rejected": "list ใช้ [] tuple ใช้ () list แก้ได้ tuple แก้ไม่ได้",
    },
    {
        "prompt": "อธิบาย decorator ใน Python",
        "chosen": (
            "Decorator คือ function ที่ wrap function อื่น เพื่อเพิ่มพฤติกรรม\n\n"
            "```python\n"
            "def timer(func):\n"
            "    import time\n"
            "    def wrapper(*args, **kwargs):\n"
            "        start = time.time()\n"
            "        result = func(*args, **kwargs)\n"
            "        print(f'ใช้เวลา: {time.time()-start:.2f}s')\n"
            "        return result\n"
            "    return wrapper\n\n"
            "@timer  # เหมือนกับ my_func = timer(my_func)\n"
            "def my_func():\n"
            "    time.sleep(1)\n\n"
            "my_func()  # ใช้เวลา: 1.00s\n"
            "```\n\n"
            "ใช้บ่อยใน: logging, authentication, caching"
        ),
        "rejected": "decorator ใช้ @ แล้วก็ wrap function",
    },
]

# สร้าง Dataset
dataset = Dataset.from_list(dpo_examples)

print(f"✅ DPO Dataset: {len(dataset)} ตัวอย่าง")
print("\n📝 ตัวอย่าง DPO data:")
print(f"Prompt:   {dataset[0]['prompt']}")
print(f"Chosen:   {dataset[0]['chosen'][:100]}...")
print(f"Rejected: {dataset[0]['rejected']}")

# ─────────────────────────────────────────────
# 🔄 SECTION 6: Format Dataset สำหรับ DPOTrainer
# ─────────────────────────────────────────────
"""
DPOTrainer รองรับ 2 รูปแบบ:

รูปแบบ 1: String format (ง่ายกว่า)
    prompt: "คำถาม"
    chosen: "คำตอบดี"
    rejected: "คำตอบแย่"

รูปแบบ 2: Chat format (แนะนำสำหรับ chat model)
    prompt: [{"role": "user", "content": "คำถาม"}]
    chosen: [{"role": "assistant", "content": "คำตอบดี"}]
    rejected: [{"role": "assistant", "content": "คำตอบแย่"}]

เราจะใช้รูปแบบ 2 เพื่อให้ตรงกับ chat template
"""

def format_dpo_chat(examples):
    """
    แปลง string format เป็น chat format
    DPOTrainer จะ apply chat template ให้อัตโนมัติ
    """
    prompts = []
    chosens = []
    rejecteds = []

    for prompt, chosen, rejected in zip(
        examples["prompt"], examples["chosen"], examples["rejected"]
    ):
        # prompt เป็น list of messages (user turn)
        prompts.append([{"role": "user", "content": prompt}])
        # chosen/rejected เป็น list of messages (assistant turn)
        chosens.append([{"role": "assistant", "content": chosen}])
        rejecteds.append([{"role": "assistant", "content": rejected}])

    return {
        "prompt": prompts,
        "chosen": chosens,
        "rejected": rejecteds,
    }


dataset = dataset.map(format_dpo_chat, batched=True)

# ─────────────────────────────────────────────
# 🏋️ SECTION 7: ตั้งค่าและรัน DPOTrainer
# ─────────────────────────────────────────────
"""
DPOConfig พารามิเตอร์สำคัญ:

- beta: temperature parameter สำหรับ DPO loss
  * ยิ่งสูง = โมเดลเปลี่ยนแปลงน้อยลงจาก reference model
  * ยิ่งต่ำ = โมเดลเปลี่ยนแปลงมากขึ้น (อาจ overfit)
  * ค่าแนะนำ: 0.1 - 0.5

- max_prompt_length: ความยาวสูงสุดของ prompt
- max_length: ความยาวสูงสุดของ prompt + response รวมกัน

DPO ต้องการ VRAM มากกว่า SFT เพราะต้องโหลด:
1. Policy model (กำลังเทรน)
2. Reference model (frozen copy ของ SFT model)
"""

dpo_config = DPOConfig(
    # ── DPO Specific ──
    beta=0.1,                    # DPO temperature (ค่าแนะนำ: 0.1-0.5)
    max_prompt_length=512,       # ความยาวสูงสุดของ prompt
    max_length=1024,             # ความยาวสูงสุดรวม (prompt + response)

    # ── Batch & Gradient ──
    per_device_train_batch_size=1,   # DPO ต้องการ VRAM มากกว่า ใช้ 1
    gradient_accumulation_steps=8,   # effective batch = 1×8 = 8

    # ── Learning Rate ──
    warmup_ratio=0.1,
    num_train_epochs=1,
    learning_rate=5e-5,              # DPO ใช้ lr ต่ำกว่า SFT
    lr_scheduler_type="cosine",

    # ── Precision ──
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),

    # ── Optimizer ──
    optim="adamw_8bit",

    # ── Logging & Saving ──
    logging_steps=5,
    output_dir="outputs",
    save_strategy="epoch",

    # ── Misc ──
    seed=42,
    report_to="none",
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,      # None = ใช้ copy ของ model เป็น reference (ประหยัด VRAM)
    args=dpo_config,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

print("\n🏋️  เริ่มการเทรนด้วย DPO...")
print("⚠️  DPO ใช้ VRAM มากกว่า SFT ถ้า OOM ให้ลด max_length หรือ max_prompt_length")

trainer_stats = trainer.train()

print(f"\n✅ DPO เทรนเสร็จสิ้น!")
print(f"⏱️  เวลาที่ใช้: {trainer_stats.metrics['train_runtime']:.2f} วินาที")

# ─────────────────────────────────────────────
# 💾 SECTION 8: บันทึกโมเดล
# ─────────────────────────────────────────────
model.save_pretrained("outputs/lora_model_dpo")
tokenizer.save_pretrained("outputs/lora_model_dpo")
print("✅ บันทึกสำเร็จที่ outputs/lora_model_dpo/")

# ─────────────────────────────────────────────
# 🧪 SECTION 9: เปรียบเทียบ SFT vs DPO
# ─────────────────────────────────────────────
"""
ทดสอบว่า DPO ช่วยให้คำตอบดีขึ้นจริงไหม
โดยถามคำถามเดิมกับทั้ง SFT model และ DPO model
"""
print("\n🧪 ทดสอบโมเดลหลัง DPO...")

FastLanguageModel.for_inference(model)

test_messages = [
    {"role": "user", "content": "อธิบาย recursion ใน Python พร้อมตัวอย่าง"},
]

input_text = tokenizer.apply_chat_template(
    test_messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=300,
    temperature=0.7,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
)

response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("\n📤 คำตอบจากโมเดลหลัง DPO:")
print("─" * 50)
print(response)
print("─" * 50)
print("\n🎉 Step 3 เสร็จสมบูรณ์! ไปต่อที่ Step 4 (Export) ได้เลย")
