"""
╔══════════════════════════════════════════════════════════════════╗
║  STEP 2: Advanced Chat Formatting & Loss Masking                ║
║  ระดับ: 🟡 กลาง                                                  ║
║  เป้าหมาย: เทรนโมเดลแบบ Multi-turn Chat + Loss Masking          ║
╚══════════════════════════════════════════════════════════════════╝

ความแตกต่างจาก Step 1:
- ใช้ Chat Template มาตรฐาน (Llama-3 format) แทน Alpaca
- ใช้ DataCollatorForCompletionOnlyLM เพื่อ mask loss ฝั่ง User
- รองรับ Multi-turn conversation (หลาย turn ใน 1 ตัวอย่าง)
"""

# ─────────────────────────────────────────────
# 📦 SECTION 1: Import Libraries
# ─────────────────────────────────────────────
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template  # helper สำหรับ chat templates
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments
from datasets import load_dataset, Dataset
import torch

# ─────────────────────────────────────────────
# ⚙️ SECTION 2: Config
# ─────────────────────────────────────────────
MODEL_NAME = "unsloth/llama-3-8b-bnb-4bit"
MAX_SEQ_LENGTH = 2048
DTYPE = None
LOAD_IN_4BIT = True

# ─────────────────────────────────────────────
# 🚀 SECTION 3: โหลดโมเดลและตั้งค่า Chat Template
# ─────────────────────────────────────────────
print("📥 กำลังโหลดโมเดล...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
)

# ตั้งค่า Chat Template ให้ tokenizer
# get_chat_template รองรับ: "llama-3", "chatml", "mistral", "gemma", "phi-3"
tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3",  # ใช้ Llama-3 format
    mapping={                  # mapping ชื่อ role
        "role": "role",
        "content": "content",
        "user": "user",
        "assistant": "assistant",
    },
)

print("✅ โหลดโมเดลและตั้งค่า Chat Template สำเร็จ")

# ─────────────────────────────────────────────
# 🔧 SECTION 4: ตั้งค่า LoRA
# ─────────────────────────────────────────────
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# ─────────────────────────────────────────────
# 📊 SECTION 5: สร้าง Multi-turn Chat Dataset
# ─────────────────────────────────────────────
"""
รูปแบบข้อมูล Multi-turn Chat:
แต่ละตัวอย่างคือ list ของ messages โดยแต่ละ message มี:
- role: "system", "user", หรือ "assistant"
- content: ข้อความ

ตัวอย่าง 1 turn:
[
    {"role": "user", "content": "คำถาม"},
    {"role": "assistant", "content": "คำตอบ"}
]

ตัวอย่าง Multi-turn (2 turns):
[
    {"role": "system", "content": "คุณคือผู้ช่วย AI"},
    {"role": "user", "content": "คำถามที่ 1"},
    {"role": "assistant", "content": "คำตอบที่ 1"},
    {"role": "user", "content": "คำถามที่ 2"},
    {"role": "assistant", "content": "คำตอบที่ 2"},
]
"""

# ── ตัวเลือก A: โหลด Dataset จาก HuggingFace ──
# dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:5000]")

# ── ตัวเลือก B: สร้าง Dataset เองสำหรับสาธิต ──
sample_conversations = [
    {
        "conversations": [
            {"role": "system", "content": "คุณคือผู้ช่วย AI ที่เชี่ยวชาญด้านการเขียนโปรแกรม"},
            {"role": "user", "content": "Python คืออะไร?"},
            {"role": "assistant", "content": "Python คือภาษาโปรแกรมมิ่งระดับสูงที่อ่านง่าย เหมาะสำหรับผู้เริ่มต้น มีไลบรารีมากมายสำหรับ Data Science, Web Development, และ AI"},
            {"role": "user", "content": "แล้ว Python ต่างจาก JavaScript ยังไง?"},
            {"role": "assistant", "content": "ความแตกต่างหลักคือ:\n1. Python ใช้สำหรับ backend, data science, AI\n2. JavaScript ใช้สำหรับ web frontend เป็นหลัก\n3. Python มี syntax ที่อ่านง่ายกว่า\n4. JavaScript รันบน browser ได้โดยตรง"},
        ]
    },
    {
        "conversations": [
            {"role": "user", "content": "อธิบาย recursion ให้หน่อย"},
            {"role": "assistant", "content": "Recursion คือการที่ฟังก์ชันเรียกตัวเองซ้ำๆ\n\nตัวอย่าง factorial:\n```python\ndef factorial(n):\n    if n == 0:  # base case\n        return 1\n    return n * factorial(n-1)  # recursive call\n```\n\nสำคัญ: ต้องมี base case เสมอ ไม่งั้น infinite loop!"},
        ]
    },
    {
        "conversations": [
            {"role": "system", "content": "ตอบเป็นภาษาไทยเสมอ"},
            {"role": "user", "content": "What is machine learning?"},
            {"role": "assistant", "content": "Machine Learning คือสาขาของ AI ที่ให้คอมพิวเตอร์เรียนรู้จากข้อมูลโดยไม่ต้องโปรแกรมทุกอย่างด้วยมือ\n\nแบ่งเป็น 3 ประเภทหลัก:\n1. Supervised Learning - เรียนจากข้อมูลที่มี label\n2. Unsupervised Learning - หา pattern จากข้อมูลที่ไม่มี label\n3. Reinforcement Learning - เรียนจาก reward/punishment"},
        ]
    },
]

# สร้าง Dataset object
raw_dataset = Dataset.from_list(sample_conversations)

# ─────────────────────────────────────────────
# 🔄 SECTION 6: Apply Chat Template
# ─────────────────────────────────────────────
def apply_chat_template(examples):
    """
    แปลง conversations เป็น text ตาม Chat Template ที่ตั้งค่าไว้
    
    tokenizer.apply_chat_template จะแปลง list of messages เป็น string
    ตาม format ของโมเดลนั้นๆ (Llama-3, ChatML, ฯลฯ)
    
    tokenize=False: คืนค่าเป็น string (ไม่ tokenize ตอนนี้)
    add_generation_prompt=False: ไม่เพิ่ม prompt สำหรับ generation
    """
    texts = tokenizer.apply_chat_template(
        examples["conversations"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": texts}


dataset = raw_dataset.map(apply_chat_template, batched=True)

print("\n📝 ตัวอย่าง text หลัง apply chat template:")
print("─" * 60)
print(dataset[0]["text"])
print("─" * 60)

# ─────────────────────────────────────────────
# 🎭 SECTION 7: Loss Masking ด้วย DataCollator
# ─────────────────────────────────────────────
"""
DataCollatorForCompletionOnlyLM คือ collator พิเศษที่:
1. หา "response template" ใน token sequence
2. Mask (ตั้งเป็น -100) ทุก token ก่อน response template
3. คำนวณ Loss เฉพาะ token หลัง response template

ทำไม -100? เพราะ PyTorch CrossEntropyLoss จะ ignore index ที่เป็น -100

สำหรับ Llama-3 format:
- response_template = "<|start_header_id|>assistant<|end_header_id|>"
- หมายความว่า Loss จะคิดเฉพาะหลัง token นี้ (คำตอบของ assistant)

ตัวอย่าง:
Input:  [user_tokens] [assistant_header] [answer_tokens] [eot]
Loss:   [-100 ...]    [-100 ...]         [loss ...]      [loss]
                                          ↑ คิด Loss เฉพาะส่วนนี้
"""

# Response template สำหรับ Llama-3
# ต้องตรงกับ token ที่ tokenizer ใช้จริงๆ
response_template = "<|start_header_id|>assistant<|end_header_id|>"

# สร้าง collator
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer,
)

# ── ทดสอบ Loss Masking ──
print("\n🔍 ทดสอบ Loss Masking:")
sample_text = dataset[0]["text"]
sample_tokens = tokenizer(sample_text, return_tensors="pt")
sample_batch = collator([{"input_ids": sample_tokens["input_ids"][0].tolist(),
                          "attention_mask": sample_tokens["attention_mask"][0].tolist()}])

labels = sample_batch["labels"][0]
masked_count = (labels == -100).sum().item()
total_count = len(labels)
print(f"  Total tokens: {total_count}")
print(f"  Masked tokens (-100): {masked_count} ({masked_count/total_count*100:.1f}%)")
print(f"  Tokens ที่คิด Loss: {total_count - masked_count} ({(total_count-masked_count)/total_count*100:.1f}%)")

# ─────────────────────────────────────────────
# 🏋️ SECTION 8: ตั้งค่าและรัน SFTTrainer
# ─────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    data_collator=collator,      # ← ใช้ collator ที่มี Loss Masking
    dataset_num_proc=2,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=100,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        logging_steps=10,
        output_dir="outputs",
        save_strategy="steps",
        save_steps=50,
        seed=42,
        report_to="none",
    ),
)

print("\n🏋️  เริ่มการเทรนด้วย Loss Masking...")
trainer_stats = trainer.train()

print(f"\n✅ เทรนเสร็จสิ้น!")
print(f"⏱️  เวลาที่ใช้: {trainer_stats.metrics['train_runtime']:.2f} วินาที")

# ─────────────────────────────────────────────
# 💾 SECTION 9: บันทึกโมเดล
# ─────────────────────────────────────────────
model.save_pretrained("outputs/lora_model_chat")
tokenizer.save_pretrained("outputs/lora_model_chat")
print("✅ บันทึกสำเร็จที่ outputs/lora_model_chat/")

# ─────────────────────────────────────────────
# 🧪 SECTION 10: ทดสอบ Multi-turn Chat
# ─────────────────────────────────────────────
print("\n🧪 ทดสอบ Multi-turn Chat...")

FastLanguageModel.for_inference(model)

# สร้าง conversation สำหรับทดสอบ
test_messages = [
    {"role": "system", "content": "คุณคือผู้ช่วย AI ที่เชี่ยวชาญด้านการเขียนโปรแกรม"},
    {"role": "user", "content": "อธิบาย list comprehension ใน Python ให้หน่อย"},
]

# Apply chat template และเพิ่ม generation prompt
input_text = tokenizer.apply_chat_template(
    test_messages,
    tokenize=False,
    add_generation_prompt=True,  # True = เพิ่ม assistant header ท้าย
)

inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.7,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
)

response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("\n📤 คำตอบจากโมเดล:")
print("─" * 50)
print(response)
print("─" * 50)
print("\n🎉 Step 2 เสร็จสมบูรณ์! ไปต่อที่ Step 3 ได้เลย")
