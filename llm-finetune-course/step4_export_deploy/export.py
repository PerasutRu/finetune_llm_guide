"""
╔══════════════════════════════════════════════════════════════════╗
║  STEP 4: Export & Deploy                                        ║
║  ระดับ: 🟡 กลาง                                                  ║
║  เป้าหมาย: Merge LoRA + Export เป็น .gguf สำหรับ Ollama         ║
╚══════════════════════════════════════════════════════════════════╝

ขั้นตอน:
1. โหลด Base Model + LoRA weights
2. Merge LoRA เข้ากับ Base Model
3. Export เป็น .gguf (16-bit และ 4-bit)
4. สร้าง Modelfile สำหรับ Ollama
"""

# ─────────────────────────────────────────────
# 📦 SECTION 1: Import Libraries
# ─────────────────────────────────────────────
from unsloth import FastLanguageModel
import os

# ─────────────────────────────────────────────
# ⚙️ SECTION 2: Config
# ─────────────────────────────────────────────

# โมเดลที่จะ export (เปลี่ยนตาม step ที่ทำ)
# - จาก Step 1: "outputs/lora_model"
# - จาก Step 2: "outputs/lora_model_chat"
# - จาก Step 3: "outputs/lora_model_dpo"
LORA_MODEL_PATH = "outputs/lora_model_dpo"

# Base model ที่ใช้เทรน (ต้องตรงกับที่ใช้ใน Step 1-3)
BASE_MODEL_NAME = "unsloth/llama-3-8b-bnb-4bit"

MAX_SEQ_LENGTH = 2048

# ─────────────────────────────────────────────
# 🚀 SECTION 3: โหลด Base Model + LoRA
# ─────────────────────────────────────────────
print("📥 กำลังโหลดโมเดล...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=LORA_MODEL_PATH,  # โหลด LoRA model โดยตรง
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

print("✅ โหลดโมเดลสำเร็จ")

# ─────────────────────────────────────────────
# 🔀 SECTION 4: Merge LoRA เข้ากับ Base Model
# ─────────────────────────────────────────────
"""
การ Merge LoRA:
- รวม LoRA weights (A × B) เข้ากับ original weights (W)
- ผลลัพธ์: W_merged = W + (lora_alpha/r) × A × B
- โมเดลที่ได้จะเหมือน fine-tuned model แต่ไม่มี adapter แยก

ข้อดี:
- Inference เร็วกว่า (ไม่ต้องคำนวณ LoRA ทุก forward pass)
- ใช้กับ llama.cpp / Ollama ได้
- ง่ายต่อการ deploy

ข้อเสีย:
- ไม่สามารถ swap adapter ได้อีก
- ไฟล์ใหญ่ขึ้น (เท่ากับ base model)
"""

print("\n🔀 กำลัง Merge LoRA weights...")

# merge_and_unload(): merge LoRA เข้ากับ base model และ unload adapter
model = model.merge_and_unload()

print("✅ Merge สำเร็จ!")

# ─────────────────────────────────────────────
# 💾 SECTION 5: บันทึก Merged Model (HuggingFace format)
# ─────────────────────────────────────────────
"""
บันทึกในรูปแบบ HuggingFace ก่อน (optional)
ใช้ได้กับ transformers library โดยตรง
"""
MERGED_DIR = "outputs/merged_model"
print(f"\n💾 บันทึก merged model ที่ {MERGED_DIR}...")

model.save_pretrained(MERGED_DIR)
tokenizer.save_pretrained(MERGED_DIR)

print(f"✅ บันทึก HuggingFace format สำเร็จ")

# ─────────────────────────────────────────────
# 📦 SECTION 6: Export เป็น .gguf (16-bit)
# ─────────────────────────────────────────────
"""
.gguf คือ format ของ llama.cpp ที่ใช้กับ:
- Ollama
- LM Studio
- Jan.ai
- llama.cpp โดยตรง

quantization_method options:
- "f16"     : 16-bit float (ขนาด ~16GB, คุณภาพสูงสุด)
- "q8_0"    : 8-bit quantized (~8GB, คุณภาพสูง)
- "q4_k_m"  : 4-bit quantized (~4.5GB, คุณภาพดี - แนะนำ)
- "q4_0"    : 4-bit quantized (~4GB, เร็วกว่า q4_k_m เล็กน้อย)
- "q2_k"    : 2-bit quantized (~2.5GB, คุณภาพต่ำสุด)

สำหรับ Colab Free Tier:
- ใช้ "q4_k_m" เพราะขนาดพอดีกับ disk space
- "f16" อาจไม่พอ disk (ต้องการ ~16GB)
"""

print("\n📦 กำลัง Export เป็น .gguf (16-bit)...")
print("⚠️  ต้องการ disk space ~16GB สำหรับ f16")

# Export 16-bit (คุณภาพสูงสุด)
model.save_pretrained_gguf(
    "outputs/gguf_f16",          # โฟลเดอร์ปลายทาง
    tokenizer,
    quantization_method="f16",   # 16-bit float
)
print("✅ Export f16 สำเร็จ: outputs/gguf_f16/")

# ─────────────────────────────────────────────
# 📦 SECTION 7: Export เป็น .gguf (4-bit quantized)
# ─────────────────────────────────────────────
"""
q4_k_m คือ quantization method ที่นิยมที่สุดสำหรับ local deployment:
- "k" = k-quant (คุณภาพดีกว่า q4_0)
- "m" = medium (balance ระหว่าง size และ quality)
- ขนาด ~4.5GB สำหรับ 8B model
- รันได้บน CPU ที่มี RAM 8GB+
"""

print("\n📦 กำลัง Export เป็น .gguf (4-bit q4_k_m)...")

model.save_pretrained_gguf(
    "outputs/gguf_q4",           # โฟลเดอร์ปลายทาง
    tokenizer,
    quantization_method="q4_k_m",  # 4-bit quantized
)
print("✅ Export q4_k_m สำเร็จ: outputs/gguf_q4/")

# ─────────────────────────────────────────────
# 📋 SECTION 8: สร้าง Modelfile สำหรับ Ollama
# ─────────────────────────────────────────────
"""
Modelfile คือ config file สำหรับ Ollama
คล้ายกับ Dockerfile แต่สำหรับ LLM

คำสั่งใน Modelfile:
- FROM: path ไปยัง .gguf file
- SYSTEM: system prompt เริ่มต้น
- PARAMETER: ตั้งค่า generation parameters
- TEMPLATE: custom prompt template (optional)
"""

modelfile_content = """# Ollama Modelfile
# สร้างโดย LLM Fine-tuning Course - Step 4

# โหลด .gguf model (เปลี่ยน path ตามที่ต้องการ)
FROM ./model-q4_k_m.gguf

# System prompt เริ่มต้น
SYSTEM \"\"\"คุณคือผู้ช่วย AI ที่มีประโยชน์ ฉลาด และซื่อสัตย์
ตอบคำถามอย่างละเอียดและถูกต้อง ถ้าไม่รู้ให้บอกว่าไม่รู้\"\"\"

# Generation parameters
PARAMETER temperature 0.7        # ความสร้างสรรค์ (0=deterministic, 1=random)
PARAMETER top_p 0.9              # nucleus sampling
PARAMETER top_k 40               # top-k sampling
PARAMETER num_predict 512        # จำนวน token สูงสุดที่ generate
PARAMETER repeat_penalty 1.1     # ลดการซ้ำคำ
"""

with open("outputs/Modelfile", "w", encoding="utf-8") as f:
    f.write(modelfile_content)

print("\n📋 สร้าง Modelfile สำเร็จ: outputs/Modelfile")

# ─────────────────────────────────────────────
# 📤 SECTION 9: Push ขึ้น HuggingFace Hub (Optional)
# ─────────────────────────────────────────────
"""
ถ้าต้องการ share โมเดลบน HuggingFace Hub:

1. Login ก่อน:
   from huggingface_hub import login
   login(token="hf_your_token_here")

2. Push LoRA adapter (เล็ก ~100MB):
   model.push_to_hub("your-username/your-model-name", tokenizer=tokenizer)

3. หรือ Push .gguf ขึ้น Hub โดยตรง:
   model.push_to_hub_gguf(
       "your-username/your-model-name",
       tokenizer,
       quantization_method=["q4_k_m", "f16"]
   )
"""

# Uncomment ถ้าต้องการ push ขึ้น Hub
# from huggingface_hub import login
# login(token="hf_your_token_here")
# model.push_to_hub_gguf(
#     "your-username/llama3-finetuned",
#     tokenizer,
#     quantization_method=["q4_k_m", "f16"],
# )

# ─────────────────────────────────────────────
# 📊 SECTION 10: สรุปไฟล์ที่ Export
# ─────────────────────────────────────────────
print("\n" + "═" * 60)
print("🎉 Export เสร็จสมบูรณ์!")
print("═" * 60)
print("\n📁 ไฟล์ที่สร้าง:")

output_files = [
    ("outputs/merged_model/", "HuggingFace format (ใช้กับ transformers)"),
    ("outputs/gguf_f16/", ".gguf 16-bit (คุณภาพสูงสุด)"),
    ("outputs/gguf_q4/", ".gguf 4-bit q4_k_m (แนะนำสำหรับ deploy)"),
    ("outputs/Modelfile", "Ollama Modelfile"),
]

for path, desc in output_files:
    exists = "✅" if os.path.exists(path) else "❌"
    print(f"  {exists} {path:<35} → {desc}")

print("\n" + "─" * 60)
print("🚀 วิธี Deploy บน Ollama:")
print("─" * 60)
print("""
# 1. ติดตั้ง Ollama (ถ้ายังไม่มี)
curl -fsSL https://ollama.com/install.sh | sh

# 2. Copy ไฟล์ .gguf ไปยัง folder เดียวกับ Modelfile
cp outputs/gguf_q4/*.gguf outputs/
cp outputs/Modelfile outputs/

# 3. สร้าง Ollama model
cd outputs
ollama create my-llama3 -f Modelfile

# 4. รัน!
ollama run my-llama3

# 5. หรือใช้ผ่าน API
curl http://localhost:11434/api/generate -d '{
  "model": "my-llama3",
  "prompt": "สวัสดี ช่วยอธิบาย Python ให้หน่อย"
}'
""")

print("─" * 60)
print("🎓 จบคอร์ส LLM Fine-tuning แล้ว!")
print("   Step 1 → Step 2 → Step 3 → Step 4 ✅")
print("─" * 60)
