# Step 4: Export & Deploy 🟡

## เป้าหมาย
Export โมเดลที่เทรนแล้วให้พร้อม Deploy บน Ollama หรือ Local machine
โดย Merge LoRA weights เข้ากับ Base Model และ Export เป็น `.gguf`

## สิ่งที่จะได้เรียน
- Merge LoRA adapter เข้ากับ Base Model
- Export เป็น `.gguf` แบบ 16-bit (คุณภาพสูง)
- Export เป็น `.gguf` แบบ 4-bit quantized (ขนาดเล็ก)
- นำไปรันบน Ollama

## ทำไมต้อง Merge?

```
ก่อน Merge:
├── base_model/     (8GB)
└── lora_adapter/   (100MB)
    → ต้องโหลดทั้งคู่ทุกครั้ง

หลัง Merge:
└── merged_model/   (8GB)
    → โหลดไฟล์เดียว เร็วกว่า
    → ใช้กับ llama.cpp / Ollama ได้
```

## ขนาดไฟล์ .gguf

| Format | ขนาด | คุณภาพ | ใช้กับ |
|--------|------|--------|--------|
| f16 | ~16GB | สูงสุด | GPU ที่มี VRAM เยอะ |
| q8_0 | ~8GB | สูง | GPU/CPU ที่มี RAM เยอะ |
| q4_k_m | ~4.5GB | ดี | CPU ทั่วไป (แนะนำ) |
| q2_k | ~2.5GB | พอใช้ | CPU ที่ RAM น้อย |

## Deploy บน Ollama

```bash
# 1. ติดตั้ง Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. สร้าง Modelfile
cat > Modelfile << EOF
FROM ./model-q4_k_m.gguf
SYSTEM "คุณคือผู้ช่วย AI ที่มีประโยชน์"
EOF

# 3. สร้างและรัน model
ollama create my-model -f Modelfile
ollama run my-model
```

## ความต้องการ
- GPU: T4 16GB (Google Colab Free)
- Disk: ~20GB สำหรับ export ไฟล์ทั้งหมด
- เวลา: ~15-20 นาที
