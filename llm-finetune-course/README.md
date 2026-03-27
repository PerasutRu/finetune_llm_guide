# 🦙 LLM Fine-tuning Course: จาก Zero ถึง Deploy

คอร์สสอน Fine-tune โมเดลภาษาขนาดใหญ่ (LLM) แบบครบวงจร ตั้งแต่พื้นฐานจนถึงนำไป Deploy จริง
ออกแบบมาให้รันบน **Google Colab Free Tier (T4 GPU 16GB)** ได้ทันที

---

## 📚 โครงสร้างคอร์ส

| Step | หัวข้อ | ระดับ | เวลาโดยประมาณ |
|------|--------|-------|----------------|
| [Step 1](./step1_sft_baseline/) | SFT Baseline ด้วย Unsloth | 🟢 เริ่มต้น | ~30 นาที |
| [Step 2](./step2_chat_formatting/) | Chat Formatting & Loss Masking | 🟡 กลาง | ~45 นาที |
| [Step 3](./step3_dpo_alignment/) | Preference Alignment (DPO) | 🔴 ขั้นสูง | ~60 นาที |
| [Step 4](./step4_export_deploy/) | Export & Deploy (.gguf) | 🟡 กลาง | ~30 นาที |

---

## 🛠️ Tech Stack

```
unsloth     - โหลดและเทรนโมเดลแบบ 4-bit อย่างรวดเร็ว (เร็วกว่า HuggingFace 2x)
trl         - Trainer สำหรับ SFT และ DPO (Transformer Reinforcement Learning)
peft        - LoRA / QLoRA adapter management
datasets    - โหลดและจัดการ dataset จาก HuggingFace Hub
transformers - Base library สำหรับโมเดลและ tokenizer
bitsandbytes - Quantization library (4-bit / 8-bit)
```

---

## ⚡ Quick Start

### 1. ติดตั้ง Dependencies

```bash
# รันใน Google Colab หรือ terminal
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
pip install datasets
```

### 2. เลือก Step ที่ต้องการเรียน

```bash
# Step 1: เริ่มต้น Fine-tune แบบง่ายที่สุด
cd step1_sft_baseline && python train.py

# Step 2: จัดการ Chat Format และ Loss Masking
cd step2_chat_formatting && python train.py

# Step 3: DPO Alignment
cd step3_dpo_alignment && python train.py

# Step 4: Export โมเดลเป็น .gguf
cd step4_export_deploy && python export.py
```

---

## 🧠 แนวคิดหลักที่จะได้เรียน

### LoRA (Low-Rank Adaptation)
แทนที่จะ Fine-tune พารามิเตอร์ทั้งหมด (8B params = ~16GB), LoRA เพิ่ม "adapter" ขนาดเล็กเข้าไป
ทำให้เทรนได้บน GPU ขนาดเล็กโดยใช้ VRAM เพียง 4-6GB

```
โมเดลเดิม (Frozen) + LoRA Adapter (Trainable) = โมเดลที่ปรับแต่งแล้ว
```

### QLoRA (Quantized LoRA)
โหลดโมเดลแบบ 4-bit quantization + LoRA = ประหยัด VRAM สูงสุด
- Llama-3 8B ปกติ: ~16GB VRAM
- Llama-3 8B แบบ 4-bit: ~5GB VRAM ✅

### SFT → DPO Pipeline
```
Base Model
    ↓ Step 1-2: SFT (Supervised Fine-Tuning)
    → โมเดลเรียนรู้รูปแบบการตอบ
    ↓ Step 3: DPO (Direct Preference Optimization)
    → โมเดลเรียนรู้ว่าคำตอบไหน "ดีกว่า"
    ↓ Step 4: Export
    → พร้อม Deploy บน Ollama / llama.cpp
```

---

## 💡 Tips สำหรับ Google Colab Free Tier

1. **เปิด GPU**: Runtime → Change runtime type → T4 GPU
2. **เช็ค VRAM**: `!nvidia-smi`
3. **ถ้า Out of Memory**: ลด `per_device_train_batch_size` เหลือ 1 และเพิ่ม `gradient_accumulation_steps`
4. **บันทึก checkpoint**: ตั้ง `save_steps` ให้บ่อยขึ้น เผื่อ session หลุด

---

## 📖 อ้างอิง

- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Llama-3 Model Card](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
