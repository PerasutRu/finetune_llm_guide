# Step 1: SFT Baseline ด้วย Unsloth 🟢

## เป้าหมาย
เรียนรู้การ Fine-tune โมเดล Llama-3 8B แบบพื้นฐาน
โดยใช้ Unsloth + LoRA/QLoRA บน Google Colab Free Tier
พร้อมเปรียบเทียบ LoRA vs QLoRA แบบ side-by-side

## ไฟล์ในโฟลเดอร์นี้

| ไฟล์ | คำอธิบาย |
|------|----------|
| `train.py` | SFT พื้นฐานด้วย QLoRA (เริ่มต้นที่นี่) |
| `compare_lora_qlora.py` | เปรียบเทียบ LoRA vs QLoRA แบบ side-by-side |

## สิ่งที่จะได้เรียน
- โหลดโมเดลแบบ 4-bit quantization ด้วย `FastLanguageModel`
- ตั้งค่า LoRA Adapter (r, lora_alpha, target_modules)
- ความแตกต่างระหว่าง LoRA และ QLoRA
- โหลด Dataset จาก HuggingFace Hub
- รัน `SFTTrainer` แบบพื้นฐาน

## ความต้องการ
- GPU: T4 16GB (Google Colab Free)
- VRAM ที่ใช้จริง: QLoRA ~6GB / LoRA ~14-16GB
- เวลาเทรน: ~20-30 นาที (100 steps)

## การติดตั้ง

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
pip install datasets
```

## รันโค้ด

```bash
# เริ่มต้น: SFT พื้นฐาน
python train.py

# เปรียบเทียบ LoRA vs QLoRA
python compare_lora_qlora.py
```

## ผลลัพธ์ที่คาดหวัง
- โฟลเดอร์ `outputs/` ที่มี LoRA/QLoRA weights บันทึกไว้
- ตารางเปรียบเทียบ VRAM, เวลา, และ loss ของแต่ละวิธี

---

## แนวคิดสำคัญ

### LoRA คืออะไร?

```
โมเดลปกติ: W (matrix ขนาดใหญ่ d×d)
LoRA:       W_new = W_frozen + (alpha/r) × A × B

A มีขนาด (d × r)   ← trainable
B มีขนาด (r × d)   ← trainable
W มีขนาด (d × d)   ← frozen (ไม่เปลี่ยน)

r = rank (เล็กมาก เช่น 16)
แทนที่จะเทรน d×d = 16,777,216 params
เราเทรนแค่ 2×d×r = 131,072 params  (ลดลง 128x!)
```

### LoRA vs QLoRA ต่างกันยังไง?

```
LoRA:
  Base Model  [bf16/fp16]  ← โหลดเต็ม 16-bit (~14GB)
      +
  LoRA Adapter [bf16/fp16] ← trainable (~100MB)
  ─────────────────────────
  VRAM รวม: ~14-16GB

QLoRA:
  Base Model  [NF4 4-bit]  ← quantized (~5GB)
      +
  LoRA Adapter [bf16/fp16] ← trainable (~100MB)
  ─────────────────────────
  VRAM รวม: ~5-6GB  ✅ รันบน T4 ได้สบาย
```

### เลือกใช้อะไร?

| สถานการณ์ | เลือกใช้ |
|-----------|---------|
| Colab Free (T4 16GB) | QLoRA |
| Colab Pro (A100 40GB) | LoRA |
| Local GPU < 12GB | QLoRA (r=8) |
| Local GPU ≥ 24GB | LoRA (r=32+) |
| Production / คุณภาพสูงสุด | LoRA r=64 บน A100 |
