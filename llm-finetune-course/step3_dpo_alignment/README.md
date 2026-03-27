# Step 3: Preference Alignment ด้วย DPO 🔴

## เป้าหมาย
เรียนรู้การใช้ DPO (Direct Preference Optimization) เพื่อสอนให้โมเดล
ตอบได้ตรงใจมนุษย์มากขึ้น โดยเรียนรู้จากคู่คำตอบ "ดี vs แย่"

## สิ่งที่จะได้เรียน
- แนวคิด Preference Learning และ RLHF
- รูปแบบ Dataset สำหรับ DPO (prompt, chosen, rejected)
- การตั้งค่าและรัน `DPOTrainer`
- ความแตกต่างระหว่าง SFT และ DPO

## แนวคิด: DPO คืออะไร?

### ปัญหาของ SFT อย่างเดียว
SFT สอนให้โมเดลเลียนแบบข้อมูล แต่ไม่รู้ว่าคำตอบไหน "ดีกว่า"
เช่น ถ้า dataset มีคำตอบที่ไม่ดีปนอยู่ โมเดลก็จะเรียนคำตอบแย่ด้วย

### DPO แก้ปัญหานี้ยังไง?
```
Dataset DPO มี 3 ส่วน:
┌─────────────────────────────────────────────┐
│ prompt:   "อธิบาย recursion ให้หน่อย"       │
│ chosen:   คำตอบที่ดี (มนุษย์เลือก)          │
│ rejected: คำตอบที่แย่ (มนุษย์ไม่ชอบ)        │
└─────────────────────────────────────────────┘

DPO Loss = สอนให้โมเดล:
✅ เพิ่มความน่าจะเป็นของ chosen
❌ ลดความน่าจะเป็นของ rejected
```

### Pipeline ที่แนะนำ
```
Base Model → SFT (Step 1-2) → DPO (Step 3) → Deploy (Step 4)
```
ต้องทำ SFT ก่อนเสมอ! DPO ต้องการโมเดลที่รู้จักรูปแบบการตอบแล้ว

## Dataset Format

```python
{
    "prompt": "คำถามหรือ conversation ก่อนหน้า",
    "chosen": "คำตอบที่มนุษย์ชอบ",
    "rejected": "คำตอบที่มนุษย์ไม่ชอบ"
}
```

## ความต้องการ
- GPU: T4 16GB (Google Colab Free)
- VRAM ที่ใช้จริง: ~10-12GB (DPO ต้องการมากกว่า SFT เพราะโหลด 2 โมเดล)
- เวลาเทรน: ~45-60 นาที (100 steps)

## หมายเหตุ
DPO ต้องการ VRAM มากกว่า SFT เพราะต้องโหลด:
1. Policy model (โมเดลที่กำลังเทรน)
2. Reference model (โมเดล SFT ที่ freeze ไว้)

ถ้า Out of Memory ให้ลด `per_device_train_batch_size` เหลือ 1
