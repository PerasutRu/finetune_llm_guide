# Step 2: Chat Formatting & Loss Masking 🟡

## เป้าหมาย
เรียนรู้การจัดการข้อมูลแบบ Multi-turn Chat และการใช้ Loss Masking
เพื่อให้โมเดลเรียนรู้เฉพาะคำตอบของ Assistant (ไม่เสียเวลาเรียน User prompt)

## สิ่งที่จะได้เรียน
- Chat Templates มาตรฐาน (ChatML / Llama-3 format)
- Multi-turn conversation formatting
- `DataCollatorForCompletionOnlyLM` สำหรับ Loss Masking
- ความแตกต่างระหว่าง SFT แบบธรรมดา vs Loss Masking

## ทำไม Loss Masking ถึงสำคัญ?

### แบบไม่มี Loss Masking (Step 1)
```
User: สวัสดี ช่วยอธิบาย Python ให้หน่อย    ← คิด Loss ด้วย (ไม่จำเป็น!)
Assistant: Python คือภาษาโปรแกรมมิ่ง...     ← คิด Loss (ถูกต้อง)
```

### แบบมี Loss Masking (Step 2) ✅
```
User: สวัสดี ช่วยอธิบาย Python ให้หน่อย    ← ไม่คิด Loss (mask = -100)
Assistant: Python คือภาษาโปรแกรมมิ่ง...     ← คิด Loss เฉพาะส่วนนี้
```

ผลลัพธ์: โมเดลเรียนรู้ได้ดีขึ้น เพราะ gradient ไหลมาจากส่วนที่สำคัญเท่านั้น

## Chat Template Format

### ChatML Format (ใช้กับ Mistral, Qwen, ฯลฯ)
```
<|im_start|>system
คุณคือผู้ช่วย AI ที่มีประโยชน์<|im_end|>
<|im_start|>user
สวัสดี<|im_end|>
<|im_start|>assistant
สวัสดีครับ มีอะไรให้ช่วยไหม?<|im_end|>
```

### Llama-3 Format
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
คุณคือผู้ช่วย AI<|eot_id|>
<|start_header_id|>user<|end_header_id|>
สวัสดี<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
สวัสดีครับ<|eot_id|>
```

## ความต้องการ
- GPU: T4 16GB (Google Colab Free)
- VRAM ที่ใช้จริง: ~7-9GB
- เวลาเทรน: ~30-40 นาที (100 steps)
