# Reference call — the behavior the agent targets

An illustrative Pakistani-Urdu customer-service exchange showing the three things
that make this hard: **natural Urdu**, **English code-switching** ("Urdish"), and a
**mid-sentence barge-in**. This is a *target* dialogue — the kind of call the system
is built to handle — not a transcript of model output.

> Channel layout in the stereo training data: **agent = left**, **caller = right**.

---

**🟦 Agent:** السلام علیکم، resilia bank میں خوش آمدید۔ میں آپ کی کیا مدد کر سکتا ہوں؟
&nbsp;&nbsp;<sub>*(Peace be upon you, welcome to resilia bank. How can I help you?)*</sub>

**🟧 Caller:** جی، میرا debit card block ہو گیا ہے، اسے کھولنا ہے۔
&nbsp;&nbsp;<sub>*(Yes, my debit card is blocked, I need it unblocked.)*</sub>

**🟦 Agent:** بالکل۔ verification کے لیے براہ مہربانی اپنا CNIC نمبر بتا دیں…
&nbsp;&nbsp;<sub>*(Of course. For verification please tell me your CNIC number…)*</sub>

**🟧 Caller:** *[cuts in]* نہیں رکیں — پہلے یہ بتائیں کہ میرا balance کتنا ہے؟
&nbsp;&nbsp;<sub>*(No wait — first tell me, what is my balance?)*</sub>

> **⏱ Barge-in.** The caller interrupts mid-sentence. The agent must go quiet within
> ~0.3–0.5 s (measured: **60 ms** stop) and switch to the new request — not finish
> its sentence. This is the full-duplex moment the whole project is about.

**🟦 Agent:** جی ضرور۔ آپ کا balance بارہ ہزار تین سو روپے ہے۔ کیا اب میں card unblock کر دوں؟
&nbsp;&nbsp;<sub>*(Sure. Your balance is twelve thousand three hundred rupees. Shall I unblock the card now?)*</sub>

**🟧 Caller:** ہاں کر دیں، شکریہ۔
&nbsp;&nbsp;<sub>*(Yes please do, thank you.)*</sub>

**🟦 Agent:** ہو گیا۔ آپ کا کارڈ فعال ہے۔ کوئی اور خدمت؟ … خدا حافظ۔
&nbsp;&nbsp;<sub>*(Done. Your card is active. Anything else? … Goodbye.)*</sub>

---

**Why this is the hard case:** the recognizer has to handle `debit card`, `CNIC`,
`balance`, `unblock` embedded in Urdu; the agent has to drop its verification
sentence the instant the caller barges in; and the voice has to sound like a
Pakistani human, not a flat TTS. The `samples/two_party_demo.wav` file shows the
stereo *format* this would be trained in (with placeholder tones for voices).
