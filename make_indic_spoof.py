"""Generate Indian-language SYNTHETIC speech -- the missing negative class.

WHY THIS EXISTS
IndicVoices is bonafide-only: every one of its ~37,000 training rows is
labelled 0. So the only thing a head can learn from Indian audio is
"Indian phonetics -> real", and that is exactly what head_full_ho learned --
it passes a cloned Hindi voice as genuine because it has never seen one.

You cannot fix that with thresholds, augmentation or calibration. The training
set needs Indian-language audio labelled 1. This produces some.

Uses Meta's MMS-TTS, which ships a separate checkpoint per language and runs
locally on any CUDA card. Output is 16 kHz mono wav, the rate the rest of the
pipeline expects.

    pip install transformers torch scipy
    python make_indic_spoof.py --lang hin --n 300 --out data/indic_spoof

    # more phonetic variety is better than more clips of one language
    python make_indic_spoof.py --lang hin --n 250 --out data/indic_spoof
    python make_indic_spoof.py --lang mar --n 150 --out data/indic_spoof
    python make_indic_spoof.py --lang ben --n 150 --out data/indic_spoof
    python make_indic_spoof.py --lang tam --n 150 --out data/indic_spoof

THEN -- and this step is not optional, see the note at the bottom of this file:
    python make_codec.py   --in data/indic_spoof --out data/indic_spoof_g711
    python make_augment.py --in data/indic_spoof --out data/indic_spoof_rir
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

# Scam-call scripts matching our impact story, plus neutral sentences so the
# model cannot key on topic instead of on synthesis artefacts.
LINES = {
    "hin": [
        "नमस्ते, मैं आपके बेटे का दोस्त बोल रहा हूँ, वह मुश्किल में है।",
        "मुझे अभी तुरंत पचास हज़ार रुपये चाहिए, कृपया जल्दी भेज दीजिए।",
        "आपका खाता बंद हो जाएगा अगर आपने अभी सत्यापन नहीं किया।",
        "पापा, मेरा फ़ोन खो गया है, यह नया नंबर है, पैसे भेज दो।",
        "मैं बैंक से बोल रहा हूँ, आपका कार्ड ब्लॉक कर दिया गया है।",
        "कृपया यह ओटीपी मुझे बताइए, वरना लेनदेन रद्द हो जाएगा।",
        "आज मौसम बहुत अच्छा है, बाहर घूमने चलते हैं।",
        "मैंने कल शाम को बाज़ार से कुछ फल खरीदे थे।",
        "यह किताब मैंने पिछले महीने पढ़ी थी, बहुत दिलचस्प थी।",
        "रेलगाड़ी सुबह छह बजे स्टेशन पहुँचेगी।",
        "उसने अपनी पढ़ाई पूरी करने के बाद नौकरी शुरू की।",
        "बच्चे स्कूल से लौटकर खेलने चले गए।",
        "हमें कल सुबह जल्दी निकलना होगा, रास्ता लंबा है।",
        "इस साल बारिश पिछले साल से कम हुई है।",
        "क्या आप मुझे यह पता समझा सकते हैं?",
        "मेरी बहन अगले हफ़्ते दिल्ली आ रही है।",
        "खाना बनकर तैयार है, सब लोग आ जाइए।",
        "उसने बहुत मेहनत की और परीक्षा में सफल हुआ।",
        "यह रास्ता सीधा बाज़ार तक जाता है।",
        "मुझे यह रंग बहुत पसंद आया।",
    ],
    "mar": [
        "नमस्कार, मी तुमच्या मुलाचा मित्र बोलतोय, तो अडचणीत आहे.",
        "मला आत्ता लगेच पैसे हवे आहेत, कृपया लवकर पाठवा.",
        "तुमचे खाते बंद होईल जर तुम्ही आत्ता पडताळणी केली नाही.",
        "आज हवामान खूप छान आहे, बाहेर फिरायला जाऊया.",
        "मी काल संध्याकाळी बाजारातून फळे विकत घेतली.",
        "गाडी सकाळी सहा वाजता स्थानकावर पोहोचेल.",
        "मुले शाळेतून परत येऊन खेळायला गेली.",
        "यावर्षी पाऊस मागील वर्षापेक्षा कमी झाला.",
        "माझी बहीण पुढच्या आठवड्यात येत आहे.",
        "जेवण तयार आहे, सर्वजण या.",
    ],
    "ben": [
        "নমস্কার, আমি আপনার ছেলের বন্ধু বলছি, সে বিপদে আছে।",
        "আমার এখনই টাকা দরকার, দয়া করে তাড়াতাড়ি পাঠান।",
        "আপনার অ্যাকাউন্ট বন্ধ হয়ে যাবে যদি এখনই যাচাই না করেন।",
        "আজ আবহাওয়া খুব সুন্দর, চলো বাইরে বেড়াতে যাই।",
        "আমি কাল বিকেলে বাজার থেকে কিছু ফল কিনেছিলাম।",
        "ট্রেনটি সকাল ছয়টায় স্টেশনে পৌঁছবে।",
        "বাচ্চারা স্কুল থেকে ফিরে খেলতে গেল।",
        "এ বছর বৃষ্টি গত বছরের চেয়ে কম হয়েছে।",
        "আমার বোন আগামী সপ্তাহে আসছে।",
        "খাবার তৈরি, সবাই আসুন।",
    ],
    "tam": [
        "வணக்கம், நான் உங்கள் மகனின் நண்பன் பேசுகிறேன், அவர் சிக்கலில் இருக்கிறார்.",
        "எனக்கு இப்போதே பணம் தேவை, தயவுசெய்து விரைவாக அனுப்புங்கள்.",
        "இப்போது சரிபார்க்கவில்லை என்றால் உங்கள் கணக்கு மூடப்படும்.",
        "இன்று வானிலை மிகவும் நன்றாக இருக்கிறது, வெளியே செல்லலாம்.",
        "நான் நேற்று மாலை சந்தையில் பழங்கள் வாங்கினேன்.",
        "ரயில் காலை ஆறு மணிக்கு நிலையத்தை அடையும்.",
        "குழந்தைகள் பள்ளியிலிருந்து திரும்பி விளையாடச் சென்றனர்.",
        "இந்த ஆண்டு மழை கடந்த ஆண்டை விட குறைவாக இருந்தது.",
        "என் சகோதரி அடுத்த வாரம் வருகிறார்.",
        "சாப்பாடு தயார், எல்லோரும் வாருங்கள்.",
    ],
    "tel": [
        "నమస్కారం, నేను మీ కొడుకు స్నేహితుడిని మాట్లాడుతున్నాను, అతను ఇబ్బందిలో ఉన్నాడు.",
        "నాకు ఇప్పుడే డబ్బు కావాలి, దయచేసి త్వరగా పంపండి.",
        "ఇప్పుడు ధృవీకరించకపోతే మీ ఖాతా మూసివేయబడుతుంది.",
        "ఈ రోజు వాతావరణం చాలా బాగుంది, బయటకు వెళ్దాం.",
        "నేను నిన్న సాయంత్రం మార్కెట్ నుండి పండ్లు కొన్నాను.",
        "రైలు ఉదయం ఆరు గంటలకు స్టేషన్‌కు చేరుకుంటుంది.",
        "పిల్లలు పాఠశాల నుండి తిరిగి ఆడుకోవడానికి వెళ్లారు.",
        "ఈ సంవత్సరం వర్షం గత సంవత్సరం కంటే తక్కువగా ఉంది.",
        "నా సోదరి వచ్చే వారం వస్తోంది.",
        "భోజనం సిద్ధంగా ఉంది, అందరూ రండి.",
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="hin", choices=sorted(LINES),
                    help="MMS-TTS language code")
    ap.add_argument("--n", type=int, default=300, help="clips to generate")
    ap.add_argument("--out", default="data/indic_spoof")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    try:
        import torch
        import scipy.io.wavfile as wav
        from transformers import VitsModel, AutoTokenizer
    except ImportError as exc:
        sys.exit(f"missing dependency: {exc}\n  pip install transformers torch scipy")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    name = f"facebook/mms-tts-{args.lang}"
    print(f"loading {name} on {device} ...")
    tok = AutoTokenizer.from_pretrained(name)
    model = VitsModel.from_pretrained(name).to(device).eval()

    if getattr(tok, "is_uroman", False):
        print("  ! this checkpoint needs romanised input (uroman). If the audio "
              "comes out as noise, try a different --lang.")

    sr = model.config.sampling_rate
    print(f"  sampling rate {sr} Hz")
    if sr != 16000:
        print("  ! not 16 kHz -- extract_embeddings will resample, which is fine.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = LINES[args.lang]
    rng = random.Random(args.seed)

    written = 0
    for i in range(args.n):
        text = lines[i % len(lines)]
        # Vary the synthesis so the set is not N copies of one prosody. These
        # are VITS knobs: pace, and how much stochastic variation the duration
        # predictor injects.
        model.speaking_rate = rng.uniform(0.85, 1.25)
        model.noise_scale = rng.uniform(0.5, 0.9)

        inputs = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            wave = model(**inputs).waveform[0].detach().cpu().numpy()

        peak = float(np.max(np.abs(wave))) or 1.0
        wave = (wave / peak * rng.uniform(0.25, 0.75)).astype(np.float32)
        if wave.size < sr:                      # skip anything under a second
            continue

        path = out / f"{args.lang}_tts_{i:05d}.wav"
        wav.write(str(path), sr, (wave * 32767).astype(np.int16))
        written += 1
        if written % 25 == 0:
            print(f"  {written} / {args.n}")

    print(f"\nwrote {written} clips to {out}")
    print()
    print("NEXT -- and do not skip the augmentation step:")
    print()
    print("  These are CLEAN studio-quality TTS. Your bonafide Indic data")
    print("  (IndicVoices) is varied, noisy, real-world audio. Train on them")
    print("  as-is and the head learns 'clean Hindi = fake' -- the exact")
    print("  shortcut you already diagnosed once, rebuilt in a new language.")
    print()
    print(f"  python augment_folder.py --in {out} --out {out}_aug")
    print()
    print("  (NOT make_codec.py / make_augment.py -- those need an ASVspoof")
    print("   protocol file and cannot take a plain folder.)")
    print()
    print("  Then extract and stamp them as SPOOF:")
    print()
    print(f"  python src\\extract_embeddings.py --split train --audio-dir {out} \\")
    print(f"      --out outputs\\embeddings_indic_spoof --batch 8")
    print(f"  python stamp_labels.py --emb-dir outputs\\embeddings_indic_spoof\\train --label 1")
    print(f"  python stamp_labels.py --emb-dir outputs\\embeddings_indic_spoof\\train --check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
