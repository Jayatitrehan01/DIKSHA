"""
pipeline.py
Discharge instructions assistant.
"""
import os

from rank_bm25 import BM25Okapi


SYSTEM_PROMPT = """You are a discharge instructions assistant that helps patients
understand their own hospital discharge summary.

Always follow these rules:
1. Answer ONLY using the discharge summary excerpts provided. If the answer is not
   there, say you cannot find it in their discharge summary and suggest they contact
   their care team.
2. Never give new medical advice, diagnoses, or opinions. Do not interpret symptoms
   or judge whether something is normal or serious.
3. If the question describes a possible emergency or warning sign, do not try to
   answer it. Tell the patient to contact their care team or emergency services
   immediately.
4. Use plain, calm language. Explain any medical terms simply.
5. For anything about medication, activity, or symptoms, end with a gentle reminder
   to confirm with their care team if unsure.

You explain existing instructions. You are not a medical advisor."""


def chunk_note(text, max_chars=600):
    lines = text.split("\n")
    sections, header, buf = [], "general", []

    def flush():
        if buf:
            body = " ".join(b.strip() for b in buf if b.strip()).strip()
            if body:
                sections.append((header, body))

    for line in lines:
        s = line.strip()
        if s.endswith(":") and 1 < len(s) <= 45:
            flush()
            header = s.rstrip(":").strip().lower()
            buf = []
        else:
            buf.append(line)
    flush()

    chunks = []
    for h, body in sections:
        if len(body) <= max_chars:
            chunks.append({"section": h, "text": body})
        else:
            words, piece = body.split(), []
            for w in words:
                piece.append(w)
                if len(" ".join(piece)) >= max_chars:
                    chunks.append({"section": h, "text": " ".join(piece)})
                    piece = []
            if piece:
                chunks.append({"section": h, "text": " ".join(piece)})

    return [c for c in chunks if len(c["text"]) > 15]


RED_FLAG_TERMS = [
    "chest pain", "cannot breathe", "can't breathe", "cant breathe",
    "shortness of breath", "trouble breathing", "difficulty breathing",
    "heavy bleeding", "bleeding won't stop", "won't stop bleeding",
    "high fever", "passed out", "fainted", "severe pain", "worst pain",
    "allergic reaction", "swelling in my face", "face is swelling",
    "slurred speech", "numb on one side", "suicidal", "want to die",
    "unconscious", "seizure",
]


def is_possible_emergency(query):
    q = query.lower()
    return any(term in q for term in RED_FLAG_TERMS)


def get_llm_client():
    if os.environ.get("GROQ_API_KEY"):
        from openai import OpenAI
        return OpenAI(base_url="https://api.groq.com/openai/v1",
                      api_key=os.environ["GROQ_API_KEY"])
    if os.environ.get("OPENAI_API_KEY"):
        from openai import OpenAI
        return OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return None


class DischargeAssistant:
    def __init__(self, note_text, llm_client=None, model="llama-3.3-70b-versatile"):
        self.note_text = note_text
        self.llm = llm_client
        self.model = model

        self.chunks = chunk_note(note_text)
        self.chunk_texts = [c["text"] for c in self.chunks]
        self.bm25 = BM25Okapi([t.lower().split() for t in self.chunk_texts])

    def retrieve(self, query, k=4):
        scores = self.bm25.get_scores(query.lower().split())
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.chunk_texts[i] for i in ranked[:k]]

    def _warning_text(self):
        for c in self.chunks:
            if any(w in c["section"] for w in ("warning", "when to call", "follow")):
                return c["text"]
        return ""

    def answer(self, query):
        if is_possible_emergency(query):
            msg = ("This sounds like it may need urgent attention. Please contact "
                   "your care team or your local emergency number right away.")
            warn = self._warning_text()
            if warn:
                msg += f"\n\nYour discharge summary notes:\n{warn}"
            return msg

        context = "\n\n".join(self.retrieve(query))


        retrieved = self.retrieve(query)
        if self.llm is None:
            top=retrieved[0] if retrieved else ''
            return ("[No language model is configured, so here is the most relevant "
                    f"text from the discharge summary:]\n\n{top}")

        resp = self.llm.chat.completions.create(
            model=self.model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content":
                    f"Discharge summary excerpts:\n{context}\n\nPatient question: {query}"},
            ],
        )
        return resp.choices[0].message.content