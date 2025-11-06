"""
Page insights service: LLM-powered extraction of per-page metadata and Q&A.

Generates:
- KeywordExtraction: entities, product_names, topics, file_type, key_phrases
- QAPair list: generated per page based on chunks

Falls back to offline heuristics when Azure OpenAI config is missing.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import List, Dict

from pydantic import BaseModel

from src.config import get_config
from src.etl.services.openai_service import OpenAIService
from src.etl.models.etl_models import KeywordExtraction, QAPair, Chunk

logger = logging.getLogger(__name__)


@dataclass
class _ExtractionResult:
    keywords: KeywordExtraction
    qna_pairs: List[QAPair]
    categories: Dict[str, List[str]]


class PageInsightsService(BaseModel):
    """LLM-backed per-page insights generator with offline heuristics."""

    offline: bool = False

    def __init__(self, **data):
        super().__init__(**data)
        cfg = get_config()
        self.offline = bool(getattr(cfg.runtime, "offline_mode", True))
        self._openai = OpenAIService(cfg.azure_openai, vision_config=None, offline=self.offline)

    def analyze_page(
        self,
        page_text: str,
        filename: str,
        doc_id: str,
        page_number: int,
        chunks: List[Chunk],
    ) -> _ExtractionResult:
        """
        Generate keyword extraction and Q&A pairs for a page.
        """
        if not page_text:
            return _ExtractionResult(keywords=KeywordExtraction(), qna_pairs=[], categories={
                "insurance_product": [],
                "insurance_term": [],
                "location": [],
                "people_occupation_user": [],
            })

        if self.offline:
            return self._analyze_offline(page_text, filename, doc_id, page_number, chunks)

        prompt = self._build_prompt(page_text, filename)
        try:
            if not self._openai.chat_client:
                raise RuntimeError("Chat client not configured for entity extraction")
            response = self._openai.chat_client.chat.completions.create(
                model=self._openai.chat_deployment,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You extract structured metadata from insurance documents. "
                            "Return STRICT JSON only per the provided schema."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=500,
                temperature=0.1,
            )
            content = (response.choices[0].message.content or "{}").strip()
            data = json.loads(content)

            # Map new structured entities JSON
            product = (data.get("product") or "")
            policy_type = (data.get("policy_type") or "")
            action = (data.get("action") or "")
            required_form_or_doc = (data.get("required_form_or_doc") or "")
            payment_channel = (data.get("payment_channel") or "")
            market = (data.get("market") or "")
            customer_role = (data.get("customer_role") or "")

            # Build categorized entities (legacy categories for downstream compatibility)
            insurance_product: List[str] = []
            if product:
                insurance_product.append(product)
            insurance_term: List[str] = []
            insurance_term.extend([t for t in [policy_type, action, required_form_or_doc, payment_channel] if t])
            loc: List[str] = []
            if market:
                loc.append(market)
            people_occupation_user: List[str] = []
            if customer_role:
                people_occupation_user.append(customer_role)
            categories = {
                "insurance_product": list(dict.fromkeys([x.strip() for x in insurance_product if x and x.strip()])),
                "insurance_term": list(dict.fromkeys([x.strip() for x in insurance_term if x and x.strip()])),
                "location": list(dict.fromkeys(loc)),
                "people_occupation_user": list(dict.fromkeys(people_occupation_user)),
            }

            # Map to KeywordExtraction (legacy fields)
            keywords = KeywordExtraction(
                entities=(categories["insurance_term"] + categories["location"] + categories["people_occupation_user"]),
                product_names=categories["insurance_product"],
                topics=[],
                file_type="",
                key_phrases=None,
                entities_en=[],
                entities_tc=[],
                product_names_en=[],
                product_names_tc=[],
                topics_en=[],
                topics_tc=[],
            )

            # Q&A generation per chunk using user-provided prompt
            qna_pairs: List[QAPair] = []
            for ch in chunks:
                qa = self._generate_qna_llm_for_chunk(ch, doc_id, page_number)
                if qa:
                    qna_pairs.extend(qa)

            return _ExtractionResult(keywords=keywords, qna_pairs=qna_pairs, categories=categories)

        except Exception as e:
            logger.warning(f"LLM extraction failed, using offline heuristics: {str(e)}")
            return self._analyze_offline(page_text, filename, doc_id, page_number, chunks)

    def _build_prompt(self, page_text: str, filename: str) -> str:
        return (
            "Extract structured entities from the text. Output JSON.\n\n"
            "Entities to extract:\n"
            "- Product name\n"
            "- Policy type (Life / Travel / Medical / Car / SME / Investment / General)\n"
            "- Action (payment, claim, reinstatement, policy change, quote request, etc.)\n"
            "- Required documents / forms\n"
            "- Channels (bank TT, credit card, online banking, 7-11, post office, etc.)\n"
            "- Customer role (policyholder / advisor / company employee)\n"
            "- Country/Market (HK / Macau / Mainland / etc.)\n\n"
            "Rules:\n"
            "- Only extract if explicitly mentioned.\n"
            "- Do NOT infer anything outside of the text.\n\n"
            "Output format:\n\n"
            "{\n"
            "  \"product\": \"\",\n"
            "  \"policy_type\": \"\",\n"
            "  \"action\": \"\",\n"
            "  \"required_form_or_doc\": \"\",\n"
            "  \"payment_channel\": \"\",\n"
            "  \"market\": \"\",\n"
            "  \"customer_role\": \"\"\n"
            "}\n\n"
            "Text:\n"
            f"{page_text}"
        )

    # def _build_qna_prompt(self, chunk_text: str) -> str:
    #     return (
    #         "Generate Q&A pairs from the text. Output JSON array.\n\n"
    #         "Rules:\n"
    #         "- Questions must be specific to the insurance content in the chunk.\n"
    #         "- NO generic questions (e.g., \"What is insurance?\", \"What is HTTPS?\").\n"
    #         "- Only ask questions where the exact answer exists in the text.\n"
    #         "- Focus on details such as coverage, eligibility, personal data usage, payment, clauses, renewal, required documents, responsibilities.\n\n"
    #         "Example types:\n"
    #         "- \"What must the applicant confirm?\"\n"
    #         "- \"Where will the policy documents be delivered?\"\n"
    #         "- \"What personal data may the company collect?\"\n\n"
    #         "Output format:\n\n"
    #         "[\n  {\"question\": \"\", \"answer\": \"\"},\n  ...\n]\n\n"
    #         f"Text:\n{chunk_text}"
    #     )
    def _build_qna_prompt(self, chunk_text: str) -> str:
        return (
            "Generate Q&A pairs from the text below.\n"
            "Return 1–5 pairs when content allows; if none are answerable, return [].\n"
            "Output STRICTLY as a JSON array. No prose. No explanation.\n\n"
            "Rules:\n"
            "- Questions must be SPECIFIC to insurance content present in the chunk.\n"
            "- NO generic questions (e.g., 'What is insurance?', 'How does HTTPS work?').\n"
            "- Answers must come ONLY from the text; do not infer beyond it.\n"
            "- Prioritize explicit facts: label:value pairs, enumerated requirements, amounts/percentages, named forms, policy actions.\n"
            "- Focus on: coverage, eligibility, payment methods, renewal, reinstatement, required documents/forms, policy changes, obligations, data/privacy usage.\n"
            "- Every Q&A must be unique, meaningful, and precisely answerable from the text.\n"
            "- Output must be valid JSON.\n\n"
            "OUTPUT FORMAT:\n"
            "[\n"
            "  {\"question\": \"...\", \"answer\": \"...\"}\n"
            "]\n\n"
            "TEXT START\n"
            f"{chunk_text}\n"
            "TEXT END"
        )


    
    def _analyze_offline(
        self, page_text: str, filename: str, doc_id: str, page_number: int, chunks: List[Chunk]
    ) -> _ExtractionResult:
        # Simple heuristics
        text = page_text.lower()
        entities: list[str] = []  # legacy aggregation
        product_names: list[str] = []
        topics: list[str] = []
        # Categorized output
        insurance_product: list[str] = []
        insurance_term: list[str] = []
        location: list[str] = []
        people_occ_user: list[str] = []

        # Organizations
        orgs = set()
        for m in re.findall(r"prudential|pru ?hk|pruhk|pru", text):
            orgs.add(m.title())
        if orgs:
            insurance_term.extend(list(orgs))  # treat org names as entities/terms for legacy union

        # People (very naive: capitalized words followed by surname-like tokens)
        people_matches = re.findall(r"([A-Z][a-z]+\s[A-Z][a-z]+)", page_text)
        people_occ_user.extend(list(set(people_matches[:5])))

        # Product name heuristics
        for m in re.findall(r"(saver|annuity|term|whole life|investment|rider|study care|critical illness)", text):
            insurance_product.append(m.title())
            product_names.append(m.title())

        # Topics
        for t, kw in {
            "Payments": ["payment", "premium", "billing"],
            "Claims": ["claim", "benefit", "submit"],
            "Compliance": ["policy", "guideline", "regulation"],
            "Customer Service": ["contact", "support", "service"],
        }.items():
            if any(k in text for k in kw):
                topics.append(t)

        # File type classification
        file_type = "Other"
        type_map = {
            "form": "Forms",
            "manual": "Manual",
            "instruction": "Instructions",
            "user guide": "User Guide",
            "guideline": "Guideline",
            "slide": "Slide",
        }
        for k, v in type_map.items():
            if k in text:
                file_type = v
                break

        # Per requirements: key phrases are removed at page level to reduce noise
        key_phrases: list[str] = []

        # Split into English vs Traditional Chinese terms
        def split_en_tc(items: list[str]) -> tuple[list[str], list[str]]:
            en, tc = [], []
            for it in items:
                if re.search(r"[\u4e00-\u9fff]", it or ""):
                    tc.append(it)
                else:
                    en.append(it)
            return en, tc

        entities_en, entities_tc = split_en_tc(entities)
        products_en, products_tc = split_en_tc(product_names)
        topics_en, topics_tc = split_en_tc(topics)

        # Build categorized output
        # Locations
        if "hong kong" in text:
            location.append("Hong Kong")
        if "macau" in text:
            location.append("Macau")

        # Common insurance terms
        for term in [
            "premium",
            "deductible",
            "sum insured",
            "benefit",
            "claim",
            "policyholder",
            "applicant",
            "agent",
        ]:
            if term in text:
                insurance_term.append(term.title())
        # People/occupation/user roles
        for role in ["applicant", "policyholder", "agent", "insured person"]:
            if role in text:
                people_occ_user.append(role.title())

        # Legacy union for backward compatibility
        entities = list(set(insurance_term + location + people_occ_user))

        keywords = KeywordExtraction(
            entities=entities,
            product_names=sorted(list(set(product_names or insurance_product))),
            topics=sorted(list(set(topics))),
            file_type=file_type,
            key_phrases=None,
            entities_en=sorted(list(set(entities_en))),
            entities_tc=sorted(list(set(entities_tc))),
            product_names_en=sorted(list(set(products_en))),
            product_names_tc=sorted(list(set(products_tc))),
            topics_en=sorted(list(set(topics_en))),
            topics_tc=sorted(list(set(topics_tc))),
        )

        # Per requirement: Q&A must be LLM-only; do not generate offline Q&A
        qna_pairs: List[QAPair] = []

        return _ExtractionResult(
            keywords=keywords,
            qna_pairs=qna_pairs,
            categories={
                "insurance_product": sorted(list(set(insurance_product))),
                "insurance_term": sorted(list(set(insurance_term))),
                "location": sorted(list(set(location))),
                "people_occupation_user": sorted(list(set(people_occ_user))),
            },
        )

    def _generate_qna_llm_for_chunk(self, ch: Chunk, doc_id: str, page_number: int) -> List[QAPair]:
        """Generate Q&A pairs using the strict JSON prompt provided."""
        text = (ch.chunk_text or "").strip()
        if not text:
            return []
        prompt = self._build_qna_prompt(text)
        try:
            if not self._openai.chat_client:
                raise RuntimeError("Chat client not configured for Q&A generation")
            response = self._openai.chat_client.chat.completions.create(
                model=self._openai.chat_deployment,
                messages=[
                    {"role": "system", "content": "You generate strict JSON arrays of Q&A pairs."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=400,
                temperature=0.0,
            )
            content = (response.choices[0].message.content or "[]").strip()
            arr = json.loads(content)
            qas: List[QAPair] = []
            for item in arr[:5]:
                q = (item.get("question") or "").strip()
                a = (item.get("answer") or "").strip()
                if q and a and ("https" not in q.lower()) and self._is_answer_anchored(a, text):
                    qas.append(
                        QAPair(
                            question=q,
                            answer=a[:500],
                            qa_doc_id=doc_id,
                            qa_page_number=page_number,
                            qa_chunk_ids=[ch.chunk_id],
                            qa_confidence=0.85,
                        )
                    )
            return qas
        except Exception as e:
            logger.warning(f"LLM Q&A generation failed for chunk {ch.chunk_id}: {str(e)}")
            # Do not fallback to offline; leave empty and let caller handle
            return []

    @staticmethod
    def _is_answer_anchored(answer: str, chunk_text: str) -> bool:
        """Ensure the answer is grounded in the chunk text to reduce hallucinations.
        Strategy: return True if a significant token (≥4 chars) from the answer appears in the chunk text,
        or the whole answer is a substring of the text (case-insensitive).
        """
        a = answer.strip().lower()
        t = (chunk_text or "").strip().lower()
        if not a or not t:
            return False
        if a in t:
            return True
        tokens = [tok for tok in re.findall(r"[a-zA-Z0-9]+", a) if len(tok) >= 4]
        return any(tok in t for tok in tokens)

    def _generate_qna_offline_for_chunk(self, ch: Chunk, doc_id: str, page_number: int) -> List[QAPair]:
        """Offline Q&A adhering to rules: specific, answerable from text, no generic/https."""
        text = (ch.chunk_text or "").strip()
        if not text:
            return []
        qas: List[QAPair] = []
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        # 1) Label:value pairs (English ':' or Chinese '：')
        label_re = re.compile(r"^([A-Za-z][\w\s\-/]+):\s*(.+)$")
        label_tc_re = re.compile(r"^([\u4e00-\u9fff]+)：\s*(.+)$")
        for ln in lines:
            m = label_re.match(ln)
            mtc = label_tc_re.match(ln)
            if m or mtc:
                label = (m.group(1) if m else mtc.group(1)).strip()
                value = (m.group(2) if m else mtc.group(2)).strip()
                if "https" in label.lower():
                    continue
                qas.append(QAPair(question=f"What is '{label}'?", answer=value[:500], qa_doc_id=doc_id, qa_page_number=page_number, qa_chunk_ids=[ch.chunk_id], qa_confidence=0.9))
                if len(qas) >= 3:
                    break

        if len(qas) < 3:
            # 2) Numeric facts with context (premium, fee, percentage)
            amt_re = re.compile(r"(?:HK\$|\$)\s?\d[\d,]*\.?\d*|\d+%|%\s?\d+")
            for ln in lines:
                if amt_re.search(ln):
                    ctx = ln.lower()
                    if "premium" in ctx:
                        question = "What is the premium amount?"
                    elif "fee" in ctx:
                        question = "What fee is specified?"
                    elif "%" in ln:
                        question = "What percentage is mentioned?"
                    else:
                        question = "What amount is specified?"
                    answer = ln.strip()[:500]
                    qas.append(QAPair(question=question, answer=answer, qa_doc_id=doc_id, qa_page_number=page_number, qa_chunk_ids=[ch.chunk_id], qa_confidence=0.8))
                    if len(qas) >= 5:
                        break

        # No generic fallback question; enforce a strict policy of only answerable specifics
        return qas
