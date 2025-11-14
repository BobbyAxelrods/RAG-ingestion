import argparse
from base64 import encode
import json
import os
from typing import Optional
from dotenv import load_dotenv
import pandas as pd
from typing import List, Dict
from openai import AzureOpenAI
import re

# Load env first to ensure artifact/.env values are available
_ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=_ENV_PATH)
_AZURE_ENDPOINT = os.getenv("CHAT_MODEL_ENDPOINT")
_AZURE_KEY = os.getenv("CHAT_MODEL_API_KEY")
_AZURE_API_VERSION = "2023-05-15"
_AZURE_CHAT_DEPLOYMENT = os.getenv("OPENAI_CHAT_DEPLOYMENT")
if not _AZURE_ENDPOINT or not _AZURE_KEY or not _AZURE_CHAT_DEPLOYMENT:
    # Defer hard failure until call time; allow metadata-only usage of this script
    client = None
else:
    client = AzureOpenAI(
        api_key=_AZURE_KEY,
        api_version=_AZURE_API_VERSION,
        azure_endpoint=_AZURE_ENDPOINT,
    )

def generate_qna_pair(chunk: str, max_pairs: int = 5) -> List[Dict[str, str]]:
    """Generate RAG-friendly Q&A pairs from a chunk using strict JSON output.

    Questions must explicitly include salient entities from the text itself.
    Output objects contain only `question` and `answer` keys; no entities list.
    """
    if not client:
        raise RuntimeError("Azure OpenAI chat client not configured (missing endpoint/key/deployment)")
    text = (chunk or "").strip()
    if not text:
        return []
    prompt = (
        "From the text below, generate 1–%d Q&A pairs optimized for retrieval.\n"
        "Return STRICTLY a JSON array; no prose or explanation.\n\n"
        "Rules:\n"
        "- Make each question highly recognizable by explicitly including salient entities from the text in the question itself.\n"
        "  Examples of entities to embed in the question: organization names, product/policy names, form titles/codes, roles, locations, dates, identifiers/labels.\n"
        "  Use exact wording from the text for entity strings; embed 1 of the most relevant entities per question.\n"
        "- Questions must be SPECIFIC to insurance/medical content explicitly present in the text. Avoid generic questions.\n"
        "- Answers must be EXACT substrings or faithful paraphrases anchored in the text; do not invent.\n"
        "- Prioritize enumerated facts: coverage items, eligibility, required documents/forms, payment channels, percentages/amounts, actions.\n"
        "- Use the same language as the text (English or Traditional Chinese). Keep answers concise (≤ 300 chars).\n\n"
        "Output format:\n"
        "[ {\"question\": \"...\", \"answer\": \"...\" } ]\n\n"
        "TEXT START\n%s\nTEXT END"
    ) % (max_pairs, text)
    try:
        resp = client.chat.completions.create(
            model=_AZURE_CHAT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You output valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=600,
            temperature=0.0,
        )
        content = (resp.choices[0].message.content or "[]").strip()
        arr = json.loads(content)
        out: List[Dict[str, str]] = []
        for item in arr[:max_pairs]:
            q = (item.get("question") or "").strip()
            a = (item.get("answer") or "").strip()
            if q and a and ("https" not in q.lower()):
                out.append({"question": q, "answer": a[:500]})
        return out
    except Exception:
        return []


def generate_llm_entities(chunk: str) -> Dict[str, List[str]]:
    """Extract bilingual entities relevant for RAG using strict JSON."""
    if not client:
        raise RuntimeError("Azure OpenAI chat client not configured (missing endpoint/key/deployment)")
    text = (chunk or "").strip()
    if not text:
        return {"entities_en": [], "entities_tc": []}
    prompt = (
        "Extract entities useful for retrieval from the text below.\n"
        "Return STRICT JSON only with two arrays: entities_en, entities_tc.\n\n"
        "Include: organizations, product names, policy types/actions, required forms/documents, payment channels, locations/markets, customer roles, medical terms.\n"
        "Rules:\n"
        "- Only include entities explicitly present in the text.\n"
        "- Deduplicate.\n"
        "- Use Traditional Chinese items in entities_tc and English items in entities_en.\n\n"
        "Output format:\n"
        "{\n  \"entities_en\": [],\n  \"entities_tc\": []\n}\n\n"
        "TEXT START\n%s\nTEXT END"
    ) % text
    try:
        resp = client.chat.completions.create(
            model=_AZURE_CHAT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You output valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=600,
            temperature=0.0,
        )
        content = (resp.choices[0].message.content or "{}").strip()
        data = json.loads(content)
        entities_en = [e.strip() for e in (data.get("entities_en") or []) if e and str(e).strip()]
        entities_tc = [e.strip() for e in (data.get("entities_tc") or []) if e and str(e).strip()]
        # Return small capped lists to keep context clean
        return {"entities_en": entities_en[:50], "entities_tc": entities_tc[:50]}
    except Exception:
        return {"entities_en": [], "entities_tc": []}


def _detect_language(text: str) -> str:
    """Return 'tc' if Traditional Chinese characters found, else 'en'."""
    if not text:
        return "en"
    if re.search(r"[\u4e00-\u9fff]", text):
        return "tc"
    return "en"


def update_entities_qna(
    json_path: str,
    inplace: bool = False,
    out_path: Optional[str] = None,
) -> str:
    """Populate chunk_metadata.chunk_entities and chunk_metadata.page_qna_pairs using chunk_content."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = data.get("chunk_data") or []
    if not isinstance(chunks, list) or not chunks:
        raise KeyError("JSON must contain non-empty 'chunk_data' array")

    for ch in chunks:
        text = (ch.get("chunk_content") or "").strip()
        meta = ch.get("chunk_metadata")
        if meta is None:
            meta = {}
            ch["chunk_metadata"] = meta

        # Entities via LLM
        ents_bi = generate_llm_entities(text)
        lang = _detect_language(text)
        selected = ents_bi["entities_tc"] if lang == "tc" else ents_bi["entities_en"]
        existing = [str(e).strip() for e in (meta.get("chunk_entities") or []) if str(e).strip()]
        merged: List[str] = []
        seen: Dict[str, bool] = {}
        for e in existing + selected:
            k = e.strip()
            if k and not seen.get(k):
                seen[k] = True
                merged.append(k)
        meta["chunk_entities"] = merged[:50]

        # Q&A via LLM
        qna = generate_qna_pair(text)
        meta["page_qna_pairs"] = qna

    # Choose output path
    if inplace:
        output_path = json_path
    else:
        if out_path:
            output_path = out_path
        else:
            base = os.path.basename(json_path)
            name, ext = os.path.splitext(base)
            output_path = os.path.join(os.path.dirname(json_path), f"{name}_entities_qna{ext}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return output_path

def load_excel(excel_path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """Load Excel into a DataFrame and normalize column names.

    If sheet_name is None and the workbook has multiple sheets, pick the first
    sheet with a 'file_name' column. Otherwise, use the provided sheet.
    """
    if sheet_name is None:
        # Load all sheets, find the one containing 'file_name'
        sheets = pd.read_excel(excel_path, sheet_name=None, engine="openpyxl")
        candidate_df: Optional[pd.DataFrame] = None
        for name, df in sheets.items():
            cols = [str(c).strip() for c in df.columns]
            if "file_name" in cols:
                df.columns = cols
                candidate_df = df
                break
        if candidate_df is None:
            # Fallback to the first sheet
            first_name = next(iter(sheets.keys()))
            candidate_df = sheets[first_name]
            candidate_df.columns = [str(c).strip() for c in candidate_df.columns]
    else:
        candidate_df = pd.read_excel(excel_path, sheet_name=sheet_name, engine="openpyxl")
        candidate_df.columns = [str(c).strip() for c in candidate_df.columns]

    if "file_name" not in candidate_df.columns:
        raise ValueError(
            "The Excel must contain a 'file_name' column for exact matching."
        )
    # Ensure 'file_name' comparisons are consistent
    candidate_df["file_name"] = candidate_df["file_name"].astype(str).str.strip()
    return candidate_df


def update_json_with_excel(
    json_path: str,
    df: pd.DataFrame,
    inplace: bool = False,
    out_path: Optional[str] = None,
) -> str:
    """Update JSON's file_index_metadata by populating all Excel columns from exact match on file_name.

    Returns the output path written to.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("file_index_metadata")
    if meta is None:
        raise KeyError(
            "JSON must contain 'file_index_metadata' or 'file_index' with a 'file_name'."
        )

    file_name = str(meta.get("file_name", "")).strip()
    if not file_name:
        raise KeyError("JSON metadata missing 'file_name' for matching.")

    matches = df[df["file_name"] == file_name]
    if matches.empty:
        raise LookupError(
            f"No Excel row found with file_name exactly matching: {file_name}"
        )

    row = matches.iloc[0]
    # Populate all columns from Excel row into JSON metadata (skip NaN)
    for col in df.columns:
        val = row[col]
        # Skip NaN/None values
        if pd.isna(val):
            continue
        # Convert numpy/pandas types to native for JSON
        if hasattr(val, "item"):
            try:
                val = val.item()
            except Exception:
                pass
        meta[str(col)] = val

    # Choose output path
    if inplace:
        output_path = json_path
    else:
        if out_path:
            output_path = out_path
        else:
            base = os.path.basename(json_path)
            name, ext = os.path.splitext(base)
            output_path = os.path.join(
                os.path.dirname(json_path), f"{name}_adjusted{ext}"
            )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Adjust JSON metadata from Excel (exact file_name match) and/or enrich chunk entities & Q&A using LLM."
        )
    )
    parser.add_argument(
        "--json",
        required=True,
        help="Path to the JSON file to adjust.",
    )
    parser.add_argument(
        "--excel",
        default=(
            "artifact/AI-PIL - LifeOps (Cashier, Claims) and GI PIL File List v1.0.xlsx"
        ),
        help="Path to the Excel file containing metadata (must include 'file_name' column).",
    )
    parser.add_argument(
        "--sheet",
        default=None,
        help="Optional Excel sheet name to read (defaults to first sheet).",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Write back to the same JSON file instead of creating *_adjusted.json.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional explicit output path for adjusted JSON.",
    )
    parser.add_argument(
        "--update_entities_qna",
        action="store_true",
        help="Populate chunk_metadata.chunk_entities and chunk_metadata.page_qna_pairs using LLM based on chunk content.",
    )
    parser.add_argument(
        "--skip_excel",
        action="store_true",
        help="Skip Excel-based metadata update; only run entities/Q&A enrichment.",
    )

    args = parser.parse_args()

    output_path = args.json
    if not args.skip_excel:
        df = load_excel(args.excel, sheet_name=args.sheet)
        output_path = update_json_with_excel(
            json_path=args.json, df=df, inplace=args.inplace, out_path=args.out
        )
        print(f"Adjusted JSON written to: {output_path}")

    if args.update_entities_qna:
        output_path = update_entities_qna(
            json_path=output_path,
            inplace=args.inplace,
            out_path=args.out,
        )
        print(f"Entities/Q&A enriched: {output_path}")


if __name__ == "__main__":
    main()

