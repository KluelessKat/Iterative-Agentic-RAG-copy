import os
import re
import json
import time
import yaml
import uuid
import shutil
import requests
import hashlib
from urllib.parse import quote_plus
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
from openai import OpenAI

#checking if the DeepResearcher search_api and text_web_browser modules are available
try:
    from search_api import web_search as deep_web_search
except Exception:
    deep_web_search = None

try:
    from text_web_browser import SimpleTextBrowser
except Exception:
    SimpleTextBrowser = None

# ------------
# Utilities
# ------------

HERE = Path(__file__).resolve().parent #path to current directory Iterative...
OUT_DIR = HERE / "outputs" #directory for output files
OUT_PATH = OUT_DIR / "rag_outputs.jsonl" #specific output file
CONFIG_PATH = HERE / "config.yaml"
SYSTEM_PROMPT_PATH = HERE / "system_prompt.txt"

#regular expressions for extracting tags from LLM responses
TAG_PATTERNS = {
    #for search + fetch page
    "helpful": re.compile(r"<helpful>\s*(yes|no)\s*</helpful>", re.I | re.S),
    "relevance_reasoning": re.compile(r"<relevance_reasoning>(.*?)</relevance_reasoning>", re.I | re.S),
    "extracted_info": re.compile(r"<extracted_info>(.*?)</extracted_info>", re.I | re.S),
    "page_down": re.compile(r"<page_down>\s*(yes|no)\s*</page_down>", re.I | re.S),
    "short_summary": re.compile(r"<short_summary>(.*?)</short_summary>", re.I | re.S),
    #for llm think loop
    "planner_decision": re.compile(r"<decision>\s*(search|answer)\s*</decision>", re.I | re.S),
    "planner_search":   re.compile(r"<search>(.*?)</search>", re.I | re.S),
    "planner_answer":   re.compile(r"<answer>(.*?)</answer>", re.I | re.S),
    "planner_conf":     re.compile(r"<confidence>\s*([0-1](?:\.\d+)?)\s*</confidence>", re.I | re.S),
}

PLANNER_PROMPT = """You are a research agent controller.
Given the MAIN QUESTION and CONTEXT SO FAR, either propose the next short web
search query (1–2 concise phrases) or give the final answer if you’re confident.

Return ONLY with this schema:
<think>brief reasoning</think>
<decision>search|answer</decision>
<search>...</search>     <!-- present only if decision=search -->
<answer>...</answer>     <!-- present only if decision=answer -->
<confidence>0.0-1.0</confidence>
"""

#Iterative Agentic RAG Helper Functions
def read_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

#create output directory if it doesn't exist
def ensure_outdir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

def append_jsonl(record: Dict[str, Any], path: Path = OUT_PATH):
    ensure_outdir()
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def parse_tag(pat: re.Pattern, text: str, default: str = "") -> str:
    m = pat.search(text)
    return m.group(1).strip() if m else default

def parse_extracted_list(block: str) -> List[str]:
    if not block: return []
    lines = []
    for raw in block.splitlines():
        line = raw.strip().lstrip("-•*").strip()
        if line: lines.append(line)
    return lines or [block.strip()]

def merge_context(ctx: str, new_items: List[str], max_chars=12000) -> str:
    if not new_items: return ctx
    seen = set(x.strip() for x in ctx.splitlines() if x.strip())
    added = [x.strip() for x in new_items if x.strip() and x.strip() not in seen]
    merged = (ctx + ("\n" if ctx and added else "") + "\n".join(added)).strip()
    return merged[-max_chars:]

def normalize_fact(s: str) -> str:
    """Normalize for dedup: lowercase, collapse whitespace."""
    return re.sub(r"\s+", " ", s.strip()).lower()

def add_citations_flat(citations: List[Dict[str, Any]],
                       facts: List[str],
                       url: str,
                       page_index: int) -> int:
    """
    Append {"text", "url", "page"} for new (fact,url,page) triples.
    Returns the number of items added.
    """
    # Build a quick membership set to avoid O(N^2)
    seen: Set[Tuple[str, str, int]] = {
        (normalize_fact(c["text"]), c["url"], c["page"]) for c in citations
    }
    start = len(citations)
    for f in facts:
        key = (normalize_fact(f), url, page_index)
        if key not in seen:
            citations.append({"text": f.strip(), "url": url, "page": page_index})
            seen.add(key)
    return len(citations) - start

def slugify(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return re.sub(r"-+", "-", s).strip("-")[:80]

#Return current UTC timestamp in ISO 8601 with a trailing Z
def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

#Logging LLM inputs
def sha12(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:12]

def clean_preview(s: str, n: int) -> str:
    return re.sub(r"\s+", " ", (s or "")[:n]).strip()

#creates an acceptable file name
def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:120]

def dump_prompt(session_id: str, step: int, system_prompt: str, user_block: str):
    d = OUT_DIR / "debug_prompts"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{session_id}-step{step:03d}-system.txt").write_text(system_prompt, encoding="utf-8")
    (d / f"{session_id}-step{step:03d}-user.txt").write_text(user_block, encoding="utf-8")

def decide_next_action(llm: "LLMClient", main_q: str, context: str, last_search: Optional[str]) -> Dict[str, Any]:
    user = f"""MAIN QUESTION: {main_q}

                CONTEXT SO FAR:
                {context or '(empty)'}

                LAST SEARCH (may be None): {last_search or 'None'}
                """
    # If you're running with the debug stub, synthesize a toy trajectory:
    if getattr(llm, "debug_fake", False):
        if getattr(llm, "debug_fake", False):
            # Generic deterministic behavior for ANY question:
            # - First time: search the main question
            # - Next: refine if context is still small
            # - Finally: answer with a generic synthesized reply
            have_some_context = len(context) >= 200 or ("-" in context)  # crude: grew a few bullets
            if last_search is None:
                # first planner step: seed with the main question text
                return {
                    "decision": "search", 
                    "search": main_q, 
                    "answer": "", 
                    "conf": 0.2}
            elif not have_some_context:
                # ask for more details using the previous query
                return {
                    "decision": "search", 
                    "search": f"{last_search} details", 
                    "answer": "", 
                    "conf": 0.3}
            else:
                # stop with a clearly marked debug answer
                return {
                    "decision": "answer", 
                    "search": "", 
                    "answer": "(debug) synthesized answer", 
                    "conf": 0.8}

    raw = llm.complete(PLANNER_PROMPT, user)
    return {
        "decision": parse_tag(TAG_PATTERNS["planner_decision"], raw, "search"),
        "search":   parse_tag(TAG_PATTERNS["planner_search"], raw, ""),
        "answer":   parse_tag(TAG_PATTERNS["planner_answer"], raw, ""),
        "conf":     float(parse_tag(TAG_PATTERNS["planner_conf"], raw, "0.0") or 0.0),
    }

# ------------
# Tracing Process
# ------------
class TraceLogger:
    def __init__(self, session_id: str, cfg: Dict[str, Any], title: str = ""):
        self.session_id = session_id
        self.cfg = cfg
        self.title = title
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        slug = slugify(title) or "session"
        # e.g., 20250818-153205__why-are-apples-different-colors__2d58446f.md
        self.basename = f"{ts}__{slug}__{session_id[:8]}"
        self.trace_dir = OUT_DIR / "traces"
        self.prompts_dir = OUT_DIR / "debug_prompts"
        self.pages_dir = OUT_DIR / "pages" / self.basename
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        self.pages_dir.mkdir(parents=True, exist_ok=True)
        self.md_path = self.trace_dir / f"{self.basename}.md"
        if cfg.get("debug_save_trace_md", False):
            self.md_path.write_text("", encoding="utf-8")

    def md(self, text: str):
        if self.cfg.get("debug_save_trace_md", False):
            with open(self.md_path, "a", encoding="utf-8") as f:
                f.write(text + "\n")

    def start(self, question: str, sys_prompt: str):
        self.md(f"# Session {self.session_id}\n\n**Question:** {question}\n")
        self.md(f"**System prompt chars:** {len(sys_prompt)}  \n`sha:{sha12(sys_prompt)}`\n")

    def log_search(self, query: str, urls: List[str]):
        self.md(f"\n## Search\n- **Query:** `{query}`\n- **URLs ({len(urls)}):**\n" + "\n".join([f"  - {u}" for u in urls]))

    def log_page_snapshot(self, url: str, page_index: int, total_pages: int, text: str):
        head = clean_preview(text, 240)
        sha = sha12(text)
        self.md(f"\n### Page {page_index}/{total_pages} — {url}\n"
                f"- chars: {len(text)}  sha:`{sha}`\n"
                f"- head: “{head}”")
        if self.cfg.get("debug_save_pages", False):
            name = f"{safe_name(url)}__p{page_index:03d}__{sha}.txt"
            (self.pages_dir / name).write_text(text, encoding="utf-8")

    def log_prompt(self, step: int, sys_prompt: str, user_block: str):
        if self.cfg.get("debug_log_prompts", False):
            N = int(self.cfg.get("debug_prompt_chars", 400))
            print(f"[PROMPT] system {len(sys_prompt)} sha={sha12(sys_prompt)} head='{clean_preview(sys_prompt, N)}'")
            print(f"[PROMPT] user   {len(user_block)} sha={sha12(user_block)} head='{clean_preview(user_block, N)}'")
        if self.cfg.get("debug_dump_full_prompts", False):
            (self.prompts_dir / f"{self.basename}-step{step:03d}-system.txt").write_text(sys_prompt, encoding="utf-8")
            (self.prompts_dir / f"{self.basename}-step{step:03d}-user.txt").write_text(user_block, encoding="utf-8")
        self.md(f"\n#### Step {step} — Prompt hashes\n- system:`{sha12(sys_prompt)}`  user:`{sha12(user_block)}`")

    def log_llm_out(self, raw: str, parsed: Dict[str,str], extracted_list: List[str]):
        self.md(f"**LLM helpful:** {parsed['helpful']}  **page_down:** {parsed['page_down']}\n")
        if extracted_list:
            self.md("**Extracted info:**\n" + "\n".join([f"- {x}" for x in extracted_list]))
        else:
            self.md("_No extracted info._")

    def log_merge(self, before: int, after: int, new_facts: List[str]):
        delta = after - before
        if delta > 0:
            self.md(f"**Context grew:** {before} → {after} (+{delta})\n**New facts merged:**\n" +
                    "\n".join([f"- {x}" for x in new_facts]))
        else:
            self.md("**No merge this step.**")

    def finish(self, context: str):
        self.md("\n## Final Context Snapshot\n```\n" + context + "\n```")

# ------------
# Pluggable LLM client
# ------------

class LLMClient:
    """
    Minimal wrapper. Replace with your provider of choice.
    Examples:
      - OpenAI Responses API (gpt-4o, gpt-4.1, etc.)
      - Together, Anthropic, local model, etc.
    """
    def __init__(self, model: str, temperature: float = 0.2, debug_fake: bool = False):
        self.model = model
        self.temperature = temperature
        self.debug_fake = debug_fake
        self._dbg_i = 0 #debug counter

    #for debugging, we can use a fake response to test the loop
    def _fake_response(self) -> str:
        """Return deterministic XML for testing the loop."""
        self._dbg_i += 1
        if self._dbg_i == 1:
            # Pretend first page was helpful and tells us to page down
            return (
                "<helpful>yes</helpful>\n"
                "<relevance_reasoning>Contains pigment basics relevant to the question.</relevance_reasoning>\n"
                "<extracted_info>\n"
                "- Red apple color largely comes from anthocyanin pigments that accumulate as fruit ripens and chlorophyll degrades.\n"
                "- Pigment expression varies by cultivar and sunlight exposure.\n"
                "</extracted_info>\n"
                "<page_down>yes</page_down>\n"
                "<short_summary>Found pigment basics; paging for more detail.</short_summary>"
            )
        elif self._dbg_i == 2:
            # Second page adds new facts and stops paging
            return (
                "<helpful>yes</helpful>\n"
                "<relevance_reasoning>Adds green coloration mechanism.</relevance_reasoning>\n"
                "<extracted_info>\n"
                "- Green apples appear green when chlorophyll is retained or degrades more slowly.\n"
                "- Temperature and light affect pigment genes; bagging/shade reduces anthocyanin.\n"
                "</extracted_info>\n"
                "<page_down>no</page_down>\n"
                "<short_summary>Captured green mechanism; moving to next source.</short_summary>"
            )
        else:
            # Anything after that: not helpful
            return (
                "<helpful>no</helpful>\n"
                "<relevance_reasoning>Nothing new.</relevance_reasoning>\n"
                "<extracted_info></extracted_info>\n"
                "<page_down>no</page_down>\n"
                "<short_summary>No new info here.</short_summary>"
            )

    def complete(self, system_prompt: str, user_content: str) -> str:
        """
        Return raw string from the model. Implement with your SDK.
        For illustration, we read from env OPENAI_API_KEY if present.
        """
        # --- debug short-circuit ---
        if self.debug_fake:
            print("[LLM] DEBUG: returning fake canned response.")
            return self._fake_response()
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("[LLM] OPENAI_API_KEY not set → returning stub response (helpful=no).")
            # Stubbed fallback so the script doesn't crash during wiring.
            # Replace with real API call.
            return ("""<helpful>no</helpful>
            <relevance_reasoning>LLM not configured. This is a stub response.</relevance_reasoning>
            <extracted_info></extracted_info>
            <page_down>no</page_down>
            <short_summary>No LLM configured; skipping.</short_summary>"""
            )

        # --- Example with openai>=1.0 style (uncomment and add to requirements) ---
        # from openai import OpenAI
       
        try:
            print(f"[LLM] Calling model={self.model} temperature={self.temperature}")
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            )
            content = resp.choices[0].message.content or ""
            print(f"[LLM] Got response ({len(content)} chars).")
            return content
        except Exception as e:
            print(f"Error calling OpenAI: {e}")
            return f"""<helpful>no</helpful>
            <relevance_reasoning>Error calling OpenAI: {e}</relevance_reasoning>
            <extracted_info></extracted_info>
            <page_down>no</page_down>
            <short_summary>Error calling OpenAI: {e}</short_summary>"""

        # If you reach here, user didn't wire an SDK. Return stub.
        return f"""<helpful>no</helpful>
<relevance_reasoning>No client wired.</relevance_reasoning>
<extracted_info></extracted_info>
<page_down>no</page_down>
<short_summary>LLM not called.</short_summary>"""

# ------------
# Very simple search/fetch placeholders
# ------------

# def search_web(query: str, k: int = 5) -> List[str]:
#     """
#     Return list of URLs. Replace with your real web.run, SerpAPI, Tavily, etc.
#     """
#     # Stub: pretend we found 2 URLs
#     return [
#         f"https://example.com/search?q={query.replace(' ', '+')}&r=1",
#         f"https://example.org/article?q={query.replace(' ', '+')}&r=2",
#     ][:k]

# def fetch_paginated(url: str, max_pages: int = 3) -> List[Tuple[int, int, str]]:
#     """
#     Return a list of (page_index, total_pages, page_content).
#     Replace with real HTTP fetch + readability + pagination logic.
#     """
#     total = max_pages
#     return [(i + 1, total, f"Stub page {i+1}/{total} content for {url}. (Likely an introduction on page 1.)")
#             for i in range(total)]

#query = the search query that you want to search for on the web
#k = the number of results you want to return
#cfg = the configuration for the search

def search_web(query: str, k: Optional[int] = None, cfg: Optional[Dict[str, Any]] = None) -> List[str]:
    k = int(k or (cfg or {}).get("search_top_k") or 5)

    # --- debug short-circuit ---
    if (cfg or {}).get("debug_fake_search"):
        print(f"[SEARCH] DEBUG: returning fake URLs for {query!r}")
        k = k or 3
        return [f"https://debug.local/source{i}" for i in range(1, k+1)]

    # 1) DeepResearcher path
    if deep_web_search and cfg is not None:
        try:
            engine = str(cfg.get("search_engine", "serper"))
            print(f"[SEARCH] deep_web_search engine={engine} q={query!r} top_k={k}")
            results = deep_web_search(query, cfg) or []
            urls: List[str] = []
            for r in results:
                if isinstance(r, dict):
                    u = r.get("link") or r.get("url") or r.get("href")
                    if u:
                        urls.append(u)
            if urls:
                print(f"[SEARCH] deep_web_search returned {len(urls)} URL(s).")
                return urls[:k]
            else:
                print("[SEARCH] deep_web_search returned 0 results.")
        except Exception as e:
            print(f"[SEARCH] deep_web_search failed: {e}")

    # 2) Serper fallback (direct HTTP) if you have a key
    serper_key = os.getenv("SERPER_API_KEY") or (cfg or {}).get("serper_api_key")
    if serper_key:
        try:
            #import requests
            url = "https://google.serper.dev/search"
            headers = {"X-API-KEY": serper_key, "Content-Type": "application/json"}
            payload = {
                "q": query,
                "num": k,
                "gl": (cfg or {}).get("search_region", "us"),
                "hl": (cfg or {}).get("search_lang", "en"),
            }
            print(f"[SEARCH] Serper fallback → POST {url} num={k}")
            resp = requests.post(url, headers=headers, json=payload, timeout=int((cfg or {}).get("http_timeout", 15)))
            if resp.status_code >= 400:
                print(f"[SEARCH] Serper HTTP {resp.status_code}: {resp.text[:200]}")
                # don't return; try the stub next
            else:
                data = resp.json()
                urls = [item.get("link") for item in (data.get("organic") or []) if item.get("link")]
                print(f"[SEARCH] Serper returned {len(urls)} URL(s).")
                if urls:
                    return urls[:k]
        except Exception as e:
            print(f"[SEARCH] Serper fallback failed: {e}")

    # 3) Stub so the loop keeps running
    print(f"[SEARCH] Using stub URLs for {query!r} (k={k})")
    return [f"https://example.com/search?q={quote_plus(query)}&r={i}" for i in range(1, k + 1)]

def fetch_pages(url: str, cfg: Dict[str, Any]) -> Tuple[List[str], int]:
    """
    Fetch and paginate readable text for a URL.

    Returns:
        (pages, total_pages)
        - pages: list of page strings (length <= max_pages_per_url)
        - total_pages: total available pages for this URL (best-effort; equals len(pages) in fallback mode)
    """
    max_pages = int(cfg.get("max_pages_per_url", 3))
    
    # --- debug short-circuit ---
    if (cfg or {}).get("debug_fake_fetch"):
        print(f"[FETCH] DEBUG: returning fake pages for {url!r}")
        pages = [
            "Page 1: Anthocyanins give red color; sun & temp regulate expression.",
            "Page 2: Chlorophyll retention explains green color in some cultivars."
        ]
        return pages, len(pages)

    # --- Preferred path: DeepResearcher's SimpleTextBrowser ---
    if SimpleTextBrowser:
        try:
            print(f"[FETCH] Using SimpleTextBrowser for: {url}")
            browser = SimpleTextBrowser(
                viewport_size=int(cfg.get("viewport_size", 1024 * 8)),
                request_kwargs={"timeout": int(cfg.get("http_timeout", 15))},
                serpapi_key=cfg.get("serpapi_api_key"),
                serper_api_key=cfg.get("serper_api_key"),
            )
            browser.address(url)
            total = len(browser.viewport_pages)
            take = min(total, max_pages)
            pages: List[str] = []
            for i in range(take):
                pages.append(browser.viewport)
                if i < take - 1:
                    browser.page_down()
            print(f"[FETCH] Browser total_pages={total}, returning={len(pages)}")
            return pages, total
        except Exception as e:
            print(f"[FETCH] SimpleTextBrowser failed, falling back. Error: {e}")

    # --- Fallback: requests + text extraction ---
    try:
        import requests  # ensure available even if you moved imports
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        timeout = int(cfg.get("http_timeout", 15))
        resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
    except Exception as e:
        print(f"[FETCH] requests.get failed for {url}: {e}")
        return [], 0

    if resp.status_code >= 400:
        print(f"[FETCH] HTTP {resp.status_code} for {url}")
        return [], 0

    ctype = resp.headers.get("Content-Type", "")
    if ("text" not in ctype) and ("html" not in ctype) and ("xml" not in ctype):
        print(f"[FETCH] Unsupported content-type '{ctype}' for {url}")
        return [], 0

    html = resp.text or ""
    text = ""

    # Try high-quality extraction if trafilatura is present
    try:
        import trafilatura  # optional dependency
        extracted = trafilatura.extract(html) or ""
        if extracted:
            text = extracted
            print(f"[FETCH] Used trafilatura extraction ({len(text)} chars).")
    except Exception:
        pass

    # Basic HTML stripping fallback
    if not text:
        try:
            # Remove script/style first
            stripped = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", " ", html, flags=re.I | re.S)
            # Remove all tags
            stripped = re.sub(r"<[^>]+>", " ", stripped)
            # Collapse whitespace
            text = re.sub(r"\s+", " ", stripped).strip()
            print(f"[FETCH] Used basic HTML strip ({len(text)} chars).")
        except Exception as e:
            print(f"[FETCH] HTML strip failed: {e}")
            return [], 0

    if not text:
        print(f"[FETCH] No extractable text for {url}")
        return [], 0

    # Chunk into "pages"
    chunk_size = int(cfg.get("fallback_page_chars", 3000))
    limit = chunk_size * max_pages
    sliced = text[:limit]
    pages = [sliced[i : i + chunk_size] for i in range(0, len(sliced), chunk_size)]
    total_pages = len(pages)  # best-effort in fallback mode
    print(f"[FETCH] Fallback returning {len(pages)} page(s) (chunk_size={chunk_size}).")

    return pages, total_pages

# ------------
# Prompt assembly (using your single-agent schema)
# ------------

def build_user_payload(
    system_prompt_template: str,
    main_question: str,
    sub_question: str,
    context_so_far: str,
    user_query: str,
    search_query: str,
    page_index: int,
    total_pages: int,
    page_content: str,
) -> str:
    """
    Fill the merged single-agent prompt template.
    Assumes the template contains the placeholders from your system_prompt.txt.
    """
    # The system prompt itself stays as the system role. The user content is the INPUTS block:
    user_block = f"""
<main_question>
{main_question}
</main_question>

<context_so_far>
{context_so_far}
</context_so_far>

<current_sub_question>
{sub_question}
</current_sub_question>

<search_meta>
    <user_query>{user_query}</user_query>
    <search_query>{search_query}</search_query>
</search_meta>

<webpage_content>
    <page_index>{page_index}</page_index>
    <total_page_number>{total_pages}</total_page_number>
    <current_page_content>
{page_content}
    </current_page_content>
</webpage_content>
"""
    return user_block.strip()


# ------------
# Agent runner
# ------------

class RAGAgent:
    def __init__(self, cfg: Dict[str, Any], sys_prompt: str):
        self.cfg = cfg
        self.sys_prompt = sys_prompt
        self.client = LLMClient(
            model=cfg.get("model", "gpt-4o-mini"),
            temperature=float(cfg.get("temperature", 0.2)),
            debug_fake=bool(self.cfg.get("debug_fake_llm", False))
        )
        self.session_id = str(uuid.uuid4())
        self.citations: List[Dict[str, Any]] = []
        self.tracer = TraceLogger(self.session_id, self.cfg)
        print(f"[INIT] LLM debug_fake={self.client.debug_fake}")

    def answer_question(self, main_question: str, user_query: Optional[str] = None) -> Dict[str, Any]:
        self.session_id = str(uuid.uuid4())
        self.citations = []
        self.tracer = TraceLogger(self.session_id, self.cfg)

        self.tracer.start(main_question, self.sys_prompt)

        context_so_far = ""
        history: List[Dict[str, Any]] = []
        last_search = user_query or None  #will update each step from the planner to issue refined searches
        pending_pages: List[Tuple[str,int,int,str]] = []  # (url, page_idx, total_pages, text)

        max_steps = int(self.cfg.get("max_steps", 5))
        stop_when_confident = bool(self.cfg.get("stop_when_confident", True))
        search_k = int(self.cfg.get("search_top_k") or self.cfg.get("search_k", 5))

        steps = 0
        final_answer = None

        while steps < max_steps:
            steps += 1
            print(f"\n========== STEP {steps} ==========")

            # 1) If we don't have pages queued, ask the planner what to do next
            if not pending_pages:
                decision = decide_next_action(self.client, main_question, context_so_far, last_search)
                history.append({"step": steps, **decision})

                # Compute the effective action after confidence gating
                action = decision["decision"]
                query  = (decision["search"] or main_question).strip()

                if action == "answer":
                    final_answer = decision["answer"].strip()
                    if final_answer and (not stop_when_confident or decision["conf"] >= 0.6):
                        print(f"[PLAN] decision=answer  conf={decision['conf']:.2f}  answer_head={final_answer[:120]!r}")
                        print("[STOP] Planner produced answer with sufficient confidence.")
                        break  # confident enough (or no early-stop gating)
                    # Not confident → fall back to one more search based on last context
                    action = "search"
                    query  = (decision["search"] or last_search or main_question).strip()

                # Run a search for the planner’s query
                if action == "search":
                    print(f"[PLAN] decision=search  conf={decision['conf']:.2f}  query={query!r}")
                    last_search = query
                    urls = search_web(query, k=search_k, cfg=self.cfg)

                    print(f"[SEARCH] query={query!r}  -> {len(urls)} URL(s)")
                    for idx, u in enumerate(urls, 1):
                        print(f"         {idx:>2}/{len(urls)}  {u}")
                    # Fetch first batch of pages (lazy: one URL at a time)
                    for url in urls:
                        pages, total_pages = fetch_pages(url, self.cfg)
                        if not pages:
                            print(f"[FETCH] (no pages) {url}")
                            continue
                        # Queue pages for triage/browse
                        head = clean_preview(pages[0], 180)
                        print(f"[FETCH] {url}  pages={len(pages)}/{total_pages}  head=“{head}”")
                        for j, pg in enumerate(pages, start=1):
                            self.tracer.log_page_snapshot(url, j, total_pages, pg)
                            pending_pages.append((url, j, total_pages, pg))
                        # Process one URL per planner step; leave others for later steps
                        if pending_pages:
                            break

            # 2) If still nothing to read, we’re stuck
            if not pending_pages:
                print("[HALT] No pages queued; stopping.")
                break

            # 3) Process exactly one page per step (keeps the loop responsive)
            url, page_index, total_pages, page_content = pending_pages.pop(0)

            # ---- build triage prompt (you already have build_user_block) ----
            sub_question = main_question  # or your planner-derived subquestion
            user_block = build_user_payload(
                system_prompt_template=self.sys_prompt,
                main_question=main_question,
                sub_question=sub_question,
                context_so_far=context_so_far,
                user_query=user_query or main_question,
                search_query=last_search or "", # actual query *this* page came from
                page_index=page_index,
                total_pages=total_pages,
                page_content=page_content,
            )

            # Log prompt previews / full dumps (you already had this)
            self.tracer.log_prompt(step=steps, sys_prompt=self.sys_prompt, user_block=user_block)

            # ---- call LLM for page triage/extraction ----
            raw = self.client.complete(self.sys_prompt, user_block)
            parsed = self._parse_llm_output(raw)
            extracted_list = parse_extracted_list(parsed["extracted_info"])
            self.tracer.log_llm_out(raw, parsed, extracted_list)

            # ---- merge new info into context ----
            before = len(context_so_far)
            existing = {normalize_fact(l) for l in context_so_far.splitlines() if l.strip()}
            new_facts = [f for f in extracted_list if normalize_fact(f) not in existing]
            if new_facts:
                context_so_far = merge_context(context_so_far, new_facts)
                self.tracer.log_merge(before, len(context_so_far), new_facts)
                add_citations_flat(self.citations, new_facts, url, page_index)
            else:
                self.tracer.log_merge(before, before, [])

            # ---- respect page_down: if "no", clear any remaining pages from this URL ----
            if parsed["page_down"].lower() != "yes":
                pending_pages = [p for p in pending_pages if p[0] != url]

        # If loop ended without a final answer, do one last planner call to produce one.
        if not final_answer:
            decision = decide_next_action(self.client, main_question, context_so_far, last_search)
            final_answer = (decision.get("answer") or "").strip() or ""

        # finalize trace
        self.tracer.finish(context_so_far)

        record = {
            "session_id": self.session_id,
            "ts": now_iso(),
            "question": main_question,
            "final_answer": final_answer,
            "context_so_far": context_so_far,
            "citations": self.citations,
            "history": history,
            "steps": steps,
        }
        append_jsonl(record)

        return record


    # def answer_question(self, main_question: str, user_query: Optional[str] = None) -> Dict[str, Any]:
    #     # --- session + tracer per question ------------------------------------
    #     self.session_id = str(uuid.uuid4())                                  
    #     self.tracer = TraceLogger(self.session_id, self.cfg, title=main_question) 

    #     # --- bootstrap ---------------------------------------------------------
    #     context_so_far = self.cfg.get("seed_context", "").strip()
    #     print(f"[RUN] Q: {main_question}")
    #     print(f"[RUN] Seed context length: {len(context_so_far)}")
    #     self.tracer.start(main_question, self.sys_prompt)

    #     search_query = user_query or main_question
    #     urls = search_web(
    #         search_query,
    #         k=int(self.cfg.get("search_k", 5)),
    #         cfg=self.cfg
    #     )
    #     print(f"[RUN] Found {len(urls)} URL(s).")

    #     max_steps = int(self.cfg.get("max_steps", 12))
    #     steps = 0
    #     sub_question = self._derive_initial_subquestion(main_question)

    #     # --- iterate URLs --------
    #     for url in urls:
    #         print(f"[URL] {url}")
    #         pages, total_pages = fetch_pages(url, self.cfg)
    #         print(f"[URL] Got {len(pages)} page(s) (total_pages={total_pages}).")
    #         if not pages:
    #             continue

    #         # page snapshots (so you can see what was actually fetched)
    #         for j, pg in enumerate(pages, start=1):
    #             self.tracer.log_page_snapshot(url, j, total_pages, pg)
            
    #         # --- iterate pages for this URL ----------
    #         for i, page_content in enumerate(pages, start=1):
    #             print(f"[PAGE] URL={url} page_index={i} chars={len(page_content)}")
    #             if steps >= max_steps:
    #                 break
    #             steps += 1

    #             # build the user payload for the LLM
    #             user_payload = build_user_payload(
    #                 self.sys_prompt, main_question, sub_question, context_so_far,
    #                 user_query or main_question, search_query, i, total_pages, page_content
    #             )

    #             # show/dump prompts
    #             self.tracer.log_prompt(steps, self.sys_prompt, user_payload)

    #             # Debug: show exactly what we send to the LLM
    #             if self.cfg.get("debug_log_prompts", False):
    #                 N = int(self.cfg.get("debug_prompt_chars", 400))
    #                 sys_hash = sha12(self.sys_prompt)
    #                 usr_hash = sha12(user_payload)
    #                 print(f"[PROMPT] system chars={len(self.sys_prompt)} sha={sys_hash} head='{clean_preview(self.sys_prompt, N)}'")
    #                 print(f"[PROMPT] user   chars={len(user_payload)} sha={usr_hash} head='{clean_preview(user_payload, N)}'")

    #             if self.cfg.get("debug_dump_full_prompts", False):
    #                 dump_prompt(self.session_id, steps, self.sys_prompt, user_payload)

    #             #call LLM
    #             raw = self.client.complete(self.sys_prompt, user_payload)
    #             print(f"[LLM] Raw head: {clean_preview(raw, 200)}")

    #             # parse & trace LLM output
    #             parsed = self._parse_llm_output(raw)
    #             print(f"[PARSE] helpful={parsed['helpful']}, page_down={parsed['page_down']}")
                
    #             extracted_list = parse_extracted_list(parsed["extracted_info"])
    #             print(f"[PARSE] extracted_info items={len(extracted_list)}")

    #             self.tracer.log_llm_out(raw, parsed, extracted_list)

    #             # compute new facts vs current context
    #             existing = {re.sub(r"\s+", " ", l.strip()).lower()
    #                         for l in context_so_far.splitlines() if l.strip()}
    #             new_facts = [f for f in extracted_list
    #                          if re.sub(r"\s+", " ", f.strip()).lower() not in existing]

    #             # merge & trace context growth
    #             before = len(context_so_far)
    #             after = before          # default if we don't merge
    #             delta = 0               # default growth
    #             merged_this_step = False

    #             if parsed["helpful"].lower() == "yes" and extracted_list:
    #                 # Compute "new facts" against existing context lines
    #                 existing = {normalize_fact(l) for l in context_so_far.splitlines() if l.strip()}
    #                 new_facts = [f for f in extracted_list if normalize_fact(f) not in existing]

    #                 if new_facts:
    #                     # Merge only the new facts into the context
    #                     context_so_far = merge_context(context_so_far, new_facts)
    #                     after = len(context_so_far)
    #                     delta = after - before
    #                     merged_this_step = delta > 0
    #                     print(f"[MERGE] context grew {before} → {after} (+{delta})")

    #                     # Add citations for the new facts (linking to this URL+page)
    #                     added_cites = add_citations_flat(self.citations, new_facts, url, i)
    #                     print(f"[CITE] added {added_cites} citation(s) for this page")
    #                 else:
    #                     print("[MERGE] Skipped: all extracted facts were already present")
    #             else:
    #                 print(f"[MERGE] Skipped (helpful={parsed['helpful']}, items={len(extracted_list)})")

    #             self.tracer.log_merge(before, after, new_facts)

    #             record = {
    #                 "session_id": self.session_id,
    #                 "ts": now_iso(),
    #                 "step": steps,
    #                 "url": url,
    #                 "page_index": i,
    #                 "total_pages": total_pages,
    #                 "helpful": parsed["helpful"],
    #                 "relevance_reasoning": parsed["relevance_reasoning"],
    #                 "extracted_info": extracted_list,
    #                 "page_down": parsed["page_down"],
    #                 "short_summary": parsed["short_summary"],
    #                 "context_chars_before": before,
    #                 "context_chars_after": after,  
    #                 "merge_delta": delta,           
    #                 "merged": merged_this_step,
    #                 "prompt_system_chars": len(self.sys_prompt),
    #                 "prompt_system_sha12": sha12(self.sys_prompt),
    #                 "prompt_user_chars": len(user_payload),
    #                 "prompt_user_sha12": sha12(user_payload),
    #             }
    #             append_jsonl(record)

    #             # paging decision ------------------------------------------------
    #             # Continue to next page only if the model explicitly says so
    #             if parsed["page_down"].lower() == "yes" and i < len(pages):
    #                 print("[NAV] Model requested page_down → next page")
    #                 # optional: refine subquestion
    #                 sub_question = self._maybe_refine_subquestion(sub_question, context_so_far)
    #                 continue
    #             else:
    #                 print("[NAV] Stop paging this URL.")
    #                 break  # move to next URL

    #         if self._enough_info(context_so_far):
    #             print("[STOP] Enough info gathered; stopping.")
    #             break

    #     # while steps < max_steps:
    #     #     step += 1


    #     final_record = {
    #         "session_id": self.session_id,
    #         "ts": now_iso(),
    #         "type": "final_context",
    #         "main_question": main_question,
    #         "context_so_far": context_so_far,
    #         "citations": self.citations,
    #     }
    #     append_jsonl(final_record)

    #     try:
    #         (OUT_DIR / "rag_citations.json").write_text(
    #             json.dumps(self.citations, ensure_ascii=False, indent=2),
    #             encoding="utf-8"
    #         )
    #         print(f"[CITE] wrote {len(self.citations)} citations → {OUT_DIR/'rag_citations.json'}")
    #     except Exception as e:
    #         print(f"[CITE] could not write rag_citations.json: {e}")

    #     return final_record

    #def run_llm_loop(self, main_question: str, user_query: Optional[str] = None)
        

    # --- helpers ---

    def _parse_llm_output(self, text: str) -> Dict[str, str]:
        return {
            "helpful": parse_tag(TAG_PATTERNS["helpful"], text, default="no"),
            "relevance_reasoning": parse_tag(TAG_PATTERNS["relevance_reasoning"], text, default=""),
            "extracted_info": parse_tag(TAG_PATTERNS["extracted_info"], text, default=""),
            "page_down": parse_tag(TAG_PATTERNS["page_down"], text, default="no"),
            "short_summary": parse_tag(TAG_PATTERNS["short_summary"], text, default=""),
        }

    def _derive_initial_subquestion(self, main_q: str) -> str:
        # Minimal heuristic; replace with a proper planner if you like.
        return main_q.strip()

    def _maybe_refine_subquestion(self, sub_q: str, context: str) -> str:
        # Hook for your iterative planner; keep as-is for now.
        return sub_q

    def _enough_info(self, context: str) -> bool:
        # Very naive stopping heuristic; tweak as needed.
        target_min_chars = int(self.cfg.get("min_context_chars", 400))
        return len(context) >= target_min_chars

# ------------
# Optional: run over an Excel input (medical_dataset.xlsx)
# ------------

def run_excel_batch(agent: RAGAgent, xlsx_path: Path):
    try:
        import pandas as pd
    except ImportError:
        raise SystemExit("Please `pip install pandas openpyxl` to use Excel batch mode.")

    df = pd.read_excel(xlsx_path)
    # Expect a column named 'question' (customize to your sheet)
    if "question" not in df.columns:
        raise SystemExit("Expected a 'question' column in the Excel file.")
    for i, row in df.iterrows():
        q = str(row["question"]).strip()
        if not q or q.lower() == "nan":
            continue
        print(f"\n[{i}] Processing question: {q[:120]}...")
        agent.answer_question(q)
        time.sleep(0.2)  # be polite to APIs

# ------------
# CLI
# ------------

def main():
    cfg = load_yaml(CONFIG_PATH) if CONFIG_PATH.exists() else {}
    os.environ.setdefault("OPENAI_API_KEY", cfg.get("openai_api_key", ""))
    os.environ.setdefault("SERPAPI_API_KEY", cfg.get("serpapi_api_key", ""))
    os.environ.setdefault("SERPER_API_KEY",  cfg.get("serper_api_key",  ""))

    sys_prompt = read_text(SYSTEM_PROMPT_PATH)
    print(f"[BOOT] Loaded config keys: {list(cfg.keys())}")
    print(f"[BOOT] system_prompt.txt length: {len(sys_prompt)} chars")

    # Setup outputs folder on first run
    ensure_outdir()
    if not OUT_PATH.exists():
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            pass  # create empty file

    agent = RAGAgent(cfg, sys_prompt)

    # Modes:
    # 1) Batch over Excel if present
    xlsx_path = HERE / "medical_dataset.xlsx"
    if xlsx_path.exists():
        print("Detected medical_dataset.xlsx — running batch mode...")
        run_excel_batch(agent, xlsx_path)
        return

    # 2) Single interactive question (fallback)
    main_q = cfg.get("single_question") or input("Enter your main question: ").strip()
    if not main_q:
        print("No question provided. Exiting.")
        return

    result = agent.answer_question(main_q)
    print("\nFinal context snapshot:\n")
    print(result["context_so_far"])
    print(f"\nAppended to {OUT_PATH}")

if __name__ == "__main__":
    main()