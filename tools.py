"""tools.py - STUDENT IMPLEMENTS.  Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json  # noqa: F401
import os  # noqa: F401
import time  # noqa: F401
import xml.etree.ElementTree as ET # noqa: F401  (arXiv answers with Atom XML)
import re
import random

import httpx  # noqa: F401
from langchain_core.tools import tool

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as e:
            retry = False
            retry_after = None
            if isinstance(e, RetryableError):
                retry = True
                retry_after = e.retry_after
            elif isinstance(e, httpx.HTTPStatusError) and e.response.status_code in (429, 500, 502, 503, 504):
                retry = True
                retry_after_header = e.response.headers.get("Retry-After")
                if retry_after_header and retry_after_header.isdigit():
                    retry_after = float(retry_after_header)
            elif isinstance(e, httpx.TransportError):
                retry = True
                
            if not retry or attempt == attempts - 1:
                raise e
                
            if retry_after is not None:
                delay = retry_after
            else:
                delay = base * (2 ** attempt) + random.uniform(0, 1)
            
            delay = min(delay, cap)
            time.sleep(delay)

last_arxiv_call_time = 0.0

@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Returns a JSON list of {id, url, published, title, summary}."""
    global last_arxiv_call_time
    
    try:
        terms = re.findall(r'\w+', query)
        if not terms:
            return "NO RESULTS"
            
        search_query = " AND ".join(f"all:{t}" for t in terms)
        max_results = max(1, min(max_results, 30))
        
        def _do_call():
            global last_arxiv_call_time
            now = time.time()
            if now - last_arxiv_call_time < 3.0:
                time.sleep(3.0 - (now - last_arxiv_call_time))
            
            last_arxiv_call_time = time.time()
            params = {
                "search_query": search_query,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
                "max_results": max_results
            }
            resp = httpx.get(ARXIV_URL, params=params, timeout=15.0)
            resp.raise_for_status()
            return resp.text
            
        xml_data = with_retry(_do_call, attempts=5, cap=60.0)
        
        root = ET.fromstring(xml_data)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        entries = root.findall('atom:entry', ns)
        
        if not entries:
            return "NO RESULTS"
            
        records = []
        for entry in entries:
            id_el = entry.find('atom:id', ns).text
            match = re.search(r'/abs/(\d+\.\d+)', id_el)
            if match:
                id_val = match.group(1)
            else:
                id_val = id_el.split('/')[-1].split('v')[0]
                
            published = entry.find('atom:published', ns).text[:10]
            title = " ".join(entry.find('atom:title', ns).text.split())
            summary = " ".join(entry.find('atom:summary', ns).text.split())
            if len(summary) > 600:
                summary = summary[:597] + "..."
                
            records.append({
                "id": id_val,
                "url": f"https://arxiv.org/abs/{id_val}",
                "published": published,
                "title": title,
                "summary": summary
            })
            
        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {str(e)}"

@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    try:
        limit = max(1, min(limit, 100))
        params = {"limit": limit}
        if date:
            params["date"] = date
            
        def _do_call():
            resp = httpx.get(HF_DAILY_URL, params=params, timeout=15.0)
            resp.raise_for_status()
            return resp.json()
            
        data = with_retry(_do_call)
        records = []
        keyword_lower = keyword.lower() if keyword else ""
        for item in data:
            paper = item.get("paper")
            if not paper or not paper.get("id"):
                continue
                
            title = paper.get("title", "")
            summary = paper.get("summary", "")
            if keyword_lower and keyword_lower not in title.lower() and keyword_lower not in summary.lower():
                continue
                
            pub = paper.get("publishedAt", "")
            if pub: pub = pub[:10]
            
            records.append({
                "id": paper.get("id"),
                "url": f"https://huggingface.co/papers/{paper.get('id')}",
                "published": pub,
                "title": title,
                "summary": summary,
                "upvotes": paper.get("upvotes", 0),
                "github": paper.get("githubRepo", ""),
                "stars": paper.get("githubStars", 0)
            })
            
        records.sort(key=lambda x: x["upvotes"], reverse=True)
        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {str(e)}"

@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}."""
    try:
        limit = max(1, min(limit, 50))
        params = {"q": query, "limit": limit}
        
        def _do_call():
            resp = httpx.get(HF_SEARCH_URL, params=params, timeout=15.0)
            resp.raise_for_status()
            return resp.json()
            
        data = with_retry(_do_call)
        records = []
        for paper in data:
            if not paper or not paper.get("id"):
                continue
                
            title = paper.get("title", "")
            summary = paper.get("ai_summary") or paper.get("summary", "")
            
            pub = paper.get("publishedAt", "")
            if pub: pub = pub[:10]
            
            records.append({
                "id": paper.get("id"),
                "url": f"https://huggingface.co/papers/{paper.get('id')}",
                "published": pub,
                "title": title,
                "summary": summary,
                "upvotes": paper.get("upvotes", 0),
                "github": paper.get("githubRepo", ""),
                "stars": paper.get("githubStars", 0)
            })
            
        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {str(e)}"

def _exa_call(method, arguments):
    exa_key = os.environ.get("EXA_API_KEY", "")
    url = EXA_URL
    if exa_key:
        url += f"?exaApiKey={exa_key}"
        
    def _do_call():
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
        }
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": method,
                "arguments": arguments
            }
        }
        
        resp = httpx.post(url, headers=headers, json=payload, timeout=20.0)
        resp.raise_for_status()
        
        text = ""
        for line in resp.text.splitlines():
            if line.startswith("data:"):
                try:
                    data = json.loads(line[5:].strip())
                    if "error" in data:
                        raise Exception(data["error"])
                    if "result" in data:
                        meta = data["result"].get("_meta", {})
                        if meta.get("rateLimit", False) or "rate limit" in str(data["result"]).lower() or "too many requests" in str(data["result"]).lower():
                            raise RetryableError("Exa rate limit hit", retry_after=10.0)
                        
                        contents = data["result"].get("content", [])
                        for c in contents:
                            if c.get("type") == "text":
                                text += c.get("text", "")
                except json.JSONDecodeError:
                    pass
        if not text and 'rateLimit' in resp.text: # Just in case it's missed
            raise RetryableError("Exa rate limit hit", retry_after=10.0)
            
        return text if text else "NO RESULTS"
    
    try:
        res = with_retry(_do_call, attempts=10, cap=60.0)
        return res
    except Exception as e:
        err_msg = str(e)
        if exa_key:
            err_msg = err_msg.replace(exa_key, "***")
        return f"ERROR: {type(e).__name__}: {err_msg}"

@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs."""
    if not objective:
        objective = query
    return _exa_call("web_search_exa", {"query": query, "objective": objective, "numResults": num_results})

@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    res = _exa_call("web_fetch_exa", {"urls": [url]})
    if not res.startswith("ERROR:") and len(res) > 12000:
        res = res[:12000] + "..."
    return res

SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]

if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        try:
            print(f"== {name}\n{fn.invoke(args)[:400]}\n")
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n")
