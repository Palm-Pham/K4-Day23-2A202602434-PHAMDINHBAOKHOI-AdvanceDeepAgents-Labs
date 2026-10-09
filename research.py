"""research.py - STUDENT IMPLEMENTS.  The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json  # noqa: F401
import os  # noqa: F401
import re  # noqa: F401
import sys
import time  # noqa: F401
from collections import Counter  # noqa: F401
from pathlib import Path

from agents import FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent  # noqa: F401
from model import make_model  # noqa: F401
from sandbox import download, open_sandbox, upload  # noqa: F401

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to your validator


def slugify(topic):
    import re
    if not topic:
        return "topic"
    slug = re.sub(r'[^\w]+', '-', topic).strip('-').lower()
    if not slug:
        return "topic"
    return slug[:60]

def build_prompt(topic):
    return f"Write a comprehensive survey report on the topic: {topic}"

def summarize(messages, elapsed, model_name):
    from collections import Counter
    subagent_calls = 0
    tool_calls_counter = Counter()
    tokens = {"input": 0, "output": 0}
    
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                name = tc.get("name")
                if name == "task":
                    subagent_calls += 1
                tool_calls_counter[name] += 1
                
        if hasattr(msg, "usage_metadata") and msg.usage_metadata:
            tokens["input"] += msg.usage_metadata.get("input_tokens", 0)
            tokens["output"] += msg.usage_metadata.get("output_tokens", 0)
            
    return {
        "model": model_name,
        "elapsed_s": round(elapsed, 1),
        "subagent_calls": subagent_calls,
        "tool_calls": dict(tool_calls_counter),
        "tokens": tokens
    }

def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    import json
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    
    if REPORT_PATH not in files or not files[REPORT_PATH]:
        raise RuntimeError("Report is missing or empty")
    if SOURCES_PATH not in files or not files[SOURCES_PATH]:
        raise RuntimeError("Sources is missing or empty")
        
    try:
        sources_data = json.loads(files[SOURCES_PATH].decode("utf-8"))
    except Exception:
        raise RuntimeError("Sources is invalid JSON")
        
    if not isinstance(sources_data, list):
        raise RuntimeError("Sources is not a JSON array")
        
    reports_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify(topic)
    
    n_sources = len(sources_data)
    families = sorted(list(set(s.get("source") for s in sources_data if isinstance(s, dict) and "source" in s)))
    
    meta = summarize(messages, elapsed, model_name)
    meta["topic"] = topic
    meta["n_sources"] = n_sources
    meta["source_families"] = families
    
    (reports_dir / f"{slug}.sources.json").write_bytes(files[SOURCES_PATH])
    (reports_dir / f"{slug}.meta.json").write_text(json.dumps(meta, indent=2))
    (reports_dir / f"{slug}.md").write_bytes(files[REPORT_PATH])
    
    return str(reports_dir / f"{slug}.md")

def main(topic):
    import time
    if not topic.strip():
        print("Usage: python research.py <topic>", file=sys.stderr)
        return 2
        
    model = make_model()
    model_name = getattr(model, "model_name", str(model.__class__.__name__))
    
    start = time.monotonic()
    with open_sandbox() as backend:
        backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
        upload(backend, {
            VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
            FINALIZER_PATH: FINALIZER_SOURCE.read_bytes()
        })
        
        agent = build_lead_agent(backend, model)
        result = agent.invoke(
            {"messages": [{"role": "user", "content": build_prompt(topic)}]},
            config={"recursion_limit": 1000}
        )
        
        elapsed = time.monotonic() - start
        try:
            report_file = save_outputs(backend, topic, result["messages"], elapsed, model_name)
        except RuntimeError as e:
            print(f"FAILED: {e}", file=sys.stderr)
            return 1
            
    print(f"Report saved to {report_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
