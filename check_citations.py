"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK)."""
    import re
    problems = []
    if not sources:
        return ["no sources in sources.json"]
        
    seen_urls = set()
    source_ns = set()
    for s in sources:
        n = s.get("n")
        if not isinstance(n, int):
            problems.append(f"source n is not an int: {n}")
        else:
            source_ns.add(n)
            
        url = s.get("url", "")
        if not url.startswith("http://") and not url.startswith("https://"):
            problems.append(f"url does not start with http(s)://: {url}")
        if url in seen_urls:
            problems.append(f"duplicated url: {url}")
        seen_urls.add(url)
        
    if "## References" not in report_text:
        problems.append("missing heading '## References'")
        return problems
        
    parts = report_text.split("## References")
    body = parts[0]
    references_text = parts[1] if len(parts) > 1 else ""
    
    cited = set(int(x) for x in re.findall(r'\[(\d+)\]', body))
    
    for c in cited:
        if c not in source_ns:
            problems.append(f"[{c}] cited but missing from sources.json")
            
    for n in source_ns:
        if n not in cited:
            problems.append(f"source [{n}] never cited")
            
    ref_lines = [line.strip() for line in references_text.strip().split('\n') if line.strip()]
    
    ref_dict = {}
    for line in ref_lines:
        match = re.match(r'^\[(\d+)\](.*)', line)
        if match:
            n = int(match.group(1))
            rest = match.group(2)
            if n in ref_dict:
                problems.append(f"reference number [{n}] appears twice in reference list")
            ref_dict[n] = rest
            if n not in source_ns:
                problems.append(f"reference list has [{n}] which is not a source")
                
    for n in source_ns:
        if n not in ref_dict:
            problems.append(f"missing reference line for source [{n}]")
            continue
            
        rest = ref_dict[n]
        urls_in_line = re.findall(r'https?://[^\s()\]>]+', rest)
        if len(urls_in_line) != 1:
            problems.append(f"reference line [{n}] does not have exactly one url")
        else:
            url_in_line = urls_in_line[0]
            source_url = next(s["url"] for s in sources if s["n"] == n)
            if url_in_line != source_url:
                problems.append(f"reference line [{n}] url {url_in_line} does not match source url {source_url}")
                
    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
