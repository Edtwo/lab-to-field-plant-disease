"""Generate BibTeX entries for arXiv IDs directly from the arXiv API (no hand-typed metadata)."""
import sys, re, urllib.request, xml.etree.ElementTree as ET
NS = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
ids = sys.argv[1:]
url = "https://export.arxiv.org/api/query?max_results=100&id_list=" + ",".join(ids)
root = ET.fromstring(urllib.request.urlopen(url, timeout=60).read())
for e in root.findall("a:entry", NS):
    aid = re.sub(r"v\d+$", "", e.find("a:id", NS).text.rsplit("/", 1)[1])
    title = " ".join(e.find("a:title", NS).text.split())
    authors = [a.find("a:name", NS).text for a in e.findall("a:author", NS)]
    year = e.find("a:published", NS).text[:4]
    key = re.sub(r"[^A-Za-z]", "", authors[0].split()[-1]).lower() + year + re.sub(r"[^a-z]", "", title.split()[0].lower())
    print(f"% url: https://arxiv.org/abs/{aid}  (verified via arXiv API)")
    print(f"@article{{{key},\n  title={{{{{title}}}}},\n  author={{{' and '.join(authors)}}},\n  journal={{arXiv preprint arXiv:{aid}}},\n  year={{{year}}}\n}}\n")
