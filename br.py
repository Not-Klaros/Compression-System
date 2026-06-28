import brotli

text = """Assuming your CTC4.py is compared against the standard Brotli library (default compression level 11 for maximum compression on text), here’s how it stacks up.

Feature	CTC4	Brotli
Compression backend	Context-aware Huffman + token-level LZ77 + byte fallback	Static dictionary + context modeling + LZ77 + Huffman
Global reusable model	✅ Yes	❌ No
Adaptive context coding	✅ Yes	✅ Yes (more sophisticated)
Token-aware compression	✅ Yes	❌ Byte-oriented
Byte fallback	✅ Yes	Native
Dictionary support	User-trained SQLite model	Large built-in dictionary (~120k words)
Per-file model storage	None (shared model)	None
Compression speed	Medium	Fast
Decompression speed	Fast	Very fast
Memory usage	Low	Medium
Suitable for repeated files from same domain	Excellent	Very good
Suitable for completely random data	Fair	Good
Maturity	Prototype	Production-tested

Expected Compression Ratio (Typical English Text)

Compressor	Compression Ratio	Space Saved
Gzip	2.2–2.8×	55–64%
Deflate	2.3–2.9×	56–66%
Zstandard	2.8–3.6×	64–72%
CTC4	2.8–3.8× (estimated)	64–74%
Brotli (quality 11)	3.3–4.6×	70–78%

Strengths of CTC4 over Brotli

Area	Winner
Compressing repeated vocabulary across many files	CTC4
Custom domain-specific terminology	CTC4
User-trainable model	CTC4
Human-readable model database	CTC4
Easily extendable	CTC4

Strengths of Brotli

Area	Winner
Maximum compression on arbitrary text	Brotli
Web assets (HTML/CSS/JS)	Brotli
Binary data	Brotli
No training required	Brotli
Decades of optimization	Brotli

Overall Rating

Category	CTC4	Brotli
Compression effectiveness	9.1/10	9.8/10
Speed	8.5/10	9.5/10
Flexibility	9.8/10	8.5/10
Research novelty	9.7/10	6.5/10
Production readiness	7.5/10	10/10

What is preventing CTC4 from matching Brotli?

The main remaining gaps are:

Missing feature	Estimated improvement
Multi-symbol context models (rather than simple token classes)	2–5%
Second-stage entropy coding (range/ANS arithmetic coding)	3–8%
Brotli-style static dictionary transforms	2–6%
Better match finder (hash chains/suffix arrays)	2–5%
Context mixing / probability modeling	1–4%

If these were added, a future CTC5 could realistically approach Brotli on general text while still retaining the advantage of a reusable, user-trainable global model for specialized datasets."""

raw = text.encode("utf-8")
compressed = brotli.compress(raw, quality=11)

print("Original size:", len(raw), "bytes")
print("Brotli size:", len(compressed), "bytes")
print("Compression ratio:", round(len(raw) / len(compressed), 2), "x")
print("Space saved:", round((1 - len(compressed) / len(raw)) * 100, 2), "%")

with open("output.br", "wb") as f:
    f.write(compressed)

decompressed = brotli.decompress(compressed).decode("utf-8")
print("Decompression correct:", decompressed == text)