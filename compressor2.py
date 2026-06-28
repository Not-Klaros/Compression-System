import argparse
import heapq
import re
import sqlite3
import struct
from collections import Counter
from pathlib import Path

ESCAPE_TOKEN = "<ESCAPE>"
MODEL_DB = "compression_model.db"

MAGIC = b"CTC1"  # Carlos Text Compressor v1
FORMAT_VERSION = 1
DEFAULT_MODEL_VERSION = 1

# File header layout:
# magic:          4 bytes  b"CTC1"
# format_version: 1 byte   compressed file format version
# model_version:  4 bytes  unsigned int, big endian
# payload_size:   8 bytes  unsigned long long, big endian
HEADER_STRUCT = struct.Struct(">4sBIQ")
HEADER_SIZE = HEADER_STRUCT.size


# ============================================================
# TOKENIZER
# ============================================================

def tokenize(text):
    """
    Splits text into word chunks and non-word chunks.

    Example:
        "hello, world" -> ["hello", ", ", "world"]
    """
    return re.findall(r"\w+|[^\w]+", text)


# ============================================================
# HUFFMAN CODING
# ============================================================

def build_huffman_codes(frequencies):
    if not frequencies:
        raise ValueError("Cannot build Huffman codes from an empty frequency table")

    heap = []
    counter = 0

    for token_id, freq in frequencies.items():
        if freq <= 0:
            continue
        heapq.heappush(heap, (freq, counter, token_id))
        counter += 1

    if not heap:
        raise ValueError("Frequency table has no positive frequencies")

    if len(heap) == 1:
        return {heap[0][2]: "0"}

    while len(heap) > 1:
        f1, _, left = heapq.heappop(heap)
        f2, _, right = heapq.heappop(heap)

        parent = (left, right)
        heapq.heappush(heap, (f1 + f2, counter, parent))
        counter += 1

    tree = heap[0][2]
    codes = {}

    def walk(node, code):
        if isinstance(node, int):
            codes[node] = code or "0"
            return

        left, right = node
        walk(left, code + "0")
        walk(right, code + "1")

    walk(tree, "")
    return codes


def rebuild_codes(model):
    id_to_code = build_huffman_codes(model["frequencies"])
    model["id_to_code"] = id_to_code
    model["code_to_id"] = {code: token_id for token_id, code in id_to_code.items()}


# ============================================================
# MODEL BUILDING
# ============================================================

def create_model(max_tokens=20000, model_version=DEFAULT_MODEL_VERSION):
    model = {
        "token_to_id": {ESCAPE_TOKEN: 0},
        "id_to_token": {0: ESCAPE_TOKEN},
        "frequencies": {0: 1},
        "max_tokens": max_tokens,
        "model_version": model_version,
    }
    rebuild_codes(model)
    return model


def build_model_from_text(
    text,
    base_tokens=3000,
    total_model_capacity=20000,
    model_version=DEFAULT_MODEL_VERSION,
):
    counts = Counter(tokenize(text))
    model = create_model(
        max_tokens=total_model_capacity,
        model_version=model_version,
    )

    next_id = 1

    for token, count in counts.most_common(base_tokens):
        if token == ESCAPE_TOKEN:
            continue

        model["token_to_id"][token] = next_id
        model["id_to_token"][next_id] = token
        model["frequencies"][next_id] = count
        next_id += 1

        if len(model["id_to_token"]) >= total_model_capacity:
            break

    rebuild_codes(model)
    return model


def build_model_from_file(
    file_path,
    base_tokens=3000,
    total_model_capacity=20000,
    model_version=DEFAULT_MODEL_VERSION,
    encoding="utf-8",
):
    text = Path(file_path).read_text(encoding=encoding)
    return build_model_from_text(
        text,
        base_tokens=base_tokens,
        total_model_capacity=total_model_capacity,
        model_version=model_version,
    )


def update_model_from_text(
    model,
    text,
    input_weight=1,
    max_new_tokens=1000,
    min_occurrences=2,
    min_length=2,
    bump_version=True,
):
    """
    Optional global-model update.

    This is NOT used during normal compression because changing the model has a cost.
    Use it only when you intentionally want to improve the shared global model.
    """
    counts = Counter(tokenize(text))
    next_id = max(model["id_to_token"].keys()) + 1
    added = 0

    for token, count in counts.most_common():
        weighted_count = count * input_weight

        if token in model["token_to_id"]:
            token_id = model["token_to_id"][token]
            model["frequencies"][token_id] += weighted_count
            continue

        if added >= max_new_tokens:
            break

        if len(model["id_to_token"]) >= model["max_tokens"]:
            break

        if count < min_occurrences or len(token) < min_length:
            continue

        model["token_to_id"][token] = next_id
        model["id_to_token"][next_id] = token
        model["frequencies"][next_id] = weighted_count
        next_id += 1
        added += 1

    if bump_version:
        model["model_version"] += 1

    rebuild_codes(model)
    return added


# ============================================================
# SQLITE SAVE / LOAD
# ============================================================

def save_model_sqlite(model, db_file=MODEL_DB):
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS metadata")
    cur.execute("DROP TABLE IF EXISTS tokens")

    cur.execute("""
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE tokens (
            id INTEGER PRIMARY KEY,
            token TEXT NOT NULL UNIQUE,
            frequency INTEGER NOT NULL
        )
    """)

    metadata_rows = [
        ("max_tokens", str(model["max_tokens"])),
        ("model_version", str(model["model_version"])),
        ("format_version", str(FORMAT_VERSION)),
        ("magic", MAGIC.decode("ascii")),
    ]

    cur.executemany(
        "INSERT INTO metadata (key, value) VALUES (?, ?)",
        metadata_rows,
    )

    rows = []
    for token_id, token in sorted(model["id_to_token"].items()):
        frequency = model["frequencies"].get(token_id, 1)
        rows.append((token_id, token, frequency))

    cur.executemany(
        "INSERT INTO tokens (id, token, frequency) VALUES (?, ?, ?)",
        rows,
    )

    cur.execute("CREATE INDEX idx_token ON tokens(token)")

    conn.commit()
    conn.close()


def load_model_sqlite(db_file=MODEL_DB):
    if not Path(db_file).exists():
        raise FileNotFoundError(
            f"Model database not found: {db_file}. Build one first with: python compressor.py train ..."
        )

    conn = sqlite3.connect(db_file)
    cur = conn.cursor()

    cur.execute("SELECT key, value FROM metadata")
    metadata = dict(cur.fetchall())

    max_tokens = int(metadata.get("max_tokens", "20000"))
    model_version = int(metadata.get("model_version", str(DEFAULT_MODEL_VERSION)))

    cur.execute("SELECT id, token, frequency FROM tokens ORDER BY id")
    rows = cur.fetchall()
    conn.close()

    if not rows:
        raise ValueError("Model database contains no tokens")

    token_to_id = {}
    id_to_token = {}
    frequencies = {}

    for token_id, token, frequency in rows:
        token_to_id[token] = token_id
        id_to_token[token_id] = token
        frequencies[token_id] = frequency

    if ESCAPE_TOKEN not in token_to_id:
        raise ValueError("Model is invalid: missing escape token")

    model = {
        "token_to_id": token_to_id,
        "id_to_token": id_to_token,
        "frequencies": frequencies,
        "max_tokens": max_tokens,
        "model_version": model_version,
    }

    rebuild_codes(model)
    return model


def model_exists(db_file=MODEL_DB):
    return Path(db_file).exists()


# ============================================================
# BIT / BYTE HELPERS
# ============================================================

def bits_to_bytes(bits):
    padding = (8 - len(bits) % 8) % 8
    bits += "0" * padding

    output = bytearray()
    output.append(padding)

    for i in range(0, len(bits), 8):
        output.append(int(bits[i:i + 8], 2))

    return bytes(output)


def bytes_to_bits(data):
    if not data:
        return ""

    padding = data[0]
    if padding > 7:
        raise ValueError("Invalid bit padding value")

    body = data[1:]
    bits = "".join(f"{byte:08b}" for byte in body)

    if padding:
        bits = bits[:-padding]

    return bits


def raw_to_bits(token):
    raw = token.encode("utf-8")
    length = len(raw)

    if length > 65535:
        raise ValueError("Raw token too large")

    return f"{length:016b}" + "".join(f"{b:08b}" for b in raw)


def bits_to_raw(bits, index):
    if index + 16 > len(bits):
        raise ValueError("Invalid raw token length")

    length = int(bits[index:index + 16], 2)
    index += 16

    raw = bytearray()

    for _ in range(length):
        if index + 8 > len(bits):
            raise ValueError("Invalid raw token bytes")

        raw.append(int(bits[index:index + 8], 2))
        index += 8

    return raw.decode("utf-8"), index


# ============================================================
# RAW PAYLOAD COMPRESS / DECOMPRESS
# ============================================================

def compress_payload(text, model):
    bits = []

    escape_id = model["token_to_id"][ESCAPE_TOKEN]
    escape_code = model["id_to_code"][escape_id]

    for token in tokenize(text):
        if token in model["token_to_id"]:
            token_id = model["token_to_id"][token]
            bits.append(model["id_to_code"][token_id])
        else:
            bits.append(escape_code)
            bits.append(raw_to_bits(token))

    return bits_to_bytes("".join(bits))


def decompress_payload(data, model):
    bits = bytes_to_bits(data)

    output = []
    current = ""
    index = 0

    while index < len(bits):
        current += bits[index]
        index += 1

        if current in model["code_to_id"]:
            token_id = model["code_to_id"][current]
            token = model["id_to_token"][token_id]

            if token == ESCAPE_TOKEN:
                raw, index = bits_to_raw(bits, index)
                output.append(raw)
            else:
                output.append(token)

            current = ""

    if current:
        raise ValueError("Compressed data ended with incomplete Huffman code")

    return "".join(output)


# Backward-compatible function names.
def compress(text, model):
    return compress_payload(text, model)


def decompress(data, model):
    return decompress_payload(data, model)


# ============================================================
# COMPRESSED FILE FORMAT
# ============================================================

def make_file_header(model_version, payload_size):
    return HEADER_STRUCT.pack(MAGIC, FORMAT_VERSION, model_version, payload_size)


def parse_file_header(data):
    if len(data) < HEADER_SIZE:
        raise ValueError("Compressed file is too small to contain a valid header")

    magic, format_version, model_version, payload_size = HEADER_STRUCT.unpack(
        data[:HEADER_SIZE]
    )

    if magic != MAGIC:
        raise ValueError("Invalid compressed file magic header")

    if format_version != FORMAT_VERSION:
        raise ValueError(
            f"Unsupported compressed file version: {format_version}. Expected: {FORMAT_VERSION}"
        )

    expected_size = HEADER_SIZE + payload_size
    if len(data) != expected_size:
        raise ValueError(
            f"Compressed file size mismatch. Header says payload is {payload_size} bytes, "
            f"but file contains {len(data) - HEADER_SIZE} bytes."
        )

    return {
        "magic": magic,
        "format_version": format_version,
        "model_version": model_version,
        "payload_size": payload_size,
        "payload": data[HEADER_SIZE:],
    }


def save_compressed_file(payload, filename, model_version):
    header = make_file_header(model_version, len(payload))
    Path(filename).write_bytes(header + payload)


def load_compressed_file(filename):
    data = Path(filename).read_bytes()
    return parse_file_header(data)


def compress_file(input_file, output_file, model_db=MODEL_DB, encoding="utf-8"):
    model = load_model_sqlite(model_db)
    text = Path(input_file).read_text(encoding=encoding)
    payload = compress_payload(text, model)
    save_compressed_file(payload, output_file, model["model_version"])
    return compression_stats(text, payload, include_header=True)


def decompress_file(input_file, output_file, model_db=MODEL_DB, encoding="utf-8"):
    model = load_model_sqlite(model_db)
    compressed = load_compressed_file(input_file)

    file_model_version = compressed["model_version"]
    current_model_version = model["model_version"]

    if file_model_version != current_model_version:
        raise ValueError(
            f"Model version mismatch. File needs model version {file_model_version}, "
            f"but loaded model is version {current_model_version}."
        )

    text = decompress_payload(compressed["payload"], model)
    Path(output_file).write_text(text, encoding=encoding)
    return text


# ============================================================
# STATS
# ============================================================

def compression_stats(text, payload, include_header=False):
    original_size = len(text.encode("utf-8"))
    compressed_size = len(payload)

    if include_header:
        compressed_size += HEADER_SIZE

    if original_size == 0:
        ratio = 0
        saved = 0
    else:
        ratio = compressed_size / original_size
        saved = 1 - ratio

    return {
        "original_size": original_size,
        "compressed_size": compressed_size,
        "ratio": ratio,
        "saved": saved,
        "header_size": HEADER_SIZE if include_header else 0,
    }


def print_stats(title, stats):
    print(f"\n{title}")
    print("Original size:", stats["original_size"], "bytes")
    print("Compressed size:", stats["compressed_size"], "bytes")
    print("Header size:", stats["header_size"], "bytes")
    print("Compression ratio:", round(stats["ratio"], 4))
    print("Space saved:", round(stats["saved"] * 100, 2), "%")


# ============================================================
# HIGH-LEVEL OPERATIONS
# ============================================================

def train_global_model(
    training_file,
    model_db=MODEL_DB,
    base_tokens=3000,
    total_model_capacity=20000,
    model_version=DEFAULT_MODEL_VERSION,
    encoding="utf-8",
):
    model = build_model_from_file(
        training_file,
        base_tokens=base_tokens,
        total_model_capacity=total_model_capacity,
        model_version=model_version,
        encoding=encoding,
    )
    save_model_sqlite(model, model_db)
    return model


def update_global_model(
    training_file,
    model_db=MODEL_DB,
    input_weight=1,
    max_new_tokens=1000,
    min_occurrences=2,
    min_length=2,
    encoding="utf-8",
):
    model = load_model_sqlite(model_db)
    text = Path(training_file).read_text(encoding=encoding)
    added = update_model_from_text(
        model,
        text,
        input_weight=input_weight,
        max_new_tokens=max_new_tokens,
        min_occurrences=min_occurrences,
        min_length=min_length,
        bump_version=True,
    )
    save_model_sqlite(model, model_db)
    return model, added


def model_info(model_db=MODEL_DB):
    model = load_model_sqlite(model_db)
    return {
        "model_db": model_db,
        "model_version": model["model_version"],
        "max_tokens": model["max_tokens"],
        "tokens": len(model["token_to_id"]),
        "format_version": FORMAT_VERSION,
        "magic": MAGIC.decode("ascii"),
        "header_size": HEADER_SIZE,
    }


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description="Versioned global-model text compressor"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Build one reusable global model")
    train.add_argument("training_file")
    train.add_argument("--model-db", default=MODEL_DB)
    train.add_argument("--base-tokens", type=int, default=3000)
    train.add_argument("--capacity", type=int, default=20000)
    train.add_argument("--model-version", type=int, default=DEFAULT_MODEL_VERSION)
    train.add_argument("--encoding", default="utf-8")

    update = sub.add_parser("update-model", help="Occasionally update the shared model and bump its version")
    update.add_argument("training_file")
    update.add_argument("--model-db", default=MODEL_DB)
    update.add_argument("--input-weight", type=int, default=1)
    update.add_argument("--max-new-tokens", type=int, default=1000)
    update.add_argument("--min-occurrences", type=int, default=2)
    update.add_argument("--min-length", type=int, default=2)
    update.add_argument("--encoding", default="utf-8")

    comp = sub.add_parser("compress", help="Compress a text file using the saved global model")
    comp.add_argument("input_file")
    comp.add_argument("output_file")
    comp.add_argument("--model-db", default=MODEL_DB)
    comp.add_argument("--encoding", default="utf-8")

    decomp = sub.add_parser("decompress", help="Decompress a file using the matching global model version")
    decomp.add_argument("input_file")
    decomp.add_argument("output_file")
    decomp.add_argument("--model-db", default=MODEL_DB)
    decomp.add_argument("--encoding", default="utf-8")

    info = sub.add_parser("info", help="Show saved model information")
    info.add_argument("--model-db", default=MODEL_DB)

    inspect = sub.add_parser("inspect", help="Show compressed file header information")
    inspect.add_argument("compressed_file")

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        model = train_global_model(
            args.training_file,
            model_db=args.model_db,
            base_tokens=args.base_tokens,
            total_model_capacity=args.capacity,
            model_version=args.model_version,
            encoding=args.encoding,
        )
        print("Global model saved:", args.model_db)
        print("Model version:", model["model_version"])
        print("Model tokens:", len(model["token_to_id"]))
        print("Max capacity:", model["max_tokens"])

    elif args.command == "update-model":
        model, added = update_global_model(
            args.training_file,
            model_db=args.model_db,
            input_weight=args.input_weight,
            max_new_tokens=args.max_new_tokens,
            min_occurrences=args.min_occurrences,
            min_length=args.min_length,
            encoding=args.encoding,
        )
        print("Global model updated:", args.model_db)
        print("New model version:", model["model_version"])
        print("Added tokens:", added)
        print("Total tokens:", len(model["token_to_id"]))

    elif args.command == "compress":
        stats = compress_file(
            args.input_file,
            args.output_file,
            model_db=args.model_db,
            encoding=args.encoding,
        )
        print("Compressed file saved:", args.output_file)
        print_stats("Compression stats:", stats)

    elif args.command == "decompress":
        decompress_file(
            args.input_file,
            args.output_file,
            model_db=args.model_db,
            encoding=args.encoding,
        )
        print("Decompressed file saved:", args.output_file)

    elif args.command == "info":
        info = model_info(args.model_db)
        for key, value in info.items():
            print(f"{key}: {value}")

    elif args.command == "inspect":
        compressed = load_compressed_file(args.compressed_file)
        print("magic:", compressed["magic"].decode("ascii"))
        print("format_version:", compressed["format_version"])
        print("model_version:", compressed["model_version"])
        print("payload_size:", compressed["payload_size"])
        print("header_size:", HEADER_SIZE)


if __name__ == "__main__":
    main()
