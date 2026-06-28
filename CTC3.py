import argparse
import heapq
import re
import sqlite3
import struct
from collections import Counter, defaultdict
from pathlib import Path

ESCAPE_TOKEN = "<ESCAPE>"
MODEL_DB = "compression_model.db"

MAGIC = b"CTC3"  # Carlos Text Compressor v3, compact raw tokens + variable LZ77
FORMAT_VERSION = 3
DEFAULT_MODEL_VERSION = 1

# File header layout:
# magic:          4 bytes  b"CTC3"
# format_version: 1 byte   compressed file format version
# model_version:  4 bytes  unsigned int, big endian
# payload_size:   8 bytes  unsigned long long, big endian
HEADER_STRUCT = struct.Struct(">4sBIQ")
HEADER_SIZE = HEADER_STRUCT.size

# LZ77 settings.
# This is token-level LZ77, not byte-level LZ77.
# Payload symbol grammar:
#   0  + Huffman known-token literal
#   10 + gamma(byte_length) + raw UTF-8 bytes
#   11 + gamma(distance) + gamma(length)
#
# Elias gamma coding is used for positive integers. It is very cheap for
# common small values and grows naturally for larger matches/distances.
DEFAULT_WINDOW_SIZE = 4095
DEFAULT_MIN_MATCH = 4
DEFAULT_MAX_MATCH = 255
MAX_GAMMA_VALUE = (1 << 32) - 1


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
            f"Model database not found: {db_file}. Build one first with: python compressor2_lz77.py train ..."
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


def gamma_to_bits(value):
    """Elias gamma code for positive integers: 1 -> '1', 2 -> '010', etc."""
    if value <= 0:
        raise ValueError("Gamma coding only supports positive integers")

    if value > MAX_GAMMA_VALUE:
        raise ValueError(f"Value {value} is too large for this format")

    binary = bin(value)[2:]
    return "0" * (len(binary) - 1) + binary


def bits_to_gamma(bits, index):
    zeros = 0

    while index < len(bits) and bits[index] == "0":
        zeros += 1
        index += 1

    if index >= len(bits):
        raise ValueError("Compressed data ended inside a gamma-coded integer")

    width = zeros + 1
    if index + width > len(bits):
        raise ValueError("Compressed data ended inside a gamma-coded integer")

    value = int(bits[index:index + width], 2)
    index += width

    if value <= 0 or value > MAX_GAMMA_VALUE:
        raise ValueError("Invalid gamma-coded integer")

    return value, index


def raw_to_bits(token):
    raw = token.encode("utf-8")
    length = len(raw)

    if length == 0:
        raise ValueError("Raw token cannot be empty")

    return gamma_to_bits(length) + "".join(f"{b:08b}" for b in raw)


def bits_to_raw(bits, index):
    length, index = bits_to_gamma(bits, index)
    raw_end = index + length * 8

    if raw_end > len(bits):
        raise ValueError("Invalid raw token bytes")

    raw = bytearray()

    while index < raw_end:
        raw.append(int(bits[index:index + 8], 2))
        index += 8

    return raw.decode("utf-8"), index


# ============================================================
# TOKEN UNITS
# ============================================================

def text_to_units(text, model):
    """
    Converts text tokens to comparable units for token-level LZ77.

    Known model tokens become integer token IDs.
    Unknown tokens become ("RAW", token) units.
    """
    units = []

    for token in tokenize(text):
        token_id = model["token_to_id"].get(token)
        if token_id is None:
            units.append(("RAW", token))
        else:
            units.append(token_id)

    return units


def unit_to_known_literal_bits(unit, model):
    if not isinstance(unit, int):
        raise ValueError("Known literal encoder received a raw token")

    escape_id = model["token_to_id"][ESCAPE_TOKEN]
    if unit == escape_id:
        raise ValueError("Escape token is reserved and cannot be emitted as a normal literal")

    return model["id_to_code"][unit]


# ============================================================
# TOKEN-LEVEL LZ77
# ============================================================

def find_lz77_match(units, position, recent_positions, window_size, min_match, max_match):
    if position + min_match > len(units):
        return None

    key = tuple(units[position:position + min_match])
    candidates = recent_positions.get(key)

    if not candidates:
        return None

    best_distance = 0
    best_length = 0
    lower_bound = max(0, position - window_size)

    # Recent matches are usually more useful, so search backwards.
    for candidate in reversed(candidates):
        if candidate < lower_bound:
            continue

        distance = position - candidate
        if distance <= 0 or distance > window_size:
            continue

        length = 0
        while (
            length < max_match
            and position + length < len(units)
            and units[candidate + length] == units[position + length]
        ):
            length += 1

        if length > best_length:
            best_length = length
            best_distance = distance

            if best_length == max_match:
                break

    if best_length >= min_match:
        return best_distance, best_length

    return None


def add_lz77_index_entries(units, start, end, recent_positions, min_match):
    """
    Adds positions [start, end) to the phrase index.
    """
    max_start = len(units) - min_match

    for pos in range(start, end):
        if pos > max_start:
            break

        key = tuple(units[pos:pos + min_match])
        recent_positions[key].append(pos)


def encode_lz77_symbols(
    units,
    window_size=DEFAULT_WINDOW_SIZE,
    min_match=DEFAULT_MIN_MATCH,
    max_match=DEFAULT_MAX_MATCH,
):
    """
    Converts token units into LZ77 symbols.

    Output symbols are tuples:
        ("LIT", unit)
        ("MATCH", distance, length)
    """
    if min_match < 2:
        raise ValueError("min_match must be at least 2")

    if window_size <= 0 or window_size > MAX_GAMMA_VALUE:
        raise ValueError(f"window_size must be between 1 and {MAX_GAMMA_VALUE}")

    if max_match <= 0 or max_match > MAX_GAMMA_VALUE:
        raise ValueError(f"max_match must be between 1 and {MAX_GAMMA_VALUE}")

    symbols = []
    recent_positions = defaultdict(list)
    position = 0

    while position < len(units):
        match = find_lz77_match(
            units,
            position,
            recent_positions,
            window_size=window_size,
            min_match=min_match,
            max_match=max_match,
        )

        if match is None:
            symbols.append(("LIT", units[position]))
            add_lz77_index_entries(units, position, position + 1, recent_positions, min_match)
            position += 1
            continue

        distance, length = match
        symbols.append(("MATCH", distance, length))

        # Insert every consumed token position so later phrases can refer to them.
        add_lz77_index_entries(units, position, position + length, recent_positions, min_match)
        position += length

    return symbols


def decode_lz77_symbols(symbols, model):
    output = []

    for symbol in symbols:
        kind = symbol[0]

        if kind == "LIT":
            unit = symbol[1]
            if isinstance(unit, int):
                output.append(model["id_to_token"][unit])
            else:
                tag, raw = unit
                if tag != "RAW":
                    raise ValueError(f"Unknown literal unit type: {tag}")
                output.append(raw)

        elif kind == "MATCH":
            distance, length = symbol[1], symbol[2]

            if distance <= 0 or distance > len(output):
                raise ValueError("Invalid LZ77 match distance")

            if length <= 0:
                raise ValueError("Invalid LZ77 match length")

            # Supports overlapping copies, like classic LZ77.
            for _ in range(length):
                output.append(output[-distance])

        else:
            raise ValueError(f"Unknown LZ77 symbol: {kind}")

    return "".join(output)


# ============================================================
# PAYLOAD COMPRESS / DECOMPRESS
# ============================================================

def compress_payload_lz77(
    text,
    model,
    window_size=DEFAULT_WINDOW_SIZE,
    min_match=DEFAULT_MIN_MATCH,
    max_match=DEFAULT_MAX_MATCH,
):
    units = text_to_units(text, model)
    symbols = encode_lz77_symbols(
        units,
        window_size=window_size,
        min_match=min_match,
        max_match=max_match,
    )

    bits = []

    for symbol in symbols:
        kind = symbol[0]

        if kind == "LIT":
            unit = symbol[1]

            if isinstance(unit, int):
                bits.append("0")
                bits.append(unit_to_known_literal_bits(unit, model))
            else:
                tag, raw = unit
                if tag != "RAW":
                    raise ValueError(f"Unknown literal unit type: {tag}")
                bits.append("10")
                bits.append(raw_to_bits(raw))

        elif kind == "MATCH":
            _, distance, length = symbol
            bits.append("11")
            bits.append(gamma_to_bits(distance))
            bits.append(gamma_to_bits(length))

        else:
            raise ValueError(f"Unknown LZ77 symbol: {kind}")

    return bits_to_bytes("".join(bits))


def decompress_payload_lz77(data, model):
    bits = bytes_to_bits(data)
    symbols = []
    index = 0

    while index < len(bits):
        marker = bits[index]
        index += 1

        if marker == "0":
            current = ""

            while index < len(bits):
                current += bits[index]
                index += 1

                if current in model["code_to_id"]:
                    token_id = model["code_to_id"][current]
                    token = model["id_to_token"][token_id]

                    if token == ESCAPE_TOKEN:
                        raise ValueError("Invalid payload: escape token used as a known literal")

                    symbols.append(("LIT", token_id))
                    break
            else:
                raise ValueError("Compressed data ended with incomplete Huffman literal")

        elif marker == "1":
            if index >= len(bits):
                raise ValueError("Compressed data ended after extended marker")

            extended_marker = bits[index]
            index += 1

            if extended_marker == "0":
                raw, index = bits_to_raw(bits, index)
                symbols.append(("LIT", ("RAW", raw)))

            elif extended_marker == "1":
                distance, index = bits_to_gamma(bits, index)
                length, index = bits_to_gamma(bits, index)
                symbols.append(("MATCH", distance, length))

            else:
                raise ValueError(f"Invalid extended marker bit: {extended_marker}")

        else:
            raise ValueError(f"Invalid LZ77 marker bit: {marker}")

    return decode_lz77_symbols(symbols, model)


# Legacy plain-Huffman payload functions kept for comparison and testing.
def compress_payload_huffman_only(text, model):
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


def decompress_payload_huffman_only(data, model):
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


# Default compressor now uses token LZ77 + Huffman.
def compress_payload(text, model, **lz77_options):
    return compress_payload_lz77(text, model, **lz77_options)


def decompress_payload(data, model):
    return decompress_payload_lz77(data, model)


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
        raise ValueError(f"Invalid compressed file magic header: {magic!r}. Expected {MAGIC!r}")

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


def compress_file(
    input_file,
    output_file,
    model_db=MODEL_DB,
    encoding="utf-8",
    window_size=DEFAULT_WINDOW_SIZE,
    min_match=DEFAULT_MIN_MATCH,
    max_match=DEFAULT_MAX_MATCH,
):
    model = load_model_sqlite(model_db)
    text = Path(input_file).read_text(encoding=encoding)
    payload = compress_payload_lz77(
        text,
        model,
        window_size=window_size,
        min_match=min_match,
        max_match=max_match,
    )
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

    text = decompress_payload_lz77(compressed["payload"], model)
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


def compare_methods(text, model, include_header=False):
    huffman_payload = compress_payload_huffman_only(text, model)
    lz77_payload = compress_payload_lz77(text, model)

    return {
        "huffman_only": compression_stats(text, huffman_payload, include_header=include_header),
        "lz77_huffman": compression_stats(text, lz77_payload, include_header=include_header),
    }


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
        "match_integer_encoding": "elias_gamma",
        "raw_length_encoding": "elias_gamma",
        "lz77_default_window_size": DEFAULT_WINDOW_SIZE,
        "lz77_default_min_match": DEFAULT_MIN_MATCH,
        "lz77_default_max_match": DEFAULT_MAX_MATCH,
    }


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description="Versioned global-model text compressor with token-level LZ77 + Huffman"
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
    comp.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    comp.add_argument("--min-match", type=int, default=DEFAULT_MIN_MATCH)
    comp.add_argument("--max-match", type=int, default=DEFAULT_MAX_MATCH)

    decomp = sub.add_parser("decompress", help="Decompress a file using the matching global model version")
    decomp.add_argument("input_file")
    decomp.add_argument("output_file")
    decomp.add_argument("--model-db", default=MODEL_DB)
    decomp.add_argument("--encoding", default="utf-8")

    compare = sub.add_parser("compare", help="Compare Huffman-only vs LZ77+Huffman on one input file")
    compare.add_argument("input_file")
    compare.add_argument("--model-db", default=MODEL_DB)
    compare.add_argument("--encoding", default="utf-8")
    compare.add_argument("--include-header", action="store_true")

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
        print("Format:", MAGIC.decode("ascii"), "version", FORMAT_VERSION)

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
            window_size=args.window_size,
            min_match=args.min_match,
            max_match=args.max_match,
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

    elif args.command == "compare":
        model = load_model_sqlite(args.model_db)
        text = Path(args.input_file).read_text(encoding=args.encoding)
        results = compare_methods(text, model, include_header=args.include_header)
        print_stats("Huffman only:", results["huffman_only"])
        print_stats("Token LZ77 + Huffman:", results["lz77_huffman"])

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
