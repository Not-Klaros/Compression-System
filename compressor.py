import re
import heapq
import sqlite3
from pathlib import Path
from collections import Counter

ESCAPE_TOKEN = "<ESCAPE>"
MODEL_DB = "compression_model.db"


# ============================================================
# TOKENIZER
# ============================================================

def tokenize(text):
    return re.findall(r"\w+|[^\w]+", text)


# ============================================================
# HUFFMAN CODING
# ============================================================

def build_huffman_codes(frequencies):
    heap = []
    counter = 0

    for token_id, freq in frequencies.items():
        heapq.heappush(heap, (freq, counter, token_id))
        counter += 1

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
    model["code_to_id"] = {
        code: token_id for token_id, code in id_to_code.items()
    }


# ============================================================
# MODEL BUILDING
# ============================================================

def create_model(max_tokens=20000):
    model = {
        "token_to_id": {ESCAPE_TOKEN: 0},
        "id_to_token": {0: ESCAPE_TOKEN},
        "frequencies": {0: 1},
        "max_tokens": max_tokens,
    }

    rebuild_codes(model)
    return model


def build_base_model_from_file(
    file_path,
    base_tokens=3000,
    total_model_capacity=20000,
    encoding="utf-8"
):
    text = Path(file_path).read_text(encoding=encoding)
    counts = Counter(tokenize(text))

    model = create_model(max_tokens=total_model_capacity)

    next_id = 1

    for token, count in counts.most_common(base_tokens):
        if token == ESCAPE_TOKEN:
            continue

        model["token_to_id"][token] = next_id
        model["id_to_token"][next_id] = token
        model["frequencies"][next_id] = count

        next_id += 1

    rebuild_codes(model)
    return model


def retrain_model_on_input(
    model,
    text,
    input_weight=100000,
    max_new_tokens=10000,
    min_occurrences=1,
    min_length=1
):
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

        if count < min_occurrences:
            continue

        if len(token) < min_length:
            continue

        model["token_to_id"][token] = next_id
        model["id_to_token"][next_id] = token
        model["frequencies"][next_id] = weighted_count

        next_id += 1
        added += 1

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

    cur.execute(
        "INSERT INTO metadata (key, value) VALUES (?, ?)",
        ("max_tokens", str(model["max_tokens"]))
    )

    rows = []

    for token_id, token in model["id_to_token"].items():
        frequency = model["frequencies"].get(token_id, 1)
        rows.append((token_id, token, frequency))

    cur.executemany(
        "INSERT INTO tokens (id, token, frequency) VALUES (?, ?, ?)",
        rows
    )

    cur.execute("CREATE INDEX idx_token ON tokens(token)")

    conn.commit()
    conn.close()


def load_model_sqlite(db_file=MODEL_DB):
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()

    cur.execute("SELECT value FROM metadata WHERE key = ?", ("max_tokens",))
    row = cur.fetchone()

    if row is None:
        max_tokens = 20000
    else:
        max_tokens = int(row[0])

    cur.execute("SELECT id, token, frequency FROM tokens ORDER BY id")
    rows = cur.fetchall()

    conn.close()

    token_to_id = {}
    id_to_token = {}
    frequencies = {}

    for token_id, token, frequency in rows:
        token_to_id[token] = token_id
        id_to_token[token_id] = token
        frequencies[token_id] = frequency

    model = {
        "token_to_id": token_to_id,
        "id_to_token": id_to_token,
        "frequencies": frequencies,
        "max_tokens": max_tokens,
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
# COMPRESS / DECOMPRESS
# ============================================================

def compress(text, model):
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


def decompress(data, model):
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


# ============================================================
# COMPRESSED FILE SAVE / LOAD
# ============================================================

def save_compressed_file(data, filename):
    Path(filename).write_bytes(data)


def load_compressed_file(filename):
    return Path(filename).read_bytes()


# ============================================================
# STATS
# ============================================================

def compression_stats(text, compressed):
    original_size = len(text.encode("utf-8"))
    compressed_size = len(compressed)

    if original_size == 0:
        return {
            "original_size": 0,
            "compressed_size": compressed_size,
            "ratio": 0,
            "saved": 0,
        }

    ratio = compressed_size / original_size
    saved = 1 - ratio

    return {
        "original_size": original_size,
        "compressed_size": compressed_size,
        "ratio": ratio,
        "saved": saved,
    }


def print_stats(title, stats):
    print(f"\n{title}")
    print("Original size:", stats["original_size"])
    print("Compressed size:", stats["compressed_size"])
    print("Compression ratio:", round(stats["ratio"], 4))
    print("Space saved:", round(stats["saved"] * 100, 2), "%")


# ============================================================
# PIPELINE
# ============================================================

def create_train_compress_save(
    text,
    training_file,
    model_db=MODEL_DB,
    compressed_output="compressed.bin",
    base_tokens=3000,
    total_model_capacity=20000,
    input_weight=100000,
    max_new_tokens=10000
):
    model = build_base_model_from_file(
        training_file,
        base_tokens=base_tokens,
        total_model_capacity=total_model_capacity,
    )

    before = compress(text, model)
    before_stats = compression_stats(text, before)

    added = retrain_model_on_input(
        model,
        text,
        input_weight=input_weight,
        max_new_tokens=max_new_tokens,
        min_occurrences=1,
        min_length=1,
    )

    compressed = compress(text, model)
    after_stats = compression_stats(text, compressed)

    save_model_sqlite(model, model_db)
    save_compressed_file(compressed, compressed_output)

    return {
        "model": model,
        "compressed": compressed,
        "added_tokens": added,
        "before_stats": before_stats,
        "after_stats": after_stats,
    }


def decompress_from_saved_files(
    model_db=MODEL_DB,
    compressed_file="compressed.bin"
):
    model = load_model_sqlite(model_db)
    compressed = load_compressed_file(compressed_file)
    return decompress(compressed, model)


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    training_file = "texts/top_english_words_mixed_500000.txt"

    text = input("Enter text to compress: ")
    name_of_file = input("Enter name of file to save compressed data: ")
    result = create_train_compress_save(
        text=text,
        training_file=training_file,
        model_db=MODEL_DB,
        compressed_output=f"{name_of_file}.bin",

        # Keep the base model modest.
        # Giant models make references more expensive.
        base_tokens=3000,

        # This leaves space for future learned tokens.
        total_model_capacity=20000,

        # Makes the current input important.
        input_weight=100000,

        # Max number of input-specific tokens to add.
        max_new_tokens=10000,
    )

    restored_now = decompress(result["compressed"], result["model"])
    restored_later = decompress_from_saved_files(
        model_db=MODEL_DB,
        compressed_file=f"{name_of_file}.bin"
    )
    print("Restored later:", restored_later)

    print("Model database:", MODEL_DB)
    print("Compressed file:", f"{name_of_file}.bin")
    print("Model tokens:", len(result["model"]["token_to_id"]))
    print("Added input-specific tokens:", result["added_tokens"])

    print_stats("Before input retraining:", result["before_stats"])
    print_stats("After input retraining:", result["after_stats"])

    print("\nCompressed hex:")
    print(result["compressed"].hex())

    print("\nRestored now correctly:", text == restored_now)
    print("Restored later correctly:", text == restored_later)