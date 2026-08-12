# CTC5 SQLite Database Tables

CTC5 stores its reusable compression model in a SQLite database named:

```text
compression_model.db
```

The database is intentionally small and simple. It only stores the shared model metadata and the token frequency table. Huffman codes and context-dependent Huffman tables are not stored directly; they are rebuilt deterministically when the model is loaded.

---

## Database Overview

| Table | Purpose |
|---|---|
| `metadata` | Stores global model and file-format information such as model version, format version, max token capacity, and magic header. |
| `tokens` | Stores every known token in the shared model, its numeric ID, and its frequency count. |

---

## Entity Relationship View

```text
compression_model.db
│
├── metadata
│   ├── key   TEXT PRIMARY KEY
│   └── value TEXT NOT NULL
│
└── tokens
    ├── id        INTEGER PRIMARY KEY
    ├── token     TEXT NOT NULL UNIQUE
    └── frequency INTEGER NOT NULL

Index:
└── idx_token ON tokens(token)
```

There is no foreign-key relationship between the two tables. The `metadata` table describes the model as a whole, while the `tokens` table stores the model vocabulary.

---

## Table: `metadata`

### Purpose

Stores key-value settings used when loading and validating the compression model.

### Schema

```sql
CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
```

### Column Description

| Column | Type | Constraint | Description |
|---|---:|---|---|
| `key` | `TEXT` | `PRIMARY KEY` | Name of the metadata setting. |
| `value` | `TEXT` | `NOT NULL` | Stored value as text. Converted back to integers where needed. |

### Example Rows

| key | value | Meaning |
|---|---:|---|
| `max_tokens` | `20000` | Maximum number of tokens the model can store. |
| `model_version` | `1` | Current reusable model version. Bumped when the model is updated. |
| `format_version` | `5` | Compressed file format version used by CTC5. |
| `magic` | `CTC5` | Magic header used to identify CTC5 compressed files. |

---

## Table: `tokens`

### Purpose

Stores the reusable token vocabulary and token frequencies used to rebuild Huffman coding tables.

### Schema

```sql
CREATE TABLE tokens (
    id INTEGER PRIMARY KEY,
    token TEXT NOT NULL UNIQUE,
    frequency INTEGER NOT NULL
);
```

### Index

```sql
CREATE INDEX idx_token ON tokens(token);
```

### Column Description

| Column | Type | Constraint | Description |
|---|---:|---|---|
| `id` | `INTEGER` | `PRIMARY KEY` | Numeric token ID used inside the compressor. `0` is reserved for `<ESCAPE>`. |
| `token` | `TEXT` | `NOT NULL`, `UNIQUE` | Actual token string, such as a word, space, punctuation, or fallback marker. |
| `frequency` | `INTEGER` | `NOT NULL` | Frequency count used to build Huffman codes. Higher frequency usually means shorter codes. |

### Example Rows

| id | token | frequency | Meaning |
|---:|---|---:|---|
| `0` | `<ESCAPE>` | `1` | Reserved escape token. Present in every model. |
| `1` | `the` | `9234` | Common word token. |
| `2` | ` ` | `8901` | Space token. |
| `3` | `, ` | `3410` | Punctuation/spacing token. |
| `4` | `compression` | `120` | Domain-specific word token. |
| `5` | `model` | `98` | Domain-specific word token. |

Actual rows depend on the training corpus used to build `compression_model.db`.

---

## How CTC5 Uses These Tables

### During Training

1. Text is tokenized into words and non-word chunks.
2. Tokens are counted.
3. The most common tokens are inserted into `tokens`.
4. General model settings are inserted into `metadata`.
5. The database is saved as `compression_model.db`.

### During Compression

1. CTC5 loads `metadata`.
2. CTC5 loads all rows from `tokens` ordered by `id`.
3. It reconstructs:
   - `token_to_id`
   - `id_to_token`
   - `frequencies`
4. It rebuilds global Huffman codes.
5. It rebuilds context-dependent Huffman codes for all two-token contexts.
6. The compressed file stores only the model version and compressed payload, not the full token table.

### During Decompression

1. CTC5 reads the compressed file header.
2. It checks the required model version.
3. It loads the same SQLite model database.
4. It rebuilds the same Huffman tables.
5. It decodes token IDs back into text.

---

## Full SQL Schema

```sql
DROP TABLE IF EXISTS metadata;
DROP TABLE IF EXISTS tokens;

CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE tokens (
    id INTEGER PRIMARY KEY,
    token TEXT NOT NULL UNIQUE,
    frequency INTEGER NOT NULL
);

CREATE INDEX idx_token ON tokens(token);
```

---

## Important Design Note

The database does not store compressed files. It stores the reusable global model only.

Each `.ctc5` compressed file stores:

```text
magic header + format version + model version + payload size + compressed bytes
```

That means the database must still exist when decompressing later. The compressed file depends on the same model version being available.
