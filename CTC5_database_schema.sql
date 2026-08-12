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

INSERT INTO metadata (key, value) VALUES
    ('max_tokens', '20000'),
    ('model_version', '1'),
    ('format_version', '5'),
    ('magic', 'CTC5');

INSERT INTO tokens (id, token, frequency) VALUES
    (0, '<ESCAPE>', 1),
    (1, 'the', 9234),
    (2, ' ', 8901),
    (3, ', ', 3410),
    (4, 'compression', 120),
    (5, 'model', 98);
