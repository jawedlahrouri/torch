import numpy as np

PIECES = "PNBRQKpnbrqk"
PIECE_INDEX = {piece: i for i, piece in enumerate(PIECES)}
BOARD_SIZE = 8
N_FEATURES = BOARD_SIZE * BOARD_SIZE * len(PIECES)
EMPTY_RUN = {str(n): n for n in range(1, BOARD_SIZE + 1)}

LABELS = ["Nothing", "Check", "Checkmate"]
LABEL_INDEX = {label: i for i, label in enumerate(LABELS)}
NOTHING = LABELS[0]

CHECKING_COLOR = {"w": "Black", "b": "White"}
OUTPUT_COLOR = True  # True ("Check White")

def _active_indices(fen_board: str) -> list:
    ranks = fen_board.split('/')
    if len(ranks) != BOARD_SIZE:
        raise ValueError(f"Invalid FEN: expected {BOARD_SIZE} ranks.")
    indices = []
    for rank_idx, rank in enumerate(ranks):
        file_idx = 0
        for char in rank:
            empty_squares = EMPTY_RUN.get(char)
            piece_idx = PIECE_INDEX.get(char)
            if empty_squares is not None:
                file_idx += empty_squares
            elif piece_idx is not None:
                if file_idx >= BOARD_SIZE:
                    raise ValueError("Invalid FEN: rank too long.")
                square = rank_idx * BOARD_SIZE + file_idx
                indices.append(square * len(PIECES) + piece_idx)
                file_idx += 1
            else:
                raise ValueError(f"Invalid FEN character: {char!r}")
        if file_idx != BOARD_SIZE:
            raise ValueError(f"Invalid FEN: each rank must have {BOARD_SIZE} squares.")
    return indices

def fen_to_vector(fen_board: str) -> np.ndarray:
    vector = np.zeros(N_FEATURES, dtype=np.float32)
    vector[_active_indices(fen_board)] = 1.0
    return vector

def format_label(label_idx: int, side_to_move) -> str:
    label = LABELS[label_idx]
    color = CHECKING_COLOR.get(side_to_move)
    if label == NOTHING or not OUTPUT_COLOR or color is None:
        return label
    return f"{label} {color}"

def _parse_line(fields: list, is_train: bool):
    indices = _active_indices(fields[0])
    side_to_move = fields[1] if len(fields) > 1 and fields[1] in CHECKING_COLOR else None
    label_idx = None
    if is_train:
        label_ids = [LABEL_INDEX[field] for field in fields[1:] if field in LABEL_INDEX]
        if len(label_ids) != 1:
            raise ValueError("missing or ambiguous label")
        label_idx = label_ids[0]
    return indices, side_to_move, label_idx

def load_dataset(filepath: str, is_train: bool):
    sample_rows = []
    feature_cols = []
    labels = []
    sides = []

    with open(filepath, "r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, 1):
            fields = line.split()
            if not fields:
                continue
            try:
                indices, side_to_move, label_idx = _parse_line(fields, is_train)
            except ValueError as err:
                raise ValueError(f"line {line_no}: {err}") from None
            sample_rows.extend([len(sides)] * len(indices))
            feature_cols.extend(indices)
            sides.append(side_to_move)
            if is_train:
                labels.append(label_idx)

    if not sides:
        raise ValueError("Dataset is empty.")

    X = np.zeros((len(sides), N_FEATURES), dtype=np.float32)
    X[sample_rows, feature_cols] = 1.0
    y = np.array(labels, dtype=np.int64) if is_train else None
    return X, y, sides

def flip_board(fen_board: str) -> str:
    ranks = fen_board.split('/')
    if len(ranks) != BOARD_SIZE:
        raise ValueError(f"Invalid FEN: expected {BOARD_SIZE} ranks.")
    return "/".join(rank.swapcase() for rank in reversed(ranks))

def flip_castling(castling: str) -> str:
    if castling == "-":
        return "-"
    swapped = castling.swapcase()
    order = "KQkq"
    result = "".join(c for c in order if c in swapped)
    return result if result else "-"

def flip_en_passant(ep: str) -> str:
    if ep == "-" or len(ep) != 2:
        return ep
    return f"{ep[0]}{9 - int(ep[1])}"

def flip_fen_line(line: str) -> str:
    fields = line.strip().split()
    if not fields:
        return ""
    fields[0] = flip_board(fields[0])
    if len(fields) >= 6 and fields[1] in ("w", "b"):
        fields[1] = "b" if fields[1] == "w" else "w"
        fields[2] = flip_castling(fields[2])
        fields[3] = flip_en_passant(fields[3])
        label_start = 6
    else:
        label_start = 1
        if len(fields) > 1 and fields[1] in ("w", "b"):
            fields[1] = "b" if fields[1] == "w" else "w"
            label_start = 2
    for i in range(label_start, len(fields)):
        if fields[i] == "White":
            fields[i] = "Black"
        elif fields[i] == "Black":
            fields[i] = "White"
    return " ".join(fields)

