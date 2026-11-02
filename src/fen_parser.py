import numpy as np

PIECES = "PNBRQKpnbrqk"
LABELS = ["Nothing", "Check", "Checkmate"]

def fen_to_vector(fen_board: str) -> np.ndarray:
    vec = np.zeros(768, dtype=np.float32)
    square_idx = 0
    for char in fen_board:
        if char == '/':
            continue
        elif char.isdigit():
            square_idx += int(char)
        elif char in PIECES:
            if square_idx >= 64:
                raise ValueError("Too many squares in FEN.")
            vec[square_idx * 12 + PIECES.index(char)] = 1.0
            square_idx += 1
        else:
            raise ValueError(f"Invalid piece: {char}")
    if square_idx != 64:
        raise ValueError("Invalid number of squares.")
    return vec

def load_dataset(filepath: str, is_train: bool):
    X_list = []
    y_list = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 1:
                continue

            fen_board = parts[0]
            X_list.append(fen_to_vector(fen_board))

            if is_train:
                if len(parts) < 2:
                    raise ValueError(f"Missing label in line: {line}")
                label = parts[-1]
                if "Checkmate" in line:
                    label = "Checkmate"
                elif "Check" in line:
                    label = "Check"
                elif "Nothing" in line:
                    label = "Nothing"

                if label not in LABELS:
                    raise ValueError(f"Unknown label: {label}")
                y_list.append(LABELS.index(label))

    if not X_list:
        raise ValueError("Dataset is empty.")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64) if is_train else None
    return X, y