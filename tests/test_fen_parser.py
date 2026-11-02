import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import fen_parser
from fen_parser import LABELS, N_FEATURES, PIECES, fen_to_vector, format_label, load_dataset

START_BOARD = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
START_FEN = START_BOARD + " w KQkq - 0 1"
CHECKMATE_FEN = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
CHECK_FEN = "rnbqkbnr/pppp2pp/8/4pp1Q/3P4/4P3/PPP2PPP/RNB1KBNR b KQkq - 1 3"


def write(tmp_path, text, name="chess.txt"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def piece_on(vector, square):
    cell = vector.reshape(64, 12)[square]
    return PIECES[int(cell.argmax())] if cell.sum() else None


def test_vector_shape_and_dtype():
    vector = fen_to_vector(START_BOARD)
    assert vector.shape == (N_FEATURES,) == (768,)
    assert vector.dtype == np.float32


def test_start_position_one_hot():
    vector = fen_to_vector(START_BOARD)
    assert vector.sum() == 32
    assert set(np.unique(vector)) == {0.0, 1.0}
    assert (vector.reshape(64, 12).sum(axis=1) <= 1).all()
    assert piece_on(vector, 0) == "r"    # a8
    assert piece_on(vector, 4) == "k"    # e8
    assert piece_on(vector, 8) == "p"    # a7
    assert piece_on(vector, 52) == "P"   # e2
    assert piece_on(vector, 60) == "K"   # e1
    assert piece_on(vector, 63) == "R"   # h1
    assert piece_on(vector, 27) is None  # d5


def test_empty_board_is_all_zero():
    assert not fen_to_vector("8/8/8/8/8/8/8/8").any()


def test_digits_skip_squares():
    vector = fen_to_vector("3k4/8/8/8/8/8/8/K7")
    assert piece_on(vector, 3) == "k"
    assert piece_on(vector, 56) == "K"
    assert vector.sum() == 2


def test_consecutive_digits_add_up():
    assert np.array_equal(fen_to_vector("11k5/8/8/8/8/8/8/44"), fen_to_vector("2k5/8/8/8/8/8/8/8"))


@pytest.mark.parametrize("board", [
    "8/8/8/8/8/8/8",              # 7 ranks
    "8/8/8/8/8/8/8/8/8",          # 9 ranks
    "7/8/8/8/8/8/8/8",           # rank too short
    "9/8/8/8/8/8/8/8",           # invalid digit
    "0/8/8/8/8/8/8/8",           # invalid digit
    "k8/8/8/8/8/8/8/8",          # rank too long
    "44k/8/8/8/8/8/8/8",         # rank too long
    "x7/8/8/8/8/8/8/8",          # unknown character
    "8/8/8/8/8/8/8/K6٣",    # Unicode digit
    "",
])
def test_invalid_boards_raise(board):
    with pytest.raises(ValueError):
        fen_to_vector(board)


def test_load_dataset_train(tmp_path):
    path = write(tmp_path, f"{START_FEN} Nothing\n{CHECKMATE_FEN} Checkmate Black\n{CHECK_FEN} Check White\n")
    X, y, sides = load_dataset(path, is_train=True)
    assert X.shape == (3, 768) and X.dtype == np.float32
    assert y.dtype == np.int64
    assert y.tolist() == [LABELS.index("Nothing"), LABELS.index("Checkmate"), LABELS.index("Check")]
    assert sides == ["w", "w", "b"]


def test_load_dataset_predict_ignores_labels(tmp_path):
    path = write(tmp_path, f"{START_FEN}\n{CHECK_FEN} Check White\n")
    X, y, sides = load_dataset(path, is_train=False)
    assert X.shape == (2, 768)
    assert y is None
    assert sides == ["w", "b"]


def test_load_dataset_skips_blank_lines_and_handles_crlf(tmp_path):
    path = write(tmp_path, f"\n{START_FEN} Nothing\r\n\n   \n{CHECK_FEN} Check White\r\n")
    X, y, _ = load_dataset(path, is_train=True)
    assert len(X) == 2 and y.tolist() == [0, 1]


def test_load_dataset_without_trailing_newline(tmp_path):
    path = write(tmp_path, START_FEN)
    X, _, _ = load_dataset(path, is_train=False)
    assert len(X) == 1


def test_load_dataset_matches_fen_to_vector(tmp_path):
    path = write(tmp_path, f"{CHECK_FEN} Check White\n{START_FEN} Nothing\n")
    X, _, _ = load_dataset(path, is_train=True)
    assert np.array_equal(X[0], fen_to_vector(CHECK_FEN.split()[0]))
    assert np.array_equal(X[1], fen_to_vector(START_BOARD))


@pytest.mark.parametrize("content", ["", "\n\n  \n"])
def test_load_dataset_empty_raises(tmp_path, content):
    with pytest.raises(ValueError, match="empty"):
        load_dataset(write(tmp_path, content), is_train=False)


@pytest.mark.parametrize("line", [
    START_FEN,                          # no label
    START_FEN + " Stalemate",           # unknown label
    START_FEN + " Check Nothing",       # ambiguous label
    START_FEN + " Checkmated",          # no substring matching
])
def test_load_dataset_bad_label_raises(tmp_path, line):
    with pytest.raises(ValueError, match="line 1"):
        load_dataset(write(tmp_path, line + "\n"), is_train=True)


def test_load_dataset_error_reports_line_number(tmp_path):
    path = write(tmp_path, f"{START_FEN}\n\nxx w\n")
    with pytest.raises(ValueError, match="line 3"):
        load_dataset(path, is_train=False)


def test_load_dataset_missing_file():
    with pytest.raises(OSError):
        load_dataset("does/not/exist.txt", is_train=False)


def test_format_label_without_color(monkeypatch):
    monkeypatch.setattr(fen_parser, "OUTPUT_COLOR", False)
    assert [format_label(i, "w") for i in range(3)] == LABELS


def test_format_label_with_color(monkeypatch):
    monkeypatch.setattr(fen_parser, "OUTPUT_COLOR", True)
    assert format_label(0, "w") == "Nothing"
    assert format_label(1, "w") == "Check Black"
    assert format_label(1, "b") == "Check White"
    assert format_label(2, "b") == "Checkmate White"
    assert format_label(2, None) == "Checkmate"
