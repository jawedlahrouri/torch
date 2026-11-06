import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fen_parser import (
    flip_board,
    flip_castling,
    flip_en_passant,
    flip_fen_line,
)

START_BOARD = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
CHECK_BLACK = "8/1p3ppk/2q4p/p3p1n1/N2n2Q1/P5PP/1P3PK1/4R3 w - - 3 28 Check Black"
CHECK_WHITE = "4r3/1p3pk1/p5pp/n2N2q1/P3P1N1/2Q4P/1P3PPK/8 b - - 3 28 Check White"
CHECKMATE_BLACK = "3r1rk1/pp3ppp/3b4/3p4/P2N1P2/2P1P1q1/1P1BBn1n/2RQR2K w - - 1 22 Checkmate Black"
CHECKMATE_WHITE = "2rqr2k/1p1bbN1N/2p1p1Q1/p2n1p2/3P4/3B4/PP3PPP/3R1RK1 b - - 1 22 Checkmate White"
NOTHING_FEN = "5B2/1b4pk/5n1p/4p3/4P3/2P2PPP/6BK/8 b - - 0 40 Nothing"


def test_flip_board_symmetric():
    assert flip_board(START_BOARD) == START_BOARD


def test_flip_board_pieces_and_ranks():
    board = "r7/8/8/8/8/8/8/7R"
    assert flip_board(board) == "7r/8/8/8/8/8/8/R7"


def test_flip_board_invalid_ranks():
    with pytest.raises(ValueError, match="expected 8 ranks"):
        flip_board("8/8/8")


def test_flip_castling():
    assert flip_castling("-") == "-"
    assert flip_castling("KQkq") == "KQkq"
    assert flip_castling("KQ") == "kq"
    assert flip_castling("k") == "K"
    assert flip_castling("Qk") == "Kq"


def test_flip_en_passant():
    assert flip_en_passant("-") == "-"
    assert flip_en_passant("e3") == "e6"
    assert flip_en_passant("e6") == "e3"
    assert flip_en_passant("c6") == "c3"


def test_flip_fen_line_check():
    assert flip_fen_line(CHECK_BLACK) == CHECK_WHITE
    assert flip_fen_line(CHECK_WHITE) == CHECK_BLACK


def test_flip_fen_line_checkmate():
    assert flip_fen_line(CHECKMATE_BLACK) == CHECKMATE_WHITE
    assert flip_fen_line(CHECKMATE_WHITE) == CHECKMATE_BLACK


def test_flip_fen_line_nothing():
    flipped = flip_fen_line(NOTHING_FEN)
    assert flipped.endswith("Nothing")
    assert flipped.split()[1] == "w"


def test_flip_fen_line_involution():
    samples = [CHECK_BLACK, CHECKMATE_BLACK, NOTHING_FEN, "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1 Nothing"]
    for s in samples:
        assert flip_fen_line(flip_fen_line(s)) == s


def test_cli_augment_script(tmp_path):
    script = Path(__file__).resolve().parent.parent / "scripts" / "augment_dataset.py"
    in_file = tmp_path / "sample.txt"
    out_file = tmp_path / "augmented.txt"

    in_file.write_text(f"{CHECK_BLACK}\n{NOTHING_FEN}\n", encoding="utf-8")

    cmd = [sys.executable, str(script), "-i", str(in_file), "-o", str(out_file), "--mode", "combine"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    lines = [line.strip() for line in out_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 4
    assert CHECK_BLACK in lines
    assert CHECK_WHITE in lines

