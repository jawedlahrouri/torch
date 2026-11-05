import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
ANALYZER = str(ROOT / "my_torch_analyzer")
GENERATOR = str(ROOT / "my_torch_generator")

sys.path.insert(0, str(ROOT / "src"))
from neural_network import NeuralNetwork  # noqa: E402

LABELS = {"Nothing", "Check", "Checkmate"}
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
CHECKMATE_FEN = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
CHECK_FEN = "rnbqkbnr/pppp2pp/8/4pp1Q/3P4/4P3/PPP2PPP/RNB1KBNR b KQkq - 1 3"


def run(script, *args, cwd=None):
    return subprocess.run([sys.executable, script, *map(str, args)],
                        capture_output=True, text=True, cwd=cwd, timeout=120)


def assert_error(result):
    assert result.returncode == 84, result.stderr
    assert result.stderr.strip() != ""
    assert result.stdout == ""
    assert "Traceback" not in result.stderr


@pytest.fixture
def env(tmp_path):
    """A valid 768-4-3 network, a 3-line file to predict and a training file."""
    (tmp_path / "tiny.conf").write_text("768,4,3\n")
    assert run(GENERATOR, "tiny.conf", 1, cwd=tmp_path).returncode == 0
    predict = tmp_path / "predict.txt"
    predict.write_text(f"{START_FEN}\n{CHECKMATE_FEN}\n{CHECK_FEN}\n")
    train = tmp_path / "train.txt"
    train.write_text(f"{START_FEN} Nothing\n{CHECKMATE_FEN} Checkmate Black\n{CHECK_FEN} Check White\n")
    return {"dir": tmp_path, "nn": tmp_path / "tiny_1.nn", "predict": predict, "train": train}


# ---------- help ----------

@pytest.mark.parametrize("script", [ANALYZER, GENERATOR])
@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_help_exits_0_on_stdout(script, flag):
    result = run(script, flag)
    assert result.returncode == 0
    assert "USAGE" in result.stdout
    assert result.stderr == ""


def test_help_wins_over_other_arguments(env):
    assert run(ANALYZER, "--bogus", "-h").returncode == 0
    assert run(GENERATOR, "tiny.conf", "-h", cwd=env["dir"]).returncode == 0


# ---------- analyzer: argument errors ----------

@pytest.mark.parametrize("args", [
    [],
    ["--predict"],
    ["--train"],
    ["--predict", "only_one.nn"],
    ["--bogus", "a", "b"],
    ["--predict=a", "b"],
    ["-p", "a", "b"],
    ["-", "a", "b"],
    ["--PREDICT", "a", "b"],
])
def test_analyzer_bad_arguments(args):
    assert_error(run(ANALYZER, *args))


def test_analyzer_requires_a_mode(env):
    assert_error(run(ANALYZER, env["nn"], env["predict"]))


def test_analyzer_conflicting_modes(env):
    assert_error(run(ANALYZER, "--predict", "--train", env["nn"], env["predict"]))
    assert_error(run(ANALYZER, "--train", "--predict", env["nn"], env["train"]))


def test_analyzer_save_with_predict(env):
    assert_error(run(ANALYZER, "--predict", "--save", env["dir"] / "x.nn", env["nn"], env["predict"]))


def test_analyzer_save_without_value(env):
    assert_error(run(ANALYZER, "--train", env["nn"], env["train"], "--save"))


def test_analyzer_save_twice(env):
    assert_error(run(ANALYZER, "--train", "--save", "a.nn", "--save", "b.nn", env["nn"], env["train"]))


def test_analyzer_too_many_positionals(env):
    assert_error(run(ANALYZER, "--predict", env["nn"], env["predict"], env["predict"]))


def test_analyzer_save_swallows_loadfile(env):
    assert_error(run(ANALYZER, "--train", "--save", env["nn"], env["train"]))


# ---------- analyzer: files ----------

def test_analyzer_missing_files(env):
    assert_error(run(ANALYZER, "--predict", env["dir"] / "nope.nn", env["predict"]))
    assert_error(run(ANALYZER, "--predict", env["nn"], env["dir"] / "nope.txt"))
    assert_error(run(ANALYZER, "--predict", "", env["predict"]))


def test_analyzer_directories_are_not_files(env):
    assert_error(run(ANALYZER, "--predict", env["dir"], env["predict"]))
    assert_error(run(ANALYZER, "--predict", env["nn"], env["dir"]))


@pytest.mark.parametrize("content", ["", "\n\n   \n"])
def test_analyzer_empty_chessfile(env, content):
    empty = env["dir"] / "empty.txt"
    empty.write_text(content)
    assert_error(run(ANALYZER, "--predict", env["nn"], empty))
    assert_error(run(ANALYZER, "--train", env["nn"], empty))


@pytest.mark.parametrize("content", [b"", b"hello\n", b"\x80\x81\x82", b'{"a": 1}', b"[]", b'[{"type": "Foo"}]'])
def test_analyzer_corrupted_loadfile(env, content):
    bad = env["dir"] / "bad.nn"
    bad.write_bytes(content)
    assert_error(run(ANALYZER, "--predict", bad, env["predict"]))


def test_analyzer_truncated_loadfile(env):
    truncated = env["dir"] / "truncated.nn"
    truncated.write_bytes(env["nn"].read_bytes()[:20])
    assert_error(run(ANALYZER, "--predict", truncated, env["predict"]))


def test_analyzer_binary_chessfile(env):
    assert_error(run(ANALYZER, "--predict", env["nn"], env["nn"]))


@pytest.mark.parametrize("line", [
    "xx w - - 0 1",
    "8/8/8/8/8/8/8 w - - 0 1",
    "8/8/8/8/8/8/8/8/8 w - - 0 1",
    "9/8/8/8/8/8/8/8 w - - 0 1",
    "7/8/8/8/8/8/8/8 w - - 0 1",
    "k8/8/8/8/8/8/8/8 w - - 0 1",
])
def test_analyzer_invalid_fen(env, line):
    bad = env["dir"] / "bad_fen.txt"
    bad.write_text(f"{START_FEN}\n{line}\n")
    result = run(ANALYZER, "--predict", env["nn"], bad)
    assert_error(result)                      # empty stdout: no partial output
    assert "line 2" in result.stderr


@pytest.mark.parametrize("conf", ["4,3", "768,5", "768,2", "768,64,3,2"])
def test_analyzer_incompatible_network(env, conf):
    (env["dir"] / "wrong.conf").write_text(conf)
    assert run(GENERATOR, "wrong.conf", 1, cwd=env["dir"]).returncode == 0
    assert_error(run(ANALYZER, "--predict", env["dir"] / "wrong_1.nn", env["predict"]))


def test_analyzer_nan_network(env):
    nn = NeuralNetwork.load(str(env["nn"]))
    nn.layers[0].weights[:] = np.nan
    nn.save(str(env["dir"] / "nan.nn"))
    assert_error(run(ANALYZER, "--predict", env["dir"] / "nan.nn", env["predict"]))


# ---------- analyzer: --predict ----------

def test_predict_stdout_is_one_label_per_line_in_order(env):
    result = run(ANALYZER, "--predict", env["nn"], env["predict"])
    assert result.returncode == 0
    assert result.stderr == ""
    lines = result.stdout.split("\n")
    assert lines.pop() == ""                   # a single trailing newline
    assert len(lines) == 3
    assert set(lines) <= LABELS


def test_predict_is_deterministic_and_order_preserving(env):
    lines = [START_FEN, CHECKMATE_FEN, CHECK_FEN]
    full = run(ANALYZER, "--predict", env["nn"], env["predict"]).stdout.split()
    for i, fen in enumerate(lines):
        single = env["dir"] / f"single_{i}.txt"
        single.write_text(fen + "\n")
        assert run(ANALYZER, "--predict", env["nn"], single).stdout.split() == [full[i]]


def test_predict_accepts_expected_labels_in_input(env):
    result = run(ANALYZER, "--predict", env["nn"], env["train"])
    assert result.returncode == 0
    assert len(result.stdout.split()) == 3


def test_predict_without_final_newline_and_crlf(env):
    path = env["dir"] / "crlf.txt"
    path.write_bytes(f"{START_FEN}\r\n{CHECK_FEN}".encode())
    result = run(ANALYZER, "--predict", env["nn"], path)
    assert result.returncode == 0
    assert len(result.stdout.split()) == 2


def test_predict_does_not_modify_loadfile(env):
    before = env["nn"].read_bytes()
    run(ANALYZER, "--predict", env["nn"], env["predict"])
    assert env["nn"].read_bytes() == before


def test_predict_argument_order_is_free(env):
    result = run(ANALYZER, env["nn"], env["predict"], "--predict")
    assert result.returncode == 0 and len(result.stdout.split()) == 3


# ---------- analyzer: --train ----------

def test_train_without_save_overwrites_loadfile(env):
    before = env["nn"].read_bytes()
    result = run(ANALYZER, "--train", env["nn"], env["train"])
    assert result.returncode == 0, result.stderr
    assert env["nn"].read_bytes() != before
    assert run(ANALYZER, "--predict", env["nn"], env["predict"]).returncode == 0


def test_train_with_save_keeps_loadfile_intact(env):
    before = env["nn"].read_bytes()
    target = env["dir"] / "trained.nn"
    result = run(ANALYZER, "--train", "--save", target, env["nn"], env["train"])
    assert result.returncode == 0, result.stderr
    assert env["nn"].read_bytes() == before
    assert target.is_file()
    assert run(ANALYZER, "--predict", target, env["predict"]).returncode == 0


def test_train_save_option_before_mode(env):
    target = env["dir"] / "trained.nn"
    assert run(ANALYZER, "--save", target, "--train", env["nn"], env["train"]).returncode == 0
    assert target.is_file()


def test_train_uses_the_labels_of_the_file(env):
    # single-class dataset: the output bias alone is enough to learn it, whatever the random init
    data = env["dir"] / "one_class.txt"
    data.write_text(f"{START_FEN} Checkmate Black\n{CHECKMATE_FEN} Checkmate Black\n{CHECK_FEN} Checkmate White\n")
    assert run(ANALYZER, "--train", env["nn"], data).returncode == 0
    assert run(ANALYZER, "--predict", env["nn"], data).stdout.split() == ["Checkmate"] * 3


@pytest.mark.parametrize("line", [
    START_FEN,                       # no label
    START_FEN + " Stalemate",        # unknown label
    START_FEN + " Check Nothing",    # ambiguous label
])
def test_train_bad_label_leaves_loadfile_untouched(env, line):
    bad = env["dir"] / "bad_label.txt"
    bad.write_text(line + "\n")
    before = env["nn"].read_bytes()
    assert_error(run(ANALYZER, "--train", env["nn"], bad))
    assert env["nn"].read_bytes() == before


def test_train_unwritable_savefile_fails_before_training(env):
    result = run(ANALYZER, "--train", "--save", env["dir"], env["nn"], env["train"])  # a directory
    assert_error(result)                      # empty stdout: training never started
    result = run(ANALYZER, "--train", "--save", env["dir"] / "no_such_dir" / "x.nn", env["nn"], env["train"])
    assert_error(result)


def test_train_leaves_no_temporary_file(env):
    run(ANALYZER, "--train", env["nn"], env["train"])
    assert sorted(p.name for p in env["dir"].iterdir() if p.name.startswith(".tmp_")) == []


# ---------- generator ----------

def test_generator_creates_numbered_files(tmp_path):
    (tmp_path / "net.conf").write_text("768,16,3\n")
    result = run(GENERATOR, "net.conf", 3, cwd=tmp_path)
    assert result.returncode == 0 and result.stdout == "" and result.stderr == ""
    assert sorted(p.name for p in tmp_path.glob("*.nn")) == ["net_1.nn", "net_2.nn", "net_3.nn"]


def test_generator_multiple_configs(tmp_path):
    (tmp_path / "a.conf").write_text("768,4,3")
    (tmp_path / "b.conf").write_text("768,8,3")
    assert run(GENERATOR, "a.conf", 1, "b.conf", 2, cwd=tmp_path).returncode == 0
    assert sorted(p.name for p in tmp_path.glob("*.nn")) == ["a_1.nn", "b_1.nn", "b_2.nn"]


def test_generator_networks_are_independent_and_random(tmp_path):
    (tmp_path / "net.conf").write_text("768,16,3")
    run(GENERATOR, "net.conf", 2, cwd=tmp_path)
    assert (tmp_path / "net_1.nn").read_bytes() != (tmp_path / "net_2.nn").read_bytes()


@pytest.mark.parametrize("content,expected", [
    ("768 64 3", [768, 64, 3]),
    ("768,64,3", [768, 64, 3]),
    ("768\n64\n3\n", [768, 64, 3]),
    ("768,64,3,", [768, 64, 3]),
])
def test_generator_config_formats(tmp_path, content, expected):
    (tmp_path / "net.conf").write_text(content)
    assert run(GENERATOR, "net.conf", 1, cwd=tmp_path).returncode == 0
    nn = NeuralNetwork.load(str(tmp_path / "net_1.nn"))
    shapes = [layer.weights.shape for layer in nn.trainable_layers]
    assert shapes == list(zip(expected, expected[1:]))


@pytest.mark.parametrize("args", [
    [],
    ["net.conf"],
    ["net.conf", "2", "other.conf"],
    ["net.conf", "0"],
    ["net.conf", "-1"],
    ["net.conf", "x"],
    ["net.conf", "1.5"],
    ["nope.conf", "1"],
    ["adir", "1"],
])
def test_generator_bad_arguments(tmp_path, args):
    (tmp_path / "net.conf").write_text("768,4,3")
    (tmp_path / "adir").mkdir()
    result = run(GENERATOR, *args, cwd=tmp_path)
    assert_error(result)
    assert list(tmp_path.glob("*.nn")) == []


@pytest.mark.parametrize("content", ["", "768", "768,0,3", "768,-5,3", "768;64;3", "abc", "768,3.5"])
def test_generator_bad_config_content(tmp_path, content):
    (tmp_path / "bad.conf").write_text(content)
    assert_error(run(GENERATOR, "bad.conf", 1, cwd=tmp_path))
    assert list(tmp_path.glob("*.nn")) == []


def test_generator_validates_everything_before_writing(tmp_path):
    (tmp_path / "ok.conf").write_text("768,4,3")
    assert_error(run(GENERATOR, "ok.conf", 2, "missing.conf", 1, cwd=tmp_path))
    assert list(tmp_path.glob("*.nn")) == []


def test_generator_refuses_output_name_collision(tmp_path):
    for folder in ("d1", "d2"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "x.conf").write_text("768,4,3")
    assert_error(run(GENERATOR, "d1/x.conf", 1, "d2/x.conf", 1, cwd=tmp_path))
    assert list(tmp_path.glob("*.nn")) == []


def test_generator_then_analyzer_pipeline(tmp_path):
    (tmp_path / "net.conf").write_text("768,8,3")
    (tmp_path / "chess.txt").write_text(f"{START_FEN}\n{CHECK_FEN}\n")
    assert run(GENERATOR, "net.conf", 1, cwd=tmp_path).returncode == 0
    result = run(ANALYZER, "--predict", tmp_path / "net_1.nn", tmp_path / "chess.txt")
    assert result.returncode == 0 and len(result.stdout.split()) == 2
