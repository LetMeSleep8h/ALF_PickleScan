#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testes rápidos do ALF_picklescan.

Todos os ficheiros de teste são gerados aqui mesmo e são inofensivos: nenhum
executa código. Um pickle "malicioso" nestes testes apenas *referencia* uma
função perigosa, sem a chamar.

Uso:
    python -m unittest discover -s tests -v
    python tests/test_smoke.py
"""

import json
import pickle
import struct
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ALF_picklescan as core  # noqa: E402


# Referencia os.system sem a chamar: serve para testar a deteção.
SUSPICIOUS_PICKLE = b"cposix\nsystem\n(S'echo hi'\ntR."


def benign_pickle() -> bytes:
    return pickle.dumps({"pesos": [1, 2, 3], "nota": "amostra inofensiva"}, protocol=4)


def write(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


def make_safetensors(path: Path, header: dict, body: bytes) -> Path:
    raw = json.dumps(header).encode("utf-8")
    return write(path, struct.pack("<Q", len(raw)) + raw + body)


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.whitelist = core.load_whitelist()

    def tearDown(self):
        self._tmp.cleanup()

    def verify(self, path: Path, sha=None):
        checks, digest = core.verificar(path, sha, self.whitelist)
        label, action, code = core.veredicto(checks)
        return checks, digest, label, code

    def state_of(self, checks, nome):
        for c in checks:
            if c["nome"] == nome:
                return c["estado"]
        return None


class TestPickle(Base):
    def test_benign_pickle_with_hash_is_green(self):
        p = write(self.tmp / "modelo.pkl", benign_pickle())
        esperado = core.sha256_of(p)
        checks, digest, label, code = self.verify(p, esperado)
        self.assertEqual(digest, esperado)
        self.assertEqual(label, "VERDE", checks)
        self.assertEqual(code, 0)
        self.assertEqual(self.state_of(checks, "Imports perigosos"), core.OK)
        self.assertEqual(self.state_of(checks, "Imports fora da whitelist"), core.OK)

    def test_without_hash_is_never_green(self):
        p = write(self.tmp / "modelo.pkl", benign_pickle())
        checks, _digest, label, code = self.verify(p)
        self.assertEqual(label, "AMARELO", checks)
        self.assertEqual(code, 1)
        self.assertEqual(self.state_of(checks, "Integridade (hash)"), core.AVISO)

    def test_hash_mismatch_is_red(self):
        p = write(self.tmp / "modelo.pkl", benign_pickle())
        _checks, _digest, label, code = self.verify(p, "0" * 64)
        self.assertEqual(label, "VERMELHO")
        self.assertEqual(code, 2)

    def test_dangerous_import_is_red(self):
        p = write(self.tmp / "mau.pkl", SUSPICIOUS_PICKLE)
        checks, _digest, label, code = self.verify(p, core.sha256_of(p))
        self.assertEqual(label, "VERMELHO", checks)
        self.assertEqual(code, 2)
        self.assertEqual(self.state_of(checks, "Imports perigosos"), core.REPROVADO)

    def test_picklescan_runs_at_all(self):
        """O picklescan tem de ser encontrado (comando, embutido ou python -m)."""
        p = write(self.tmp / "modelo.pkl", benign_pickle())
        code, saida = core.run_picklescan(p)
        self.assertIsNotNone(code, "picklescan não foi encontrado: " + str(saida))

    def test_picklescan_effective_on_dangerous_pickle(self):
        """A amostra perigosa tem de ser detetada, não apenas o ALF_picklescan."""
        p = write(self.tmp / "mau.pkl", SUSPICIOUS_PICKLE)
        code, _saida = core.run_picklescan(p)
        self.assertEqual(code, 1)

    def test_executable_inside_pytorch_zip_is_red(self):
        p = self.tmp / "modelo.bin"
        with zipfile.ZipFile(p, "w") as z:
            z.writestr("archive/data.pkl", benign_pickle())
            z.writestr("archive/lib.so", b"\x7fELF falso")
        checks, _digest, label, code = self.verify(p, core.sha256_of(p))
        self.assertEqual(label, "VERMELHO", checks)
        self.assertEqual(self.state_of(checks, "Ficheiros executáveis dentro do modelo"),
                         core.REPROVADO)

    def test_clean_pytorch_zip_has_no_executable_alert(self):
        p = self.tmp / "limpo.bin"
        with zipfile.ZipFile(p, "w") as z:
            z.writestr("archive/data.pkl", benign_pickle())
            z.writestr("archive/version", b"3\n")
        checks, _digest, _label, _code = self.verify(p, core.sha256_of(p))
        self.assertEqual(self.state_of(checks, "Ficheiros executáveis dentro do modelo"), core.OK)

    def test_report_is_written(self):
        p = write(self.tmp / "modelo.pkl", benign_pickle())
        checks, digest, label, code = self.verify(p, core.sha256_of(p))
        report = core.gerar_relatorio(p, digest, label, "Aprovado", checks)
        self.assertTrue(Path(report).is_file())
        texto = Path(report).read_text(encoding="utf-8")
        self.assertIn(digest, texto)
        self.assertIn("Limitações", texto)
        Path(report).unlink()


class TestSafetensors(Base):
    def test_valid_safetensors(self):
        p = make_safetensors(self.tmp / "modelo.safetensors",
                             {"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
                             b"\x00\x00\x80?")
        checks, _digest, label, _code = self.verify(p, core.sha256_of(p))
        self.assertEqual(self.state_of(checks, "Estrutura safetensors"), core.OK)
        self.assertEqual(label, "VERDE", checks)

    def test_offsets_outside_file(self):
        p = make_safetensors(self.tmp / "modelo.safetensors",
                             {"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 999]}},
                             b"\x00\x00\x80?")
        estado, detalhe = core.check_safetensors(p)
        self.assertEqual(estado, core.REPROVADO, detalhe)

    def test_header_bigger_than_file(self):
        p = write(self.tmp / "modelo.safetensors", struct.pack("<Q", 10**9) + b"{}")
        estado, _detalhe = core.check_safetensors(p)
        self.assertEqual(estado, core.REPROVADO)

    def test_invalid_json_header(self):
        p = write(self.tmp / "modelo.safetensors", struct.pack("<Q", 4) + b"nope")
        estado, _detalhe = core.check_safetensors(p)
        self.assertEqual(estado, core.REPROVADO)


class TestGGUF(Base):
    def test_valid_gguf(self):
        p = write(self.tmp / "modelo.gguf", b"GGUF" + struct.pack("<I", 3) + b"\x00" * 32)
        checks, _digest, label, _code = self.verify(p, core.sha256_of(p))
        self.assertEqual(self.state_of(checks, "Estrutura GGUF"), core.OK)
        self.assertEqual(label, "VERDE", checks)

    def test_unknown_version_warns(self):
        p = write(self.tmp / "modelo.gguf", b"GGUF" + struct.pack("<I", 9) + b"\x00" * 32)
        estado, _detalhe = core.check_gguf(p)
        self.assertEqual(estado, core.AVISO)

    def test_bad_magic(self):
        p = write(self.tmp / "modelo.gguf", b"NOPE" + struct.pack("<I", 3))
        estado, _detalhe = core.check_gguf(p)
        self.assertEqual(estado, core.REPROVADO)


class TestMisc(Base):
    def test_unknown_format_warns(self):
        p = write(self.tmp / "modelo.xyz", b"texto qualquer")
        checks, _digest, label, _code = self.verify(p, core.sha256_of(p))
        self.assertEqual(self.state_of(checks, "Formato"), core.AVISO)
        self.assertEqual(label, "AMARELO")

    def test_sha256_known_vector(self):
        p = write(self.tmp / "a.bin", b"abc")
        self.assertEqual(core.sha256_of(p),
                         "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")

    def test_fmt_size(self):
        self.assertEqual(core.fmt_size(512), "512 B")
        self.assertEqual(core.fmt_size(2048), "2.00 KB")


class TestI18n(unittest.TestCase):
    def test_every_language_has_every_key(self):
        from i18n import LANGUAGES, STRINGS
        base = set(STRINGS["en"])
        for code, _name in LANGUAGES:
            self.assertIn(code, STRINGS, f"idioma em falta: {code}")
            faltam = base - set(STRINGS[code])
            self.assertFalse(faltam, f"{code}: chaves em falta {sorted(faltam)}")

    def test_missing_key_falls_back_to_english(self):
        from i18n import t
        self.assertEqual(t("verify", "en"), "Verify model")
        self.assertEqual(t("chave_inexistente", "fr"), "chave_inexistente")


if __name__ == "__main__":
    unittest.main(verbosity=2)
