#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ALF_picklescan v2.1
Verificação de modelos de IA antes de os usar.

Formatos suportados:
  - pickle / PyTorch (.bin, .pt, .pth, .pkl)  -> picklescan + análise de imports
  - safetensors (.safetensors)                 -> validação de cabeçalho
  - GGUF (.gguf)                               -> validação de assinatura e versão

Uso:
  python ALF_picklescan.py                              (modo interativo)
  python ALF_picklescan.py modelo.bin
  python ALF_picklescan.py modelo.bin --sha256 <hash>   (hash publicado pelo autor)

Requisitos:
  pip install picklescan==1.0.5

Saída:
  Semáforo no ecrã + relatório HTML em ALF_relatorios/
  Códigos de saída: 0 = verde, 1 = amarelo, 2 = vermelho, 3 = erro
"""

import argparse
import hashlib
import html
import json
import os
import pickletools
import shutil
import struct
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

VERSION = "2.1"
if getattr(sys, "frozen", False):
    # Executável empacotado: usar a pasta onde está o .exe (não a pasta temporária do PyInstaller)
    BASE_DIR = Path(sys.executable).resolve().parent
else:
    BASE_DIR = Path(__file__).resolve().parent
WHITELIST_FILE = BASE_DIR / "ALF_whitelist.txt"
REPORT_DIR = BASE_DIR / "ALF_relatorios"

PICKLE_EXT = {".bin", ".pt", ".pth", ".pkl", ".pickle"}
SAFETENSORS_EXT = {".safetensors"}

# Ficheiros executáveis não devem vir dentro de um modelo
CODE_EXT_BLOCK = {".so", ".dll", ".exe", ".dylib", ".pyd", ".sh", ".bat", ".ps1", ".jar"}

# Módulos que nunca devem aparecer num pickle de modelo
DANGEROUS_MODULES = {
    "os", "posix", "nt", "subprocess", "sys", "socket", "shutil", "runpy",
    "importlib", "pty", "ctypes", "pickle", "_pickle", "code", "codeop",
    "webbrowser", "multiprocessing", "asyncio", "commands", "platform",
}
DANGEROUS_BUILTINS = {"eval", "exec", "compile", "__import__", "getattr", "open", "input", "breakpoint"}

# Funções conhecidas por serem usadas em payloads maliciosos em pickle.
# Qualquer destas num pickle de modelo é reprovação.
KNOWN_MALICIOUS = {
    "os.system": "executa comandos do sistema operativo",
    "os.popen": "executa comandos e lê a saída",
    "posix.system": "executa comandos do sistema (Linux/macOS)",
    "nt.system": "executa comandos do sistema (Windows)",
    "subprocess.Popen": "inicia processos",
    "subprocess.call": "inicia processos",
    "subprocess.run": "inicia processos",
    "subprocess.check_output": "inicia processos e lê a saída",
    "builtins.eval": "executa código Python arbitrário",
    "builtins.exec": "executa código Python arbitrário",
    "builtins.compile": "compila código Python arbitrário",
    "builtins.__import__": "carrega módulos arbitrários",
    "builtins.getattr": "acede a atributos dinamicamente (usado para contornar filtros)",
    "runpy.run_path": "executa um ficheiro Python",
    "socket.socket": "abre ligações de rede (possível exfiltração de dados)",
    "urllib.request.urlopen": "faz pedidos de rede",
    "pty.spawn": "abre uma shell interativa",
    "webbrowser.open": "abre URLs",
}

# Opcodes que chamam funções ao carregar o pickle.
# Normais em PyTorch: o risco avalia-se pelo que é chamado (ver imports).
CALL_OPCODES = {"REDUCE", "BUILD", "INST", "OBJ", "NEWOBJ", "NEWOBJ_EX"}

STRING_OPS = {"SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE",
              "SHORT_BINSTRING", "BINSTRING", "STRING"}

# Lista inicial de imports considerados normais em modelos PyTorch/NumPy.
# Editável em ALF_whitelist.txt. Só acrescentar o que confirmares como normal.
DEFAULT_WHITELIST = """\
# Imports normais em checkpoints PyTorch e NumPy.
# Um import por linha, formato modulo.nome. Linhas com # são comentários.
# Só acrescenta um import depois de confirmares que é normal para o modelo.
collections.OrderedDict
torch._utils._rebuild_tensor
torch._utils._rebuild_tensor_v2
torch._utils._rebuild_parameter
torch.FloatStorage
torch.HalfStorage
torch.BFloat16Storage
torch.DoubleStorage
torch.LongStorage
torch.IntStorage
torch.ShortStorage
torch.CharStorage
torch.ByteStorage
torch.BoolStorage
torch.Size
numpy.core.multiarray._reconstruct
numpy._core.multiarray._reconstruct
numpy.ndarray
numpy.dtype
"""

OK, AVISO, REPROVADO, INFO = "OK", "ATENÇÃO", "REPROVADO", "INFO"

LIMITACOES = (
    "Esta ferramenta não garante que o modelo é seguro. Faz apenas testes ao ficheiro: "
    "integridade (SHA-256), estrutura e imports em ficheiros pickle. Não analisa o "
    "comportamento do modelo. Um modelo malicioso treinado para um gatilho específico "
    "pode passar em todos os testes. Se o modelo for usado com ferramentas ou acesso a "
    "dados, executa-o isolado (contentor ou VM sem rede e sem credenciais)."
)


# ---------------------------------------------------------------- utilitários

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_whitelist() -> set:
    try:
        if not WHITELIST_FILE.exists():
            WHITELIST_FILE.write_text(DEFAULT_WHITELIST, encoding="utf-8")
        text = WHITELIST_FILE.read_text(encoding="utf-8")
    except OSError:
        text = DEFAULT_WHITELIST  # pasta só de leitura: usa a lista embutida
    items = set()
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            items.add(line)
    return items


def fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.2f} GB"


# ------------------------------------------------------ análise de pickle

def _analyse_stream(fh) -> dict:
    """Lê os opcodes sem executar nada e recolhe os globals (imports)."""
    globals_found = []
    opcodes = {}
    unresolved = 0
    recent_strings = []
    for opcode, arg, _pos in pickletools.genops(fh):
        name = opcode.name
        opcodes[name] = opcodes.get(name, 0) + 1
        if name in STRING_OPS:
            recent_strings.append(arg)
            recent_strings = recent_strings[-2:]
        elif name in ("GLOBAL", "INST"):
            mod, _, attr = str(arg).partition(" ")
            globals_found.append((mod, attr))
        elif name == "STACK_GLOBAL":
            if len(recent_strings) == 2:
                globals_found.append((recent_strings[0], recent_strings[1]))
            else:
                unresolved += 1
    return {"globals": globals_found, "opcodes": opcodes, "unresolved": unresolved}


def analyse_pickle(path: Path) -> list:
    """Devolve [(nome_interno, resultado)] para cada pickle do ficheiro."""
    results = []
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.endswith(".pkl")]
            if not names:
                raise ValueError("ZIP sem nenhum ficheiro .pkl (data.pkl)")
            for n in names:
                with z.open(n) as fh:
                    results.append((n, _analyse_stream(fh)))
    else:
        with open(path, "rb") as fh:
            results.append(("(ficheiro completo)", _analyse_stream(fh)))
    return results


def check_globals(globals_found, whitelist):
    danger, unknown = set(), set()
    for mod, attr in globals_found:
        full = f"{mod}.{attr}"
        root = mod.split(".")[0]
        if root in DANGEROUS_MODULES or full in KNOWN_MALICIOUS:
            danger.add(full)
        elif root in ("builtins", "__builtin__") and attr in DANGEROUS_BUILTINS:
            danger.add(full)
        elif full not in whitelist:
            unknown.add(full)
    return sorted(danger), sorted(unknown)


def _picklescan_library(path: Path):
    """Corre o picklescan dentro deste processo.

    É o caminho usado pelo executável empacotado (o picklescan vai lá dentro) e
    também serve quando o comando `picklescan` não está no PATH.
    """
    try:
        from picklescan import scanner
    except ImportError:
        return None
    try:
        result = scanner.scan_file_path(str(path))
    except Exception as ex:
        return 2, f"A análise não foi concluída: {ex}"
    globais = "\n".join(
        f"{g.module}.{g.name} [{getattr(g.safety, 'value', g.safety)}]" for g in result.globals
    )
    if result.scan_err:
        return 2, "picklescan não conseguiu ler uma ou mais partes do ficheiro.\n" + globais
    return (1 if result.issues_count else 0), (globais or "Nenhum global suspeito encontrado.")


def run_picklescan(path: Path):
    """Devolve (código, saída).

    0 = limpo, 1 = padrão malicioso, 2 = análise incompleta, None = indisponível.
    Tenta, por ordem: o comando `picklescan`, a biblioteca no próprio processo e
    `python -m picklescan`. Assim funciona com pip install, sem o comando no PATH,
    e dentro do .exe/.bin que já leva o picklescan embutido.
    """
    exe = shutil.which("picklescan")
    if exe:
        try:
            r = subprocess.run([exe, "--path", str(path)], capture_output=True,
                               text=True, timeout=7200,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except subprocess.TimeoutExpired:
            return 2, "picklescan excedeu o tempo limite"
        except OSError as ex:
            return 2, f"Não foi possível executar o picklescan: {ex}"
        return r.returncode, (r.stdout + r.stderr).strip()

    embutido = _picklescan_library(path)
    if embutido is not None:
        return embutido

    if not getattr(sys, "frozen", False):
        try:
            import picklescan  # noqa: F401
        except ImportError:
            pass
        else:
            try:
                r = subprocess.run([sys.executable, "-m", "picklescan", "--path", str(path)],
                                   capture_output=True, text=True, timeout=7200,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            except subprocess.TimeoutExpired:
                return 2, "picklescan excedeu o tempo limite"
            return r.returncode, (r.stdout + r.stderr).strip()

    return None, ("picklescan não está instalado (pip install picklescan==1.0.5). "
                  "As restantes verificações continuam disponíveis.")


def pickle_checks(path: Path, whitelist: set) -> list:
    checks = []

    # 1. Conteúdo do ZIP (formato PyTorch)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
        bad = [n for n in names if Path(n).suffix.lower() in CODE_EXT_BLOCK]
        if bad:
            checks.append(dict(nome="Ficheiros executáveis dentro do modelo", estado=REPROVADO,
                               detalhe="\n".join(bad[:20])))
        else:
            checks.append(dict(nome="Ficheiros executáveis dentro do modelo", estado=OK,
                               detalhe="Nenhum"))
        py = [n for n in names if n.lower().endswith(".py")]
        if py:
            checks.append(dict(nome="Código Python dentro do modelo", estado=AVISO,
                               detalhe="\n".join(py[:20]) + "\nCódigo destes ficheiros pode correr ao carregar o modelo."))

    # 2. picklescan
    code, saida = run_picklescan(path)
    if code == 0:
        checks.append(dict(nome="picklescan", estado=OK, detalhe="Nenhum padrão malicioso detetado."))
    elif code == 1:
        checks.append(dict(nome="picklescan", estado=REPROVADO,
                           detalhe="picklescan detetou padrão malicioso.\n" + saida))
    elif code is None:
        checks.append(dict(nome="picklescan", estado=AVISO, detalhe=saida))
    else:
        checks.append(dict(nome="picklescan", estado=AVISO,
                           detalhe="A análise não foi concluída (código %s).\n%s" % (code, saida)))

    # 3. Análise própria de imports (whitelist)
    try:
        streams = analyse_pickle(path)
    except Exception as e:
        checks.append(dict(nome="Leitura da estrutura pickle", estado=AVISO,
                           detalhe=f"Não foi possível ler o pickle: {e}"))
        return checks

    all_globals, unresolved = [], 0
    ops_total = {}
    for _name, r in streams:
        all_globals += r["globals"]
        unresolved += r["unresolved"]
        for k, v in r["opcodes"].items():
            ops_total[k] = ops_total.get(k, 0) + v

    danger, unknown = check_globals(all_globals, whitelist)
    if danger:
        linhas = [f"{d} - {KNOWN_MALICIOUS.get(d, 'módulo de execução, sistema ou rede')}"
                  for d in danger]
        checks.append(dict(nome="Imports perigosos", estado=REPROVADO,
                           detalhe="\n".join(linhas)))
    else:
        checks.append(dict(nome="Imports perigosos", estado=OK, detalhe="Nenhum."))

    unique_imports = sorted({f"{m}.{a}" for m, a in all_globals})
    if unknown:
        lista = unknown[:40]
        extra = f"\n(+{len(unknown) - 40} mais)" if len(unknown) > 40 else ""
        checks.append(dict(nome="Imports fora da whitelist", estado=AVISO,
                           detalhe="\n".join(lista) + extra +
                           "\nSe forem normais para este modelo, acrescenta-os a ALF_whitelist.txt."))
    else:
        checks.append(dict(nome="Imports fora da whitelist", estado=OK,
                           detalhe="Todos os imports estão na whitelist."))

    if unresolved:
        checks.append(dict(nome="Referências não resolvidas", estado=AVISO,
                           detalhe=f"{unresolved} instruções STACK_GLOBAL não puderam ser resolvidas. "
                                   "Revisão manual recomendada."))

    checks.append(dict(nome="Imports encontrados (informativo)", estado=INFO,
                       detalhe=f"{len(unique_imports)} distintos\n" + "\n".join(unique_imports[:40])))
    chamadas = {k: ops_total[k] for k in sorted(CALL_OPCODES) if ops_total.get(k)}
    if chamadas:
        suspeito = bool(unknown or unresolved or not all_globals)
        texto = ("Presentes: " + ", ".join(f"{k}={v}" for k, v in chamadas.items()) + ".\n"
                 "REDUCE e BUILD são normais em PyTorch. O risco está no que é chamado, "
                 "e a verificação de imports acima indica quais são essas funções.")
        if not all_globals:
            texto += "\nHá chamadas sem nenhum import identificado. Revisão manual recomendada."
        checks.append(dict(nome="Opcodes que chamam funções", estado=AVISO if suspeito else INFO,
                           detalhe=texto))
    checks.append(dict(nome="Opcodes (informativo)", estado=INFO,
                       detalhe=", ".join(f"{k}={v}" for k, v in sorted(ops_total.items()))))
    return checks


# ------------------------------------------------- safetensors e GGUF

def check_safetensors(path: Path):
    size = path.stat().st_size
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) < 8:
            return REPROVADO, "Ficheiro demasiado pequeno para ser safetensors."
        n = struct.unpack("<Q", raw)[0]
        if n > 100 * 1024 * 1024 or 8 + n > size:
            return REPROVADO, "Tamanho do cabeçalho inválido."
        try:
            cab = json.loads(f.read(n).decode("utf-8"))
        except Exception as e:
            return REPROVADO, f"Cabeçalho JSON inválido: {e}"
    limite = size - 8 - n
    count = 0
    for nome, info in cab.items():
        if nome == "__metadata__":
            continue
        try:
            ini, fim = info["data_offsets"]
        except Exception:
            return REPROVADO, f"Tensor '{nome}' sem data_offsets."
        if not (0 <= ini <= fim <= limite):
            return REPROVADO, f"Tensor '{nome}' fora dos limites do ficheiro."
        count += 1
    return OK, f"{count} tensores; cabeçalho e offsets coerentes."


def check_gguf(path: Path):
    with open(path, "rb") as f:
        magic = f.read(4)
        ver = f.read(4)
    if magic != b"GGUF" or len(ver) < 4:
        return REPROVADO, "Assinatura GGUF em falta."
    v = struct.unpack("<I", ver)[0]
    if v not in (2, 3):
        return AVISO, f"Versão GGUF {v} inesperada (esperado 2 ou 3)."
    return OK, f"GGUF v{v}, assinatura correta."


# ------------------------------------------------------- orquestração

def verificar(path: Path, sha_ref, whitelist: set):
    checks = []
    size = path.stat().st_size
    checks.append(dict(nome="Ficheiro", estado=INFO, detalhe=f"{path.name} ({fmt_size(size)})"))

    digest = sha256_of(path)
    checks.append(dict(nome="SHA-256 calculado", estado=INFO, detalhe=digest))
    if sha_ref:
        if digest.lower() == sha_ref.strip().lower():
            checks.append(dict(nome="Integridade (hash)", estado=OK,
                               detalhe="Confere com o hash de referência."))
        else:
            checks.append(dict(nome="Integridade (hash)", estado=REPROVADO,
                               detalhe="NÃO confere com o hash de referência. O ficheiro é diferente "
                                       "do publicado ou foi alterado."))
    else:
        checks.append(dict(nome="Integridade (hash)", estado=AVISO,
                           detalhe="Nenhum hash de referência fornecido. Sem isto não é possível "
                                   "confirmar que o ficheiro é o original."))

    with open(path, "rb") as f:
        head = f.read(4)
    ext = path.suffix.lower()
    if head == b"GGUF":
        fmt = "gguf"
    elif ext in SAFETENSORS_EXT:
        fmt = "safetensors"
    elif ext in PICKLE_EXT or zipfile.is_zipfile(path) or head[:1] == b"\x80":
        fmt = "pickle"
    else:
        fmt = "desconhecido"
    checks.append(dict(nome="Formato detetado", estado=INFO, detalhe=fmt))

    if fmt == "pickle":
        checks += pickle_checks(path, whitelist)
    elif fmt == "safetensors":
        est, det = check_safetensors(path)
        checks.append(dict(nome="Estrutura safetensors", estado=est, detalhe=det))
    elif fmt == "gguf":
        est, det = check_gguf(path)
        checks.append(dict(nome="Estrutura GGUF", estado=est, detalhe=det))
    else:
        checks.append(dict(nome="Formato", estado=AVISO,
                           detalhe="Formato não reconhecido. O ficheiro não foi analisado."))
    return checks, digest


def veredicto(checks):
    estados = {c["estado"] for c in checks}
    if REPROVADO in estados:
        return "VERMELHO", "NÃO USAR", 2
    if AVISO in estados:
        return "AMARELO", "USAR SÓ DEPOIS DE REVISÃO MANUAL", 1
    return "VERDE", "APROVADO COM LIMITAÇÕES", 0


# ------------------------------------------------------------- relatório

CSS_CLASS = {OK: "ok", AVISO: "aviso", REPROVADO: "erro", INFO: "info"}
COR = {"VERDE": "#2e7d32", "AMARELO": "#b28704", "VERMELHO": "#c62828"}


def gerar_relatorio(path: Path, digest: str, label: str, acao: str, checks: list) -> Path:
    REPORT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = REPORT_DIR / f"relatorio_{path.stem}_{stamp}.html"
    e = html.escape
    linhas = "".join(
        f"<tr><td>{e(c['nome'])}</td>"
        f"<td class='{CSS_CLASS[c['estado']]}'>{e(c['estado'])}</td>"
        f"<td><pre>{e(c['detalhe'])}</pre></td></tr>"
        for c in checks
    )
    doc = f"""<!doctype html>
<html lang="pt-PT"><head><meta charset="utf-8">
<title>ALF_picklescan - {e(path.name)}</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 980px; margin: 2rem auto; padding: 0 1rem; color: #222; }}
.veredicto {{ color: #fff; background: {COR[label]}; padding: 1rem 1.2rem; border-radius: 8px; font-size: 1.2rem; }}
.limites {{ background: #f4f4f4; padding: .8rem 1rem; border-radius: 8px; margin-top: 1rem; font-size: .92rem; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
td, th {{ border: 1px solid #ddd; padding: .5rem; vertical-align: top; text-align: left; }}
pre {{ white-space: pre-wrap; word-break: break-all; margin: 0; font-size: .85rem; }}
.ok {{ color: #2e7d32; font-weight: bold; }} .aviso {{ color: #b28704; font-weight: bold; }}
.erro {{ color: #c62828; font-weight: bold; }} .info {{ color: #555; }}
</style></head><body>
<h1>ALF_picklescan - Relatório de verificação</h1>
<p><strong>Ficheiro:</strong> {e(path.name)}<br>
<strong>Data:</strong> {datetime.now().strftime('%d/%m/%Y %H:%M')}<br>
<strong>SHA-256:</strong> <code>{e(digest)}</code></p>
<div class="veredicto">{e(label)} - {e(acao)}</div>
<div class="limites"><strong>Limitações:</strong> {e(LIMITACOES)}</div>
<table><tr><th>Verificação</th><th>Estado</th><th>Detalhe</th></tr>{linhas}</table>
<p style="color:#777;font-size:.8rem">ALF_picklescan v{VERSION}</p>
</body></html>"""
    out.write_text(doc, encoding="utf-8")
    return out


# --------------------------------------------------------------- consola

def mostrar(checks, label, acao, report):
    simbolo = {OK: "[OK]     ", AVISO: "[AVISO]  ", REPROVADO: "[REPROV.] ", INFO: "[INFO]   "}
    print()
    for c in checks:
        print(f"{simbolo[c['estado']]}{c['nome']}")
        for linha in c["detalhe"].splitlines()[:6]:
            print(f"            {linha}")
    print()
    print("=" * 60)
    print(f" RESULTADO: {label} - {acao}")
    print("=" * 60)
    print(f" Relatório: {report}")
    print()
    print(LIMITACOES)


def pedir_entrada():
    print(f"ALF_picklescan v{VERSION}")
    print("-" * 40)
    ficheiro = input("Caminho do modelo (arrasta o ficheiro para aqui): ").strip().strip('"').strip("'")
    sha = input("SHA-256 publicado pelo autor (Enter para saltar): ").strip() or None
    return ficheiro, sha


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Verifica modelos de IA antes de os usar.")
    ap.add_argument("ficheiro", nargs="?", help="Caminho do modelo (.bin, .pt, .safetensors, .gguf, ...)")
    ap.add_argument("--sha256", default=None, help="SHA-256 publicado pelo autor do modelo")
    args = ap.parse_args()

    interativo = args.ficheiro is None
    if interativo:
        ficheiro, sha = pedir_entrada()
    else:
        ficheiro, sha = args.ficheiro, args.sha256

    path = Path(ficheiro).expanduser()
    if not path.is_file():
        print(f"Erro: ficheiro não encontrado: {path}")
        return 3

    print("\nA analisar... (modelos grandes podem demorar vários minutos)")
    try:
        whitelist = load_whitelist()
        checks, digest = verificar(path, sha, whitelist)
    except Exception as ex:
        print(f"Erro inesperado durante a análise: {ex}")
        return 3

    label, acao, code = veredicto(checks)
    report = gerar_relatorio(path, digest, label, acao, checks)
    mostrar(checks, label, acao, report)

    if interativo and sys.stdin.isatty():
        input("\nPrima Enter para fechar...")
    return code


if __name__ == "__main__":
    sys.exit(main())
