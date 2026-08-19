#!/usr/bin/env python3
"""인스턴스별 포트 배정 (Act 5 · C안 — 도메인당 콘솔 N대).

install.sh가 venv 前에 시스템 python으로 호출하므로 **stdlib 전용**
(scripts/parse_preset.py와 같은 제약).

한 호스트에 콘솔 N대를 세우면 server(8443)·Qdrant(6333)·llama-server(8181~3)가
전부 충돌한다. 지식 컬렉션명은 "bge"로 고정이라 Qdrant를 공유할 수 없고,
인스턴스마다 자기 Qdrant를 자기 포트로 띄워야 한다(스토리지는 클론 디렉토리별
./storage 로 자연 분리).

레지스트리(~/.ai-console/instances.tsv)에 인스턴스↔포트를 남기는 이유:
LISTEN 검사만으로는 **중지된** 형제 인스턴스의 포트를 비어 있다고 오판한다.
재실행 시에는 기록된 포트를 그대로 돌려준다(멱등 — 재설치가 포트를 흔들지 않음).

usage: alloc_ports.py <instance> <root>
출력:  PORT_server=8443 … 한 줄씩 (install.sh가 export)
"""
from __future__ import annotations

import os
import re
import socket
import sys

# 역할 → 기본 포트. config.ini.template의 값과 일치시킬 것.
#
# qdrant_grpc 만 config에 대응 섹션이 없다(install.sh의 CONFIG_ROLES 참고).
# Qdrant는 HTTP와 별개로 gRPC를 6334 고정으로 연다 — 실측: 두 번째 인스턴스가
# "Error while starting gRPC server: transport error" 를 남긴다. 이 콘솔은 gRPC를
# 쓰지 않아(코드에 grpc 참조 0건) 치명적이진 않지만, 포트를 배정해 잡음을 없앤다.
ROLES = [
    ("server", 8443),
    ("qdrant", 6333),
    ("qdrant_grpc", 6334),
    ("llama_server", 8181),
    ("llama_server_translation", 8182),
    ("llama_server_gemma", 8183),
]

REGISTRY = os.path.join(
    os.environ.get("AI_CONSOLE_HOME", os.path.expanduser("~/.ai-console")),
    "instances.tsv",
)
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def read_registry() -> list[dict]:
    if not os.path.isfile(REGISTRY):
        return []
    rows = []
    with open(REGISTRY, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            name, root, portspec = parts[0], parts[1], parts[2]
            ports = {}
            for pair in portspec.split(","):
                k, _, v = pair.partition("=")
                if v.isdigit():
                    ports[k] = int(v)
            rows.append({"name": name, "root": root, "ports": ports})
    return rows


def write_registry(rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(REGISTRY), exist_ok=True)
    tmp = REGISTRY + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("# ai-console 인스턴스 레지스트리 — name\troot\trole=port,…\n")
        f.write("# install.sh(scripts/alloc_ports.py)가 관리. 인스턴스 제거 시 해당 줄 삭제.\n")
        for r in rows:
            spec = ",".join(f"{k}={v}" for k, v in r["ports"].items())
            f.write(f"{r['name']}\t{r['root']}\t{spec}\n")
    os.replace(tmp, REGISTRY)


def ports_from_config(root: str) -> dict:
    """기존 설치본의 config.ini 포트를 씨앗으로 채택.

    레지스트리 도입 前에 설치된 콘솔은 기록이 없다. 그대로 신규 배정하면
    자기 자신이 쓰는 포트를 '사용 중'으로 보고 옆으로 비켜서(8443→8444)
    config와 어긋난다. 업그레이드 경로를 위해 config를 먼저 읽는다.
    """
    path = os.path.join(root, "config.ini")
    if not os.path.isfile(path):
        return {}
    import configparser
    cp = configparser.ConfigParser()
    try:
        cp.read(path, encoding="utf-8")
    except configparser.Error:
        return {}
    out = {}
    for role, _base in ROLES:
        v = cp.get(role, "port", fallback="").strip()
        if v.isdigit():
            out[role] = int(v)
    # gRPC는 config에 섹션이 없다. 기존 인스턴스는 Qdrant 기본 규약대로 HTTP+1 을
    # 이미 쓰고 있으므로 그 값을 승계한다 — 안 그러면 자기 gRPC 포트를 '사용 중'으로
    # 보고 옆으로 밀려 실제 기동 포트와 어긋난다.
    if "qdrant" in out:
        out.setdefault("qdrant_grpc", out["qdrant"] + 1)
    return out


def in_use(port: int) -> bool:
    """누가 듣고 있으면 True. connect 성공(=응답) 또는 bind 실패 둘 중 하나면 사용 중."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        if s.connect_ex(("127.0.0.1", port)) == 0:
            return True
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("0.0.0.0", port))
    except OSError:
        return True
    return False


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: alloc_ports.py <instance> <root>", file=sys.stderr)
        return 2
    name, root = sys.argv[1], os.path.abspath(sys.argv[2])
    if not NAME_RE.match(name):
        print(f"인스턴스 이름이 올바르지 않습니다: {name!r} "
              "(영숫자로 시작, 이후 영숫자·. _ - 만)", file=sys.stderr)
        return 2

    rows = read_registry()
    mine = next((r for r in rows if r["name"] == name), None)

    # 한 디렉토리를 두 인스턴스명으로 등록하면 포트·카트리지·Qdrant가 뒤엉킨다.
    # (인스턴스 = 클론 1개가 C안의 전제 — docs/multi-instance.md)
    clash = next((r for r in rows if r["root"] == root and r["name"] != name), None)
    if clash:
        print(f"이 디렉토리는 이미 인스턴스 '{clash['name']}' 로 등록돼 있습니다: {root}\n"
              f"인스턴스를 새로 세우려면 별도 클론을 쓰세요. "
              f"이름만 바꾸려면 {REGISTRY} 의 해당 줄을 지우고 다시 실행하세요.",
              file=sys.stderr)
        return 1

    if mine and all(role in mine["ports"] for role, _ in ROLES):
        # 재설치 — 기록된 포트 유지. root만 갱신(클론을 옮겼을 수 있음).
        mine["root"] = root
        ports = mine["ports"]
        reused = True
    else:
        reserved = {p for r in rows if r["name"] != name for p in r["ports"].values()}
        seed = {k: v for k, v in ports_from_config(root).items() if v not in reserved}
        ports, reused = {}, False
        for role, base in ROLES:
            if role in seed:
                ports[role] = seed[role]      # 기존 설치본 — 쓰던 포트 유지
                continue
            p = base
            while p in reserved or p in ports.values() or in_use(p):
                p += 1
                if p > base + 200:
                    print(f"{role}: {base}~{base+200} 범위에 빈 포트가 없습니다", file=sys.stderr)
                    return 1
            ports[role] = p
        if mine:
            mine["root"], mine["ports"] = root, ports
        else:
            rows.append({"name": name, "root": root, "ports": ports})

    write_registry(rows)

    for role, _ in ROLES:
        print(f"PORT_{role}={ports[role]}")
    print(f"INSTANCE_REUSED={'1' if reused else '0'}")
    print(f"INSTANCE_REGISTRY={REGISTRY}")
    print(f"INSTANCE_COUNT={len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
