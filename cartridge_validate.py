"""cartridge.yaml + knowledge/*.yaml 경량 스키마 검증 (Phase 4).

jsonschema 의존성 없이 PyYAML 만으로 수동 검증한다.
스펙: docs/cartridge-mount-design.md "스키마 검증 (validate) — UNION 스키마" 절.
지식 doc 검증의 권위는 README 가 아니라 엔진 화이트리스트
(aibot_prompts_functions.py:475 save_the_yaml).

읽기 전용 · 부작용 없음. aibot_cli.py 의 `cartridge validate` 가 소비한다.
"""
from __future__ import annotations

import os
import re

import yaml

# name: kebab-case
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,40}$")
# prompts 슬롯 — ⓐ 구조형 / ⓑ 단일형
STRUCTURED_SLOTS = ("intent", "qna", "action", "plan", "playbook")
SINGLE_SLOT = "system"

OK, WARN, ERROR = "ok", "warn", "error"
_MAX_EXAMPLES = 5


class Report:
    """검사 결과(✓/⚠/✗) 누적기."""

    def __init__(self) -> None:
        self.checks: list[dict] = []

    def ok(self, label: str, detail: str = "") -> None:
        self.checks.append({"status": OK, "label": label, "detail": detail})

    def warn(self, label: str, detail: str = "") -> None:
        self.checks.append({"status": WARN, "label": label, "detail": detail})

    def error(self, label: str, detail: str = "") -> None:
        self.checks.append({"status": ERROR, "label": label, "detail": detail})

    @property
    def n_errors(self) -> int:
        return sum(1 for c in self.checks if c["status"] == ERROR)

    @property
    def n_warnings(self) -> int:
        return sum(1 for c in self.checks if c["status"] == WARN)

    def as_dict(self, path: str) -> dict:
        return {
            "path": path,
            "ok": self.n_errors == 0,
            "errors": self.n_errors,
            "warnings": self.n_warnings,
            "checks": self.checks,
        }


def validate_cartridge(path: str) -> Report:
    """카트리지 디렉토리(또는 cartridge.yaml 경로)를 검증해 Report 반환."""
    rep = Report()
    target = os.path.abspath(path)
    if os.path.isfile(target) and target.endswith((".yaml", ".yml")):
        manifest_path = target
        cart_dir = os.path.dirname(target)
    else:
        cart_dir = target
        manifest_path = os.path.join(cart_dir, "cartridge.yaml")

    if not os.path.isdir(cart_dir):
        rep.error("카트리지 디렉토리", f"디렉토리 없음: {cart_dir}")
        return rep
    if not os.path.isfile(manifest_path):
        rep.error("cartridge.yaml", f"매니페스트 없음: {manifest_path}")
        return rep

    try:
        with open(manifest_path, encoding="utf-8") as f:
            manifest = yaml.safe_load(f)
    except (yaml.YAMLError, OSError) as e:
        rep.error("cartridge.yaml 파싱", str(e))
        return rep
    if not isinstance(manifest, dict):
        rep.error("cartridge.yaml 구조", "최상위가 맵(map)이 아님")
        return rep

    _check_name(rep, manifest)
    _check_version(rep, manifest)
    _check_description(rep, manifest)
    _check_prompts(rep, manifest, cart_dir)
    _check_knowledge(rep, manifest, cart_dir)
    _check_model(rep, manifest)
    _check_plugins(rep, manifest, cart_dir)
    return rep


def _check_name(rep: Report, m: dict) -> None:
    name = m.get("name")
    if name is None:
        rep.error("name", "필수 필드 누락")
    elif not isinstance(name, str):
        rep.error("name", f"문자열이어야 함 (got {type(name).__name__})")
    elif not NAME_RE.match(name):
        rep.error("name", f"kebab-case 위반 '^[a-z0-9][a-z0-9-]{{1,40}}$': {name!r}")
    else:
        rep.ok("name", name)


def _check_version(rep: Report, m: dict) -> None:
    v = m.get("version")
    if v is None:
        rep.error("version", "필수 필드 누락")
    elif not isinstance(v, str):
        # YAML 에서 1.0 은 float 로 파싱됨 — 문자열(따옴표) 요구
        rep.error("version", f"문자열이어야 함 (따옴표 필요, got {type(v).__name__}: {v!r})")
    else:
        rep.ok("version", v)


def _check_description(rep: Report, m: dict) -> None:
    if "description" in m and not isinstance(m["description"], str):
        rep.warn("description", "문자열 권장")


def _check_prompts(rep: Report, m: dict, cart_dir: str) -> None:
    prompts = m.get("prompts")
    if prompts is None:
        rep.error("prompts", "필수 필드 누락")
        return
    if not isinstance(prompts, dict):
        rep.error("prompts", f"맵이어야 함 (got {type(prompts).__name__})")
        return
    if not prompts:
        rep.error("prompts", "비어 있음 — 슬롯 최소 1개 필요")
        return

    recognized = set(STRUCTURED_SLOTS) | {SINGLE_SLOT}
    present = [k for k in prompts if k in recognized]
    if not present:
        rep.error(
            "prompts",
            f"인식되는 슬롯 없음 (허용: {', '.join(STRUCTURED_SLOTS)} 또는 {SINGLE_SLOT})",
        )
    else:
        shape = "단일형(system)" if SINGLE_SLOT in prompts else "구조형"
        rep.ok("prompts 형태", f"{shape} · 슬롯 {', '.join(present)}")

    for k in prompts:
        if k not in recognized:
            rep.warn("prompts 미지 슬롯", f"{k} (무시됨)")

    for slot in present:
        val = prompts[slot]
        if not isinstance(val, str):
            rep.error(f"prompts.{slot}", f"경로 문자열이어야 함 (got {type(val).__name__})")
            continue
        if os.path.isfile(os.path.join(cart_dir, val)):
            rep.ok(f"prompts.{slot}", val)
        else:
            rep.error(f"prompts.{slot}", f"파일 없음: {val}")


def _check_knowledge(rep: Report, m: dict, cart_dir: str) -> None:
    kn = m.get("knowledge")
    if kn is None:
        return
    if not isinstance(kn, dict):
        rep.error("knowledge", f"맵이어야 함 (got {type(kn).__name__})")
        return
    d = kn.get("dir")
    if d is None:
        rep.error("knowledge.dir", "knowledge 있으면 dir 필수")
        return
    if not isinstance(d, str):
        rep.error("knowledge.dir", f"경로 문자열이어야 함 (got {type(d).__name__})")
        return
    kn_dir = os.path.join(cart_dir, d)
    if not os.path.isdir(kn_dir):
        rep.error("knowledge.dir", f"디렉토리 없음: {d}")
        return
    rep.ok("knowledge.dir", d)

    bd = kn.get("breakdown")
    if bd is not None:
        if not isinstance(bd, dict):
            rep.warn("knowledge.breakdown", "정수 맵 권장")
        else:
            bad = [str(k) for k, v in bd.items() if not isinstance(v, int)]
            if bad:
                rep.warn("knowledge.breakdown", f"정수 아님: {', '.join(bad)}")

    _check_knowledge_docs(rep, kn_dir)


def _check_knowledge_docs(rep: Report, kn_dir: str) -> None:
    """knowledge/**/*.yaml 을 엔진 화이트리스트로 검증. 건수만 보고."""
    yaml_files = [
        os.path.join(root, fn)
        for root, _dirs, files in os.walk(kn_dir)
        for fn in files
        if fn.endswith((".yaml", ".yml"))
    ]
    total = len(yaml_files)
    if total == 0:
        rep.warn("knowledge docs", "YAML 문서 0개")
        return

    counts = {"parse": 0, "missing_q": 0, "missing_a": 0, "bad_list": 0, "disabled": 0}
    examples: dict[str, list[str]] = {k: [] for k in counts}

    def note(cat: str, rel: str) -> None:
        counts[cat] += 1
        if len(examples[cat]) < _MAX_EXAMPLES:
            examples[cat].append(rel)

    for fp in yaml_files:
        rel = os.path.relpath(fp, kn_dir)
        try:
            with open(fp, encoding="utf-8") as f:
                doc = yaml.safe_load(f)
        except (yaml.YAMLError, OSError):
            note("parse", rel)
            continue
        if not isinstance(doc, dict):
            note("parse", f"{rel} (맵 아님)")
            continue
        if doc.get("enabled") is False:
            # 의도적으로 꺼둔 문서 — 검색에서 제외된다(Qdrant enabled 필터). 내용 검사 대상 아님.
            note("disabled", rel)
            continue
        if not doc.get("question"):
            note("missing_q", rel)
        # 답변 콘텐츠는 answer 또는 spec 이다. action 형식은 쿼리 명세(spec)가 답이며
        # 임베딩 파이프라인도 spec 을 '명세'로 동등 색인한다(aibot_embedding.py 필드 표).
        # answer 만 보면 정상 action 문서 전부가 오탐으로 잡힌다.
        if not doc.get("answer") and not doc.get("spec"):
            note("missing_a", rel)
        for key in ("aliases", "tags"):
            v = doc.get(key)
            if v is not None and not isinstance(v, list):
                note("bad_list", f"{rel}:{key}")

    scanned = f"{total}개 스캔"
    if counts["disabled"]:
        scanned += f" (enabled:false {counts['disabled']}개 제외)"
    rep.ok("knowledge docs", scanned)
    if counts["parse"]:
        rep.error("knowledge docs 파싱 실패", _fmt(counts["parse"], examples["parse"]))
    if counts["missing_q"]:
        rep.error("knowledge docs question 누락", _fmt(counts["missing_q"], examples["missing_q"]))
    if counts["bad_list"]:
        rep.error("knowledge docs aliases/tags 타입", _fmt(counts["bad_list"], examples["bad_list"]))
    if counts["missing_a"]:
        rep.warn("knowledge docs 답변 콘텐츠 없음 (answer·spec 둘 다 없음)",
                 _fmt(counts["missing_a"], examples["missing_a"]))


def _fmt(n: int, exs: list[str]) -> str:
    tail = " …" if n > len(exs) else ""
    return f"{n}건 (예: {', '.join(exs)}{tail})"


def _check_model(rep: Report, m: dict) -> None:
    model = m.get("model")
    if model is None:
        return
    if not isinstance(model, dict):
        rep.error("model", f"맵이어야 함 (got {type(model).__name__})")
        return
    rec = model.get("recommended")
    if rec is None:
        rep.error("model.recommended", "model 있으면 recommended 필수")
    elif not isinstance(rec, str):
        rep.error("model.recommended", f"문자열이어야 함 (got {type(rec).__name__})")
    else:
        rep.ok("model.recommended", rec)
    for opt in ("minimum_tier", "notes"):
        if opt in model and not isinstance(model[opt], str):
            rep.warn(f"model.{opt}", "문자열 권장")


def _check_plugins(rep: Report, m: dict, cart_dir: str) -> None:
    if "plugins" not in m:
        return
    plugins = m.get("plugins")
    if plugins is None or plugins == []:
        rep.ok("plugins", "없음")
        return
    if not isinstance(plugins, list):
        rep.error("plugins", f"리스트여야 함 (got {type(plugins).__name__})")
        return
    for i, p in enumerate(plugins):
        if not isinstance(p, dict):
            rep.error(f"plugins[{i}]", "객체(map)여야 함")
            continue
        name = p.get("name")
        pth = p.get("path")
        if not name or not isinstance(name, str):
            rep.error(f"plugins[{i}].name", "필수 (문자열)")
        if not pth or not isinstance(pth, str):
            rep.error(f"plugins[{i}].path", "필수 (문자열)")
            continue
        label = name if isinstance(name, str) else f"plugins[{i}]"
        # path 는 카트리지 밖(공유 플러그인)을 가리킬 수 있음 → 부재는 경고
        if os.path.exists(os.path.join(cart_dir, pth)):
            rep.ok(f"plugin {label}", pth)
        else:
            rep.warn(f"plugin {label}", f"경로 없음: {pth}")
