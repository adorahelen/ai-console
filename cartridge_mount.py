"""카트리지 장착/해제 (Phase 4). cartridge_validate 위에 얹는다.

설계: docs/cartridge-mount-design.md (A안 — 도메인당 콘솔 1대·전역 장착).
- config 패치는 configparser read-modify-write. install.sh가 이미 configparser로 config를
  생성하므로(주석 없음) 다른 섹션([openai] 등)은 그대로 보존된다.
- 장착 상태는 config 밖 cartridges/.mounted.json (gitignore) — 재설치가 장착을 깨지 않도록
  install.sh가 이 파일로 재적용(apply_state_to_config는 그 공용 함수).
- v1 범위: 프롬프트 슬롯 배선 + 지식 bulk 업로드 + 상태 기록. [model] 핸들러 변경(서버
  재기동=install 영역)과 플러그인 module 자동유추(이름 불일치)는 경고만.
"""
from __future__ import annotations

import configparser
import glob
import json
import os
import time

import cartridge_validate
from config_utils import qdrant_collection

STATE_PATH = "cartridges/.mounted.json"
CONFIG_PATH = "config.ini"

# 지식 컬렉션명 — aibot_rag_module_BGE / aibot_embedding_BGE 가 쓰는 고정값("bge").
# purge는 이 컬렉션을 통째로 지운다(추적 guid 단위가 아님).
KNOWLEDGE_COLLECTION = qdrant_collection()

MODELS_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models.yaml")


def _preset_handler(preset: str):
    """models.yaml 의 프리셋명을 핸들러 키로 변환. 못 찾으면 None."""
    try:
        import yaml
        with open(MODELS_YAML, encoding="utf-8") as f:
            spec = yaml.safe_load(f) or {}
        entry = (spec.get("presets") or {}).get(preset) or {}
        return entry.get("handler")
    except Exception:
        return None

# 슬롯 번역표 — cartridge 일반명 슬롯 → 핸들러별 config [prompts] 키.
# 각 핸들러가 실제 읽는 키를 handler_*.py 에서 확인해 매핑(intent는 IntentAnalyzer 공통).
SLOT_MAP = {
    "gemma": {
        "intent": "intent", "qna": "qna_gemma", "action": "action_gemma",
        "plan": "plan_gpt", "playbook": "playbook", "system": "gemma",
    },
    "gpt-oss": {
        "intent": "intent", "qna": "qna_gpt", "action": "action_gpt",
        "plan": "plan_gpt", "playbook": "playbook", "system": "gpt_oss",
    },
    "llama": {
        "intent": "intent", "qna": "qna_cve", "action": "action",
        "plan": "plan", "system": "llama",
    },
}


class MountError(Exception):
    pass


# ─────────────────────────────────────────────────────────────
# config / state IO
# ─────────────────────────────────────────────────────────────
def _load_config(config_path: str) -> configparser.ConfigParser:
    cp = configparser.ConfigParser()
    if not os.path.isfile(config_path):
        raise MountError(f"config.ini 없음: {config_path} (설치 후 장착)")
    cp.read(config_path, encoding="utf-8")
    return cp


def active_handler(config_path: str = CONFIG_PATH) -> str:
    return _load_config(config_path).get("model", "model", fallback="gemma").strip()


def qdrant_endpoint(config_path: str = CONFIG_PATH) -> str:
    """config [qdrant] → Qdrant REST 베이스 URL. purge는 콘솔을 거치지 않고 여기 직접 친다."""
    cp = _load_config(config_path)
    host = cp.get("qdrant", "host", fallback="localhost").strip()
    port = cp.get("qdrant", "port", fallback="6333").strip()
    return f"http://{host}:{port}"


def read_state(state_path: str = STATE_PATH) -> dict | None:
    if not os.path.isfile(state_path):
        return None
    with open(state_path, encoding="utf-8") as f:
        return json.load(f)


def _write_state(state: dict, state_path: str = STATE_PATH) -> None:
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=1)


# ─────────────────────────────────────────────────────────────
# 슬롯 번역 + 장착 계획
# ─────────────────────────────────────────────────────────────
def _cart_dir(path: str) -> tuple[str, dict]:
    """카트리지 경로 → (cwd 기준 상대 디렉토리, manifest dict)."""
    target = os.path.abspath(path)
    if os.path.isfile(target):
        cart_abs, manifest_path = os.path.dirname(target), target
    else:
        cart_abs, manifest_path = target, os.path.join(target, "cartridge.yaml")
    import yaml
    with open(manifest_path, encoding="utf-8") as f:
        manifest = yaml.safe_load(f)
    rel = os.path.relpath(cart_abs, os.getcwd())
    return rel, manifest


def plan_mount(path: str, config_path: str = CONFIG_PATH) -> dict:
    """장착 계획 수립(부작용 없음). validate 실패면 MountError."""
    rep = cartridge_validate.validate_cartridge(path)
    if rep.n_errors:
        raise MountError(f"validate 실패 — 에러 {rep.n_errors}건. `cartridge validate {path}` 확인 후 장착")

    cart_rel, manifest = _cart_dir(path)
    handler = active_handler(config_path)
    table = SLOT_MAP.get(handler)
    if table is None:
        raise MountError(
            f"핸들러 '{handler}'의 슬롯 번역표 미정 (지원: {', '.join(sorted(SLOT_MAP))}). "
            f"지원 핸들러 프리셋에서 장착하세요."
        )

    # 프롬프트 슬롯 → config [prompts] 키 = cwd 기준 상대경로(콘솔은 repo 루트에서 실행)
    prompts = manifest.get("prompts", {}) or {}
    wiring: dict[str, str] = {}
    dropped: list[str] = []
    for slot, relpath in prompts.items():
        key = table.get(slot)
        if key and isinstance(relpath, str):
            wiring[key] = os.path.join(cart_rel, relpath).replace(os.sep, "/")
        elif isinstance(relpath, str):
            # 핸들러가 지원하지 않는 슬롯 — 조용히 버리면 해당 intent가 소리 없이 죽는다.
            # 예: llama 는 playbook 미지원(handler_llama 가 그 키를 읽지 않음).
            dropped.append(slot)

    warnings: list[str] = []
    if dropped:
        warnings.append(
            f"핸들러 '{handler}'가 지원하지 않는 슬롯 {len(dropped)}개 미배선: {', '.join(sorted(dropped))} "
            f"— 해당 intent는 카트리지 프롬프트 없이 동작합니다"
        )
    # [model] 핸들러 정합성 — 강제 변경 안 함(서버 재기동 필요=install 영역)
    # 프리셋명과 핸들러명은 서로 다른 네임스페이스다. models.yaml 로 프리셋→핸들러를 풀어
    # "실제로 다를 때만" 경고한다. [2026-08-19] 이전 판은 recommended 가 있기만 하면
    # 무조건 경고해서, 기본 카트리지를 기본 프리셋에 장착해도 매번 경고가 떴다.
    rec = (manifest.get("model") or {}).get("recommended")
    if rec:
        rec_handler = _preset_handler(rec)
        if rec_handler is None:
            warnings.append(
                f"권장 프리셋 '{rec}' 를 models.yaml 에서 찾을 수 없습니다 — 오타이거나 삭제된 프리셋"
            )
        elif rec_handler != handler:
            warnings.append(
                f"권장 프리셋 '{rec}'(핸들러 '{rec_handler}') ≠ 현재 핸들러 '{handler}' "
                f"— 모델 전환은 install 영역입니다(mount 는 [model] 미변경)"
            )
    # 플러그인 — module 이름 자동유추 불가(카트리지 name≠config plugin_module)
    plugins = manifest.get("plugins") or []
    if plugins:
        warnings.append(
            f"플러그인 {len(plugins)}개 — [validation] plugin_module 수동 설정 필요(자동유추 불가)"
        )

    kn = manifest.get("knowledge") or {}
    kn_dir = os.path.join(cart_rel, kn.get("dir", "knowledge")) if kn else None

    return {
        "cartridge": manifest.get("name"),
        "handler": handler,
        "prompts": wiring,          # config_key → path
        "knowledge_dir": kn_dir,
        "warnings": warnings,
    }


# ─────────────────────────────────────────────────────────────
# 공용: 상태 → config 적용 (mount와 install.sh가 함께 호출)
# ─────────────────────────────────────────────────────────────
def rewire_state_for_handler(state: dict, config_path: str = CONFIG_PATH) -> dict:
    """저장된 장착 상태를 **현재 [model] 핸들러 기준으로** 재배선한 상태를 돌려준다.

    프리셋을 바꾸면(install.sh --preset …) 핸들러가 바뀌는데, `.mounted.json` 의 배선은
    장착 당시 핸들러의 config 키(예 llama)로 굳어 있다. 그대로 재적용하면 새 핸들러가
    읽는 키(예 gemma)는 순정 프롬프트인 채로 남고, 그런데도 `cartridge status` 는
    "장착됨"이라고 보고한다 — 카트리지가 죽은 걸 아무도 모르는 상태가 된다.

    재계산에 실패하면(카트리지 디렉터리 이동·삭제 등) 원본 상태를 그대로 돌려주고
    warnings 에 사유를 남긴다 — 재적용 자체를 실패시키지는 않는다.
    """
    stored_handler = state.get("handler")
    current = active_handler(config_path)
    if not stored_handler or stored_handler == current:
        return state

    kn_dir = state.get("knowledge_dir") or ""
    cart_dir = os.path.dirname(kn_dir.rstrip("/")) if kn_dir else ""
    if not cart_dir or not os.path.isdir(cart_dir):
        out = dict(state)
        out["warnings"] = list(state.get("warnings") or []) + [
            f"핸들러가 '{stored_handler}'→'{current}' 로 바뀌었으나 카트리지 경로를 찾지 못해 "
            f"재배선하지 못했습니다 — `aibotctl cartridge unmount` 후 다시 mount 하세요"
        ]
        return out

    try:
        plan = plan_mount(cart_dir, config_path)
    except MountError as e:
        out = dict(state)
        out["warnings"] = list(state.get("warnings") or []) + [
            f"핸들러 '{stored_handler}'→'{current}' 재배선 실패({e}) — 다시 mount 하세요"
        ]
        return out

    out = dict(state)
    out["handler"] = plan["handler"]
    out["prompts"] = plan["prompts"]
    # 이전 핸들러 키에 남은 배선은 되돌린다 — 안 그러면 죽은 배선이 config 에 남는다
    out["_stale_prompt_keys"] = [
        k for k in (state.get("prompts") or {}) if k not in plan["prompts"]
    ]
    out["warnings"] = list(plan.get("warnings") or []) + [
        f"핸들러 변경 감지 ('{stored_handler}' → '{current}') — [prompts] 배선을 다시 계산했습니다"
    ]
    return out


def apply_state_to_config(state: dict, config_path: str = CONFIG_PATH) -> None:
    """mount 상태의 [prompts] 배선 + [paths] 지식 루트를 config.ini에 적용. 다른 섹션은 보존."""
    cp = _load_config(config_path)
    if not cp.has_section("prompts"):
        cp.add_section("prompts")

    # 이전 핸들러 키에 남은 죽은 배선을 template 기본값으로 되돌린다
    stale = state.get("_stale_prompt_keys") or []
    if stale:
        tmpl = configparser.ConfigParser()
        tmpl.read("config.ini.template", encoding="utf-8")
        for key in stale:
            if tmpl.has_option("prompts", key):
                cp.set("prompts", key, tmpl.get("prompts", key))

    for key, path in (state.get("prompts") or {}).items():
        cp.set("prompts", key, path)

    # 지식 루트도 카트리지를 따라가야 한다. 이걸 안 옮기면 config 의 aibot_docs_dir 가
    # 원본 운영 서버 레이아웃 `docs/aibot/yaml/` 을 계속 가리키고, 그 경로를 읽는 쪽
    # (회귀 심판의 load_context 등)이 **장착한 카트리지가 아닌 다른 지식으로 채점**한다.
    # 충실성(환각) 축은 근거 대조가 전부라 이 어긋남이 곧 오채점이다.
    kn_dir = state.get("knowledge_dir")
    if kn_dir:
        if not cp.has_section("paths"):
            cp.add_section("paths")
        cp.set("paths", "aibot_docs_dir", str(kn_dir).replace(os.sep, "/"))

    with open(config_path, "w", encoding="utf-8") as f:
        cp.write(f)


def _restore_config_prompts(state: dict, config_path: str = CONFIG_PATH) -> None:
    """해제 — 장착이 바꾼 [prompts] 키를 template 기본값으로 되돌린다."""
    tmpl = configparser.ConfigParser()
    tmpl.read("config.ini.template", encoding="utf-8")
    cp = _load_config(config_path)
    for key in (state.get("prompts") or {}):
        if tmpl.has_option("prompts", key):
            cp.set("prompts", key, tmpl.get("prompts", key))
    # 지식 루트도 함께 되돌린다 (apply_state_to_config 와 대칭)
    if state.get("knowledge_dir") and tmpl.has_option("paths", "aibot_docs_dir"):
        cp.set("paths", "aibot_docs_dir", tmpl.get("paths", "aibot_docs_dir"))
    with open(config_path, "w", encoding="utf-8") as f:
        cp.write(f)


# ─────────────────────────────────────────────────────────────
# 지식 업로드/삭제 (콘솔 필요)
# ─────────────────────────────────────────────────────────────
def _knowledge_files(kn_dir: str) -> list[str]:
    return sorted(
        f for f in glob.glob(os.path.join(kn_dir, "**", "*.yaml"), recursive=True)
    )


def upload_knowledge(kn_dir: str, base_url: str, api_key: str) -> list[str]:
    """POST /api/ai/prompts/bulk — 반환 guid 목록. requests 사용."""
    import requests
    files_meta = _knowledge_files(kn_dir)
    if not files_meta:
        return []
    guids: list[str] = []
    # 한 번에 다 올림(bulk). 파일명 그대로 전송.
    payload = [("files", (os.path.basename(f), open(f, "rb"), "application/x-yaml")) for f in files_meta]
    try:
        r = requests.post(
            f"{base_url}/api/ai/prompts/bulk",
            headers={"Authorization": f"Bearer {api_key}"},
            files=payload, verify=False, timeout=600,
        )
        r.raise_for_status()
        for d in r.json().get("details", []):
            if d.get("guid"):
                guids.append(d["guid"])
    finally:
        for _n, (_fn, fh, _ct) in payload:
            fh.close()
    return guids


def delete_knowledge(guids: list[str], base_url: str, api_key: str) -> None:
    import requests
    if not guids:
        return
    r = requests.delete(
        f"{base_url}/api/ai/prompts",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"guids": ",".join(guids)}, verify=False, timeout=120,
    )
    r.raise_for_status()


# ─────────────────────────────────────────────────────────────
# mount / unmount 오케스트레이션
# ─────────────────────────────────────────────────────────────
def mount(path: str, base_url: str, api_key: str, dry_run: bool = False,
          config_path: str = CONFIG_PATH, state_path: str = STATE_PATH) -> dict:
    plan = plan_mount(path, config_path)
    if dry_run:
        plan["dry_run"] = True
        return plan
    if read_state(state_path):
        raise MountError("이미 장착된 카트리지가 있습니다 — 먼저 `cartridge unmount`")

    apply_state_to_config(plan, config_path)
    guids = upload_knowledge(plan["knowledge_dir"], base_url, api_key) if plan["knowledge_dir"] else []

    state = {
        "cartridge": plan["cartridge"], "handler": plan["handler"],
        "prompts": plan["prompts"], "knowledge_dir": plan["knowledge_dir"],
        "knowledge_guids": guids, "mounted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _write_state(state, state_path)
    plan["knowledge_uploaded"] = len(guids)
    return plan


def unmount(base_url: str, api_key: str,
            config_path: str = CONFIG_PATH, state_path: str = STATE_PATH) -> dict:
    state = read_state(state_path)
    if not state:
        raise MountError("장착된 카트리지 없음")
    delete_knowledge(state.get("knowledge_guids", []), base_url, api_key)
    _restore_config_prompts(state, config_path)
    os.remove(state_path)
    return {"cartridge": state.get("cartridge"), "knowledge_removed": len(state.get("knowledge_guids", []))}


def purge(config_path: str = CONFIG_PATH, state_path: str = STATE_PATH,
          collection: str = KNOWLEDGE_COLLECTION, dry_run: bool = False) -> dict:
    """클린 콘솔 provisioning — 지식 컬렉션 통삭제 + 배선 복원 + 상태 제거.

    unmount와의 차이: unmount는 `.mounted.json`에 기록된 guid만 콘솔 REST로 지운다.
    purge는 **Qdrant를 직접** 쳐서 컬렉션째 날린다 — 추적 밖 잔여물(수동 업로드, 중단된
    mount, 이전 카트리지 흔적)까지 제거하고 콘솔이 죽어 있어도 동작한다.
    멱등: 컬렉션이 없거나 장착 상태가 없어도 에러가 아니다.
    """
    import requests

    endpoint = qdrant_endpoint(config_path)
    state = read_state(state_path)
    result = {
        "endpoint": endpoint, "collection": collection,
        "cartridge": (state or {}).get("cartridge"),
        "prompts_restored": len((state or {}).get("prompts") or {}),
        "dry_run": dry_run,
    }

    try:
        r = requests.get(f"{endpoint}/collections/{collection}", timeout=30)
    except requests.RequestException as e:
        raise MountError(f"Qdrant 접속 실패({endpoint}) — 기동 확인 후 재시도: {e}")
    result["existed"] = r.status_code == 200
    result["points"] = r.json().get("result", {}).get("points_count") if result["existed"] else 0

    if dry_run:
        return result

    if result["existed"]:
        d = requests.delete(f"{endpoint}/collections/{collection}", timeout=120)
        if not d.ok:
            raise MountError(f"컬렉션 삭제 실패({d.status_code}): {d.text[:200]}")
    if state:
        _restore_config_prompts(state, config_path)
        os.remove(state_path)
    return result


def list_cartridges(cart_root: str = "cartridges") -> list[dict]:
    mounted = (read_state() or {}).get("cartridge")
    out = []
    for d in sorted(glob.glob(os.path.join(cart_root, "*"))):
        name = os.path.basename(d)
        if name.startswith(".") or name == "_template" or not os.path.isfile(os.path.join(d, "cartridge.yaml")):
            continue
        out.append({"name": name, "mounted": name == mounted})
    return out
