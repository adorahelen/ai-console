"""위저드 API — 온보딩(캐릭터 생성 · 지식 채우기 · 카트리지 저장).

docs/onboarding-design.md 구현. 원칙(자기부트스트랩): 사용자에게 내부 포맷을
직접 요구하지 않고, 원재료 → 내부 포맷 변환은 설치된 로컬 LLM이 담당한다.

기존 엔진 위의 부가 라우터 — qa_llm.py에서 include_router 2줄로 장착.
"""
import json
import os
import re
from typing import Any, Callable, Dict, List, Optional

import yaml
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from config_utils import load_prompt_file

# ── 메타 프롬프트 폴백 (파일 미설정 시) ──────────────────────────
_DRAFT_FALLBACK = (
    "You are a prompt engineer. Based on the user's answers below, write a complete "
    "system prompt (in the same language as the answers) for an AI assistant. Include: "
    "role definition, audience, tone, hard rules, and output format. "
    "Return ONLY the system prompt text, no commentary."
)
_EXTRACT_FALLBACK = (
    "You are a knowledge engineer. Read the document below and produce Q&A knowledge "
    "units a user might ask about it. Write question/answer in the document's language; "
    "answer must be self-contained. Output plain text blocks, NOT JSON — one block per "
    "unit, three tagged lines then a blank line (quotes need no escaping):\n"
    "Q: <question>\nA: <answer in one paragraph>\nAL: <variant 1> | <variant 2>"
)


# ── 요청 스키마 ──────────────────────────────────────────────────
class DraftRequest(BaseModel):
    role: str                    # 1. 이 에이전트는 무엇을 하나
    audience: str = ""           # 2. 누가 쓰나
    tone: str = ""               # 3. 말투
    rules: str = ""              # 4. 반드시 지킬 규칙
    needs_action: bool = False   # 5. [고급] 작업 생성 intent 필요 여부


class TestChatRequest(BaseModel):
    system_prompt: str
    message: str
    history: List[Dict[str, str]] = []


class KnowledgeConvertRequest(BaseModel):
    text: str
    count: int = 5               # 문서에서 뽑을 qna 개수 (가이드값)


class KnowledgeItem(BaseModel):
    question: str
    answer: str
    aliases: List[str] = []


class CartridgeSaveRequest(BaseModel):
    name: str                    # kebab-case 카트리지 이름
    description: str = ""
    system_prompt: str
    knowledge: List[KnowledgeItem] = []
    model_preset: str = ""


# 카트리지 이름 규칙 — 저장·장착이 같은 규칙을 쓴다. 장착 쪽에서는 이 검증이
# **경로 탈출 방어**를 겸한다(name 이 os.path.join("cartridges", name) 에 들어간다).
_CART_NAME_RE = re.compile(r"[a-z0-9][a-z0-9-]{1,40}")


class CartridgeMountRequest(BaseModel):
    name: str
    # 장착은 config.ini [prompts] 를 고쳐 쓴다 = 관리 평면. 사용자 Bearer 로는 부족하다
    # (security-review.md S-3 와 같은 규칙).
    admin_key: str = Field(..., min_length=32, max_length=64,
                           description="관리자 키 (api_keys/admin.key)")


class CartridgeUnmountRequest(BaseModel):
    admin_key: str = Field(..., min_length=32, max_length=64,
                           description="관리자 키 (api_keys/admin.key)")


_QNA_TAG_TO_FIELD = {"q": "question", "질문": "question", "question": "question",
                     "a": "answer", "답변": "answer", "answer": "answer",
                     "al": "aliases", "변형": "aliases", "aliases": "aliases"}
_QNA_TAG_RE = re.compile(r"^(Q|A|AL|질문|답변|변형|Question|Answer|Aliases)\s*[:：]\s*(.*)$",
                         re.IGNORECASE)


def _parse_qna_blocks(text: str) -> List[Dict[str, Any]]:
    """Q:/A:/AL: 라인 태그 포맷 파싱 — 본문에 따옴표·특수문자가 있어도 절대 안 깨진다.

    소형 모델은 JSON 이스케이프(문자열 안 따옴표)를 안정적으로 못 한다(T2 실측:
    "쾅-탓탁" 같은 인용이 많은 문서에서 8B가 전 항목 파싱 실패). 태그 라인 기반이라
    내용에 어떤 문자가 와도 안전하고, 여러 줄 답변은 이어붙인다."""
    items, cur, field = [], None, None

    def _commit():
        if cur and cur.get("question") and cur.get("answer"):
            items.append(cur)

    for raw in text.splitlines():
        line = raw.strip().strip("`").lstrip("-*# ").strip()
        m = _QNA_TAG_RE.match(line)
        if m:
            key = _QNA_TAG_TO_FIELD[m.group(1).lower()]
            val = m.group(2).strip()
            if key == "question":
                _commit()
                cur, field = {"question": val, "answer": "", "aliases": []}, "question"
            elif cur is not None:
                field = key
                if key == "aliases":
                    cur["aliases"] = [a.strip() for a in val.split("|") if a.strip()]
                else:
                    cur[key] = val
        elif cur is not None and field in ("question", "answer") and line:
            cur[field] = (cur[field] + "\n" + line).strip()  # 여러 줄 필드 이어붙임
    _commit()
    return items


def _extract_qna_items(text: str) -> List[Dict[str, Any]]:
    """LLM 응답에서 qna 항목을 관대하게 추출 — JSON 배열 → JSONL → Q:/A: 태그 3단 폴백.

    전체를 json.loads 한 방에 걸면 모델이 따옴표 하나만 틀려도 전부 502가 된다
    (T2 실측). 어떤 경로로든 살릴 수 있는 항목은 살리고, 전부 실패할 때만 에러."""
    start, end = text.find("["), text.rfind("]")
    if start >= 0 and end > start:
        try:
            arr = json.loads(text[start:end + 1])
            # aliases 같은 내부 배열(["x"])을 qna 배열로 오인하지 않도록 dict 배열만 인정
            if isinstance(arr, list) and arr and all(isinstance(x, dict) for x in arr):
                return arr
        except json.JSONDecodeError:
            pass
    items = []
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        s, e = line.find("{"), line.rfind("}")
        if s < 0 or e <= s:
            continue
        try:
            obj = json.loads(line[s:e + 1])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            items.append(obj)
    if items:
        return items
    items = _parse_qna_blocks(text)
    if not items:
        raise ValueError("응답에서 Q&A 항목을 찾지 못했습니다 (JSON/Q·A 태그 모두 불일치)")
    return items


def create_wizard_router(get_handler: Callable, auth_dependency: Callable) -> APIRouter:
    """get_handler: 활성 핸들러(agent_complete 보유)를 반환하는 callable."""
    router = APIRouter(prefix="/api/wizard", tags=["온보딩 위저드"])

    async def _run(messages: List[Dict]) -> Dict[str, Any]:
        """활성 핸들러로 생성. 미초기화/미로드는 503으로 통일(500 누출 방지)."""
        handler = get_handler()
        if handler is None:
            raise HTTPException(status_code=503, detail="LLM 핸들러가 아직 초기화되지 않았습니다")
        try:
            return await handler.agent_complete(messages, {}) or {}
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=f"모델을 사용할 수 없습니다: {e}")

    async def _generate(system: str, user: str) -> str:
        result = await _run(
            [{"role": "system", "content": system}, {"role": "user", "content": user}])
        return result.get("content", "")

    @router.post("/prompt-draft", description="5문답 → 시스템 프롬프트 초안 (로컬 LLM 생성)")
    async def prompt_draft(req: DraftRequest, user_info: Dict = Depends(auth_dependency)):
        meta = load_prompt_file("wizard_prompt_draft", _DRAFT_FALLBACK)
        answers = (
            f"1. 역할: {req.role}\n"
            f"2. 사용자: {req.audience or '일반 사용자'}\n"
            f"3. 말투: {req.tone or '정중하고 간결하게'}\n"
            f"4. 규칙: {req.rules or '(없음)'}\n"
            f"5. 작업 생성(쿼리·코드 등) 필요: {'예' if req.needs_action else '아니오'}\n"
        )
        draft = await _generate(meta, answers)
        if not draft.strip():
            raise HTTPException(status_code=502, detail="초안 생성 실패 (빈 응답)")
        return {"draft": draft.strip(), "needs_action": req.needs_action}

    @router.post("/test-chat", description="초안 프롬프트로 테스트 대화")
    async def test_chat(req: TestChatRequest, user_info: Dict = Depends(auth_dependency)):
        messages = ([{"role": "system", "content": req.system_prompt}]
                    + req.history[-6:]
                    + [{"role": "user", "content": req.message}])
        result = await _run(messages)
        return {"reply": result.get("content", "")}

    @router.post("/knowledge-convert", description="문서 텍스트 → qna 지식 초안 (큐레이션 모드)")
    async def knowledge_convert(req: KnowledgeConvertRequest,
                                user_info: Dict = Depends(auth_dependency)):
        meta = load_prompt_file("wizard_knowledge_extract", _EXTRACT_FALLBACK)
        doc = req.text[:20000]  # 컨텍스트 보호 — 초과분은 다음 호출로
        raw = await _generate(meta, f"[목표 개수: 약 {req.count}개]\n\n{doc}")
        try:
            items = _extract_qna_items(raw)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"qna 파싱 실패: {e}", headers=None)
        cleaned = []
        for it in items:
            if not isinstance(it, dict):
                continue  # 소형 모델이 문자열 배열 등을 뱉을 수 있음 — 무시
            q, a = str(it.get("question", "")).strip(), str(it.get("answer", "")).strip()
            if q and a:
                raw_aliases = it.get("aliases", [])
                aliases = ([str(x).strip() for x in raw_aliases if str(x).strip()]
                           if isinstance(raw_aliases, list) else [])
                cleaned.append({"question": q, "answer": a, "aliases": aliases})
        return {"items": cleaned, "truncated": len(req.text) > 20000}

    @router.post("/cartridge-save", description="위저드 결과를 카트리지 디렉토리로 저장")
    async def cartridge_save(req: CartridgeSaveRequest,
                             user_info: Dict = Depends(auth_dependency)):
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,40}", req.name):
            raise HTTPException(status_code=400,
                                detail="이름은 kebab-case (소문자·숫자·하이픈, 2~41자)")
        base = os.path.join("cartridges", req.name)
        if os.path.exists(base):
            raise HTTPException(status_code=409, detail=f"이미 존재하는 카트리지: {req.name}")

        os.makedirs(os.path.join(base, "prompts"))
        os.makedirs(os.path.join(base, "knowledge"))

        # 슬롯 1: 프롬프트
        with open(os.path.join(base, "prompts", "system.txt"), "w", encoding="utf-8") as f:
            f.write(req.system_prompt.strip() + "\n")

        # 슬롯 2: 지식 (qna YAML — knowledge/README.md 규격)
        files = []
        for i, item in enumerate(req.knowledge, 1):
            slug = re.sub(r"[^a-z0-9]+", "-", item.question.lower())[:40].strip("-") or "item"
            fname = f"{i:03d}-{slug}.yaml"
            doc = {"type": "qna", "name": f"{req.name}-{i:03d}", "enabled": True,
                   "question": item.question, "answer": item.answer,
                   "aliases": item.aliases}
            with open(os.path.join(base, "knowledge", fname), "w", encoding="utf-8") as f:
                yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False,
                               default_flow_style=False)
            files.append(fname)

        # 매니페스트 (슬롯 3 포함)
        manifest = {
            "name": req.name, "version": "0.1.0",
            "description": req.description or f"{req.name} — 위저드로 생성된 카트리지",
            "prompts": {"system": "prompts/system.txt"},
            "knowledge": {"dir": "knowledge/"},
            "model": {"recommended": req.model_preset or "(설치 프리셋)"},
        }
        with open(os.path.join(base, "cartridge.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(manifest, f, allow_unicode=True, sort_keys=False)

        return {
            "saved": base,
            "knowledge_files": files,
            "next": f"장착: 위저드 4단계에서 '{req.name}' 장착 "
                    "(또는 aibotctl cartridge mount) → 콘솔 재시작. "
                    "장착이 프롬프트 배선과 지식 적재를 함께 수행한다.",
        }

    # ── 4단계: 카트리지 장착 ────────────────────────────────────────
    # Phase 4 CLI(aibotctl cartridge)와 **같은 코드 경로**를 UI 에서 부른다.
    # UI 전용 로직을 새로 만들지 않는다 — 두 경로가 갈리면 상태(.mounted.json)가 어긋난다.
    #
    # ⚠ mount/unmount 는 **동기 def** 여야 한다. cartridge_mount.mount() 가 지식을
    # 자기 콘솔의 /api/ai/prompts/bulk 로 HTTP 업로드하는데, async 안에서 블로킹 호출을
    # 하면 이벤트 루프가 막혀 그 자기요청을 처리하지 못하고 데드락 난다.
    # 동기 def 는 FastAPI 가 스레드풀에서 돌리므로 루프가 살아 있다.

    def _self_endpoint() -> tuple:
        """자기 콘솔 base_url + 지식 업로드용 사용자 키 (CLI `_bearer_key` 와 같은 출처)."""
        import configparser
        cp = configparser.ConfigParser()
        cp.read("config.ini", encoding="utf-8")
        port = cp.get("server", "port", fallback="8443").strip()
        try:
            with open("api_keys/default.key", encoding="utf-8") as f:
                key = f.read().strip()
        except OSError:
            raise HTTPException(status_code=500,
                                detail="api_keys/default.key 없음 — 지식 적재에 필요합니다")
        return f"https://localhost:{port}", key

    def _require_admin(admin_key: str) -> None:
        from aibot_restapi_auth import verify_admin_key
        if not verify_admin_key(admin_key):
            raise HTTPException(status_code=401, detail="관리자 인증이 필요합니다.")

    @router.get("/cartridges", description="카트리지 목록 + 현재 장착 상태")
    async def cartridges_list(user_info: Dict = Depends(auth_dependency)):
        import cartridge_mount
        items = []
        for c in cartridge_mount.list_cartridges():
            desc, n_know = "", 0
            try:
                with open(os.path.join("cartridges", c["name"], "cartridge.yaml"),
                          encoding="utf-8") as f:
                    m = yaml.safe_load(f) or {}
                desc = m.get("description", "")
                kn = (m.get("knowledge") or {}).get("dir")
                if kn:
                    n_know = len(cartridge_mount._knowledge_files(
                        os.path.join("cartridges", c["name"], kn)))
            except Exception:
                pass   # 매니페스트가 깨져도 목록 자체는 보여준다 (validate 가 진단 담당)
            items.append({**c, "description": desc, "knowledge_files": n_know})

        state = cartridge_mount.read_state() or {}
        return {
            "cartridges": items,
            "mounted": {
                "cartridge": state.get("cartridge"),
                "handler": state.get("handler"),
                "mounted_at": state.get("mounted_at"),
                "knowledge_uploaded": len(state.get("knowledge_guids", []) or []),
                "warnings": state.get("warnings", []),
            } if state else None,
            "active_handler": cartridge_mount.active_handler(),
        }

    @router.post("/cartridge-mount", description="카트리지 장착 (프롬프트 배선 + 지식 적재)")
    def cartridge_mount_ep(req: CartridgeMountRequest,          # noqa: 동기 def 의도적
                           user_info: Dict = Depends(auth_dependency)):
        _require_admin(req.admin_key)
        if not _CART_NAME_RE.fullmatch(req.name):
            raise HTTPException(status_code=400, detail="잘못된 카트리지 이름")
        import cartridge_mount
        path = os.path.join("cartridges", req.name)
        if not os.path.isfile(os.path.join(path, "cartridge.yaml")):
            raise HTTPException(status_code=404, detail=f"카트리지 없음: {req.name}")
        base_url, key = _self_endpoint()
        try:
            plan = cartridge_mount.mount(path, base_url, key)
        except cartridge_mount.MountError as e:
            raise HTTPException(status_code=409, detail=str(e))
        # 프롬프트는 핸들러가 기동 시 인스턴스에 굳는다 — 그래서 재시작이 필요했다.
        # reload_cartridge_prompts() 가 config.ini 재파싱 + 베이스 프롬프트 재적용을 하므로
        # 성공하면 재시작이 불필요하다. 실패(콘솔 미초기화 등)면 종전대로 재시작 안내.
        reload = _reload_runtime_prompts()
        return {
            "mounted": plan["cartridge"],
            "handler": plan["handler"],
            "prompt_slots": len(plan.get("prompts", {})),
            "knowledge_uploaded": plan.get("knowledge_uploaded", 0),
            "warnings": plan.get("warnings", []),
            "reload": reload,
            "restart_required": not reload.get("ok", False),
        }

    @router.post("/cartridge-unmount", description="장착 해제 (지식 삭제 + 배선 복원)")
    def cartridge_unmount_ep(req: CartridgeUnmountRequest,      # noqa: 동기 def 의도적
                             user_info: Dict = Depends(auth_dependency)):
        _require_admin(req.admin_key)
        import cartridge_mount
        base_url, key = _self_endpoint()
        try:
            out = cartridge_mount.unmount(base_url, key)
        except cartridge_mount.MountError as e:
            raise HTTPException(status_code=409, detail=str(e))
        reload = _reload_runtime_prompts()
        out["reload"] = reload
        out["restart_required"] = not reload.get("ok", False)
        return out

    return router


def _reload_runtime_prompts() -> dict:
    """장착/해제 뒤 [prompts] 배선을 런타임에 재적용 (콘솔의 전역 llm_handler 대상).

    같은 프로세스 안이라 REST(/api/cartridge/reload)를 자기 자신에게 쏘지 않고 직접 호출한다.

    ⚠️ `import qa_llm` 로 잡으면 안 된다 — 콘솔은 `python qa_llm.py` 로 기동하므로 실행 중인
    모듈 객체는 `__main__` 이고, 새로 import 하면 llm_handler=None 인 **두 번째 사본**이 잡힌다
    (실측: restart_required 가 항상 True 로 나오던 원인). sys.modules 에서 살아있는 쪽을 찾는다.
    """
    import sys
    candidates = [sys.modules.get("__main__"), sys.modules.get("qa_llm")]
    fallback = None
    for mod in candidates:
        fn = getattr(mod, "reload_cartridge_prompts", None)
        if fn is None:
            continue
        if getattr(mod, "llm_handler", None) is not None:
            try:
                return fn()
            except Exception as e:
                return {"ok": False, "detail": f"런타임 리로드 실패 ({type(e).__name__}: {e})"}
        fallback = fallback or fn

    if fallback is None:
        return {"ok": False, "detail": "reload_cartridge_prompts 미탑재 — 재시작 필요"}
    try:
        return fallback()          # 핸들러가 아직 없는 기동 직후 등 — ok:False 로 정직하게 답한다
    except Exception as e:
        return {"ok": False, "detail": f"런타임 리로드 실패 ({type(e).__name__}: {e})"}
