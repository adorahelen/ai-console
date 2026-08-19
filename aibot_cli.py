#!/usr/bin/env python3
"""AI QA System 관리 CLI.

사용 예:
    ./aibotctl keys generate alice marketing --model gpt-5-mini --length high
    ./aibotctl keys list
    ./aibotctl keys show <api_key>
    ./aibotctl keys set <api_key> --model gpt-4o-mini
    ./aibotctl keys renew <api_key> 6 --unit month
    ./aibotctl keys delete <api_key> --yes

전역 옵션:
    --json          JSON 출력
    --base-url URL  서버 주소 override (기본 https://localhost:<server.port>)
    --admin-key KEY 관리자 키 override (기본 환경변수 ADMIN_KEY)

자동완성 설치 (bash):
    eval "$(_AIBOTCTL_COMPLETE=bash_source ./aibotctl)"
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any

import click
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ─────────────────────────────────────────────────────────────────────────────
# 모델/길이/추론 선택지
# ─────────────────────────────────────────────────────────────────────────────
MODEL_CHOICES = [
    "gpt-oss",       # 기본 (로컬)
    "gpt-4o-mini",   # OpenAI gpt-4o 계열
    "gpt-4o",
    "gpt-5-nano",    # OpenAI gpt-5 계열 (추론 모델)
    "gpt-5-mini",
    "gpt-5",
]
LENGTH_CHOICES = ["low", "medium", "high"]
REASONING_CHOICES = ["minimal", "low", "medium", "high"]
UNIT_CHOICES = ["day", "month", "year"]


# ─────────────────────────────────────────────────────────────────────────────
# 설정 로드 + REST 클라이언트
# ─────────────────────────────────────────────────────────────────────────────
def _load_config() -> tuple[str, str]:
    """config.ini 에서 base_url 과 admin_key 추출."""
    try:
        from config_utils import ConfigManager
        cm = ConfigManager()
        port = cm.config.get("server", "port", fallback="5443")
    except Exception:
        port = "5443"
    base_url = f"https://localhost:{port}"
    # '0'*32 더미 기본값 제거 — 서버가 verify_admin_key fail-closed(S-2)라 더미면 무조건 401.
    # run.sh get_admin_key와 동일하게 api_keys/admin.key를 기본으로 읽는다.
    admin_key = os.environ.get("ADMIN_KEY", "")
    if not admin_key:
        try:
            _kp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_keys", "admin.key")
            with open(_kp, encoding="utf-8") as _f:
                admin_key = _f.read().strip()
        except OSError:
            admin_key = ""
    return base_url, admin_key


class APIError(click.ClickException):
    """REST 호출 에러를 사용자 친화 메시지로 변환."""
    def __init__(self, msg: str, exit_code: int = 1):
        super().__init__(msg)
        self.exit_code = exit_code


def _request(method: str, base_url: str, path: str, payload: dict | None = None,
             timeout: int = 30) -> dict:
    url = f"{base_url.rstrip('/')}{path}"
    try:
        resp = requests.request(method, url, json=payload, verify=False, timeout=timeout)
    except requests.exceptions.ConnectionError:
        raise APIError(f"서버에 연결할 수 없습니다 ({url}). './run.sh start' 로 서버를 먼저 띄워주세요.")
    except requests.exceptions.Timeout:
        raise APIError(f"요청 타임아웃 ({timeout}s)")
    except requests.exceptions.RequestException as e:
        raise APIError(f"요청 실패: {e}")

    if not resp.ok:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        raise APIError(f"HTTP {resp.status_code}: {detail}")

    try:
        return resp.json()
    except ValueError:
        return {"raw": resp.text}


# ─────────────────────────────────────────────────────────────────────────────
# 출력 포맷
# ─────────────────────────────────────────────────────────────────────────────
def _json_option(f):
    """subcommand 에서도 --json 을 받게 하는 데코레이터.
    글로벌 --json 과 동일하게 ctx.obj['json'] 을 토글한다."""
    def _cb(ctx, _param, value):
        if value:
            ctx.ensure_object(dict)
            ctx.obj["json"] = True
        return value
    return click.option(
        "--json", "as_json", is_flag=True, expose_value=False,
        callback=_cb, help="JSON 출력 (스크립트 친화)",
    )(f)


def _emit(data: Any, ctx: click.Context, human_renderer=None) -> None:
    """JSON 모드면 JSON 으로, 아니면 human_renderer 호출."""
    if ctx.obj.get("json"):
        click.echo(json.dumps(data, ensure_ascii=False, indent=2, default=str))
    elif human_renderer is not None:
        human_renderer(data)
    else:
        click.echo(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def _print_kv(d: dict, fields: list[tuple[str, str]] | None = None) -> None:
    """단일 dict 를 key: value 로 출력. fields=[(key,label),...]"""
    if fields is None:
        fields = [(k, k) for k in d.keys()]
    width = max(len(label) for _, label in fields) + 2
    for key, label in fields:
        val = d.get(key)
        if val is None:
            val = "-"
        click.echo(f"  {label:<{width}} {val}")


def _print_table(rows: list[dict], columns: list[tuple[str, str]]) -> None:
    """rows: list[dict], columns: [(key,header),...] 로 단순 테이블 출력."""
    if not rows:
        click.echo("(no rows)")
        return
    headers = [h for _, h in columns]
    widths = [len(h) for h in headers]
    for r in rows:
        for i, (key, _) in enumerate(columns):
            v = r.get(key)
            v = "-" if v is None else str(v)
            widths[i] = max(widths[i], len(v))
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    click.echo(fmt.format(*headers))
    click.echo(fmt.format(*("-" * w for w in widths)))
    for r in rows:
        vals = []
        for key, _ in columns:
            v = r.get(key)
            vals.append("-" if v is None else str(v))
        click.echo(fmt.format(*vals))


def _mask_key(api_key: str) -> str:
    if not api_key or len(api_key) < 16:
        return api_key
    return f"{api_key[:8]}…{api_key[-4:]}"


_STATUS_MARK = {"ok": ("✓", "green"), "warn": ("⚠", "yellow"), "error": ("✗", "red")}


def _render_cartridge_report(d: dict) -> None:
    """cartridge validate 결과를 ✓/⚠/✗ 리포트로 출력."""
    click.secho(f"카트리지 검증: {d['path']}", bold=True)
    for c in d["checks"]:
        mark, color = _STATUS_MARK.get(c["status"], ("·", None))
        line = f"  {mark} {c['label']}"
        if c.get("detail"):
            line += f": {c['detail']}"
        click.secho(line, fg=color)
    click.echo()
    if d["errors"]:
        click.secho(f"실패 — 에러 {d['errors']}건, 경고 {d['warnings']}건", fg="red", bold=True)
    elif d["warnings"]:
        click.secho(f"통과 (경고 {d['warnings']}건)", fg="yellow", bold=True)
    else:
        click.secho("통과 — 모든 검사 정상", fg="green", bold=True)


# ─────────────────────────────────────────────────────────────────────────────
# 최상위 그룹
# ─────────────────────────────────────────────────────────────────────────────
@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--json", "as_json", is_flag=True, help="JSON 으로 출력 (스크립트 친화)")
@click.option("--base-url", default=None, help="서버 주소 (기본: config.ini 의 server.port)")
@click.option("--admin-key", default=None, help="관리자 키 (기본: 환경변수 ADMIN_KEY)")
@click.version_option("0.1.0", prog_name="aibotctl")
@click.pass_context
def cli(ctx: click.Context, as_json: bool, base_url: str | None, admin_key: str | None) -> None:
    """AI QA System 관리 CLI."""
    default_base, default_admin = _load_config()
    ctx.ensure_object(dict)
    ctx.obj["base_url"] = base_url or default_base
    ctx.obj["admin_key"] = admin_key or default_admin
    ctx.obj["json"] = as_json


# ─────────────────────────────────────────────────────────────────────────────
# keys 서브그룹 (sub-commands 는 별도 task 에서 추가)
# ─────────────────────────────────────────────────────────────────────────────
@cli.group()
def keys() -> None:
    """구독키 관리 (generate / list / show / set / renew / delete)."""


# ─────────────────────────────────────────────────────────────────────────────
# keys generate
# ─────────────────────────────────────────────────────────────────────────────
@keys.command("generate")
@_json_option
@click.argument("name", required=False)
@click.argument("account", required=False)
@click.option("--description", default=None, help="설명 (200자 이내)")
@click.option("--acl", default=None, help="IP 화이트리스트 CIDR (콤마 구분)")
@click.option("--model", type=click.Choice(MODEL_CHOICES), default=None,
              help=f"기본 모델 (기본: {MODEL_CHOICES[0]})")
@click.option("--length", type=click.Choice(LENGTH_CHOICES), default=None,
              help="응답 길이 (기본: medium)")
@click.option("--reasoning", type=click.Choice(REASONING_CHOICES), default=None,
              help="추론 깊이 (기본: 미설정 = 서버 config 폴백)")
@click.option("--no-prompt", is_flag=True, help="누락 필드도 묻지 않음 (자동화용)")
@click.pass_context
def keys_generate(ctx, name, account, description, acl, model, length, reasoning, no_prompt):
    """새 구독키 생성. 필수 필드 누락 시 자동으로 물어봄."""
    if no_prompt:
        if not name or not account:
            raise APIError("--no-prompt 모드는 name 과 account 가 필수입니다.")
    else:
        if not name:
            name = click.prompt("Subscription name")
        if not account:
            account = click.prompt("Account")
        if description is None:
            description = click.prompt("Description (blank to skip)", default="", show_default=False) or None
        if acl is None:
            acl = click.prompt("ACL CIDR (blank for any IP)", default="", show_default=False) or None
        if model is None:
            model = click.prompt(
                f"Model ({'/'.join(MODEL_CHOICES)}, blank=default)",
                type=click.Choice(MODEL_CHOICES + [""]),
                default="", show_choices=False, show_default=False,
            ) or None
        if length is None:
            length = click.prompt(
                f"Length ({'/'.join(LENGTH_CHOICES)}, blank=default)",
                type=click.Choice(LENGTH_CHOICES + [""]),
                default="", show_choices=False, show_default=False,
            ) or None
        if reasoning is None:
            reasoning = click.prompt(
                f"Reasoning ({'/'.join(REASONING_CHOICES)}, blank=default)",
                type=click.Choice(REASONING_CHOICES + [""]),
                default="", show_choices=False, show_default=False,
            ) or None

    payload = {
        "admin_key": ctx.obj["admin_key"],
        "name": name,
        "account": account,
        "description": description,
        "acl": acl,
        "model": model,
        "length": length,
        "reasoning_effort": reasoning,
    }
    # null 값 제거 (서버에서 검증 시 깔끔)
    payload = {k: v for k, v in payload.items() if v is not None}

    data = _request("POST", ctx.obj["base_url"], "/api/generate", payload)

    def _render(d):
        click.secho("✅ Created subscription:", fg="green")
        _print_kv(d, fields=[
            ("api_key", "api_key"),
            ("sub_id", "sub_id"),
            ("guid", "guid"),
            ("created_at", "created_at"),
            ("expires_at", "expires_at"),
            ("message", "message"),
        ])
        click.echo("\nApplied settings:")
        _print_kv({
            "model": model, "length": length, "reasoning_effort": reasoning,
        }, fields=[("model", "model"), ("length", "length"), ("reasoning_effort", "reasoning")])

    _emit(data, ctx, _render)


# ─────────────────────────────────────────────────────────────────────────────
# keys list
# ─────────────────────────────────────────────────────────────────────────────
@keys.command("list")
@_json_option
@click.argument("name_pos", required=False)
@click.argument("account_pos", required=False)
@click.option("--name", default=None, help="이름으로 필터")
@click.option("--account", default=None, help="계정으로 필터")
@click.pass_context
def keys_list(ctx, name_pos, account_pos, name, account):
    """구독키 목록.

    Positional 사용 시: NAME [ACCOUNT] — flag 사용 시: --name / --account.
    """
    payload = {"admin_key": ctx.obj["admin_key"]}
    eff_name = name or name_pos
    eff_account = account or account_pos
    if eff_name:
        payload["name"] = eff_name
    if eff_account:
        payload["account"] = eff_account

    data = _request("POST", ctx.obj["base_url"], "/api/list", payload)
    rows = data.get("subscriptions", [])

    def _render(_):
        if not rows:
            click.echo("(no subscriptions)")
            return
        click.echo(f"Total: {data.get('count', len(rows))}\n")
        # 보기 좋게 일부 컬럼만
        view = []
        for r in rows:
            view.append({
                "id": r.get("id"),
                "name": r.get("name"),
                "account": r.get("account"),
                "model": r.get("model"),
                "length": r.get("length"),
                "reasoning": r.get("reasoning_effort"),
                "api_key": r.get("api_key", ""),
                "expires_at": r.get("expires_at"),
            })
        _print_table(view, columns=[
            ("id", "ID"),
            ("name", "NAME"),
            ("account", "ACCOUNT"),
            ("model", "MODEL"),
            ("length", "LENGTH"),
            ("reasoning", "REASONING"),
            ("api_key", "API_KEY"),
            ("expires_at", "EXPIRES_AT"),
        ])

    _emit(data, ctx, _render)


# ─────────────────────────────────────────────────────────────────────────────
# keys show
# ─────────────────────────────────────────────────────────────────────────────
@keys.command("show")
@_json_option
@click.argument("api_key", required=False)
@click.pass_context
def keys_show(ctx, api_key):
    """구독키 상세 (= verify)."""
    if not api_key:
        api_key = click.prompt("API key")

    data = _request("POST", ctx.obj["base_url"], "/api/verify", {"api_key": api_key})

    # /api/verify 응답은 valid/id/user_id/guid/permissions 만 — DB 의 model/length/reasoning
    # 까지 보고 싶으면 /api/list 로 추가 조회
    list_data = _request("POST", ctx.obj["base_url"], "/api/list", {"admin_key": ctx.obj["admin_key"]})
    sub = next((s for s in list_data.get("subscriptions", []) if s.get("api_key") == api_key), None)

    merged = dict(data)
    if sub:
        merged.update({
            "name": sub.get("name"),
            "account": sub.get("account"),
            "description": sub.get("description"),
            "acl": sub.get("acl"),
            "model": sub.get("model"),
            "length": sub.get("length"),
            "reasoning_effort": sub.get("reasoning_effort"),
            "expires_at": sub.get("expires_at"),
            "created_at": sub.get("created_at"),
        })

    def _render(d):
        click.secho("Subscription details:", bold=True)
        _print_kv(d, fields=[
            ("id", "id"),
            ("name", "name"),
            ("account", "account"),
            ("description", "description"),
            ("acl", "acl"),
            ("model", "model"),
            ("length", "length"),
            ("reasoning_effort", "reasoning_effort"),
            ("expires_at", "expires_at"),
            ("created_at", "created_at"),
            ("guid", "guid"),
            ("user_id", "user_id"),
            ("valid", "valid"),
            ("permissions", "permissions"),
        ])

    _emit(merged, ctx, _render)


# ─────────────────────────────────────────────────────────────────────────────
# keys set (PATCH /api/subscriptions/settings)
# ─────────────────────────────────────────────────────────────────────────────
@keys.command("set")
@_json_option
@click.argument("api_key", required=False)
@click.option("--model", type=click.Choice(MODEL_CHOICES), default=None,
              help="기본 모델 지정")
@click.option("--length", type=click.Choice(LENGTH_CHOICES), default=None,
              help="응답 길이")
@click.option("--reasoning", type=click.Choice(REASONING_CHOICES), default=None,
              help="추론 깊이")
@click.option("--clear-model", is_flag=True, help="model NULL 로 (config 폴백)")
@click.option("--clear-length", is_flag=True, help="length NULL 로 (medium 폴백)")
@click.option("--clear-reasoning", is_flag=True, help="reasoning NULL 로 (config 폴백)")
@click.pass_context
def keys_set(ctx, api_key, model, length, reasoning, clear_model, clear_length, clear_reasoning):
    """구독의 model/length/reasoning 갱신. clear-* 는 NULL 폴백."""
    if not api_key:
        api_key = click.prompt("API key")

    # --field 와 --clear-field 동시 지정 방지
    if (model and clear_model) or (length and clear_length) or (reasoning and clear_reasoning):
        raise APIError("같은 필드에 값 지정과 --clear-* 를 동시에 줄 수 없습니다.")

    payload: dict[str, Any] = {"api_key": api_key}
    if model:
        payload["model"] = model
    elif clear_model:
        payload["model"] = None
    if length:
        payload["length"] = length
    elif clear_length:
        payload["length"] = None
    if reasoning:
        payload["reasoning_effort"] = reasoning
    elif clear_reasoning:
        payload["reasoning_effort"] = None

    if set(payload.keys()) == {"api_key"}:
        raise APIError("갱신할 필드를 하나 이상 지정해주세요 "
                       "(--model/--length/--reasoning 또는 --clear-*).")

    data = _request("PATCH", ctx.obj["base_url"], "/api/subscriptions/settings", payload)

    def _render(d):
        applied = d.get("applied", {})
        click.secho("✅ Settings updated:", fg="green")
        # PATCH 에 포함된 필드만 표시. NULL 은 "(cleared)" 로 명시
        rendered = {}
        for k in ("model", "length", "reasoning_effort"):
            if k in applied:
                rendered[k] = "(cleared)" if applied[k] is None else applied[k]
        rendered["updated_at"] = d.get("updated_at")
        _print_kv(rendered)

    _emit(data, ctx, _render)


# ─────────────────────────────────────────────────────────────────────────────
# keys renew (PUT /api/renew)
# ─────────────────────────────────────────────────────────────────────────────
@keys.command("renew")
@_json_option
@click.argument("api_key", required=False)
@click.argument("duration", required=False, type=int)
@click.option("--unit", type=click.Choice(UNIT_CHOICES), default=None,
              help="기간 단위 (day/month/year, 기본 year)")
@click.pass_context
def keys_renew(ctx, api_key, duration, unit):
    """만료일 연장. 기본 +1 year. duration 은 1~100 범위."""
    if not api_key:
        api_key = click.prompt("API key")
    if unit is None:
        unit = click.prompt(
            "Unit",
            type=click.Choice(UNIT_CHOICES), default="year",
        )
    if duration is None:
        duration = click.prompt(
            f"Duration in {unit}s (1-100)", type=int, default=1,
        )

    if duration < 1 or duration > 100:
        raise APIError("duration 은 1~100 범위여야 합니다.")

    payload = {"api_key": api_key, "duration": duration, "unit": unit}
    data = _request("PUT", ctx.obj["base_url"], "/api/renew", payload)

    def _render(d):
        click.secho(f"✅ Renewed (+{duration} {unit}):", fg="green")
        _print_kv(d, fields=[
            ("expires_at", "expires_at"),
            ("api_key", "api_key"),
            ("message", "message"),
        ])

    _emit(data, ctx, _render)


# ─────────────────────────────────────────────────────────────────────────────
# keys delete (DELETE /api/delete)
# ─────────────────────────────────────────────────────────────────────────────
@keys.command("delete")
@_json_option
@click.argument("api_key", required=False)
@click.option("-y", "--yes", is_flag=True, help="확인 프롬프트 건너뜀")
@click.pass_context
def keys_delete(ctx, api_key, yes):
    """구독키 삭제. 임베딩 데이터까지 cascade."""
    if not api_key:
        api_key = click.prompt("API key")

    if not yes:
        if not click.confirm(
            f"❗ API key {_mask_key(api_key)} 를 정말 삭제할까요? (구독 데이터도 함께 삭제됨)",
            default=False,
        ):
            click.echo("Cancelled.")
            ctx.exit(0)

    data = _request("DELETE", ctx.obj["base_url"], "/api/delete", {"api_key": api_key})

    def _render(d):
        click.secho("✅ Deleted.", fg="green")
        if d:
            _print_kv(d)

    _emit(data, ctx, _render)


# ─────────────────────────────────────────────────────────────────────────────
# cartridge 서브그룹 (Phase 4)
# ─────────────────────────────────────────────────────────────────────────────
@cli.group()
def cartridge() -> None:
    """카트리지 관리 (validate · list · status · mount · unmount · purge)."""


# ─────────────────────────────────────────────────────────────────────────────
# cartridge validate
# ─────────────────────────────────────────────────────────────────────────────
@cartridge.command("validate")
@_json_option
@click.argument("path", type=click.Path(exists=True))
@click.pass_context
def cartridge_validate(ctx, path):
    """카트리지(cartridge.yaml + knowledge/*.yaml)를 UNION 스키마로 검증.

    읽기 전용. 슬롯 경로·knowledge.dir 실재까지 확인해 깨진 배선을 장착 전에 잡는다.
    ERROR 가 있으면 종료코드 1, WARNING 만이면 0.
    """
    from cartridge_validate import validate_cartridge

    rep = validate_cartridge(path)
    data = rep.as_dict(path)
    _emit(data, ctx, _render_cartridge_report)
    if data["errors"]:
        ctx.exit(1)


def _bearer_key() -> str:
    """지식 업로드용 사용자 API 키 (api_keys/default.key)."""
    try:
        with open("api_keys/default.key", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        raise click.ClickException("api_keys/default.key 없음 — 콘솔 설치 후 장착")


def _render_plan(d: dict) -> None:
    tag = " [DRY-RUN]" if d.get("dry_run") else ""
    click.secho(f"장착 계획: {d['cartridge']}{tag}", bold=True)
    click.echo(f"  핸들러: {d['handler']}")
    click.secho("  프롬프트 배선 (config [prompts]):", bold=True)
    for k, v in (d.get("prompts") or {}).items():
        click.echo(f"    {k} → {v}")
    if d.get("knowledge_dir"):
        click.echo(f"  지식: {d['knowledge_dir']}" + (f" · 업로드 {d['knowledge_uploaded']}건" if "knowledge_uploaded" in d else ""))
    for w in d.get("warnings") or []:
        click.secho(f"  ⚠ {w}", fg="yellow")


@cartridge.command("list")
@_json_option
@click.pass_context
def cartridge_list(ctx):
    """cartridges/ 목록 + 현재 장착 표시."""
    import cartridge_mount
    rows = cartridge_mount.list_cartridges()
    def _render(rs):
        for r in rs:
            mark = click.style("● 장착", fg="green") if r["mounted"] else "  "
            click.echo(f"  {mark}  {r['name']}")
    _emit(rows, ctx, _render)


@cartridge.command("status")
@_json_option
@click.pass_context
def cartridge_status(ctx):
    """현재 장착된 카트리지 상세 (.mounted.json)."""
    import cartridge_mount
    st = cartridge_mount.read_state()
    def _render(s):
        if not s:
            click.echo("장착된 카트리지 없음")
            return
        click.secho(f"장착: {s['cartridge']}", bold=True, fg="green")
        click.echo(f"  핸들러: {s.get('handler')} · 장착시각: {s.get('mounted_at')}")
        click.echo(f"  프롬프트 슬롯: {len(s.get('prompts') or {})} · 지식: {len(s.get('knowledge_guids') or [])}건")
    _emit(st or {}, ctx, _render)


@cartridge.command("mount")
@_json_option
@click.argument("path", type=click.Path(exists=True))
@click.option("--dry-run", is_flag=True, help="변경 없이 장착 계획만 출력")
@click.pass_context
def cartridge_mount_cmd(ctx, path, dry_run):
    """카트리지 장착 — 프롬프트 배선 + 지식 업로드 + 상태 기록. 콘솔이 떠 있으면 자동 반영."""
    import cartridge_mount
    try:
        if dry_run:
            plan = cartridge_mount.plan_mount(path)
            plan["dry_run"] = True
            _emit(plan, ctx, _render_plan)
            return
        plan = cartridge_mount.mount(path, ctx.obj["base_url"], _bearer_key())
    except cartridge_mount.MountError as e:
        raise click.ClickException(str(e))
    plan["reload"] = _try_runtime_reload(ctx)
    _emit(plan, ctx, _render_plan)
    if not ctx.obj["json"]:
        _echo_reload_result(plan["reload"])


def _try_runtime_reload(ctx) -> dict:
    """장착/해제 후 런타임 반영을 시도한다 (콘솔이 죽어 있으면 조용히 건너뛴다).

    장착 자체는 콘솔이 없어도 되어야 하므로(첫 설치 직후) 로컬에서 하고,
    반영만 REST 로 요청한다. 실패는 에러가 아니라 "재시작 필요" 안내로 처리.
    """
    try:
        return _request("POST", ctx.obj["base_url"], "/api/cartridge/reload",
                        {"admin_key": ctx.obj["admin_key"]})
    except Exception as e:
        return {"ok": False, "detail": f"콘솔 미응답 — 재시작 필요 ({type(e).__name__})"}


def _echo_reload_result(reload: dict) -> None:
    if reload.get("ok"):
        n = len(reload.get("reloaded") or [])
        click.secho(f"✓ 장착 완료 — 런타임 반영됨 (핸들러 {n}개 리로드, 재시작 불필요)", fg="green", bold=True)
    else:
        click.secho(f"✓ 장착 완료 — 반영하려면 콘솔 재시작: ./run.sh restart", fg="yellow", bold=True)
        if reload.get("detail"):
            click.echo(f"  ({reload['detail']})")


@cartridge.command("unmount")
@_json_option
@click.pass_context
def cartridge_unmount_cmd(ctx):
    """현재 장착 해제 — 지식 삭제 + config 복원 + 상태 제거."""
    import cartridge_mount
    try:
        r = cartridge_mount.unmount(ctx.obj["base_url"], _bearer_key())
    except cartridge_mount.MountError as e:
        raise click.ClickException(str(e))
    r["reload"] = _try_runtime_reload(ctx)
    _emit(r, ctx, lambda d: click.secho(
        f"✓ 해제: {d['cartridge']} (지식 {d['knowledge_removed']}건 삭제)"
        + (" — 런타임 반영됨" if d["reload"].get("ok") else " — ./run.sh restart"), fg="green"))


@cartridge.command("purge")
@_json_option
@click.option("--dry-run", is_flag=True, help="변경 없이 삭제 대상만 출력")
@click.option("--yes", "-y", is_flag=True, help="확인 프롬프트 생략")
@click.pass_context
def cartridge_purge_cmd(ctx, dry_run, yes):
    """지식 컬렉션 통삭제 + 배선 복원 — 클린 콘솔로 되돌린다. (파괴적)

    unmount는 장착 기록에 있는 지식만 지우지만, purge는 Qdrant 컬렉션째 날린다.
    추적 밖 잔여물까지 지우므로 새 도메인 provisioning·재현 테스트의 출발점.
    콘솔이 죽어 있어도 동작(Qdrant 직접 호출).
    """
    import cartridge_mount
    try:
        preview = cartridge_mount.purge(dry_run=True)
        if not dry_run and not yes:
            # --json(비대화)에서 확인을 우회하면 안 된다 — 되돌릴 수 없는 삭제라 --yes 명시 요구
            if ctx.obj["json"]:
                raise click.ClickException("purge는 되돌릴 수 없습니다 — JSON/비대화 모드에선 --yes 를 명시하세요")
            click.secho(
                f"⚠ {preview['endpoint']} 의 컬렉션 '{preview['collection']}'"
                + (f" (지식 {preview['points']}건)" if preview["existed"] else " (없음)")
                + " 을 삭제합니다. 되돌릴 수 없습니다.", fg="yellow")
            click.confirm("계속할까요?", abort=True)
        r = cartridge_mount.purge(dry_run=dry_run)
    except cartridge_mount.MountError as e:
        raise click.ClickException(str(e))

    def _render(d):
        tag = " [DRY-RUN]" if d.get("dry_run") else ""
        if d["existed"]:
            click.secho(f"✓ 컬렉션 '{d['collection']}' 삭제 (지식 {d['points']}건){tag}", fg="green")
        else:
            click.echo(f"컬렉션 '{d['collection']}' 없음 — 건너뜀{tag}")
        if d["cartridge"]:
            click.secho(f"✓ 장착 해제: {d['cartridge']} (슬롯 {d['prompts_restored']}개 복원){tag}", fg="green")
        if not d.get("dry_run"):
            click.secho("클린 콘솔 — 반영하려면 ./run.sh restart", bold=True)

    _emit(r, ctx, _render)


if __name__ == "__main__":
    cli(prog_name="aibotctl")
