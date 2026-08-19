#!/usr/bin/env python3
"""models.yaml 프리셋 파서 — install.sh용 (stdlib만, venv 前 시스템 python으로 실행).

pyyaml에 의존하지 않는다(설치 초기 단계라 venv 미생성). models.yaml의 고정
들여쓰기 구조(0/2/4/6칸)를 직접 파싱하되, 4칸 빈-값 키(server_overrides)의
6칸 자식(n_ctx 등)을 그 키 아래 중첩 dict로 올바르게 담는다.

  parse_preset.py <tier>              → CANDIDATES 목록 출력
  parse_preset.py <tier> <preset>     → SELECTED 상세(export용) 출력

CANDIDATES 는 --vram/--ram (단위 GB) 을 주면 min_vram_gb/min_ram_gb 미달 프리셋을
걸러낸다. 둘 다 생략하면 필터 없이 전량 출력한다(CI 의 전 티어 순회가 이 경로).
필터 결과가 0건이면 헤더만 출력하므로 호출자가 그 경우를 처리해야 한다.
"""
import os
import sys


_DEFAULT_MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'models.yaml')


def load(path=None):
    path = path or _DEFAULT_MODELS
    data, cur, sub, nested = {}, None, None, None
    for line in open(path, encoding='utf-8'):
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        indent = len(line) - len(line.lstrip())
        key, _, val = line.strip().partition(':')
        val = val.strip()
        if ' #' in val:  # 인라인 주석 제거 (값에 '#'을 쓰려면 별도 행 주석으로)
            val = val.split(' #', 1)[0].rstrip()
        val = val.strip('"')
        if indent == 0:
            cur, sub, nested = key, None, None
            data[cur] = {}
        elif indent == 2:
            sub, nested = key, None
            data[cur][sub] = val if val else {}
        elif indent == 4:
            if not isinstance(data[cur].get(sub), dict):
                data[cur][sub] = {}
            if val:
                data[cur][sub][key] = val
                nested = None
            else:
                # 빈 값 4칸 키 = 중첩 컨테이너 (예: server_overrides:)
                data[cur][sub][key] = {}
                nested = key
        elif indent >= 6 and nested is not None:
            data[cur][sub][nested][key] = val
    return data


def _int(val, default=0):
    try:
        return int(str(val).strip())
    except (TypeError, ValueError):
        return default


def main():
    argv = sys.argv[1:]
    vram_gb = ram_gb = None
    rest = []
    i = 0
    while i < len(argv):
        if argv[i] == '--vram' and i + 1 < len(argv):
            vram_gb = _int(argv[i + 1]); i += 2
        elif argv[i] == '--ram' and i + 1 < len(argv):
            ram_gb = _int(argv[i + 1]); i += 2
        else:
            rest.append(argv[i]); i += 1

    tier = rest[0]
    preset = rest[1] if len(rest) > 1 else ''
    data = load()
    tiers, presets = data['tiers'], data['presets']
    if tier not in tiers:
        sys.exit(f"unknown tier: {tier}")

    if not preset:
        print("CANDIDATES")
        cands = [p.strip() for p in tiers[tier].get('presets', '').strip('[]').split(',')]
        n = 0
        for name in cands:
            p = presets.get(name, {})
            need_v, need_r = _int(p.get('min_vram_gb')), _int(p.get('min_ram_gb'))
            # 하드 제약 미달이면 제외. HW 인자가 없으면(=CI) 필터를 걸지 않는다.
            if vram_gb is not None and vram_gb < need_v:
                continue
            if ram_gb is not None and ram_gb < need_r:
                continue
            n += 1
            print(f"{n}|{name}|{need_v}|{need_r}|{p.get('vram', '')}|{p.get('note', '')}")
        return

    if preset not in presets:
        sys.exit(f"unknown preset: {preset}")
    p = presets[preset]
    print("SELECTED")
    for k in ['handler', 'runtime', 'repo', 'include', 'local_dir',
              'path_key', 'server_section', 'extra_args']:
        print(f"{k}={p.get(k, '')}")
    ov = p.get('server_overrides', {})
    if isinstance(ov, dict) and ov:
        print("overrides=" + ",".join(f"{k}:{v}" for k, v in ov.items()))
    emb = data.get('embedding', {})
    print(f"emb_repo={emb.get('repo', 'BAAI/bge-m3')}")
    print(f"emb_dir={emb.get('local_dir', 'models/bge-m3')}")


if __name__ == '__main__':
    main()
