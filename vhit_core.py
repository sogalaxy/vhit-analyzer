"""
vhit_core.py — vHIT FFT 분석 순수 코어 (tkinter/matplotlib 비의존)
===================================================================
data_fft_folder_clean11.py 에서 분석 로직만 추출한 모듈.

설계 원칙
---------
1. tkinter / matplotlib / filedialog / messagebox 의존성 0
2. 입력은 파일경로가 아닌 **문자열(파일 내용)** 과 **dict**
3. 출력은 GUI가 아닌 **list / dict / DataFrame** (직렬화 가능)
4. numpy / scipy / pandas 만 사용 → Pyodide WASM 호환

인코딩 주의
-----------
원본은 파이썬이 직접 cp949/euc-kr 파일을 열었으나, 웹에서는
브라우저(FileReader/TextDecoder)가 디코딩한 **유니코드 문자열**을 넘긴다.
따라서 이 모듈의 파서는 인코딩을 다루지 않고 디코딩된 str만 받는다.

핵심 도메인 지식 (유실 금지)
----------------------------
- CSV 두 버전: 신버전(ABP, HC)은 임펄스 블록 행 첫 컬럼에 키워드 직접 위치,
  구버전(AHS, VN)은 첫 컬럼이 빈 문자열. 파싱 루프 최초에
  cols[0]=='' 이면 한 칸 shift 하여 단일 코드로 처리.
- FFT 표준 길이 250 (TARGET_FFT_LEN): zero-padding / trimming.
- ROI 기본 10–50 Hz: 보상성 사케이드의 고주파 에너지 포착.
- Power Ratio = Σ|FFT_eye|² / Σ|FFT_head|²  (ROI 대역)
  Area Ratio  = Σ|FFT_eye|  / Σ|FFT_head|   (ROI 대역)
- 분포가 넓고 skewed → 비모수 검정(Mann-Whitney/Wilcoxon/Dunn) 일관 사용.
- 임펄스단위 유의성은 임펄스 개수로 부풀려질 수 있음 → subject_means·paired 신뢰.
"""

import csv
import re
import io
from datetime import datetime
from itertools import combinations
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats


# ══════════════════════════════════════════════════════════════
# 전역 상수
# ══════════════════════════════════════════════════════════════
TARGET_FFT_LEN = 250
DEFAULT_FS = 250

# 카테고리 정의 (4분할 / 2분할)
CAT_DEFS = [
    ('HIMP Left',   'HIMP',  'Left'),
    ('HIMP Right',  'HIMP',  'Right'),
    ('SHIMP Left',  'SHIMP', 'Left'),
    ('SHIMP Right', 'SHIMP', 'Right'),
]
CAT_DEFS_MERGED = [
    ('HIMP (L+R)',  'HIMP',  'All'),
    ('SHIMP (L+R)', 'SHIMP', 'All'),
]


def get_cat_defs(merged=False):
    """카테고리 모드에 따른 정의 반환."""
    return CAT_DEFS_MERGED if merged else CAT_DEFS


def match_dir(imp_dir, def_dir):
    """임펄스 방향이 카테고리 정의 방향에 매칭되는지. 'All'은 L·R 모두."""
    if def_dir == 'All':
        return imp_dir in ('Left', 'Right')
    return imp_dir == def_dir


def cat_label(t, d):
    """사람-읽기용 카테고리 라벨."""
    return f"{t} (L+R)" if d == 'All' else f"{t} {d}"


# ══════════════════════════════════════════════════════════════
# FFT / Ratio (원본 calc_fft / hf_ratio 그대로)
# ══════════════════════════════════════════════════════════════
def _standardize(data):
    """FFT 표준 길이로 trim / zero-pad."""
    n = len(data)
    if n >= TARGET_FFT_LEN:
        return data[:TARGET_FFT_LEN]
    return np.pad(data, (0, TARGET_FFT_LEN - n), mode='constant')


def calc_fft(data, fs=DEFAULT_FS):
    """rfft 진폭 스펙트럼 (정규화: /n)."""
    n = len(data)
    return (np.fft.rfftfreq(n, d=1 / fs),
            np.abs(np.fft.rfft(data)) / n)


def hf_ratio(head, eye, method='power', low=10, high=50, fs=DEFAULT_FS):
    """
    ROI 대역(low–high Hz)에서 eye/head 비율.
      power  : Σ|FFT|²  (Power Ratio)
      linear : Σ|FFT|   (Area Ratio)
    """
    fh, mh = calc_fft(head, fs)
    fe, me = calc_fft(eye, fs)
    mask = (fh >= low) & (fh <= high)
    if method == 'power':
        ph = np.sum(mh[mask] ** 2)
        pe = np.sum(me[mask] ** 2)
    else:
        ph = np.sum(mh[mask])
        pe = np.sum(me[mask])
    return pe / ph if ph > 0 else 0.0


# ══════════════════════════════════════════════════════════════
# 통계 유틸 (원본 그대로 — 이미 순수 함수)
# ══════════════════════════════════════════════════════════════
def rank_biserial_r(a, b):
    """Mann-Whitney 기반 효과크기."""
    if len(a) < 1 or len(b) < 1:
        return float('nan')
    U, _ = stats.mannwhitneyu(a, b, alternative='two-sided')
    return 1 - (2 * U) / (len(a) * len(b))


def dunn_bonferroni(groups, names):
    """Dunn's post-hoc + Bonferroni (scipy만 사용)."""
    all_data = np.concatenate(groups)
    N = len(all_data)
    if N < 3:
        return {}
    ranks = stats.rankdata(all_data)
    ns = [len(g) for g in groups]
    mean_ranks, idx = [], 0
    for n in ns:
        mean_ranks.append(np.mean(ranks[idx:idx + n]))
        idx += n
    _, counts = np.unique(all_data, return_counts=True)
    T_corr = np.sum(counts ** 3 - counts)
    pairs = list(combinations(range(len(groups)), 2))
    raw_ps, zs = {}, {}
    for i, j in pairs:
        var = (N * (N + 1) / 12 - T_corr / (12 * (N - 1))) * (1 / ns[i] + 1 / ns[j])
        if var <= 0:
            z, p = 0.0, 1.0
        else:
            z = (mean_ranks[i] - mean_ranks[j]) / np.sqrt(var)
            p = 2 * stats.norm.sf(abs(z))
        raw_ps[(names[i], names[j])] = p
        zs[(names[i], names[j])] = z
    m = len(pairs)
    result = {}
    for key, p in raw_ps.items():
        g1, g2 = key
        result[key] = {
            'p_raw': p,
            'p_adj': min(p * m, 1.0),
            'z': zs[key],
            'r': rank_biserial_r(groups[names.index(g1)], groups[names.index(g2)]),
        }
    return result


def compute_pairwise_stats(vals_by_group, active):
    """
    vals_by_group: {group_name: [values]}
    active: 그룹명 리스트 (정의 순서)
    Returns: {'kw': (stat,p) or None, 'pairs': {(g1,g2):{p_raw,p_adj,z,r}}}
    """
    non_empty = [(g, vals_by_group.get(g, [])) for g in active
                 if len(vals_by_group.get(g, [])) >= 2]
    result = {'kw': None, 'pairs': {}}
    if len(non_empty) < 2:
        return result
    if len(non_empty) >= 3:
        kw_stat, kw_p = stats.kruskal(*[d for _, d in non_empty])
        result['kw'] = (kw_stat, kw_p)
        result['pairs'] = dunn_bonferroni(
            [d for _, d in non_empty], [g for g, _ in non_empty])
    else:
        g1, d1 = non_empty[0]
        g2, d2 = non_empty[1]
        _, p = stats.mannwhitneyu(d1, d2, alternative='two-sided')
        result['pairs'][(g1, g2)] = {
            'p_raw': p, 'p_adj': p, 'z': float('nan'),
            'r': rank_biserial_r(d1, d2)}
    return result


def wilcoxon_paired(a, b):
    """Wilcoxon signed-rank + matched rank-biserial r. n<4면 (None,None,None)."""
    if len(a) < 4 or len(b) < 4 or len(a) != len(b):
        return None, None, None
    try:
        stat, p = stats.wilcoxon(a, b, alternative='two-sided')
        n = len(a)
        r = 1 - (2 * stat) / (n * (n + 1) / 2)
        return stat, p, r
    except Exception:
        return None, None, None


# ══════════════════════════════════════════════════════════════
# 파서 — 문자열 입력 (핵심 변경점)
# ══════════════════════════════════════════════════════════════
def _parse_floats(tokens):
    r = []
    for t in tokens:
        try:
            if t:
                r.append(float(t))
        except ValueError:
            pass
    return r


def parse_csv_content(content, filename, group_name, fft_low=10, fft_high=50,
                      fs=DEFAULT_FS):
    """
    CSV 파일 '내용 문자열'을 파싱하여 임펄스 dict 리스트 반환.

    원본 load_single_file 과의 차이:
      - 파일경로/인코딩 처리 제거 → 디코딩된 str(content)을 직접 받음
      - GUI 사이드이펙트 없음, 반환값만 사용

    Parameters
    ----------
    content   : str   디코딩된 CSV 전체 텍스트
    filename  : str   환자ID 추출용 (예: 강점순_1901971_2021_05_31.csv)
    group_name: str   'HC' / 'VN acute' 등
    fft_low/high : ROI 대역
    fs        : 샘플링레이트

    Returns
    -------
    (impulses: list[dict], deleted_count: int, error: str|None)
    """
    deleted_count = 0
    impulses = []
    try:
        lines = list(csv.reader(io.StringIO(content)))
        if not lines:
            return [], 0, f"{filename} (빈 파일)"

        # 파일명에서 환자 이름 추출 (첫 '_' 앞 토큰)
        stem = filename.rsplit('.', 1)[0]
        patient_name = stem.split('_')[0].strip()

        cur_type = "HIMP"
        cur_date = "Unknown"
        in_block = False
        b_dir = "Other"
        b_gain = 0.0
        b_pv = 0.0
        b_del = False
        b_eye = None
        b_head = None
        imp_id = 0
        b_saccades = []

        def commit():
            nonlocal imp_id, deleted_count, b_eye, b_head
            if b_eye is None or b_head is None:
                return
            if b_del:
                deleted_count += 1
                return
            e_raw = np.array(b_eye, dtype=float)
            h_raw = np.array(b_head, dtype=float)
            h_std = _standardize(h_raw)
            e_std = _standardize(e_raw)
            ratio = hf_ratio(h_std, e_std, 'power', fft_low, fft_high, fs)
            aratio = hf_ratio(h_std, e_std, 'linear', fft_low, fft_high, fs)
            peak_h = b_pv if b_pv > 0 else float(np.max(np.abs(h_raw)))
            imp_id += 1
            impulses.append({
                'group': group_name,
                'filename': filename,
                'patient_name': patient_name,
                'id': imp_id,
                'type': cur_type,
                'date': cur_date,
                'visit_date': None,
                'visit_order': 1,
                'days_from_first': 0,
                'visit_order_global': 1,
                'days_from_first_global': 0,
                'direction': b_dir,
                'head': h_raw,
                'eye': e_raw,
                'ratio': ratio,
                'area_ratio': aratio,
                'gain': b_gain,
                'peak_h_vel': peak_h,
                'saccades': list(b_saccades),
                'saccade_count': len(b_saccades),
            })

        for row in lines:
            if not row:
                continue
            cols = [str(c).strip() for c in row]

            # 구버전 호환: 첫 컬럼이 빈 문자열이면 한 칸 shift
            if cols[0] == '' and len(cols) > 1:
                cols = cols[1:]

            lc0 = cols[0].lower()

            if "test type" in lc0:
                cur_type = ("SHIMP"
                            if len(cols) > 1 and
                               ("suppression" in cols[1].lower() or
                                "shimp" in cols[1].lower())
                            else "HIMP")
                continue
            if "test date" in lc0:
                cur_date = cols[1].split(' ')[0] if len(cols) > 1 else "Unknown"
                continue
            if re.match(r'^impulse\s+\d+', lc0):
                if in_block:
                    commit()
                in_block = True
                b_del = False
                b_gain = 0.0
                b_pv = 0.0
                b_eye = None
                b_head = None
                b_dir = "Other"
                b_saccades = []
                joined = " ".join(cols).lower()
                if "left" in joined:
                    b_dir = "Left"
                elif "right" in joined:
                    b_dir = "Right"
                continue
            if not in_block:
                continue

            if "direction" in lc0:
                dv = cols[1].lower() if len(cols) > 1 else ""
                b_dir = ("Left" if "left" in dv or dv == "l" else
                         "Right" if "right" in dv or dv == "r" else "Other")
            elif lc0 == "gain":
                try:
                    b_gain = float(cols[1])
                except (ValueError, IndexError):
                    b_gain = 0.0
            elif "peak velocity" in lc0 or "peak head velocity" in lc0:
                try:
                    b_pv = float(cols[1])
                except (ValueError, IndexError):
                    b_pv = 0.0
            elif "deleted" in lc0:
                b_del = len(cols) > 1 and cols[1].strip().lower() == "yes"
            elif lc0 == "eye":
                pts = _parse_floats(cols[1:])
                if len(pts) > 20:
                    b_eye = pts
            elif lc0 == "head":
                pts = _parse_floats(cols[1:])
                if len(pts) > 20:
                    b_head = pts
            elif re.match(r'^saccade\s+\d+', lc0):
                try:
                    lat_str = cols[2] if len(cols) > 2 else ""
                    amp_str = cols[4] if len(cols) > 4 else ""
                    lat_m = re.search(r'[-\d.]+', lat_str)
                    amp_m = re.search(r'[-\d.]+', amp_str)
                    b_saccades.append({
                        'n': int(re.search(r'\d+', cols[0]).group()),
                        'latency_ms': float(lat_m.group()) if lat_m else float('nan'),
                        'amplitude': float(amp_m.group()) if amp_m else float('nan'),
                    })
                except Exception:
                    pass

        if in_block:
            commit()

    except Exception as e:
        return impulses, deleted_count, f"{filename} ({e})"

    return impulses, deleted_count, None


def load_group(file_dict, group_name, fft_low=10, fft_high=50, fs=DEFAULT_FS):
    """
    그룹 하나를 로드. file_dict = {filename: content_str}.

    Returns
    -------
    dict {
      'impulses': list[dict],
      'n_files': int,
      'n_impulses': int,
      'deleted': int,
      'errors': list[str],
    }
    """
    all_imp = []
    total_deleted = 0
    errors = []
    for fname, content in file_dict.items():
        imps, deleted, err = parse_csv_content(
            content, fname, group_name, fft_low, fft_high, fs)
        all_imp.extend(imps)
        total_deleted += deleted
        if err:
            errors.append(err)
    return {
        'impulses': all_imp,
        'n_files': len(file_dict),
        'n_impulses': len(all_imp),
        'deleted': total_deleted,
        'errors': errors,
    }


# ══════════════════════════════════════════════════════════════
# 방문 회차 / 경과일 부여 (원본 _assign_visit_orders 그대로)
# ══════════════════════════════════════════════════════════════
_DATE_FORMATS = [
    '%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y',
    '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S',
    '%Y.%m.%d', '%Y.%m.%d %H:%M:%S',
]


def _parse_date(s):
    s = str(s).strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    tok = s.split(' ')[0]
    for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%Y.%m.%d']:
        try:
            return datetime.strptime(tok, fmt).date()
        except ValueError:
            continue
    return None


def assign_visit_orders(all_impulses):
    """
    날짜 기반 visit_order / days_from_first 부여 (in-place).
      (A) 그룹 내 : visit_order, days_from_first
      (B) 환자 전역(cross-group) : visit_order_global, days_from_first_global
          → 종단(acute/followup 다른 그룹) 분석에 필수
    """
    for imp in all_impulses:
        imp['visit_date'] = _parse_date(imp['date'])

    # (A) 그룹 내
    key_grp = defaultdict(set)
    for imp in all_impulses:
        if imp['visit_date']:
            key_grp[(imp['group'], imp['patient_name'])].add(imp['visit_date'])
    for (grp, pname), dset in key_grp.items():
        sorted_d = sorted(dset)
        first = sorted_d[0]
        d2o = {d: i + 1 for i, d in enumerate(sorted_d)}
        for imp in all_impulses:
            if imp['group'] == grp and imp['patient_name'] == pname:
                vd = imp['visit_date']
                imp['visit_order'] = d2o.get(vd, 1) if vd else 1
                imp['days_from_first'] = (vd - first).days if vd else 0

    # (B) 환자 전역
    key_pt = defaultdict(set)
    for imp in all_impulses:
        if imp['visit_date']:
            key_pt[imp['patient_name']].add(imp['visit_date'])
    for pname, dset in key_pt.items():
        sorted_d = sorted(dset)
        first = sorted_d[0]
        d2o = {d: i + 1 for i, d in enumerate(sorted_d)}
        for imp in all_impulses:
            if imp['patient_name'] == pname:
                vd = imp['visit_date']
                imp['visit_order_global'] = d2o.get(vd, 1) if vd else 1
                imp['days_from_first_global'] = (vd - first).days if vd else 0


# ══════════════════════════════════════════════════════════════
# 필터 (원본 get_filtered 그대로)
# ══════════════════════════════════════════════════════════════
def apply_filter(impulses, active=True, min_head_vel=150.0,
                 max_ratio=1.5, gain_filter=True, gain_max=1.2):
    """이상치 필터. active=False면 원본 그대로."""
    if not active:
        return impulses
    return [i for i in impulses
            if i['peak_h_vel'] >= min_head_vel
            and i['ratio'] <= max_ratio
            and (not gain_filter or i['gain'] <= gain_max)]


# ══════════════════════════════════════════════════════════════
# 그룹별 데이터 추출 (원본 _groups_data 단순화 — trim 옵션 포함)
# ══════════════════════════════════════════════════════════════
def _trim_values(values, use_trim, trim_pct):
    """양측 trim. OFF거나 n<10이면 원본."""
    if not use_trim or not values or len(values) < 10:
        return values
    lo = float(np.percentile(values, trim_pct))
    hi = float(np.percentile(values, 100 - trim_pct))
    trimmed = [v for v in values if lo <= v <= hi]
    return trimmed if len(trimmed) >= 2 else values


def groups_data(valid, active, dk, merged=False, per_person=False,
                use_trim=False, trim_pct=5.0):
    """
    그룹별 × 카테고리별 값 추출.
      per_person=False : 임펄스 단위
      per_person=True  : 파일(사람)별 평균
    Returns: {group: [list per category]}
    """
    cat_defs = get_cat_defs(merged)
    if not per_person:
        return {g: [_trim_values(
                        [i[dk] for i in valid
                         if i['group'] == g and i['type'] == t
                         and match_dir(i['direction'], d)],
                        use_trim, trim_pct)
                    for _, t, d in cat_defs]
                for g in active}
    result = {}
    for g in active:
        files = sorted(set(i['filename'] for i in valid if i['group'] == g))
        result[g] = []
        for _, t, d in cat_defs:
            pool = [i for i in valid
                    if i['group'] == g and i['type'] == t
                    and match_dir(i['direction'], d)]
            pool_vals = [i[dk] for i in pool]
            if (use_trim and len(pool_vals) >= 10):
                trimmed = _trim_values(pool_vals, use_trim, trim_pct)
                if 2 <= len(trimmed) < len(pool_vals):
                    lo = float(np.percentile(pool_vals, trim_pct))
                    hi = float(np.percentile(pool_vals, 100 - trim_pct))
                    pool = [i for i in pool if lo <= i[dk] <= hi]
            means = []
            for f in files:
                vv = [i[dk] for i in pool if i['filename'] == f]
                if vv:
                    means.append(np.mean(vv))
            result[g].append(means)
    return result


# ══════════════════════════════════════════════════════════════
# Paired 데이터 추출 (원본 _get_paired_data 그대로)
# ══════════════════════════════════════════════════════════════
def get_paired_data(valid, acute_g, fu_g, dk='ratio', merged=False):
    """
    acute_g / fu_g 의 동일 patient_name 매칭 → 카테고리별 paired 값.
    Returns: {cat_label: {'acute':[], 'fu':[], 'names':[]}}
    """
    cat_defs = get_cat_defs(merged)
    acute_names = set(i['patient_name'] for i in valid if i['group'] == acute_g)
    fu_names = set(i['patient_name'] for i in valid if i['group'] == fu_g)
    matched = sorted(acute_names & fu_names)

    result = {}
    for label, t, d in cat_defs:
        cat_key = cat_label(t, d)
        a_vals, f_vals, names = [], [], []
        for pname in matched:
            av = [i[dk] for i in valid if i['group'] == acute_g
                  and i['patient_name'] == pname
                  and i['type'] == t and match_dir(i['direction'], d)]
            fv = [i[dk] for i in valid if i['group'] == fu_g
                  and i['patient_name'] == pname
                  and i['type'] == t and match_dir(i['direction'], d)]
            if av and fv:
                a_vals.append(float(np.mean(av)))
                f_vals.append(float(np.mean(fv)))
                names.append(pname)
        result[cat_key] = {'acute': a_vals, 'fu': f_vals, 'names': names}
    return result


# ══════════════════════════════════════════════════════════════
# Claude 리포트 생성 (원본 export_claude_report → 문자열 반환)
# ══════════════════════════════════════════════════════════════
def _sanitize(obj):
    """NaN/Inf → None, numpy 타입 → 기본 타입 (표준 JSON 보장)."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(x) for x in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (np.isnan(v) or np.isinf(v)) else v
    return obj


def build_report(valid, active, acute_g, fu_g, *, merged=False,
                 fft_low=10, fft_high=50, fs=DEFAULT_FS,
                 filter_active=False, use_trim=False, trim_pct=5.0,
                 tool_version="vhit_core"):
    """
    구조화 분석 리포트 dict 생성 (export_claude_report 로직).
    파일 쓰기 없음. report dict 반환 → 호출측에서 json.dumps.
    """
    cat_defs = get_cat_defs(merged)

    meta = {
        "tool_version": tool_version,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "fft_roi_hz": [fft_low, fft_high],
        "fft_length_samples": TARGET_FFT_LEN,
        "sampling_rate_hz": fs,
        "category_mode": "L+R merged" if merged else "L/R separated",
        "filter_active": filter_active,
        "trim_active": use_trim,
        "trim_pct_each_side": trim_pct if use_trim else 0,
        "groups": active,
        "metrics_explained": {
            "PowerRatio": "FFT eye/head power ratio in ROI band (squared magnitude). "
                          "High value = strong high-freq eye energy relative to head, "
                          "typically reflecting compensatory saccade energy.",
            "AreaRatio": "FFT eye/head linear magnitude ratio in ROI band.",
            "Gain": "Conventional VOR gain (eye velocity / head velocity). "
                    "Normal ~1.0, reduced in vestibular hypofunction.",
            "HIMP": "Head Impulse Test (standard). Low gain + covert/overt "
                    "saccades indicate VOR deficit.",
            "SHIMP": "Suppression HIMP. Saccades appear when VOR is intact; "
                     "fewer saccades in patients than normals.",
        },
    }

    def _descr(vals):
        if not vals:
            return None
        arr = np.array(vals, dtype=float)
        return {
            "n": int(len(arr)),
            "mean": round(float(np.mean(arr)), 4),
            "median": round(float(np.median(arr)), 4),
            "sd": round(float(np.std(arr, ddof=1)), 4) if len(arr) > 1 else 0.0,
            "iqr": [round(float(np.percentile(arr, 25)), 4),
                    round(float(np.percentile(arr, 75)), 4)],
        }

    # 기술통계
    descriptives = {}
    for dk, dlbl in [('ratio', 'PowerRatio'), ('area_ratio', 'AreaRatio'), ('gain', 'Gain')]:
        descriptives[dlbl] = {}
        for label, t, d in cat_defs:
            cat = cat_label(t, d)
            descriptives[dlbl][cat] = {}
            for g in active:
                vals = [i[dk] for i in valid
                        if i['group'] == g and i['type'] == t
                        and match_dir(i['direction'], d)]
                descriptives[dlbl][cat][g] = _descr(vals)

    # 그룹간 비교 (임펄스 + 사람평균)
    group_comparisons = {}
    for dk, dlbl in [('ratio', 'PowerRatio'), ('area_ratio', 'AreaRatio'), ('gain', 'Gain')]:
        group_comparisons[dlbl] = {}
        for per_p, mlbl in [(False, 'all_impulses'), (True, 'subject_means')]:
            gd = groups_data(valid, active, dk, merged=merged, per_person=per_p,
                             use_trim=use_trim, trim_pct=trim_pct)
            group_comparisons[dlbl][mlbl] = {}
            for ci, (label, t, d) in enumerate(cat_defs):
                cat = cat_label(t, d)
                vals = {g: gd[g][ci] for g in active}
                sr = compute_pairwise_stats(vals, active)
                entry = {"omnibus": None, "pairwise": []}
                if sr['kw']:
                    entry["omnibus"] = {
                        "test": "Kruskal-Wallis",
                        "statistic": round(sr['kw'][0], 4),
                        "p": round(sr['kw'][1], 6)}
                for (g1, g2), pv in sr['pairs'].items():
                    entry["pairwise"].append({
                        "group1": g1, "group2": g2,
                        "test": "Dunn" if sr['kw'] else "Mann-Whitney",
                        "p_raw": round(pv.get('p_raw', float('nan')), 6),
                        "p_adj": round(pv.get('p_adj', float('nan')), 6),
                        "effect_r": round(pv.get('r', float('nan')), 4),
                        "significant": pv.get('p_adj', 1) < 0.05})
                group_comparisons[dlbl][mlbl][cat] = entry

    # 종단 Paired
    longitudinal = None
    paired_skipped_reason = None
    if acute_g not in active:
        paired_skipped_reason = f"acute 그룹 '{acute_g}'이(가) 로드된 그룹에 없음"
    elif fu_g not in active:
        paired_skipped_reason = f"followup 그룹 '{fu_g}'이(가) 로드된 그룹에 없음"
    elif acute_g == fu_g:
        paired_skipped_reason = "acute와 followup 그룹이 동일함"
    else:
        longitudinal = {"acute_group": acute_g, "followup_group": fu_g, "metrics": {}}
        for dk, dlbl in [('ratio', 'PowerRatio'), ('area_ratio', 'AreaRatio'), ('gain', 'Gain')]:
            paired = get_paired_data(valid, acute_g, fu_g, dk, merged=merged)
            longitudinal["metrics"][dlbl] = {}
            for cat, data in paired.items():
                a_vals, f_vals, names = data['acute'], data['fu'], data['names']
                stat, p, r = wilcoxon_paired(a_vals, f_vals)
                days = []
                for nm in names:
                    dd = [i.get('days_from_first_global', 0) for i in valid
                          if i['group'] == fu_g and i['patient_name'] == nm]
                    days.append(int(np.mean(dd)) if dd else None)
                deltas = [f - a for a, f in zip(a_vals, f_vals)]
                vd = [(dd, de) for dd, de in zip(days, deltas) if dd is not None]
                if len(vd) >= 4:
                    rho_d, p_d = stats.spearmanr([x[0] for x in vd], [x[1] for x in vd])
                else:
                    rho_d, p_d = None, None
                n_dec = sum(1 for de in deltas if de < 0)
                n_inc = sum(1 for de in deltas if de > 0)
                longitudinal["metrics"][dlbl][cat] = {
                    "n_pairs": len(a_vals),
                    "acute_mean": round(np.mean(a_vals), 4) if a_vals else None,
                    "followup_mean": round(np.mean(f_vals), 4) if f_vals else None,
                    "delta_mean": round(np.mean(deltas), 4) if deltas else None,
                    "delta_pct": round(np.mean(deltas) / np.mean(a_vals) * 100, 2)
                                 if a_vals and np.mean(a_vals) else None,
                    "wilcoxon_p": round(p, 6) if p is not None else None,
                    "wilcoxon_r": round(r, 4) if r is not None else None,
                    "significant": (p < 0.05) if p is not None else None,
                    "direction_decreased": n_dec,
                    "direction_increased": n_inc,
                    "days_interval_mean": round(np.mean([d for d in days if d]), 0)
                                          if any(days) else None,
                    "spearman_days_vs_delta": round(rho_d, 4) if rho_d is not None else None,
                    "p_days_vs_delta": round(p_d, 4) if p_d is not None else None,
                }

    # Gain × PowerRatio 상관 (HIMP만)
    gain_fft_corr = {}
    for g in active:
        gain_fft_corr[g] = {}
        for label, t, d in cat_defs:
            if t != 'HIMP':
                continue
            cat = cat_label(t, d)
            imp_g = [i for i in valid if i['group'] == g and i['type'] == t
                     and match_dir(i['direction'], d)]
            if len(imp_g) >= 3:
                rho, p = stats.spearmanr([i['gain'] for i in imp_g],
                                         [i['ratio'] for i in imp_g])
                gain_fft_corr[g][cat] = {
                    "spearman_rho": round(rho, 4),
                    "p": round(p, 6),
                    "n": len(imp_g),
                    "interpretation": ("strong_negative" if rho < -0.5 else
                                       "weak_negative" if rho < 0 else "positive")}

    report = {
        "metadata": meta,
        "descriptive_statistics": descriptives,
        "group_comparisons": group_comparisons,
        "longitudinal_paired_analysis": longitudinal,
        "gain_vs_fft_correlation": gain_fft_corr,
        "_paired_skipped_reason": paired_skipped_reason,
    }
    return _sanitize(report)
