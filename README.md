# vHIT 웹 데모 (2단계: Pyodide 데모)

## 구성
- `index.html` — 단일 페이지 데모 (폴더 업로드 → 파싱 → 통계 → JSON 다운로드)
- `vhit_core.py` — tkinter 제거한 순수 분석 코어 (1단계 산출물)

## 로컬 실행
브라우저에서 `file://`로 바로 열면 `fetch('vhit_core.py')`가 CORS로 막힙니다.
간단한 정적 서버가 필요합니다:

```bash
cd vhit_web
python -m http.server 8000
# 브라우저에서 http://localhost:8000 접속
```

## 배포
정적 호스팅(GitHub Pages 등)에 두 파일을 그대로 올리면 됩니다.
**서버는 파일을 받지 않습니다** — 모든 계산은 브라우저 WASM(Pyodide)에서 실행되고
환자 데이터는 PC를 벗어나지 않습니다.

## 인코딩
환자 한글 파일명·내용(cp949/euc-kr)은 JS `TextDecoder`가 디코딩한 뒤
유니코드 문자열만 파이썬에 전달합니다. 파이썬 코어는 인코딩을 다루지 않습니다.

## 현재 데모 범위
- 두 CSV 포맷(신/구) 동시 파싱
- 그룹간 비모수 비교 (사람평균 우선 표시)
- 종단 paired Wilcoxon (acute → followup 자동추론)
- Claude 해석용 JSON + 프롬프트 다운로드

## 다음 단계 (미구현)
- 3단계: figure 시각화 Plotly.js 이식
- 필터 ON/OFF · trim · 카테고리 모드 UI 토글
- CSV/Excel 내보내기
