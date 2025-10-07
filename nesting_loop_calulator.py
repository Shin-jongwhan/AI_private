# ─────────────────────────────────────────────────────────────────────────────
# 다중 중첩 합을 O(∑|L_i|)로 빠르게 계산하는 유틸
#   - 입력: lsNesting  (2중 이상 중첩 리스트, 예: [[0,1],[3,4], ...])
#   - 정의:
#       * 덧셈:  Σ(x1 + x2 + ... + xk)
#       * 뺄셈:  Σ(x1 - x2 - ... - xk)   # 첫 리스트는 +, 나머지는 -
#       * 곱셈:  Σ(x1 * x2 * ... * xk)
#       * 나눗셈: Σ(x1 / (x2 * ... * xk))  # 분모 0은 예외 처리(제외)
#
# 시간 복잡도 요약(각 함수 공통 가정):
#   - k = 리스트 개수, n_m = |L_m| (m=1..k)
#   - 나이브(중첩루프): O(∏ n_m)        ← 모든 조합을 직접 순회
#   - 본 구현(공식 이용): O(∑ n_m + k)  ← 각 리스트 합/역수합만 계산하면 됨
#     (∑ n_m 가 지배적. 길이 곱 대신 길이들의 합으로 감소 → 큰 성능 이득)
# 공간 복잡도: O(1) (입력 제외, 상수 보조 변수만 사용)
#
# ∏ 는 각 항목의 곱을 의미함
# ∑ 은 각 항목의 합산을 의미함
# ─────────────────────────────────────────────────────────────────────────────

def nested_add(lsNesting):
    """
    모든 조합에 대해 (x1 + x2 + ... + xk)를 합산.
    공식: sum_m [ (sum(L_m)) * (∏_{r≠m} |L_r|) ]

    시간 복잡도:
      - 나이브: O(∏ n_m)
      - 본 함수: O(∑ n_m + k)
        * 각 리스트 합(sum)을 구하는 데 O(∑ n_m)
        * 전체 길이 곱과 각 항의 '다른 리스트 길이들의 곱' 계산은 O(k)
    """
    lsLens = [len(lsList) for lsList in lsNesting]                 # 각 리스트 길이
    if any(nLen == 0 for nLen in lsLens):                          # 빈 리스트가 있으면 조합 수 0
        return 0

    nProd_all = 1
    for nLen in lsLens:
        nProd_all *= nLen                                          # 전체 길이 곱 (조합 수)

    nTotal = 0
    for nIdx, lsList in enumerate(lsNesting):
        nSum_current = sum(lsList)                                 # 해당 리스트의 합  → O(|L_m|)
        nMult_others = nProd_all // lsLens[nIdx]                   # 다른 리스트 길이들의 곱 → O(1)
        nTotal += nSum_current * nMult_others
    return nTotal


def nested_sub(lsNesting):
    """
    모든 조합에 대해 (x1 - x2 - ... - xk)를 합산.
    공식: +term(L1) - term(L2) - ... - term(Lk)
          term(Lm) = (sum(L_m)) * (∏_{r≠m} |L_r|)

    시간 복잡도:
      - 나이브: O(∏ n_m)
      - 본 함수: O(∑ n_m + k)
        * 합계 계산 비용이 지배 → O(∑ n_m)
    """
    lsLens = [len(lsList) for lsList in lsNesting]
    if any(nLen == 0 for nLen in lsLens):
        return 0

    nProd_all = 1
    for nLen in lsLens:
        nProd_all *= nLen

    nTotal = 0
    for nIdx, lsList in enumerate(lsNesting):
        nSum_current = sum(lsList)
        nMult_others = nProd_all // lsLens[nIdx]
        if nIdx == 0:                                              # 첫 리스트는 +
            nTotal += nSum_current * nMult_others
        else:                                                      # 나머지는 -
            nTotal -= nSum_current * nMult_others
    return nTotal


def nested_mul(lsNesting):
    """
    모든 조합에 대해 (x1 * x2 * ... * xk)를 합산.
    공식: ∏_m (sum(L_m))

    시간 복잡도:
      - 나이브: O(∏ n_m)
      - 본 함수: O(∑ n_m)
        * 각 리스트의 합을 한 번씩만 구함 → O(∑ n_m)
        * 합들의 곱은 O(k)
    """
    if any(len(lsList) == 0 for lsList in lsNesting):              # 곱의 합 정의상, 빈 리스트면 0
        return 0
    nProd_sums = 1
    for lsList in lsNesting:
        nProd_sums *= sum(lsList)                                  # 각 리스트 합 → O(|L_m|)
    return nProd_sums


def nested_div(lsNesting):
    """
    모든 조합에 대해 x1 / (x2 * x3 * ... * xk)를 합산.
    정의:
      - 첫 리스트(lsNesting[0])는 분자.
      - 나머지 리스트들은 분모. 이들에서 '0'은 예외 처리(제외)하여 역수 합을 계산.
      - 공식: (sum(L1)) * ∏_{m=2..k} (sum(1 / y for y in L_m if y != 0))

    시간 복잡도:
      - 나이브: O(∏ n_m)
      - 본 함수: O(∑ n_m + k)
        * 분자 합 계산 O(|L1|), 각 분모 리스트의 역수 합 O(|L_m|)
        * 리스트 전체로 O(∑ n_m), 곱/상수 연산은 O(k)
    주의:
      - 분모 리스트가 전부 0이거나 비어 역수합이 0이면 해당 곱 항이 0 → 전체 결과 0
    """
    if len(lsNesting) == 0:
        return 0.0

    lsNumerator = lsNesting[0]
    lsDenoms = lsNesting[1:]

    flSum_num = float(sum(lsNumerator))                            # 분자 합(실수형)
    flProd_inv = 1.0                                               # 분모 역수합들의 곱

    for lsD in lsDenoms:
        flSum_inv = 0.0
        for nVal in lsD:
            if nVal != 0:                                          # 0은 예외 처리(제외)
                flSum_inv += 1.0 / nVal                            # O(|L_m|)
        flProd_inv *= flSum_inv                                    # 어떤 분모 리스트라도 역수합=0이면 전체 0

    return flSum_num * flProd_inv


# ─────────────────────────────────────────────────────────────────────────────
# '분리 가능한 곱들의 유한 합' (단순형: T=1, 항등함수) 계산 함수. 
#   정의: 모든 조합에 대해 Π_m x_m 를 합산
#         Σ_{모든 조합} Π_m x_m  =  Π_m ( Σ_{x∈L_m} x )
#
# 사용처
#   - 행렬 곱연산에서 각 원소의 연산. 2개 행렬 곱은 효과가 별로 없지만, 3개 이상으로 가면 폭발적인 연산 효과.
#
# 입력:
#   - lsNesting : [L1, L2, ..., Lk]  (각 Lm은 수들의 리스트)
#
# 시간복잡도:
#   - 나이브(중첩 전부 순회): O(∏|L_m|)
#   - 본 함수(공식 이용)   : O(∑|L_m| + k)  ← 각 리스트를 한 번씩만 훑어 합을 구한 뒤 (k-1) 번 곱함
#     * k: nested list 개수 (len(lsNesting))
#     * 쉽게 설명하자면 [ls1, ls2, ls3] 이 [[1,2,3], [1,2], [3,4]] 이런 형태라면 -> O(3+2+2)
# 공간복잡도: O(1)
# 주의:
#   - 어떤 L_m이라도 빈 리스트면 가능한 조합이 없으므로 결과는 0.0
# ─────────────────────────────────────────────────────────────────────────────
# ───────────────────────────────────────────────────────────────
# 1) 원래 방식(나이브): 모든 조합을 직접 순회해 x*y*z 합산
#    시간복잡도 O(∏ |L_m|) = O(2*1*1) = O(2)
# 예시 코드
'''
# 예시 데이터: [[1, 2], [3], [4]]
lsNesting = [[1, 2], [3], [4]]

print("▶ 원래 방식(나이브) 진행")
nTotal_naive = 0
for nX in lsNesting[0]:
    for nY in lsNesting[1]:
        for nZ in lsNesting[2]:
            nTerm = nX * nY * nZ
            nTotal_naive += nTerm
            print(f"  조합 ({nX},{nY},{nZ}): {nX}*{nY}*{nZ} = {nTerm}  | 누적합 = {nTotal_naive}")
print(f"  → 최종 합(나이브) = {nTotal_naive}\n")
'''
# ───────────────────────────────────────────────────────────────
# 추가 아이디어 : 희소곱(값이 0이 있는 곱이라면, 그 값은 제외하고 연산시키면 loop를 덜 돌 수 있음)
def separable_product_sum(lsNesting):
    # 빈 입력 처리
    if not lsNesting:
        return 0.0

    # 가능한 조합이 하나도 없는 경우(어느 한 리스트가 비어 있음)
    for lsLm in lsNesting:
        if len(lsLm) == 0:
            return 0.0

    # Π_m ( Σ_{x∈L_m} x )
    flTotal = 1.0
    for lsLm in lsNesting:
        flSum = 0.0
        for nVal in lsLm:
            flSum += nVal
        flTotal *= flSum

    return flTotal


# ─────────────────────────────────────────────────────────────────────────────
# 간단 검증 예시 (원한다면 주석 해제하여 확인)
# ─────────────────────────────────────────────────────────────────────────────
'''
if __name__ == "__main__":
    lsNesting = [[1, 2], [3, 4]]

    nSum_loop = 0
    nSub_loop = 0
    nMul_loop = 0
    flDiv_loop = 0.0
    for nA in lsNesting[0]:
        for nB in lsNesting[1]:
            nSum_loop += (nA + nB)
            nSub_loop += (nA - nB)
            nMul_loop += (nA * nB)
            if nB != 0:
                flDiv_loop += (nA / nB)

    print("add:", nested_add(lsNesting), "==", nSum_loop)
    print("sub:", nested_sub(lsNesting), "==", nSub_loop)
    print("mul:", nested_mul(lsNesting), "==", nMul_loop)
    print("div:", nested_div(lsNesting), "==", flDiv_loop)
'''