import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 기본 설정
# =========================

st.set_page_config(
    page_title="Block Overhang Simulation",
    layout="wide"
)

st.title("블록 쌓기 문제: 조화급수와 로그 근사 시뮬레이션")

st.markdown(
    r"""
    전체 돌출량은 다음과 같이 정의한다.

    $$
    D_N=\frac{L}{2}H_N
    =
    \frac{L}{2}\sum_{n=1}^{N}\frac{1}{n}
    $$

    조화수는 오일러-마스케로니 상수 \(\gamma\)를 이용해 다음처럼 근사할 수 있다.

    $$
    H_N
    \approx
    \ln N+\gamma+\frac{1}{2N}
    -\frac{1}{12N^2}
    +\frac{1}{120N^4}
    $$
    """
)

# =========================
# 상수
# =========================

gamma = 0.5772156649015329

# =========================
# 사이드바 입력
# =========================

st.sidebar.header("변수 조절")

L = st.sidebar.slider(
    "블록 길이 L",
    min_value=0.5,
    max_value=5.0,
    value=1.0,
    step=0.1
)

N_max = st.sidebar.slider(
    "왼쪽 그래프의 최대 블록 수 N_max",
    min_value=10,
    max_value=5000,
    value=500,
    step=10
)

n_selected = st.sidebar.slider(
    "오른쪽 그래프에서 볼 블록 수 n",
    min_value=1,
    max_value=N_max,
    value=min(50, N_max),
    step=1
)

# =========================
# 함수 정의
# =========================

def harmonic_numbers(N):
    """
    H_1, H_2, ..., H_N 계산
    """
    n = np.arange(1, N + 1)
    return np.cumsum(1 / n)


def exact_overhang(N, L):
    """
    D_N = L/2 * H_N
    """
    H = harmonic_numbers(N)
    return (L / 2) * H


def log_approx(N_values, L):
    """
    D_N ≈ L/2 (ln N + gamma)
    """
    N_values = np.array(N_values)
    return (L / 2) * (np.log(N_values) + gamma)


def precise_approx(N_values, L):
    """
    D_N ≈ L/2 (ln N + gamma + 1/(2N) - 1/(12N^2) + 1/(120N^4))
    """
    N_values = np.array(N_values)
    return (L / 2) * (
        np.log(N_values)
        + gamma
        + 1 / (2 * N_values)
        - 1 / (12 * N_values**2)
        + 1 / (120 * N_values**4)
    )


# =========================
# 데이터 계산
# =========================

N_values = np.arange(1, N_max + 1)

D_exact = exact_overhang(N_max, L)
D_log = log_approx(N_values, L)
D_precise = precise_approx(N_values, L)

D_selected = D_exact[n_selected - 1]
H_selected = 2 * D_selected / L

# =========================
# 수치 결과 표시
# =========================

col_info1, col_info2, col_info3 = st.columns(3)

with col_info1:
    st.metric("선택한 블록 수 n", f"{n_selected}")

with col_info2:
    st.metric("전체 돌출량 Dₙ", f"{D_selected:.6f}")

with col_info3:
    st.metric("블록 길이 대비 돌출량 Dₙ/L", f"{D_selected / L:.6f}")

# =========================
# 그래프 영역
# =========================

left_col, right_col = st.columns(2)

# =========================
# 왼쪽 그래프
# =========================

with left_col:
    st.subheader("1. 정확값과 오일러-마스케로니 근사 비교")

    fig1, ax1 = plt.subplots(figsize=(7, 5))

    ax1.plot(
        N_values,
        D_exact,
        label=r"Exact: $D_N=\frac{L}{2}H_N$",
        linewidth=2
    )

    ax1.plot(
        N_values,
        D_log,
        linestyle="--",
        label=r"Log approx: $\frac{L}{2}(\ln N+\gamma)$",
        linewidth=2
    )

    ax1.plot(
        N_values,
        D_precise,
        linestyle=":",
        label=r"Precise approx: $\frac{L}{2}(\ln N+\gamma+\frac{1}{2N}-\frac{1}{12N^2}+\frac{1}{120N^4})$",
        linewidth=2
    )

    ax1.set_xlabel("Number of blocks N")
    ax1.set_ylabel(r"Overhang $D_N$")
    ax1.set_title("Exact Overhang vs Euler-Mascheroni Approximation")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    st.pyplot(fig1)

    st.markdown(
        r"""
        왼쪽 그래프는 정확한 돌출량

        $$
        D_N=\frac{L}{2}H_N
        $$

        과 로그 근사

        $$
        D_N\approx \frac{L}{2}(\ln N+\gamma)
        $$

        그리고 보정항을 포함한 정밀 근사를 비교한다.
        """
    )

# =========================
# 오른쪽 그래프
# =========================

with right_col:
    st.subheader("2. 슬라이더로 선택한 n에 따른 Dₙ 변화")

    fig2, ax2 = plt.subplots(figsize=(7, 5))

    ax2.plot(
        N_values[:n_selected],
        D_exact[:n_selected],
        linewidth=2,
        label=r"$D_k=\frac{L}{2}H_k$"
    )

    ax2.scatter(
        [n_selected],
        [D_selected],
        s=80,
        zorder=5,
        label=rf"Selected: $D_{{{n_selected}}}$"
    )

    ax2.axvline(n_selected, linestyle="--", alpha=0.5)
    ax2.axhline(D_selected, linestyle="--", alpha=0.5)

    ax2.set_xlabel("Number of blocks k")
    ax2.set_ylabel(r"Overhang $D_k$")
    ax2.set_title(rf"Current Value: $D_{{{n_selected}}}={D_selected:.4f}$")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

    st.pyplot(fig2)

    st.markdown(
        rf"""
        현재 선택한 값:

        $$
        n={n_selected}
        $$

        $$
        H_n={H_selected:.6f}
        $$

        $$
        D_n=\frac{{L}}{{2}}H_n={D_selected:.6f}
        $$

        따라서 블록 길이 대비 돌출량은

        $$
        \frac{{D_n}}{{L}}={D_selected / L:.6f}
        $$
        """
    )

# =========================
# 아래쪽: 항별 돌출량 그래프
# =========================

st.subheader("3. 각 단계별 추가 돌출량")

a_values = L / (2 * N_values[:n_selected])

fig3, ax3 = plt.subplots(figsize=(10, 4))

ax3.bar(
    N_values[:n_selected],
    a_values,
    width=0.8,
    label=r"$a_n=\frac{L}{2n}$"
)

ax3.set_xlabel("Step n")
ax3.set_ylabel(r"Additional overhang $a_n$")
ax3.set_title("Additional Overhang at Each Step")
ax3.grid(True, axis="y", alpha=0.3)
ax3.legend()

st.pyplot(fig3)

st.markdown(
    r"""
    각 단계에서 추가로 얻는 돌출량은

    $$
    a_n=\frac{L}{2n}
    $$

    이므로 점점 작아진다. 하지만 그 합

    $$
    \sum_{n=1}^{N}\frac{L}{2n}
    $$

    은 조화급수이므로 매우 느리게 발산한다.
    """
)