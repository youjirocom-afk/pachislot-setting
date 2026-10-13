"""
パチスロ設定判別プロトタイプ（Streamlit）

小役・ボーナスの理論確率は架空の値です。実機の解析値ではありません。
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="パチスロ設定判別",
    layout="centered"
)

# Streamlitのデフォルトのフッターやメニューを非表示にする設定
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 架空の確率テーブル（分母 = 1/n の n）
# 設定が上がるほど小役・ボーナスが出やすい想定
# ---------------------------------------------------------------------------
SETTINGS = [1, 2, 3, 4, 5, 6]

THEORETICAL_DENOM: Dict[str, Dict[int, float]] = {
    "ベル": {1: 6.50, 2: 6.35, 3: 6.20, 4: 6.05, 5: 5.90, 6: 5.75},
    "スイカ": {1: 82.0, 2: 72.0, 3: 64.0, 4: 56.0, 5: 50.0, 6: 44.0},
    "チェリー": {1: 38.0, 2: 33.0, 3: 29.0, 4: 25.5, 5: 22.5, 6: 19.8},
    "BB": {1: 399.0, 2: 341.0, 3: 292.0, 4: 249.0, 5: 213.0, 6: 182.0},
    "RB": {1: 586.0, 2: 498.0, 3: 421.0, 4: 355.0, 5: 298.0, 6: 251.0},
}

PRIZE_KEYS = ["ベル", "スイカ", "チェリー", "BB", "RB"]
PRIZE_ICONS = {
    "ベル": "🔔",
    "スイカ": "🍉",
    "チェリー": "🍒",
    "BB": "🎰",
    "RB": "⭐",
}


def denom_to_prob(denom: float) -> float:
    return 1.0 / denom if denom > 0 else 0.0


def format_one_over(games: int, count: int) -> str:
    if count <= 0:
        return "—"
    return f"1/{games / count:.1f}"


def multinomial_log_likelihood(counts: List[int], probs: List[float]) -> float:
    """多項分布の対数尤度（定数項 n! は設定間比較で相殺するので省略）。"""
    ll = 0.0
    for n, p in zip(counts, probs):
        if n == 0:
            continue
        if p <= 0:
            return -math.inf
        ll += n * math.log(p)
    return ll


def estimate_settings(games: int, observed: Dict[str, int]) -> pd.DataFrame:
    other = games - sum(observed[k] for k in PRIZE_KEYS)
    counts = [observed[k] for k in PRIZE_KEYS] + [other]

    log_likes = []
    for setting in SETTINGS:
        probs = [denom_to_prob(THEORETICAL_DENOM[k][setting]) for k in PRIZE_KEYS]
        p_other = 1.0 - sum(probs)
        if p_other <= 0:
            log_likes.append(-math.inf)
            continue
        log_likes.append(multinomial_log_likelihood(counts, probs + [p_other]))

    arr = np.array(log_likes, dtype=float)
    finite = np.isfinite(arr)
    posterior = np.zeros_like(arr)
    if finite.any():
        shifted = arr[finite] - np.max(arr[finite])
        weights = np.exp(shifted)
        posterior[finite] = weights / weights.sum()

    best = int(SETTINGS[int(np.argmax(posterior))]) if posterior.sum() > 0 else None
    rows = []
    for setting, ll, post in zip(SETTINGS, log_likes, posterior):
        rows.append(
            {
                "設定": f"設定{setting}",
                "設定番号": setting,
                "推定確率(%)": round(float(post) * 100, 1),
                "対数尤度": None if not math.isfinite(ll) else round(float(ll), 2),
                "最有力": setting == best,
            }
        )
    return pd.DataFrame(rows)


def inject_mobile_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.1rem;
            padding-bottom: 3rem;
            max-width: 720px;
        }
        h1 { font-size: 1.55rem !important; letter-spacing: -0.02em; }
        h2 { font-size: 1.15rem !important; }
        h3 { font-size: 1.02rem !important; }
        div[data-testid="stMetric"] {
            background: #fff;
            border: 1px solid #e8e4df;
            border-radius: 14px;
            padding: 0.55rem 0.7rem;
        }
        div[data-testid="stMetric"] label { font-size: 0.78rem; }
        .best-badge {
            background: linear-gradient(135deg, #1f6feb, #3d8bfd);
            color: white;
            border-radius: 16px;
            padding: 1rem 1.1rem;
            margin: 0.4rem 0 1rem 0;
        }
        .best-badge .label { font-size: 0.8rem; opacity: 0.9; }
        .best-badge .value { font-size: 1.7rem; font-weight: 700; margin-top: 0.15rem; }
        .note {
            color: #6b6560;
            font-size: 0.82rem;
            line-height: 1.5;
        }
        @media (max-width: 640px) {
            .block-container { padding-left: 0.85rem; padding-right: 0.85rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="パチスロ設定判別",
        page_icon="🎰",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    inject_mobile_css()

    st.title("🎰 パチスロ設定判別")
    st.markdown(
        '<p class="note">回転数と役の回数を入れると、出現率（1/〇〇）と'
        "設定1〜6の推定割合を表示します。理論値はプロトタイプ用の仮データです。</p>",
        unsafe_allow_html=True,
    )

    with st.form("counts_form"):
        games = st.number_input(
            "総回転数（ゲーム数）",
            min_value=0,
            max_value=100_000,
            value=2000,
            step=1,
        )
        col_a, col_b = st.columns(2)
        with col_a:
            bell = st.number_input("ベル回数", min_value=0, max_value=100_000, value=310, step=1)
            suika = st.number_input("スイカ回数", min_value=0, max_value=100_000, value=32, step=1)
            cherry = st.number_input("チェリー回数", min_value=0, max_value=100_000, value=72, step=1)
        with col_b:
            bb = st.number_input("BB（ビッグボーナス）回数", min_value=0, max_value=10_000, value=7, step=1)
            rb = st.number_input("RB（レギュラーボーナス）回数", min_value=0, max_value=10_000, value=5, step=1)

        st.form_submit_button("判定する", use_container_width=True, type="primary")

    observed = {
        "ベル": int(bell),
        "スイカ": int(suika),
        "チェリー": int(cherry),
        "BB": int(bb),
        "RB": int(rb),
    }
    total_hits = sum(observed.values())

    if games <= 0:
        st.warning("総回転数を1以上にしてください。")
        return
    if total_hits > games:
        st.error("役の合計回数が総回転数を超えています。入力を確認してください。")
        return

    st.subheader("出現率")
    rate_cols = st.columns(2)
    items = [("総回転数", f"{games:,}")] + [
        (f"{PRIZE_ICONS[k]} {k}", format_one_over(games, observed[k])) for k in PRIZE_KEYS
    ]
    for i, (label, value) in enumerate(items):
        with rate_cols[i % 2]:
            st.metric(label, value)

    observed_denoms = {k: (games / observed[k] if observed[k] else None) for k in PRIZE_KEYS}

    compare_rows = []
    for k in PRIZE_KEYS:
        row = {"役": f"{PRIZE_ICONS[k]} {k}", "実測": format_one_over(games, observed[k])}
        for s in SETTINGS:
            row[f"設定{s}"] = f"1/{THEORETICAL_DENOM[k][s]:.1f}"
        compare_rows.append(row)
    st.caption("実測と各設定の理論確率（1/n）")
    st.dataframe(pd.DataFrame(compare_rows), use_container_width=True, hide_index=True)

    result = estimate_settings(games, observed)
    best_row = result.loc[result["推定確率(%)"].idxmax()]
    st.markdown(
        f"""
        <div class="best-badge">
            <div class="label">現時点の最有力</div>
            <div class="value">{best_row["設定"]}　{best_row["推定確率(%)"]}%</div>
            <div class="label">一様事前分布 × 多項分布のベイズ推定（サンプルが少ないと差は小さく出ます）</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("設定ごとの推定割合")
    chart_df = result[["設定", "推定確率(%)"]].copy()
    fig = px.bar(
        chart_df,
        x="設定",
        y="推定確率(%)",
        text="推定確率(%)",
        color="推定確率(%)",
        color_continuous_scale="Blues",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
    fig.update_layout(
        margin=dict(l=8, r=8, t=24, b=8),
        height=320,
        coloraxis_showscale=False,
        yaxis_title="推定確率 (%)",
        xaxis_title="",
        yaxis_range=[0, max(100, chart_df["推定確率(%)"].max() * 1.2)],
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        result[["設定", "推定確率(%)"]].assign(**{"推定確率(%)": lambda d: d["推定確率(%)"].map(lambda x: f"{x:.1f}%")}),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("理論確率テーブル（仮）を見る"):
        table_rows = []
        for k in PRIZE_KEYS:
            row = {"役": k}
            for s in SETTINGS:
                row[f"設定{s}"] = f"1/{THEORETICAL_DENOM[k][s]:.1f}"
            table_rows.append(row)
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
        st.caption("実機の解析値ではありません。app.py の THEORETICAL_DENOM を差し替えてください。")

    if any(v is None for v in observed_denoms.values()):
        st.info("回数が0の役は出現率を「—」と表示しています。データが増えると推定が安定します。")


if __name__ == "__main__":
    main()
