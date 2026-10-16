import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import streamlit as st

# 1. 페이지 및 브라우저 탭 설정
st.set_page_config(
    page_title="영화 유형 나누기", page_icon="🎬", layout="wide"
)

st.title("🎬 영화 유형 나누기 (K-Means 군집화)")

# 2. 데이터 로드 및 전처리
DATA_URL = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    total_raw_count = len(df)

    # 필수 열 결측치 제거 및 first_week_audi > 0 조건 적용
    df = df.dropna(
        subset=[
            "first_scrn",
            "total_audi",
            "days_in_top10",
            "first_week_audi",
            "movieNm",
        ]
    )
    df = df[df["first_week_audi"] > 0].copy()

    # 파생 속성 생성: 롱런 지수 (최대 20으로 클리핑)
    df["long_run_idx"] = (df["total_audi"] / df["first_week_audi"]).clip(
        upper=20
    )

    # 스크린 수 및 누적 관객 상용로그 변환
    df["log_first_scrn"] = np.log10(df["first_scrn"].replace(0, np.nan))
    df["log_total_audi"] = np.log10(df["total_audi"].replace(0, np.nan))

    # 로그 변환 후 결측치 행 정제
    df = df.dropna(
        subset=["log_first_scrn", "log_total_audi", "long_run_idx"]
    )
    filtered_count = len(df)

    return df, total_raw_count, filtered_count


df, raw_count, clean_count = load_data()

# 전체 영화 편수와 분석 대상 편수 안내
st.write(
    f"**전체 영화 편수:** {raw_count:,}편 | **분석 대상 편수(조건 정제 후):** {clean_count:,}편"
)

st.divider()

# 3. 군집화 속성 및 묶음 수(k) 선택
feature_dict = {
    "스크린 수(로그)": "log_first_scrn",
    "누적 관객(로그)": "log_total_audi",
    "10위권 일수": "days_in_top10",
    "롱런 지수": "long_run_idx",
}

col_feat, col_k = st.columns([2, 1])

with col_feat:
    st.subheader("1. 군집화 속성 선택")
    selected_feature_names = st.multiselect(
        "군집화에 활용할 속성을 선택하세요 (최소 2개 이상):",
        options=list(feature_dict.keys()),
        default=list(feature_dict.keys()),
    )

with col_k:
    st.subheader("2. 묶음 수(k) 선택")
    n_clusters = st.slider(
        "묶음 수를 선택하세요 (2~7):", min_value=2, max_value=7, value=3
    )

if len(selected_feature_names) < 2:
    st.warning("⚠️ 군집화를 위해 속성을 최소 2개 이상 선택해 주세요.")
    st.stop()

selected_features = [feature_dict[name] for name in selected_feature_names]

# 4. 데이터 표준화 및 K-Means 군집화
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[selected_features])

# 선택한 k로 K-Means 실행 (난수 고정)
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(scaled_data)
df["raw_cluster"] = cluster_labels

# 묶음 기호 생성 (㉮ ~ ㉴)
k_symbols = ["㉮", "㉯", "㉰", "㉱", "㉲", "㉳", "㉴"]

# 누적 관객(total_audi) 평균이 큰 묶음부터 ㉮, ㉯, ... 순으로 매핑
avg_audi_per_cluster = df.groupby("raw_cluster")["total_audi"].mean()
sorted_clusters = avg_audi_per_cluster.sort_values(ascending=False).index

label_map = {
    sorted_clusters[i]: k_symbols[i] for i in range(len(sorted_clusters))
}
df["cluster"] = df["raw_cluster"].map(label_map)
cluster_order = [k_symbols[i] for i in range(n_clusters)]

# 5. 시각화 (2차원 & 3차원 산점도)
st.divider()
st.subheader("3. 군집 시각화")

# 2차원 산점도
col1, col2 = st.columns(2)
with col1:
    x_axis_2d = st.selectbox(
        "2차원 X축 속성 선택:",
        options=list(feature_dict.keys()),
        index=0,
        key="2d_x",
    )
with col2:
    y_axis_2d = st.selectbox(
        "2차원 Y축 속성 선택:",
        options=list(feature_dict.keys()),
        index=1,
        key="2d_y",
    )

fig_2d = px.scatter(
    df,
    x=feature_dict[x_axis_2d],
    y=feature_dict[y_axis_2d],
    color="cluster",
    category_orders={"cluster": cluster_order},
    hover_name="movieNm",
    labels={
        feature_dict[x_axis_2d]: x_axis_2d,
        feature_dict[y_axis_2d]: y_axis_2d,
        "cluster": "유형",
    },
    title=f"2차원 산점도 ({x_axis_2d} vs {y_axis_2d})",
    color_discrete_sequence=px.colors.qualitative.Set2,
)
st.plotly_chart(fig_2d, use_container_width=True)

# 3차원 산점도
st.write("---")
if len(selected_feature_names) >= 3:
    col3, col4, col5 = st.columns(3)
    with col3:
        x_axis_3d = st.selectbox(
            "3차원 X축 속성 선택:",
            options=list(feature_dict.keys()),
            index=0,
            key="3d_x",
        )
    with col4:
        y_axis_3d = st.selectbox(
            "3차원 Y축 속성 선택:",
            options=list(feature_dict.keys()),
            index=1,
            key="3d_y",
        )
    with col5:
        z_axis_3d = st.selectbox(
            "3차원 Z축 속성 선택:",
            options=list(feature_dict.keys()),
            index=2,
            key="3d_z",
        )

    fig_3d = px.scatter_3d(
        df,
        x=feature_dict[x_axis_3d],
        y=feature_dict[y_axis_3d],
        z=feature_dict[z_axis_3d],
        color="cluster",
        category_orders={"cluster": cluster_order},
        hover_name="movieNm",
        labels={
            feature_dict[x_axis_3d]: x_axis_3d,
            feature_dict[y_axis_3d]: y_axis_3d,
            feature_dict[z_axis_3d]: z_axis_3d,
            "cluster": "유형",
        },
        title=f"3차원 산점도 ({x_axis_3d} vs {y_axis_3d} vs {z_axis_3d})",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_3d.update_traces(marker=dict(size=3))
    st.plotly_chart(fig_3d, use_container_width=True)
else:
    st.info("3차원 산점도를 그리려면 속성을 3개 이상 선택해주세요.")

# 6. 묶음별 통계 표 및 대표 영화 목록
st.divider()
st.subheader("4. 묶음(유형)별 통계 및 대표 영화")

# 원래 단위 평균 표 생성
summary_df = (
    df.groupby("cluster")
    .agg(
        편수=("movieNm", "count"),
        스크린수_평균=("first_scrn", "mean"),
        누적관객수_평균=("total_audi", "mean"),
        상위10위_일수_평균=("days_in_top10", "mean"),
        롱런지수_평균=("long_run_idx", "mean"),
    )
    .reindex(cluster_order)
)

st.dataframe(
    summary_df.style.format(
        {
            "편수": "{:,}편",
            "스크린수_평균": "{:,.1f}개",
            "누적관객수_평균": "{:,.0f}명",
            "상위10위_일수_평균": "{:,.1f}일",
            "롱런지수_평균": "{:,.2f}",
        }
    ),
    use_container_width=True,
)

# 묶음별 누적 관객수 상위 5개 영화 목록 (동적 컬럼 배치)
st.write("#### 🏆 묶음별 누적 관객수 Top 5 영화")
cols = st.columns(n_clusters)

for idx, cluster_name in enumerate(cluster_order):
    with cols[idx]:
        st.markdown(f"##### 묶음 {cluster_name}")
        top5_movies = (
            df[df["cluster"] == cluster_name]
            .sort_values(by="total_audi", ascending=False)
            .head(5)
        )
        for i, (_, row) in enumerate(top5_movies.iterrows(), 1):
            st.write(f"**{i}. {row['movieNm']}**")
            st.caption(f"({row['total_audi']:,}명)")

# 7. 엘보우 기법(Inertia) 평가 및 실루엣 점수 계산
st.divider()
st.subheader("5. 적정 묶음 수 진단 (엘보우 기법 & 실루엣 점수)")

# k=1부터 7까지 Inertia 진단
k_range = range(1, 8)
inertia_list = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(scaled_data)
    inertia_list.append(km.inertia_)

# 선택한 k에 대한 실루엣 점수 계산
current_silhouette = silhouette_score(scaled_data, cluster_labels)
st.info(
    f"📊 **현재 선택한 묶음 수({n_clusters}개)의 실루엣 점수:** `{current_silhouette:.3f}`  \n"
    f"*(실루엣 점수는 -1에서 1 사이의 값을 가지며, 1에 가까울수록 군집 간 경계가 명확하고 잘 묶였음을 의미합니다.)*"
)

# Inertia 꺾은선 그래프
fig_elbow = go.Figure()

fig_elbow.add_trace(
    go.Scatter(
        x=list(k_range),
        y=inertia_list,
        mode="lines+markers",
        marker=dict(size=8, color="#1f77b4"),
        line=dict(width=2),
        name="Inertia (거리 제곱합)",
    )
)

# 현재 고른 n_clusters 위치에 세로선 표시
fig_elbow.add_vline(
    x=n_clusters,
    line_width=2,
    line_dash="dash",
    line_color="red",
    annotation_text=f"현재 선택 (k={n_clusters})",
    annotation_position="top right",
)

fig_elbow.update_layout(
    title="묶음 수(k)에 따른 중심 거리 제곱합(Inertia) 변화",
    xaxis_title="묶음 수 (k)",
    yaxis_title="Inertia (중심 거리 제곱합)",
    xaxis=dict(tickmode="linear", tick0=1, dtick=1),
)

st.plotly_chart(fig_elbow, use_container_width=True)

# 감소량 표 작성
diff_list = [None]  # k=1일 때는 비교 대상이 없으므로 빈값/None
for i in range(1, len(inertia_list)):
    reduction = inertia_list[i - 1] - inertia_list[i]
    diff_list.append(reduction)

elbow_df = pd.DataFrame(
    {
        "묶음 수 (k)": list(k_range),
        "중심 거리 제곱합 (Inertia)": [f"{val:,.2f}" for val in inertia_list],
        "전 단계 대비 감소량": [
            f"{val:,.2f}" if val is not None else "" for val in diff_list
        ],
    }
)

st.dataframe(elbow_df, use_container_width=True, hide_index=True)
