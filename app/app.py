from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from streamlit_extras.metric_cards import style_metric_cards

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "messages_labeled.csv"
TOP_N_OVERVIEW = 50

# Validated colorblind-safe pair (the project's original teal/orange failed
# the accessibility check — low contrast on two colors, one pair too close
# together for colorblind viewers)
BLUE = "#2a78d6"
ORANGE = "#eb6834"

st.set_page_config(page_title="Banking Message Topic Explorer", layout="wide")


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_data
def compute_validation_metrics(df: pd.DataFrame) -> tuple[float, float]:
    labeled = df[df["cluster"] != -1]
    nmi = normalized_mutual_info_score(labeled["true_intent"], labeled["cluster"])
    ari = adjusted_rand_score(labeled["true_intent"], labeled["cluster"])
    return nmi, ari


df = load_data()
nmi, ari = compute_validation_metrics(df)

summary = (
    df.groupby(["cluster", "ai_label", "ai_rationale"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)
n_topics = int((summary["cluster"] != -1).sum())
outlier_pct = (df["cluster"] == -1).mean()

st.title("Banking Message Topic Explorer")

tab_overview, tab_detail = st.tabs(["Overview", "Topic Detail"])

with tab_overview:
    st.markdown(
        """
        These are ~13,000 real banking customer service messages from
        **BANKING77** ([Casanueva et al., 2020](https://github.com/PolyAI-LDN/task-specific-datasets)),
        each carrying a true intent label that was held out and never shown
        to the clustering step. Three unsupervised methods were compared
        against those held-out labels — TF-IDF, sentence embeddings, and
        BERTopic — and **BERTopic's topic assignment**, explored below, is
        the one that scored best. An LLM (Claude) then turned each
        discovered topic into the human-readable label and rationale shown
        here.
        """
    )

    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Messages", f"{len(df):,}")
    kpi_cols[1].metric("Topics discovered", f"{n_topics}")
    kpi_cols[2].metric("Unclustered", f"{outlier_pct:.1%}")
    kpi_cols[3].metric("NMI vs. true intent", f"{nmi:.2f}")
    style_metric_cards(border_left_color=BLUE)

    st.write("")

    top_topics = summary[summary["cluster"] != -1].head(TOP_N_OVERVIEW)
    fig = px.bar(
        top_topics,
        x="count",
        y="ai_label",
        orientation="h",
        title=f"Message volume — top {TOP_N_OVERVIEW} of {len(summary)} discovered topics",
        labels={"count": "Messages", "ai_label": "Topic"},
        template="plotly_white",
    )
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        showlegend=False,
        font=dict(family="sans serif", size=13),
        title_font_size=16,
        height=max(500, TOP_N_OVERVIEW * 18),  # ~18px/bar keeps labels legible
    )
    fig.update_traces(marker_color=BLUE)
    st.plotly_chart(fig, use_container_width=True)

    st.write("")

    topic_level = summary[summary["cluster"] != -1].copy()
    topic_purity = (
        df[df["cluster"] != -1]
        .groupby("cluster")["true_intent"]
        .apply(lambda s: s.value_counts(normalize=True).iloc[0])
    )
    topic_level["purity"] = topic_level["cluster"].map(topic_purity)

    dist_cols = st.columns(2)

    with dist_cols[0]:
        fig_sizes = px.histogram(
            topic_level, x="count", nbins=30,
            title="Topic size distribution",
            labels={"count": "Messages per topic"},
            template="plotly_white",
        )
        fig_sizes.update_traces(marker_color=BLUE)
        fig_sizes.update_layout(showlegend=False, bargap=0.05)
        st.plotly_chart(fig_sizes, use_container_width=True)
        st.caption(
            f"Median topic has {int(topic_level['count'].median())} messages; "
            f"the largest real topic has {int(topic_level['count'].max())}. "
            "(Excludes the unclustered bucket — it isn't a real topic.)"
        )

    with dist_cols[1]:
        fig_purity = px.histogram(
            topic_level, x="purity", nbins=20,
            title="Topic purity distribution",
            labels={"purity": "Share of top true intent"},
            template="plotly_white",
        )
        fig_purity.update_traces(marker_color=ORANGE)
        fig_purity.update_layout(showlegend=False, bargap=0.05, xaxis_tickformat=".0%")
        st.plotly_chart(fig_purity, use_container_width=True)
        st.caption(
            f"{(topic_level['purity'] >= 0.7).mean():.0%} of topics are at least 70% "
            "pure — one true intent dominates. Lower-purity ones are worth a closer "
            "look in the Topic Detail tab."
        )

with tab_detail:
    option_labels = summary["ai_label"] + " (n=" + summary["count"].astype(str) + ")"
    label_to_cluster = dict(zip(option_labels, summary["cluster"]))

    selected_option = st.selectbox("Choose a topic to explore", option_labels)
    selected_cluster = label_to_cluster[selected_option]
    cluster_row = summary[summary["cluster"] == selected_cluster].iloc[0]
    topic_df = df[df["cluster"] == selected_cluster]

    with st.container(border=True):
        st.subheader(cluster_row["ai_label"])
        st.write(cluster_row["ai_rationale"])

        purity = topic_df["true_intent"].value_counts(normalize=True).iloc[0]
        n_distinct_intents = topic_df["true_intent"].nunique()

        metric_cols = st.columns(3)
        metric_cols[0].metric("Messages in this topic", int(cluster_row["count"]))
        metric_cols[1].metric("Share of all messages", f"{cluster_row['count'] / len(df):.1%}")
        metric_cols[2].metric("Purity (top intent share)", f"{purity:.0%}")

    style_metric_cards(border_left_color=BLUE)
    st.caption(f"Spans {n_distinct_intents} distinct true intents.")

    st.write("")

    col_messages, col_intents = st.columns(2)

    with col_messages:
        st.markdown("**Sample messages**")
        sample_size = min(10, int(cluster_row["count"]))
        sample = topic_df["message"].sample(sample_size, random_state=1)
        for msg in sample:
            st.write(f"- {msg}")

    with col_intents:
        st.markdown("**True intent breakdown**")
        st.caption(
            "BANKING77's real labels, held out during clustering -- shown here "
            "so you can see how coherent (or mixed) this topic actually is."
        )
        intent_counts = topic_df["true_intent"].value_counts().head(6).reset_index()
        intent_counts.columns = ["true_intent", "count"]
        fig_intents = px.bar(
            intent_counts, x="count", y="true_intent", orientation="h",
            template="plotly_white",
        )
        fig_intents.update_layout(
            yaxis={"categoryorder": "total ascending"},
            showlegend=False,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        fig_intents.update_traces(marker_color=ORANGE)
        st.plotly_chart(fig_intents, use_container_width=True)