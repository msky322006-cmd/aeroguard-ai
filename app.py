
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# =========================================================
# 1. تنظیمات اصلی
# =========================================================

st.set_page_config(
    page_title="AeroGuard AI | پایش موتور هواپیما",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "aeroguard_ai_20x15.csv"
AIRPLANE_PATH = BASE_DIR / "airplane.jpg"
ENGINE_PATH = BASE_DIR / "engine.jpg"


# =========================================================
# 2. طراحی فارسی، راست چین و فشرده
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #031126;
    --panel: #071c37;
    --border: #075d9a;
    --blue: #139cff;
    --cyan: #18d5f5;
    --muted: #a9c8e6;
}

html, body, [class*="css"], [data-testid="stApp"] {
    font-family: 'Vazirmatn', Tahoma, sans-serif !important;
}

.stApp {
    background: linear-gradient(135deg, #020b19, #041a35 65%, #020d1e);
    color: #edf7ff;
}

.block-container {
    max-width: 100% !important;
    padding: 0.7rem 1rem 0.4rem 1rem !important;
}

[data-testid="stHeader"] {
    background: transparent;
    height: 0;
}

[data-testid="stSidebar"] {
    background: #04152b;
    border-left: 1px solid #075d9a;
}

[data-testid="stSidebar"] > div {
    padding-top: 0.8rem;
}

h1, h2, h3, p, label {
    color: #edf7ff;
    font-family: 'Vazirmatn', Tahoma, sans-serif !important;
}

h1 {
    font-size: clamp(22px, 2vw, 34px) !important;
    font-weight: 800 !important;
}

h2 { font-size: 22px !important; }
h3 { font-size: 17px !important; }

[data-testid="stVerticalBlock"] {
    gap: 0.55rem;
}

[data-testid="stHorizontalBlock"] {
    gap: 0.65rem;
}

.panel {
    background: linear-gradient(145deg, #081f3c, #04152b);
    border: 1px solid var(--border);
    border-radius: 13px;
    padding: 12px;
}

.hero {
    background: linear-gradient(100deg, #062448, #031329);
    border: 1px solid #0a83c9;
    border-radius: 14px;
    padding: 15px 18px;
    margin-bottom: 10px;
}

.hero-subtitle {
    color: #a8dcff;
    font-size: 13px;
}

.metric-card {
    background: linear-gradient(145deg, #092544, #04152b);
    border: 1px solid #0876b9;
    border-radius: 12px;
    padding: 12px 8px;
    text-align: center;
    min-height: 105px;
}

.metric-label {
    color: #b7d8f6;
    font-size: 12px;
}

.metric-value {
    color: #ffffff;
    font-size: clamp(20px, 1.8vw, 29px);
    font-weight: 800;
    margin: 5px 0;
}

.metric-unit {
    color: #4fd4ff;
    font-size: 11px;
}

.section-title {
    color: #39c8ff;
    font-weight: 700;
    font-size: 15px;
    margin-bottom: 6px;
}

.small-muted {
    color: #a1c1df;
    font-size: 11px;
}

div.stButton > button,
div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(90deg, #0879f9, #08b4e8);
    color: white;
    border: 0;
    border-radius: 9px;
    font-weight: 700;
    min-height: 40px;
}

div.stButton > button:hover,
div[data-testid="stFormSubmitButton"] button:hover {
    color: white;
    border: 1px solid #74ddff;
}

div[data-testid="stNumberInput"] input {
    background: #061b34;
    color: white;
    border-color: #155584;
}

[data-testid="stDataFrame"] {
    border: 1px solid #155584;
    border-radius: 8px;
}

hr {
    border-color: #164165;
}

footer, #MainMenu {
    visibility: hidden;
}

@media (max-width: 900px) {
    .block-container {
        padding: 0.5rem !important;
    }
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. بارگذاری و بررسی دیتاست
# =========================================================

@st.cache_data
def load_dataset(path_string, modified_time):
    data = pd.read_csv(path_string)
    data.columns = data.columns.str.strip()

    if "RUL_demo" not in data.columns:
        raise ValueError(
            "ستون RUL_demo در فایل CSV وجود ندارد."
        )

    # اصلاح مقدارهایی مثل 50s
    data["RUL_demo"] = (
        data["RUL_demo"]
        .astype(str)
        .str.strip()
        .str.replace("s", "", regex=False)
    )

    data["RUL_demo"] = pd.to_numeric(
        data["RUL_demo"], errors="coerce"
    )

    # تبدیل ستون‌های عددی
    for column in data.columns:
        if column != "Engine_ID":
            data[column] = pd.to_numeric(
                data[column], errors="coerce"
            )

    data = data.dropna(subset=["RUL_demo"]).copy()
    data = data.replace([np.inf, -np.inf], np.nan)

    if data.empty:
        raise ValueError("دیتاست هیچ ردیف معتبری ندارد.")

    # ستون‌هایی که نباید ورودی مدل باشند
    excluded = {"Engine_ID", "RUL_demo"}

    features = [
        column for column in data.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(data[column])
        and data[column].notna().any()
    ]

    if not features:
        raise ValueError(
            "هیچ ستون عددی مناسبی برای آموزش مدل پیدا نشد."
        )

    # پر کردن خانه‌های خالی با میانه هر ستون
    data[features] = data[features].fillna(
        data[features].median()
    )

    data = data.dropna(subset=["RUL_demo"]).copy()

    if len(data) < 5:
        raise ValueError(
            "برای آموزش مدل حداقل ۵ ردیف معتبر لازم است."
        )

    return data, features


if not CSV_PATH.exists():
    st.error(
        "فایل aeroguard_ai_20x15.csv پیدا نشد. "
        "فایل CSV را کنار app.py قرار بده."
    )
    st.stop()

try:
    df, FEATURES = load_dataset(
        str(CSV_PATH),
        CSV_PATH.stat().st_mtime
    )
except Exception as error:
    st.error(f"خطا در خواندن دیتاست: {error}")
    st.stop()


# =========================================================
# 4. آموزش مدل
# =========================================================

@st.cache_resource
def train_model(data, feature_names):
    X = data[feature_names].copy()
    y = data["RUL_demo"].copy()

    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=5,
        min_samples_leaf=1,
        random_state=42
    )

    model.fit(X, y)

    # ارزیابی آموزشی با جدا کردن بخشی از داده‌ها
    if len(data) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

        evaluation_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )

        evaluation_model.fit(X_train, y_train)
        test_prediction = evaluation_model.predict(X_test)

        mae = float(
            mean_absolute_error(y_test, test_prediction)
        )
    else:
        mae = None

    return model, mae


model, model_mae = train_model(df, FEATURES)


# =========================================================
# 5. ابزارهای کمکی
# =========================================================

def get_number(row, column, fallback=0.0):
    if column not in row.index:
        return float(fallback)

    value = row[column]

    if pd.isna(value):
        return float(fallback)

    return float(value)


def metric_card(label, value, unit, icon):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{icon} {label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-unit">{unit}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def style_chart(fig, height=270):
    fig.update_layout(
        height=height,
        autosize=True,
        margin=dict(l=12, r=12, t=38, b=35),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(3,18,38,0.55)",
        font=dict(
            color="#e4f3ff",
            family="Vazirmatn, Tahoma, sans-serif",
            size=11
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            gridcolor="#173b5c",
            zerolinecolor="#173b5c"
        ),
        yaxis=dict(
            gridcolor="#173b5c",
            zerolinecolor="#173b5c"
        )
    )
    return fig


# =========================================================
# 6. منوی فارسی
# =========================================================

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center;padding:8px 0 18px">
            <div style="font-size:27px;font-weight:800;color:#20c8ff">
                ✈ AeroGuard AI
            </div>
            <div class="small-muted">
                پایش هوشمند موتور هواپیما
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    page = st.radio(
        "بخش موردنظر",
        [
            "داشبورد",
            "پیش‌بینی عمر موتور",
            "تحلیل داده‌ها",
            "درباره پروژه"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### 🟢 وضعیت سامانه")
    st.success("مدل آماده استفاده است")
    st.caption(f"تعداد رکوردها: {len(df)}")
    st.caption(f"تعداد ویژگی‌ها: {len(FEATURES)}")

    if model_mae is not None:
        st.caption(
            f"میانگین خطای مطلق آزمایشی: {model_mae:.2f} ساعت"
        )

    st.markdown("---")
    st.caption("نسخه آموزشی — نه برای تعمیر واقعی هواپیما")


# =========================================================
# 7. سربرگ با عکس محلی
# =========================================================

hero_left, hero_right = st.columns([1.65, 1], gap="small")

with hero_left:
    st.markdown(
        """
        <div class="hero">
            <h1>AeroGuard <span style="color:#19c9ff">AI</span></h1>
            <div style="font-size:17px;font-weight:700">
                سامانه هوشمند پایش موتور هواپیما
            </div>
            <p class="hero-subtitle">
                پیش‌بینی عمر باقی‌مانده موتور با یادگیری ماشین
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with hero_right:
    if AIRPLANE_PATH.exists():
        st.image(
            str(AIRPLANE_PATH),
            use_container_width=True
        )
    else:
        st.warning(
            "فایل airplane.jpg پیدا نشد."
        )


# =========================================================
# 8. داشبورد اصلی
# =========================================================

latest = df.iloc[-1]
latest_input = df[FEATURES].iloc[[-1]]
latest_prediction = max(
    0.0, float(model.predict(latest_input)[0])
)

# وضعیت داشبورد: ابتدا از آخرین ردیف دیتاست شروع می‌کند،
# اما پس از پیش‌بینی با مشخصات واردشده توسط کاربر جایگزین می‌شود.
if "dashboard_values" not in st.session_state:
    st.session_state.dashboard_values = {
        feature: float(latest[feature]) for feature in FEATURES
    }

if "dashboard_prediction" not in st.session_state:
    st.session_state.dashboard_prediction = latest_prediction

if page == "داشبورد":
    dashboard_values = st.session_state.dashboard_values
    dashboard_prediction = st.session_state.dashboard_prediction

    # کارت‌های شاخص
    c1, c2, c3, c4 = st.columns(4, gap="small")

    with c1:
        metric_card(
            "ساعات کارکرد",
            f"{dashboard_values.get('Operating_Hours', 0):,.0f}",
            "ساعت",
            "⏱"
        )

    with c2:
        metric_card(
            "دمای موتور",
            f"{dashboard_values.get('Temperature', 0):g}",
            "درجه سانتی‌گراد",
            "🌡️"
        )

    with c3:
        metric_card(
            "لرزش موتور",
            f"{dashboard_values.get('Vibration', 0):g}",
            "مقدار حسگر",
            "〰️"
        )

    with c4:
        metric_card(
            "عمر باقی‌مانده",
            f"{dashboard_prediction:.1f}",
            "ساعت تخمینی",
            "🛠️"
        )

    st.write("")

    # بخش نمودار و عکس موتور
    chart_col, engine_col = st.columns(
        [2.15, 1], gap="small"
    )

    with chart_col:
        st.markdown(
            '<div class="section-title">📈 روند داده‌های موتور</div>',
            unsafe_allow_html=True
        )

        fig = go.Figure()

        sensor_config = [
            ("Temperature", "دمای موتور", "#ff9138"),
            ("Vibration", "لرزش موتور", "#18c9ff"),
            ("Pressure", "فشار", "#00d5ad")
        ]

        for column, label, color in sensor_config:
            if column in df.columns:
                # داده‌های دیتاست حفظ می‌شوند و آخرین نقطه،
                # مقدار جدید واردشده در صفحه پیش‌بینی است.
                chart_x = list(range(1, len(df) + 1))
                chart_y = df[column].tolist()

                if column in dashboard_values:
                    chart_x.append(len(df) + 1)
                    chart_y.append(dashboard_values[column])

                fig.add_trace(
                    go.Scatter(
                        x=chart_x,
                        y=chart_y,
                        name=label,
                        mode="lines+markers",
                        line=dict(color=color, width=2),
                        marker=dict(size=5)
                    )
                )

        fig.update_layout(
            title="روند حسگرها (آخرین نقطه: مشخصات فعلی واردشده)",
            xaxis_title="شماره رکورد",
            yaxis_title="مقدار حسگر",
            hovermode="x unified"
        )

        style_chart(fig, 285)
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False, "responsive": True}
        )

    with engine_col:
        st.markdown(
            '<div class="section-title">🛩️ تصویر موتور هواپیما</div>',
            unsafe_allow_html=True
        )

        if ENGINE_PATH.exists():
            st.image(
                str(ENGINE_PATH),
                use_container_width=True
            )
        else:
            st.warning("فایل engine.jpg پیدا نشد.")

        st.markdown(
            f"""
            <div class="panel">
                <div class="metric-label">وضعیت مدل</div>
                <div style="font-size:17px;font-weight:700;color:#24d8b0">
                    ● آماده پیش‌بینی
                </div>
                <div class="small-muted">
                    مدل: Random Forest
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button(
            "⚙️ رفتن به پیش‌بینی",
            use_container_width=True
        ):
            st.session_state["requested_page"] = "پیش‌بینی عمر موتور"
            st.rerun()

    # جدول کوچک برای جلوگیری از طولانی شدن صفحه
    st.markdown(
        '<div class="section-title">🗃️ آخرین داده‌های ثبت‌شده</div>',
        unsafe_allow_html=True
    )

    display_columns = (
        ["Engine_ID"] if "Engine_ID" in df.columns else []
    ) + [
        col for col in [
            "Operating_Hours",
            "Temperature",
            "Vibration",
            "Pressure",
            "RUL_demo"
        ]
        if col in df.columns
    ]

    st.dataframe(
        df[display_columns].tail(4),
        use_container_width=True,
        hide_index=True,
        height=155
    )


# =========================================================
# 9. صفحه پیش‌بینی
# =========================================================

elif page == "پیش‌بینی عمر موتور":

    st.markdown("## ⚙️ پیش‌بینی عمر باقی‌مانده موتور")
    st.caption(
        "مقادیر حسگرها را وارد کن و دکمه پیش‌بینی را بزن."
    )

    st.markdown(
        '<div class="section-title">اطلاعات ورودی موتور</div>',
        unsafe_allow_html=True
    )

    user_values = {}

    with st.form("engine_prediction_form"):

        columns = st.columns(3, gap="small")

        for i, feature in enumerate(FEATURES):
            default = float(latest[feature])

            with columns[i % 3]:
                user_values[feature] = st.number_input(
                    feature,
                    value=default,
                    key=f"prediction_{feature}"
                )

        submitted = st.form_submit_button(
            "✦ پیش‌بینی عمر باقی‌مانده",
            use_container_width=True
        )

    if submitted:
        input_df = pd.DataFrame(
            [user_values],
            columns=FEATURES
        )

        prediction = max(
            0.0, float(model.predict(input_df)[0])
        )

        # ذخیره ورودی‌ها و نتیجه برای نمایش در داشبورد
        st.session_state.dashboard_values = {
            feature: float(user_values[feature])
            for feature in FEATURES
        }
        st.session_state.dashboard_prediction = prediction

        st.markdown("---")
        st.markdown("### نتیجه پیش‌بینی")

        metric_card(
            "عمر باقی‌مانده تخمینی",
            f"{prediction:.1f}",
            "ساعت",
            "🛠️"
        )

        if prediction < 50:
            st.warning(
                "مقدار پیش‌بینی‌شده کم است. این نتیجه صرفاً آموزشی است."
            )
        else:
            st.info(
                "نتیجه بر اساس مدل آموزش‌دیده روی دیتاست پروژه است."
            )

        st.success(
            "مشخصات جدید ذخیره شد. حالا از منوی سمت چپ وارد «داشبورد» شو "
            "تا کارت‌ها و نمودار با مقادیر جدید نمایش داده شوند."
        )

        st.caption(
            "این مدل و دیتاست کوچک برای تصمیم‌گیری واقعی درباره "
            "ایمنی پرواز یا تعمیر هواپیما معتبر نیستند."
        )


# =========================================================
# 10. صفحه تحلیل داده‌ها
# =========================================================

elif page == "تحلیل داده‌ها":

    st.markdown("## 📊 تحلیل داده‌های آموزشی")

    left, right = st.columns(2, gap="medium")

    with left:
        st.markdown(
            '<div class="section-title">عمر باقی‌مانده در دیتاست</div>',
            unsafe_allow_html=True
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(df) + 1)),
                y=df["RUL_demo"],
                mode="lines+markers",
                name="عمر باقی‌مانده",
                line=dict(color="#a46bff", width=3)
            )
        )

        fig.update_layout(
            xaxis_title="شماره رکورد",
            yaxis_title="ساعت"
        )

        style_chart(fig, 330)
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False}
        )

    with right:
        st.markdown(
            '<div class="section-title">اهمیت ویژگی‌ها در مدل</div>',
            unsafe_allow_html=True
        )

        importance = pd.DataFrame({
            "ویژگی": FEATURES,
            "اهمیت": model.feature_importances_
        }).sort_values("اهمیت", ascending=True)

        fig2 = go.Figure(
            go.Bar(
                x=importance["اهمیت"],
                y=importance["ویژگی"],
                orientation="h",
                marker_color="#18baff"
            )
        )

        fig2.update_layout(
            xaxis_title="اهمیت نسبی",
            yaxis_title=""
        )

        style_chart(
            fig2,
            max(330, 24 * len(FEATURES))
        )

        st.plotly_chart(
            fig2,
            use_container_width=True,
            config={"displayModeBar": False}
        )

    st.markdown("### جدول کامل دیتاست")
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        height=300
    )


# =========================================================
# 11. صفحه درباره پروژه
# =========================================================

elif page == "درباره پروژه":

    st.markdown("## درباره AeroGuard AI")

    info_col, image_col = st.columns(
        [1.2, 1], gap="medium"
    )

    with info_col:
        st.markdown(
            """
            <div class="panel">
                <div class="section-title">معرفی سامانه</div>
                AeroGuard AI یک پروژه آموزشی برای پیش‌بینی عمر
                باقی‌مانده موتور هواپیماست.

                <br><br>

                <div class="section-title">فناوری‌های استفاده‌شده</div>
                • زبان برنامه‌نویسی Python
                <br>
                • رابط کاربری Streamlit
                <br>
                • الگوریتم یادگیری ماشین Random Forest
                <br>
                • نمودارهای تعاملی Plotly
                <br>
                • داده‌های آموزشی حسگر موتور

                <br><br>

                <div class="section-title">هدف پروژه</div>
                نمایش نحوه استفاده از داده‌های حسگر برای تخمین
                عمر باقی‌مانده موتور در یک نمونه آموزشی.
            </div>
            """,
            unsafe_allow_html=True
        )

    with image_col:
        if AIRPLANE_PATH.exists():
            st.image(
                str(AIRPLANE_PATH),
                use_container_width=True
            )
        if ENGINE_PATH.exists():
            st.image(
                str(ENGINE_PATH),
                use_container_width=True
            )

    st.warning(
        "این پروژه نمایشی است و برای نگهداری واقعی هواپیما "
        "یا تأیید ایمنی پرواز طراحی نشده است."
    )


# =========================================================
# 12. پانوشت
# =========================================================

st.markdown("---")
st.markdown(
    """
    <div style="text-align:center;color:#83aaca;font-size:11px">
        AeroGuard AI | سامانه آموزشی پیش‌بینی عمر موتور هواپیما
    </div>
    """,
    unsafe_allow_html=True
)
