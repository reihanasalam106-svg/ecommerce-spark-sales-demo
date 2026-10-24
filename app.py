from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import altair as alt
import streamlit as st
from py4j.protocol import Py4JJavaError
from pyspark.errors import PySparkException
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


APP_DIR = Path(__file__).resolve().parent
SAMPLE_CSV = APP_DIR / "sales_demo.csv"
REQUIRED_COLUMNS = (
    "OrderID",
    "OrderDate",
    "Product",
    "Category",
    "Quantity",
    "UnitPrice",
    "CustomerID",
)


st.set_page_config(
    page_title="E-Commerce Sales Analysis",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def add_dashboard_styles() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(ellipse at 5% 0%, #e8f4ff 0, transparent 34rem),
                    #f6f8fc;
            }
            [data-testid="stHeader"] { background: transparent; }
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #10233f 0%, #17375e 100%);
            }
            [data-testid="stSidebar"] * { color: #f4f7fb; }
            [data-testid="stSidebar"] [data-baseweb="upload-dropzone"] * {
                color: #17375e;
            }
            .hero {
                padding: 1.5rem 1.75rem;
                margin: 0.35rem 0 1.25rem;
                border: 1px solid #e1e9f3;
                border-radius: 20px;
                background: linear-gradient(120deg, #ffffff 0%, #f2f8ff 100%);
                box-shadow: 0 10px 30px rgba(26, 54, 93, 0.06);
            }
            .hero-kicker {
                color: #3478c8;
                font-size: 0.78rem;
                font-weight: 700;
                letter-spacing: 0.12em;
                text-transform: uppercase;
            }
            .hero h1 {
                margin: 0.35rem 0 0.4rem;
                color: #12243b;
                font-size: clamp(1.8rem, 4vw, 2.65rem);
                line-height: 1.15;
            }
            .hero p { margin: 0; color: #64748b; font-size: 1rem; }
            [data-testid="stMetric"] {
                padding: 1rem 1.1rem;
                border: 1px solid #e5ebf3;
                border-radius: 16px;
                background: #fff;
                box-shadow: 0 8px 22px rgba(26, 54, 93, 0.05);
            }
            [data-testid="stMetricLabel"] { color: #64748b; }
            [data-testid="stMetricValue"] { color: #132942; }
            [data-testid="stDataFrame"] {
                border: 1px solid #e5ebf3;
                border-radius: 14px;
                overflow: hidden;
            }
            .section-note { color: #718096; font-size: 0.9rem; }
            @media (max-width: 700px) {
                .hero { padding: 1.2rem; border-radius: 16px; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def create_spark_session() -> SparkSession:
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    warehouse_dir = Path(tempfile.gettempdir()) / "ecommerce_sales_spark_warehouse"
    (warehouse_dir / "default.db").mkdir(parents=True, exist_ok=True)
    return (
        SparkSession.builder.master("local[2]")
        .appName("EcommerceSalesAnalysis")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.sql.warehouse.dir", warehouse_dir.as_uri())
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )


def read_sales_csv(spark: SparkSession, csv_path: Path) -> DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path.name}")
    return spark.read.option("header", True).option("inferSchema", True).option(
        "mode", "FAILFAST"
    ).csv(str(csv_path))


def validate_and_prepare(raw_df: DataFrame) -> DataFrame:
    missing_columns = [name for name in REQUIRED_COLUMNS if name not in raw_df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(
            "The CSV is missing required column(s): "
            f"{missing}. Required columns: {', '.join(REQUIRED_COLUMNS)}."
        )

    if raw_df.limit(1).count() == 0:
        raise ValueError("The CSV has a header but no sales records.")

    prepared = (
        raw_df.withColumn("OrderDate", F.expr("try_cast(`OrderDate` AS DATE)"))
        .withColumn("_quantity_value", F.expr("try_cast(`Quantity` AS DOUBLE)"))
        .withColumn("_unit_price_value", F.expr("try_cast(`UnitPrice` AS DOUBLE)"))
    )

    invalid_row = (
        F.col("OrderID").isNull()
        | F.col("OrderDate").isNull()
        | F.col("Product").isNull()
        | (F.trim(F.col("Product").cast("string")) == "")
        | F.col("Category").isNull()
        | (F.trim(F.col("Category").cast("string")) == "")
        | F.col("CustomerID").isNull()
        | (F.trim(F.col("CustomerID").cast("string")) == "")
        | F.col("_quantity_value").isNull()
        | F.isnan(F.col("_quantity_value"))
        | (F.col("_quantity_value") < 1)
        | (F.col("_quantity_value") != F.floor(F.col("_quantity_value")))
        | F.col("_unit_price_value").isNull()
        | F.isnan(F.col("_unit_price_value"))
        | (F.col("_unit_price_value") < 0)
    )
    if prepared.filter(invalid_row).limit(1).count() > 0:
        raise ValueError(
            "Some rows contain missing or invalid values. Check that dates are "
            "valid, Quantity is a positive whole number, UnitPrice is non-negative, "
            "and the other required fields are filled in."
        )

    return (
        prepared.withColumn("Quantity", F.col("_quantity_value").cast("long"))
        .withColumn("UnitPrice", F.col("_unit_price_value"))
        .withColumn("Revenue", F.col("Quantity") * F.col("UnitPrice"))
        .drop("_quantity_value", "_unit_price_value")
    )


def format_revenue(value: Any) -> str:
    return f"₹{float(value or 0):,.2f}"


def searchable_table(df: DataFrame) -> None:
    search = st.text_input(
        "Search records",
        placeholder="Try a product, category, order ID, or customer ID",
        help="Search checks all columns in the currently loaded sales data.",
    ).strip()

    if search:
        searchable_text = F.lower(
            F.concat_ws(
                " ",
                *[
                    F.coalesce(F.col(name).cast("string"), F.lit(""))
                    for name in df.columns
                ],
            )
        )
        display_df = df.filter(searchable_text.contains(search.lower()))
    else:
        display_df = df

    table_pdf = display_df.toPandas()
    st.caption(f"{len(table_pdf):,} matching record(s)")
    st.dataframe(table_pdf, use_container_width=True, hide_index=True)


def render_dashboard(spark: SparkSession, csv_path: Path, source_name: str) -> None:
    raw_df = read_sales_csv(spark, csv_path)
    df = validate_and_prepare(raw_df)

    totals = df.agg(
        F.countDistinct("OrderID").alias("orders"),
        F.sum("Quantity").alias("units"),
        F.sum("Revenue").alias("revenue"),
        F.countDistinct("CustomerID").alias("customers"),
    ).first()

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Business Data Analytics · Mini Project</div>
            <h1>E-Commerce Sales Data Analysis</h1>
            <p>Explore sales performance with Apache Spark-powered analysis.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Data source: {source_name}  ·  Revenue = Quantity × UnitPrice")

    metric_columns = st.columns(4)
    metric_columns[0].metric("Total Orders", f"{totals['orders']:,}")
    metric_columns[1].metric("Total Units Sold", f"{int(totals['units'] or 0):,}")
    metric_columns[2].metric("Total Revenue", format_revenue(totals["revenue"]))
    metric_columns[3].metric("Number of Customers", f"{totals['customers']:,}")

    st.markdown("### Sales insights")
    st.markdown(
        '<p class="section-note">Charts update when you load a different CSV. '
        "Hover for values; use the chart controls to zoom or pan.</p>",
        unsafe_allow_html=True,
    )
    chart_left, chart_right = st.columns(2)

    product_pdf = (
        df.groupBy("Product")
        .agg(F.sum("Quantity").alias("Units Sold"))
        .orderBy(F.desc("Units Sold"), F.asc("Product"))
        .limit(10)
        .toPandas()
    )
    product_chart = (
        alt.Chart(product_pdf)
        .mark_bar(cornerRadiusEnd=5, color="#3978c5")
        .encode(
            x=alt.X("Units Sold:Q", title="Units sold"),
            y=alt.Y("Product:N", sort="-x", title=None),
            tooltip=[
                alt.Tooltip("Product:N", title="Product"),
                alt.Tooltip("Units Sold:Q", title="Units sold", format=","),
            ],
        )
        .properties(height=300)
        .interactive(bind_y=False)
    )
    with chart_left:
        st.markdown("#### Top-selling products")
        st.altair_chart(product_chart, use_container_width=True)

    category_pdf = (
        df.groupBy("Category")
        .agg(F.sum("Revenue").alias("Revenue"))
        .orderBy(F.desc("Revenue"))
        .toPandas()
    )
    category_chart = (
        alt.Chart(category_pdf)
        .mark_bar(cornerRadiusEnd=5, color="#36a68a")
        .encode(
            x=alt.X("Revenue:Q", title="Revenue (₹)", axis=alt.Axis(format=",.0f")),
            y=alt.Y("Category:N", sort="-x", title=None),
            tooltip=[
                alt.Tooltip("Category:N", title="Category"),
                alt.Tooltip("Revenue:Q", title="Revenue (₹)", format=",.2f"),
            ],
        )
        .properties(height=300)
        .interactive(bind_y=False)
    )
    with chart_right:
        st.markdown("#### Revenue by category")
        st.altair_chart(category_chart, use_container_width=True)

    monthly_pdf = (
        df.withColumn("Month", F.date_format("OrderDate", "yyyy-MM"))
        .groupBy("Month")
        .agg(F.sum("Revenue").alias("Revenue"))
        .orderBy("Month")
        .toPandas()
    )
    monthly_chart = (
        alt.Chart(monthly_pdf)
        .mark_line(point=alt.OverlayMarkDef(size=70), color="#e28a3b")
        .encode(
            x=alt.X("Month:O", sort="ascending", title="Month"),
            y=alt.Y("Revenue:Q", title="Revenue (₹)", axis=alt.Axis(format=",.0f")),
            tooltip=[
                alt.Tooltip("Month:O", title="Month"),
                alt.Tooltip("Revenue:Q", title="Revenue (₹)", format=",.2f"),
            ],
        )
        .properties(height=300)
        .interactive(bind_x=False)
    )
    st.markdown("#### Monthly revenue")
    st.altair_chart(monthly_chart, use_container_width=True)

    st.markdown("### Sales dataset")
    st.markdown(
        '<p class="section-note">Revenue is calculated in Spark. '
        "Search filters records across every column.</p>",
        unsafe_allow_html=True,
    )
    searchable_table(df)


def main() -> None:
    add_dashboard_styles()
    with st.sidebar:
        st.markdown("## 🛍️ Sales dashboard")
        st.markdown("Upload a sales CSV, or use the included sample data.")
        uploaded_file = st.file_uploader(
            "Upload sales data",
            type=["csv"],
            help=f"Your CSV must include: {', '.join(REQUIRED_COLUMNS)}.",
        )
        st.divider()
        st.markdown("**Required columns**")
        st.caption(", ".join(REQUIRED_COLUMNS))
        st.markdown("**Analysis engine**")
        st.caption("Apache Spark · local mode")
        st.markdown("**Revenue formula**")
        st.caption("Quantity × UnitPrice")

    if uploaded_file is not None:
        with tempfile.TemporaryDirectory(prefix="ecommerce_sales_") as temp_dir:
            csv_path = Path(temp_dir) / "uploaded_sales.csv"
            csv_path.write_bytes(uploaded_file.getvalue())
            source_name = uploaded_file.name
            try:
                spark = create_spark_session()
                render_dashboard(spark, csv_path, source_name)
            except (FileNotFoundError, ValueError) as error:
                st.error(str(error))
            except (PySparkException, Py4JJavaError, OSError) as error:
                st.error(
                    "Spark could not read or analyse this CSV. Check that the file "
                    f"is a valid CSV and try again. Details: {error}"
                )
    else:
        try:
            spark = create_spark_session()
            render_dashboard(spark, SAMPLE_CSV, SAMPLE_CSV.name)
        except FileNotFoundError as error:
            st.error(
                f"{error}. Restore sales_demo.csv to the same folder as app.py "
                "or upload another CSV in the sidebar."
            )
        except ValueError as error:
            st.error(str(error))
        except (PySparkException, Py4JJavaError, OSError) as error:
            st.error(
                "Spark could not read or analyse the sample data. Check your Java "
                f"installation and try again. Details: {error}"
            )


if __name__ == "__main__":
    main()
