# eda_page.py
import re
import unicodedata
from typing import Dict, Tuple, Any, Optional, List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

try:
    import ftfy
except Exception:
    ftfy = None

# =========================
# VIS STYLE (Storytelling)
# =========================
ACCENT = "#1f77b4"   # énfasis (usar poco)
NEUTRAL = "#9aa0a6"  # contexto
RISK = "#d62728"     # alerta (solo cuando amerita)

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.15,
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

def kpi_tile_sm(title: str, value_text: str, subtitle: str, alert: bool = False):
    """Tile compacto para 3 por fila."""
    fig, ax = plt.subplots(figsize=(4.2, 1.9))
    ax.axis("off")
    color = RISK if alert else ACCENT
    ax.text(0.5, 0.62, value_text, ha="center", va="center",
            fontsize=26, fontweight="bold", color=color, transform=ax.transAxes)
    ax.text(0.5, 0.23, subtitle, ha="center", va="center",
            fontsize=10, color="#333", transform=ax.transAxes)
    ax.set_title(title, pad=6)
    plt.tight_layout()
    return fig

def barh_series(s: pd.Series, title: str, xlabel: str, highlight_last=True, percent=False):
    """Barras horizontales (mejor para categorías)."""
    s = s.dropna()
    if s.empty:
        return None
    s = s.sort_values(ascending=True)
    colors = [NEUTRAL] * len(s)
    if highlight_last and len(colors) > 0:
        colors[-1] = ACCENT

    fig, ax = plt.subplots(figsize=(8, 4.8))
    vals = (s.values * 100) if percent else s.values
    ax.barh(s.index.astype(str), vals, color=colors)
    ax.set_title(title)
    ax.set_xlabel(xlabel + (" (%)" if percent else ""))
    plt.tight_layout()
    return fig

def scatter_clean(x, y, title, xlabel, ylabel, add_h0=False):
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.scatter(x, y, s=14, color=NEUTRAL, alpha=0.6)
    if add_h0:
        ax.axhline(0, linewidth=2, color=RISK, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    return fig

def line_daily(df: pd.DataFrame, date_col: str, y_col: str, agg: str, title: str, ylabel: str):
    """Serie temporal diaria: una sola variable, agregación definida."""
    tmp = df.dropna(subset=[date_col]).copy()
    if tmp.empty:
        return None
    tmp["day"] = tmp[date_col].dt.date

    y = pd.to_numeric(tmp[y_col], errors="coerce") if y_col in tmp.columns else pd.Series(dtype=float)
    tmp["_y"] = y

    if agg == "sum":
        s = tmp.groupby("day")["_y"].sum()
    elif agg == "mean":
        s = tmp.groupby("day")["_y"].mean()
    else:
        s = tmp.groupby("day")["_y"].apply(lambda s: s.notna().sum())

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(s.index.astype(str), s.values, marker="o", color=ACCENT)
    ax.set_title(title)
    ax.set_xlabel("Día")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    return fig


# =========================
# TEXT NORMALIZATION
# =========================
def normalize_text_full(s):
    if pd.isna(s):
        return s
    s = str(s)
    if ftfy is not None:
        s = ftfy.fix_text(s)
    s = s.lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def safe_sum(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return float("nan")
    return float(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())


def ensure_str(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()


def ensure_dt(df: pd.DataFrame, col: str, new_col: str) -> None:
    if col in df.columns:
        df[new_col] = pd.to_datetime(df[col], errors="coerce")
    else:
        df[new_col] = pd.NaT


def ensure_bool_ticket(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        return
    if df[col].dtype == bool:
        return
    t = df[col].astype(str).str.strip().str.lower()
    df[col] = t.isin(["si", "sí", "s", "yes", "true", "1"])


def duplicate_count(df: pd.DataFrame) -> int:
    return int(df.duplicated().sum())


def null_profile(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "nulos_%": (df.isna().mean() * 100).round(2),
        "nulos_n": df.isna().sum().astype(int),
        "tipo": df.dtypes.astype(str)
    }).sort_values("nulos_%", ascending=False)
    return out


def numeric_cols(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include="number").columns.tolist()


def cat_cols(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(exclude="number").columns.tolist()


# =========================
# BUILD UNIFIED DATASET (JOIN + FE)
# =========================
@st.cache_data(show_spinner=False)
def build_unified_dataset(inv: pd.DataFrame, trx: pd.DataFrame, fb: pd.DataFrame,
                          join_trx_fb: str = "inner",
                          join_inv: str = "left") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    meta: Dict[str, Any] = {"warnings": [], "counts": {}, "join_params": {}}
    meta["join_params"] = {"join_trx_fb": join_trx_fb, "join_inv": join_inv}

    inv2 = inv.copy()
    trx2 = trx.copy()
    fb2 = fb.copy()

    # llaves
    ensure_str(inv2, "SKU_ID")
    ensure_str(trx2, "SKU_ID")
    ensure_str(trx2, "Transaccion_ID")
    ensure_str(fb2, "Transaccion_ID")

    # ---------- FB AGG ----------
    if "Transaccion_ID" not in fb2.columns:
        meta["warnings"].append("Feedback no tiene Transaccion_ID; no se puede agregar ni unir.")
        fb_agg = pd.DataFrame(columns=["Transaccion_ID"])
    else:
        agg_map = {}
        for c in ["Rating_Producto", "Rating_Logistica", "Satisfaccion_NPS", "Edad_Cliente"]:
            if c in fb2.columns:
                agg_map[c] = "mean"

        if "Ticket_Soporte_Abierto" in fb2.columns:
            t = fb2["Ticket_Soporte_Abierto"].astype(str).str.strip().str.lower()
            fb2["_Ticket_Soporte_bool"] = t.isin(["si", "sí", "s", "yes", "true", "1"])
            agg_map["_Ticket_Soporte_bool"] = "any"

        fb_agg = (
            fb2.groupby("Transaccion_ID", as_index=False).agg(agg_map)
            if agg_map else fb2[["Transaccion_ID"]].drop_duplicates()
        )

        if "_Ticket_Soporte_bool" in fb_agg.columns:
            fb_agg = fb_agg.rename(columns={"_Ticket_Soporte_bool": "Ticket_Soporte_Abierto"})

    # ---------- TRX + FB ----------
    if "Transaccion_ID" not in trx2.columns or "Transaccion_ID" not in fb_agg.columns:
        meta["warnings"].append("Transacciones o Feedback agregado no tienen Transaccion_ID; no se puede unir.")
        trx_fb = trx2.copy()
    else:
        trx_fb = trx2.merge(
            fb_agg,
            on="Transaccion_ID",
            how=join_trx_fb,
            validate="many_to_one"
        )

    # ---------- (TRX+FB) + INV ----------
    if "SKU_ID" not in trx_fb.columns or "SKU_ID" not in inv2.columns:
        meta["warnings"].append("Transacciones/Inventario no tienen SKU_ID; no se puede unir inventario.")
        dataset = trx_fb.copy()
    else:
        dataset = trx_fb.merge(
            inv2,
            on="SKU_ID",
            how=join_inv,
            validate="many_to_one"
        )

    # ---------- Feature Engineering ----------
    if {"Cantidad_Vendida", "Precio_Venta_Final"}.issubset(dataset.columns):
        dataset["Ingreso_Total"] = (
            pd.to_numeric(dataset["Cantidad_Vendida"], errors="coerce")
            * pd.to_numeric(dataset["Precio_Venta_Final"], errors="coerce")
        )
    else:
        dataset["Ingreso_Total"] = np.nan

    if {"Cantidad_Vendida", "Costo_Unitario_USD"}.issubset(dataset.columns):
        dataset["Costo_Producto"] = (
            pd.to_numeric(dataset["Cantidad_Vendida"], errors="coerce")
            * pd.to_numeric(dataset["Costo_Unitario_USD"], errors="coerce")
        )
    else:
        dataset["Costo_Producto"] = np.nan

    if "Costo_Envio" in dataset.columns:
        dataset["Costo_Logistico"] = pd.to_numeric(dataset["Costo_Envio"], errors="coerce")
    else:
        dataset["Costo_Logistico"] = np.nan

    dataset["Margen_Utilidad"] = dataset["Ingreso_Total"] - dataset["Costo_Producto"] - dataset["Costo_Logistico"]

    dataset["Margen_%"] = np.where(
        pd.to_numeric(dataset["Ingreso_Total"], errors="coerce") > 0,
        dataset["Margen_Utilidad"] / dataset["Ingreso_Total"],
        np.nan
    )

    if "Costo_Unitario_USD" in dataset.columns:
        dataset["SKU_Fantasma"] = dataset["Costo_Unitario_USD"].isna()
    else:
        dataset["SKU_Fantasma"] = False

    if {"Stock_Actual", "Cantidad_Vendida"}.issubset(dataset.columns):
        dataset["Stock_Insuficiente"] = (
            pd.to_numeric(dataset["Stock_Actual"], errors="coerce")
            < pd.to_numeric(dataset["Cantidad_Vendida"], errors="coerce")
        )
    else:
        dataset["Stock_Insuficiente"] = False

    if {"Tiempo_Entrega_Real", "Lead_Time_Dias"}.issubset(dataset.columns):
        dataset["Entrega_Tardia"] = (
            pd.to_numeric(dataset["Tiempo_Entrega_Real"], errors="coerce")
            > pd.to_numeric(dataset["Lead_Time_Dias"], errors="coerce")
        )
    else:
        dataset["Entrega_Tardia"] = False

    dataset["Riesgo_Operacion"] = np.where(
        (dataset["Entrega_Tardia"] == True) | (dataset["Stock_Insuficiente"] == True),
        "Alto",
        "Normal"
    )

    for col in ["Ciudad_Destino", "Canal_Venta"]:
        if col in dataset.columns:
            dataset[f"{col}_norm"] = dataset[col].apply(normalize_text_full)

    ensure_dt(dataset, "Fecha_Venta", "Fecha_Venta_dt")
    ensure_dt(dataset, "Ultima_Revision", "Ultima_Revision_dt")

    if "Ticket_Soporte_Abierto" in dataset.columns:
        ensure_bool_ticket(dataset, "Ticket_Soporte_Abierto")
        dataset["Ticket_Indicador"] = dataset["Ticket_Soporte_Abierto"].astype(int)
    else:
        dataset["Ticket_Indicador"] = 0

    meta["counts"] = {
        "inv_rows": int(len(inv2)),
        "trx_rows": int(len(trx2)),
        "fb_rows": int(len(fb2)),
        "fb_agg_rows": int(len(fb_agg)),
        "dataset_rows": int(len(dataset)),
        "sku_fantasma_rows": int(dataset["SKU_Fantasma"].sum()) if "SKU_Fantasma" in dataset.columns else 0,
    }
    return dataset, meta


# =========================
# RENDER
# =========================
def render_eda(inv_df: Optional[pd.DataFrame],
               trx_df: Optional[pd.DataFrame],
               fb_df: Optional[pd.DataFrame],
               inv_name: str = "Inventario",
               trx_name: str = "Transacciones",
               fb_name: str = "Feedback"):
    st.subheader("📈 EDA dinámico (sobre datasets limpios)")

    if inv_df is None or trx_df is None or fb_df is None:
        st.info("Para el EDA consolidado se requieren los 3 datasets.")
        st.caption("Cargue Inventario + Transacciones + Feedback.")
        return

    # =========================
    # CONTROLES (solo lo esencial)
    # =========================
    with st.sidebar:
        st.subheader("EDA: Controles")

        join_trx_fb = st.selectbox(
            "Join TRX + Feedback",
            options=["left", "inner"],
            index=0,  # left por defecto
            help="left: conserva transacciones sin feedback (recomendado DSS). inner: solo con feedback."
        )

        join_inv = st.selectbox(
            "Join (TRX+FB) + Inventario",
            options=["left", "inner"],
            index=0,
            help="left: conserva ventas aunque el SKU no exista en inventario (SKU fantasma)."
        )

        show_only_controlados = st.checkbox("Solo SKUs controlados (no fantasma)", value=False)
        enable_filters = st.checkbox("Activar filtros (canal/categoría/fechas)", value=True)

    dataset, meta = build_unified_dataset(inv_df, trx_df, fb_df, join_trx_fb=join_trx_fb, join_inv=join_inv)

    # Explicación (requerida)
    with st.expander("📌 ¿Por qué agregamos Feedback por Transaccion_ID?", expanded=True):
        st.write(
            "La razón para agrupar el feedback por **Transaccion_ID** es que el objetivo es evaluar **cada transacción**, "
            "no cada comentario del cliente. Una misma compra puede tener múltiples comentarios, calificaciones o reclamos; "
            "si se juntan esos registros directamente, la venta se repetiría varias veces, distorsionando ingresos, costos, "
            "entregas y calidad del servicio. Al concentrar el feedback en **una sola fila por transacción** —promediando "
            "calificaciones y marcando si hubo soporte— logramos que cada fila represente una compra real con toda su información, "
            "permitiendo métricas y decisiones imparciales."
        )
        st.caption("Implementación: agregación (mean/any) por Transaccion_ID antes del join con Transacciones.")

    if meta.get("warnings"):
        for w in meta["warnings"]:
            st.warning(w)

    # =========================
    # FILTROS
    # =========================
    df = dataset.copy()

    if show_only_controlados and "SKU_Fantasma" in df.columns:
        df = df[df["SKU_Fantasma"] == False].copy()

    if enable_filters:
        cols = st.columns(3)

        with cols[0]:
            if "Canal_Venta_norm" in df.columns:
                canales = ["(Todos)"] + sorted([c for c in df["Canal_Venta_norm"].dropna().unique().tolist() if str(c).strip() != ""])
                sel_canal = st.selectbox("Canal (norm)", options=canales, index=0)
                if sel_canal != "(Todos)":
                    df = df[df["Canal_Venta_norm"] == sel_canal].copy()
            else:
                st.caption("Sin Canal_Venta_norm")

        with cols[1]:
            if "Categoria" in df.columns:
                cats = ["(Todas)"] + sorted([c for c in df["Categoria"].dropna().unique().tolist() if str(c).strip() != ""])
                sel_cat = st.selectbox("Categoría", options=cats, index=0)
                if sel_cat != "(Todas)":
                    df = df[df["Categoria"] == sel_cat].copy()
            else:
                st.caption("Sin Categoria")

        with cols[2]:
            if "Fecha_Venta_dt" in df.columns and df["Fecha_Venta_dt"].notna().any():
                min_d = df["Fecha_Venta_dt"].min().date()
                max_d = df["Fecha_Venta_dt"].max().date()
                r = st.date_input("Rango Fecha_Venta", value=(min_d, max_d))
                if isinstance(r, tuple) and len(r) == 2:
                    d1, d2 = r
                    df = df[(df["Fecha_Venta_dt"].dt.date >= d1) & (df["Fecha_Venta_dt"].dt.date <= d2)].copy()
            else:
                st.caption("Sin Fecha_Venta válida")
                
    st.session_state["eda_filtered_df"] = df.copy()
    # =========================
    # EDA TOP: RESUMEN (cuantitativo)
    # =========================
    st.markdown("## 🧪 Resumen del dataset consolidado (EDA)")
    n_rows, n_cols = df.shape
    dups = duplicate_count(df)
    n_num = len(numeric_cols(df))
    n_cat = len(cat_cols(df))
    null_cells = int(df.isna().sum().sum())
    null_pct = float((null_cells / (n_rows * n_cols) * 100) if (n_rows * n_cols) else 0)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Filas", f"{n_rows:,}")
    c2.metric("Columnas", f"{n_cols:,}")
    c3.metric("Duplicados (fila exacta)", f"{dups:,}")
    c4.metric("Numéricas", f"{n_num:,}")
    c5.metric("Categóricas/Otras", f"{n_cat:,}")

    st.caption(f"Nulos (celdas): {null_cells:,} ({null_pct:.2f}%)")
    st.caption(f"Filas tras filtros: {len(df):,}")

    # Expander: describe primero
    with st.expander("📋 Describe (cuantitativo + cualitativo)", expanded=True):
        st.dataframe(df.describe(include="all").T, use_container_width=True, height=420)

    with st.expander("👀 Vista previa (top 30)", expanded=False):
        st.dataframe(df.head(30), use_container_width=True, height=380)

    with st.expander("🧾 Perfil de nulos por columna", expanded=False):
        st.dataframe(null_profile(df).head(40), use_container_width=True, height=420)

    st.divider()

    # =========================
    # DESCARGA
    # =========================
    st.markdown("### ⬇️ Descargar dataset consolidado")
    st.caption("Incluye joins + variables derivadas. Respeta filtros activos.")
    st.download_button(
        "Descargar dataset_consolidado.csv",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="dataset_consolidado.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.divider()

    # =========================
    # KPIs (sin margen% medio)
    # =========================
    st.markdown("## 📌 KPIs ejecutivos")

    controlados = df[df["SKU_Fantasma"] == False].copy() if "SKU_Fantasma" in df.columns else df.copy()
    fantasmas = df[df["SKU_Fantasma"] == True].copy() if "SKU_Fantasma" in df.columns else df.iloc[0:0].copy()

    # --- cálculos ---
    total_ing = safe_sum(df, "Ingreso_Total")
    ctrl_ing = safe_sum(controlados, "Ingreso_Total")
    fan_ing = safe_sum(fantasmas, "Ingreso_Total")
    ctrl_margen = safe_sum(controlados, "Margen_Utilidad")

    tardia_pct = float(df["Entrega_Tardia"].mean() * 100) if "Entrega_Tardia" in df.columns and len(df) else np.nan
    stock_pct = float(df["Stock_Insuficiente"].mean() * 100) if "Stock_Insuficiente" in df.columns and len(df) else np.nan
    tickets_sum = int(df["Ticket_Indicador"].sum()) if "Ticket_Indicador" in df.columns else 0

    fan_pct = (fan_ing / total_ing * 100) if total_ing > 0 else np.nan

    # --- fila 1 ---
    a, b, c = st.columns(3)
    with a:
        st.pyplot(kpi_tile_sm("Ingreso total", f"{total_ing:,.0f}", "USD", alert=False))
    with b:
        st.pyplot(kpi_tile_sm("Margen controlados", f"{ctrl_margen:,.0f}", "USD", alert=(ctrl_margen < 0)))
    with c:
        st.pyplot(kpi_tile_sm("Confiabilidad logística", f"{tardia_pct:.1f}%", "entrega tardía", alert=(tardia_pct >= 20)))

    # --- fila 2 ---
    d, e, f = st.columns(3)
    with d:
        st.pyplot(kpi_tile_sm("Stock insuficiente", f"{stock_pct:.1f}%", "transacciones", alert=(stock_pct >= 10)))
    with e:
        st.pyplot(kpi_tile_sm("Venta invisible", f"{fan_pct:.1f}%", "ingreso SKU fantasma", alert=(fan_pct >= 10)))
    with f:
        st.pyplot(kpi_tile_sm("Tickets", f"{tickets_sum:,}", "abiertos", alert=(tickets_sum >= 100)))

        # KPI tile 1: Confiabilidad logística
    if "Entrega_Tardia" in df.columns and len(df) > 0:
        tardia = float(df["Entrega_Tardia"].mean() * 100)
        fig = kpi_tile(
            "Confiabilidad logística",
            f"{tardia:.1f}%",
            "de transacciones tienen entrega tardía",
            alert=(tardia >= 20)  # umbral ejemplo
        )
        st.pyplot(fig)
    
    # KPI tile 2: Venta invisible (SKU fantasma)
    if "SKU_Fantasma" in df.columns and "Ingreso_Total" in df.columns and len(df) > 0:
        total_ing = safe_sum(df, "Ingreso_Total")
        ing_fant = safe_sum(df[df["SKU_Fantasma"] == True], "Ingreso_Total")
        pct = (ing_fant / total_ing * 100) if total_ing > 0 else np.nan
        fig = kpi_tile(
            "Venta invisible",
            f"{pct:.1f}%",
            "del ingreso está fuera del inventario (SKU fantasma)",
            alert=(pct >= 10)
        )
        st.pyplot(fig)

    st.divider()

    # =========================
    # ANÁLISIS CUANTITATIVO / CUALITATIVO + VIZ INTERACTIVA
    # =========================
    st.markdown("## 📊 Análisis")
    
    tab_exec, tab_ops, tab_data = st.tabs([
        "Ejecutivo",
        "Operación",
        "Exploración"
    ])


    # =========================================================
    # TAB 1: EJECUTIVO (VALOR)
    # Enfocado en rentabilidad y venta invisible
    # =========================================================
    with tab_exec:
    st.markdown("### Ejecutivo")

    option = st.selectbox(
        "Seleccione vista",
        ["Rentabilidad por canal", "Venta invisible", "Concentración de pérdidas"],
        index=0
    )

    if option == "Rentabilidad por canal":
        if {"Margen_%", "Canal_Venta_norm"}.issubset(df.columns):
            ctrl = df[df["SKU_Fantasma"] == False].copy() if "SKU_Fantasma" in df.columns else df.copy()
            by = ctrl.groupby("Canal_Venta_norm")["Margen_%"].mean().sort_values(ascending=False).head(12)
            fig = barh_series(by, "Canales más rentables (margen % promedio)", "Margen %", percent=True)
            if fig: st.pyplot(fig)
        else:
            st.info("Faltan columnas (Canal_Venta_norm, Margen_%).")

    elif option == "Venta invisible":
        if {"SKU_Fantasma", "Ingreso_Total"}.issubset(df.columns):
            ctrl_ing = safe_sum(df[df["SKU_Fantasma"] == False], "Ingreso_Total")
            fan_ing = safe_sum(df[df["SKU_Fantasma"] == True], "Ingreso_Total")

            fig, ax = plt.subplots(figsize=(6.5, 3.8))
            ax.bar(["Controlado", "Fantasma"], [ctrl_ing, fan_ing], color=[NEUTRAL, ACCENT])
            ax.set_title("Ingreso fuera del inventario (SKU fantasma)")
            ax.set_ylabel("Ingreso total (USD)")
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Faltan columnas (SKU_Fantasma, Ingreso_Total).")

    else:  # Concentración de pérdidas
        if {"SKU_ID", "Margen_Utilidad", "Ingreso_Total", "Transaccion_ID"}.issubset(df.columns):
            ctrl = df[df["SKU_Fantasma"] == False].copy() if "SKU_Fantasma" in df.columns else df.copy()
            base = ctrl.dropna(subset=["SKU_ID", "Margen_Utilidad", "Ingreso_Total", "Transaccion_ID"]).copy()

            sku_kpi = (base.groupby("SKU_ID", as_index=False)
                       .agg(Margen=("Margen_Utilidad", "sum")))
            neg = sku_kpi[sku_kpi["Margen"] < 0].sort_values("Margen").head(15)

            if not neg.empty:
                s = neg.set_index("SKU_ID")["Margen"]
                fig = barh_series(s, "SKUs con pérdidas (margen total negativo)", "Margen total (USD)", highlight_last=False)
                if fig:
                    # pintar en rojo (pérdidas) -> ajuste dentro de la función para este caso
                    st.pyplot(fig)
            else:
                st.caption("No hay pérdidas (según filtros).")
        else:
            st.info("Faltan columnas (SKU_ID, Margen_Utilidad, Ingreso_Total, Transaccion_ID).")

    
    # =========================================================
    # TAB 2: OPERACIÓN (LOGÍSTICA / INVENTARIO)
    # Enfocado en confiabilidad y cuellos de botella
    # =========================================================
      with tab_ops:
        st.markdown("### Operación")
    
        # Selector para no hacer la sección extensa
        op_view = st.selectbox(
            "Seleccione vista",
            [
                "Impacto de entrega tardía en NPS",
                "Quiebre de stock (tasa y categorías)",
                "Cuellos de botella (ciudad / bodega)"
            ],
            index=0,
            key="ops_view"
        )
    
        # =========================================================
        # 1) Impacto de entrega tardía en NPS
        # =========================================================
        if op_view == "Impacto de entrega tardía en NPS":
            if {"Entrega_Tardia", "Satisfaccion_NPS"}.issubset(df.columns):
                tmp = df.dropna(subset=["Entrega_Tardia", "Satisfaccion_NPS"]).copy()
                if tmp.empty:
                    st.info("No hay datos suficientes para comparar NPS por entrega tardía.")
                else:
                    grp = tmp.groupby("Entrega_Tardia")["Satisfaccion_NPS"].mean().reindex([False, True]).dropna()
                    if grp.empty:
                        st.info("No hay datos suficientes para calcular el NPS promedio por estado de entrega.")
                    else:
                        fig, ax = plt.subplots(figsize=(6.5, 3.5))
                        labels = ["Normal", "Tardía"][:len(grp)]
                        colors = [NEUTRAL, ACCENT][:len(grp)]
                        ax.bar(labels, grp.values, color=colors)
                        ax.set_title("NPS promedio según entrega tardía")
                        ax.set_ylabel("NPS promedio")
                        plt.tight_layout()
                        st.pyplot(fig)
    
                        # diferencia (si están ambos grupos)
                        if (False in grp.index) and (True in grp.index):
                            delta = float(grp.loc[False] - grp.loc[True])
                            st.caption(f"Diferencia (Normal - Tardía): {delta:.2f} puntos de NPS.")
            else:
                st.info("Faltan columnas (Entrega_Tardia, Satisfaccion_NPS).")
    
        # =========================================================
        # 2) Quiebre de stock: tasa y categorías
        # =========================================================
        elif op_view == "Quiebre de stock (tasa y categorías)":
            if {"Stock_Insuficiente", "Categoria"}.issubset(df.columns):
                tmp = df.dropna(subset=["Stock_Insuficiente", "Categoria"]).copy()
                if tmp.empty:
                    st.info("No hay datos suficientes para evaluar quiebre de stock.")
                else:
                    tasa = float(tmp["Stock_Insuficiente"].mean() * 100) if len(tmp) else np.nan
                    fig = kpi_tile_sm(
                        "Riesgo de quiebre de stock",
                        f"{tasa:.1f}%",
                        "de transacciones con stock insuficiente",
                        alert=(pd.notna(tasa) and tasa >= 10)
                    )
                    st.pyplot(fig)
    
                    # Por categoría (solo controlados si existe SKU_Fantasma)
                    ctrl = tmp[tmp["SKU_Fantasma"] == False].copy() if "SKU_Fantasma" in tmp.columns else tmp.copy()
                    if len(ctrl) > 0:
                        top = (ctrl.assign(q=ctrl["Stock_Insuficiente"].astype(int))
                                  .groupby("Categoria")["q"].mean()
                                  .sort_values(ascending=False)
                                  .head(10))
                        fig = barh_series(
                            top,
                            "Categorías con mayor quiebre de stock",
                            "Tasa stock insuficiente",
                            percent=True
                        )
                        if fig:
                            st.pyplot(fig)
                    else:
                        st.caption("No hay transacciones controladas disponibles con los filtros actuales.")
            else:
                st.info("Faltan columnas (Stock_Insuficiente, Categoria).")
    
        # =========================================================
        # 3) Cuellos de botella por ciudad / bodega
        # =========================================================
        else:
            # Config mínimo para robustez
            min_n = st.slider("Mínimo de transacciones por grupo", 20, 200, 50, step=10, key="ops_min_n")
            top_k = st.slider("Mostrar Top", 5, 20, 10, step=1, key="ops_top_k")
    
            # -------------------------
            # Ciudades
            # -------------------------
            st.markdown("#### Ciudad destino")
            if {"Ciudad_Destino_norm", "Tiempo_Entrega_Real", "Satisfaccion_NPS", "Transaccion_ID"}.issubset(df.columns):
                log = df.dropna(subset=["Ciudad_Destino_norm", "Tiempo_Entrega_Real", "Satisfaccion_NPS", "Transaccion_ID"]).copy()
                if log.empty:
                    st.info("No hay datos suficientes para análisis por ciudad.")
                else:
                    log["NPS_bajo"] = (pd.to_numeric(log["Satisfaccion_NPS"], errors="coerce") <= 0).astype(int)
    
                    stats_city = (log.groupby("Ciudad_Destino_norm")
                                    .agg(
                                        n=("Transaccion_ID", "count"),
                                        tiempo_prom=("Tiempo_Entrega_Real", "mean"),
                                        tasa_nps_bajo=("NPS_bajo", "mean")
                                    ))
    
                    stats_city = stats_city[stats_city["n"] >= min_n].copy()
                    stats_city["score"] = stats_city["tasa_nps_bajo"].fillna(0) * stats_city["tiempo_prom"].fillna(0)
                    stats_city = stats_city.sort_values("score", ascending=False).head(top_k)
    
                    if stats_city.empty:
                        st.caption("No hay ciudades con suficiente volumen bajo los filtros actuales.")
                    else:
                        fig = barh_series(
                            stats_city["tasa_nps_bajo"],
                            "Ciudades con mayor proporción de NPS bajo",
                            "Tasa NPS bajo",
                            percent=True,
                            highlight_last=False
                        )
                        if fig:
                            st.pyplot(fig)
                        st.dataframe(stats_city[["n", "tiempo_prom", "tasa_nps_bajo", "score"]].round(3),
                                     use_container_width=True)
            else:
                st.caption("No hay columnas suficientes para análisis por ciudad (Ciudad_Destino_norm, Tiempo_Entrega_Real, Satisfaccion_NPS, Transaccion_ID).")
    
            # -------------------------
            # Bodegas (solo controlados)
            # -------------------------
            st.markdown("#### Bodega origen")
            if {"Bodega_Origen", "Tiempo_Entrega_Real", "Satisfaccion_NPS", "Transaccion_ID"}.issubset(df.columns):
                base = df.copy()
                if "SKU_Fantasma" in base.columns:
                    base = base[base["SKU_Fantasma"] == False].copy()
    
                bod = base.dropna(subset=["Bodega_Origen", "Tiempo_Entrega_Real", "Satisfaccion_NPS", "Transaccion_ID"]).copy()
                if bod.empty:
                    st.info("No hay datos suficientes para análisis por bodega.")
                else:
                    bod["NPS_bajo"] = (pd.to_numeric(bod["Satisfaccion_NPS"], errors="coerce") <= 0).astype(int)
    
                    stats_bod = (bod.groupby("Bodega_Origen")
                                   .agg(
                                       n=("Transaccion_ID", "count"),
                                       tiempo_prom=("Tiempo_Entrega_Real", "mean"),
                                       tasa_nps_bajo=("NPS_bajo", "mean")
                                   ))
    
                    stats_bod = stats_bod[stats_bod["n"] >= min_n].copy()
                    stats_bod["score"] = stats_bod["tasa_nps_bajo"].fillna(0) * stats_bod["tiempo_prom"].fillna(0)
                    stats_bod = stats_bod.sort_values("score", ascending=False).head(top_k)
    
                    if stats_bod.empty:
                        st.caption("No hay bodegas con suficiente volumen bajo los filtros actuales.")
                    else:
                        fig = barh_series(
                            stats_bod["tasa_nps_bajo"],
                            "Bodegas con mayor proporción de NPS bajo",
                            "Tasa NPS bajo",
                            percent=True,
                            highlight_last=False
                        )
                        if fig:
                            st.pyplot(fig)
                        st.dataframe(stats_bod[["n", "tiempo_prom", "tasa_nps_bajo", "score"]].round(3),
                                     use_container_width=True)
            else:
                st.caption("No hay columnas suficientes para análisis por bodega (Bodega_Origen, Tiempo_Entrega_Real, Satisfaccion_NPS, Transaccion_ID).")

    
    # =========================================================
    # TAB 4: EXPLORACIÓN CONTROLADA (DINÁMICO, PERO RESTRINGIDO)
    # Solo 4 gráficos permitidos, con variables pre-curadas
    # =========================================================
    with tab_data:
        st.markdown("### Exploración controlada (dinámico pero útil)")
        st.caption("Se limita a gráficos que suelen aportar valor en DSS: no se permite cualquier combinación.")
    
        # Lista curada de variables numéricas útiles (si existen)
        curated_num = [c for c in [
            "Ingreso_Total", "Margen_Utilidad", "Margen_%", "Costo_Envio",
            "Tiempo_Entrega_Real", "Lead_Time_Dias", "Stock_Actual", "Cantidad_Vendida",
            "Satisfaccion_NPS", "Rating_Producto", "Rating_Logistica", "Ticket_Indicador"
        ] if c in df.columns]
    
        curated_cat = [c for c in [
            "Canal_Venta_norm", "Categoria", "Ciudad_Destino_norm", "Bodega_Origen", "Riesgo_Operacion"
        ] if c in df.columns]
    
        chart = st.selectbox(
            "Seleccione visualización",
            options=[
                "Distribución (histograma) de una métrica",
                "Top categorías por una métrica",
                "Relación: costo envío vs margen",
                "Serie temporal: métrica diaria"
            ],
            index=0
        )
    
        if chart == "Distribución (histograma) de una métrica":
            if not curated_num:
                st.info("No hay variables numéricas curadas disponibles.")
            else:
                x = st.selectbox("Métrica", curated_num, index=0)
                series = pd.to_numeric(df[x], errors="coerce").dropna()
                fig, ax = plt.subplots(figsize=(7, 4.2))
                ax.hist(series, bins=30, color=NEUTRAL)
                if x in ["Margen_Utilidad", "Margen_%"]:
                    ax.axvline(0, linewidth=2, color=RISK, alpha=0.8)
                ax.set_title(f"Distribución de {x}")
                ax.set_xlabel(x)
                ax.set_ylabel("Frecuencia")
                plt.tight_layout()
                st.pyplot(fig)
    
        elif chart == "Top categorías por una métrica":
            if not curated_cat or not curated_num:
                st.info("Se requiere al menos 1 categórica y 1 numérica curada.")
            else:
                cat = st.selectbox("Categoría", curated_cat, index=0)
                metric = st.selectbox("Métrica", curated_num, index=0)
                agg = st.selectbox("Agregación", ["sum", "mean", "count"], index=0)
                topn = st.slider("Top N", 5, 20, 10)
    
                tmp = df[[cat, metric]].copy()
                tmp[cat] = tmp[cat].astype(str)
                tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
    
                if agg == "sum":
                    s = tmp.groupby(cat)[metric].sum().sort_values(ascending=False).head(topn)
                    xlabel = f"{metric} (sum)"
                elif agg == "mean":
                    s = tmp.groupby(cat)[metric].mean().sort_values(ascending=False).head(topn)
                    xlabel = f"{metric} (mean)"
                else:
                    s = tmp.groupby(cat)[metric].apply(lambda s: s.notna().sum()).sort_values(ascending=False).head(topn)
                    xlabel = f"{metric} (count)"
    
                fig = barh_series(s, f"Top {topn} {cat} por {xlabel}", xlabel, highlight_last=True, percent=False)
                if fig: st.pyplot(fig)
    
        elif chart == "Relación: costo envío vs margen":
            if not {"Costo_Envio", "Margen_Utilidad"}.issubset(df.columns):
                st.info("Faltan columnas (Costo_Envio, Margen_Utilidad).")
            else:
                ctrl = df[df["SKU_Fantasma"] == False].copy() if "SKU_Fantasma" in df.columns else df.copy()
                xx = pd.to_numeric(ctrl["Costo_Envio"], errors="coerce")
                yy = pd.to_numeric(ctrl["Margen_Utilidad"], errors="coerce")
                mask = xx.notna() & yy.notna()
                fig = scatter_clean(xx[mask], yy[mask],
                                   "El costo de envío está empujando ventas por debajo de margen 0",
                                   "Costo_Envio (USD)", "Margen_Utilidad (USD)", add_h0=True)
                st.pyplot(fig)
    
        else:  # Serie temporal
            if "Fecha_Venta_dt" not in df.columns or not df["Fecha_Venta_dt"].notna().any():
                st.info("No hay Fecha_Venta_dt válida.")
            elif not curated_num:
                st.info("No hay métricas numéricas curadas disponibles.")
            else:
                x = st.selectbox("Métrica", curated_num, index=0)
                agg = st.selectbox("Agregación", ["sum", "mean", "count"], index=0)
                fig = line_daily(df, "Fecha_Venta_dt", x, agg,
                                 title=f"Serie temporal diaria: {x} ({agg})",
                                 ylabel=f"{x} ({agg})")
                if fig: st.pyplot(fig)

