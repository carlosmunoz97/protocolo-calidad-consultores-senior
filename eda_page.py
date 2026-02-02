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
            options=["inner", "left"],
            index=0,
            help="inner: solo transacciones con feedback. left: conserva transacciones sin feedback."
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
    st.markdown("## 📌 KPIs ejecutivos (según filtros)")

    controlados = df[df["SKU_Fantasma"] == False].copy() if "SKU_Fantasma" in df.columns else df.copy()
    fantasmas = df[df["SKU_Fantasma"] == True].copy() if "SKU_Fantasma" in df.columns else df.iloc[0:0].copy()

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Ingreso total", f"{safe_sum(df,'Ingreso_Total'):,.2f}")
    with k2:
        st.metric("Ingreso controlados", f"{safe_sum(controlados,'Ingreso_Total'):,.2f}")
    with k3:
        st.metric("Ingreso fantasmas", f"{safe_sum(fantasmas,'Ingreso_Total'):,.2f}")
    with k4:
        st.metric("Margen controlados", f"{safe_sum(controlados,'Margen_Utilidad'):,.2f}")

    k5, k6, k7 = st.columns(3)
    with k5:
        st.metric("Entrega tardía %", f"{float(df['Entrega_Tardia'].mean()*100) if 'Entrega_Tardia' in df.columns else np.nan:,.2f}%")
    with k6:
        st.metric("Stock insuficiente %", f"{float(df['Stock_Insuficiente'].mean()*100) if 'Stock_Insuficiente' in df.columns else np.nan:,.2f}%")
    with k7:
        st.metric("Tickets (suma)", f"{int(df['Ticket_Indicador'].sum()) if 'Ticket_Indicador' in df.columns else 0}")

    st.divider()

    # =========================
    # ANÁLISIS CUANTITATIVO / CUALITATIVO + VIZ INTERACTIVA
    # =========================
    st.markdown("## 📊 Análisis (cuantitativo y cualitativo)")

    tab_q, tab_c, tab_viz = st.tabs(["Cuantitativo", "Cualitativo", "Visualización interactiva"])

    # ---------- CUANTITATIVO ----------
    with tab_q:
        st.markdown("### Distribuciones y relaciones (numéricas)")

        num = numeric_cols(df)
        if not num:
            st.info("No se detectaron columnas numéricas.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                x = st.selectbox("Variable numérica (X)", options=num, index=0)
            with col2:
                y = st.selectbox("Variable numérica (Y, opcional)", options=["(Ninguna)"] + num, index=0)

            chart_type = st.selectbox(
                "Tipo de gráfico",
                options=["Histograma", "Boxplot", "Scatter (X vs Y)", "Serie temporal (por fecha)"],
                index=0
            )

            if chart_type == "Histograma":
                fig = plt.figure()
                plt.hist(pd.to_numeric(df[x], errors="coerce").dropna(), bins=30)
                plt.title(f"Histograma: {x}")
                plt.xlabel(x)
                plt.ylabel("Frecuencia")
                plt.tight_layout()
                st.pyplot(fig)

            elif chart_type == "Boxplot":
                fig = plt.figure()
                plt.boxplot(pd.to_numeric(df[x], errors="coerce").dropna(), vert=True)
                plt.title(f"Boxplot: {x}")
                plt.ylabel(x)
                plt.tight_layout()
                st.pyplot(fig)

            elif chart_type == "Scatter (X vs Y)":
                if y == "(Ninguna)":
                    st.info("Seleccione Y para usar scatter.")
                else:
                    xx = pd.to_numeric(df[x], errors="coerce")
                    yy = pd.to_numeric(df[y], errors="coerce")
                    mask = xx.notna() & yy.notna()
                    fig = plt.figure()
                    plt.scatter(xx[mask], yy[mask], s=10)
                    plt.title(f"Scatter: {x} vs {y}")
                    plt.xlabel(x)
                    plt.ylabel(y)
                    plt.tight_layout()
                    st.pyplot(fig)

                    corr = xx[mask].corr(yy[mask])
                    st.caption(f"Correlación Pearson (sobre pares válidos): {corr:.4f}" if pd.notna(corr) else "Correlación no disponible.")

            elif chart_type == "Serie temporal (por fecha)":
                if "Fecha_Venta_dt" not in df.columns or df["Fecha_Venta_dt"].notna().sum() == 0:
                    st.info("No hay Fecha_Venta_dt válida para serie temporal.")
                else:
                    agg = st.selectbox("Agregación", options=["sum", "mean", "count"], index=0)
                    # agrupar por día
                    tmp = df.dropna(subset=["Fecha_Venta_dt"]).copy()
                    tmp["day"] = tmp["Fecha_Venta_dt"].dt.date
                    if agg == "sum":
                        s = tmp.groupby("day")[x].apply(lambda s: pd.to_numeric(s, errors="coerce").sum())
                    elif agg == "mean":
                        s = tmp.groupby("day")[x].apply(lambda s: pd.to_numeric(s, errors="coerce").mean())
                    else:
                        s = tmp.groupby("day")[x].apply(lambda s: pd.to_numeric(s, errors="coerce").notna().sum())

                    fig = plt.figure()
                    plt.plot(s.index.astype(str), s.values, marker="o")
                    plt.title(f"Serie temporal ({agg}) de {x}")
                    plt.xlabel("Día")
                    plt.ylabel(f"{x} ({agg})")
                    plt.xticks(rotation=45, ha="right")
                    plt.tight_layout()
                    st.pyplot(fig)

        st.markdown("### Matriz de correlación (Top)")
        if len(num) >= 2:
            corr = df[num].apply(pd.to_numeric, errors="coerce").corr()
            st.dataframe(corr.round(3), use_container_width=True, height=320)
        else:
            st.caption("No hay suficientes numéricas para correlación.")

    # ---------- CUALITATIVO ----------
    with tab_c:
        st.markdown("### Frecuencias / composición (categóricas)")
        cats = cat_cols(df)
        if not cats:
            st.info("No se detectaron columnas categóricas.")
        else:
            c = st.selectbox("Variable categórica", options=cats, index=0)
            topn = st.slider("Top N", 5, 30, 10)
            vc = df[c].astype(str).value_counts(dropna=False).head(topn)

            st.dataframe(vc.rename("conteo").to_frame(), use_container_width=True)

            fig = plt.figure()
            vc.plot(kind="bar")
            plt.title(f"Top {topn}: {c}")
            plt.xlabel(c)
            plt.ylabel("Conteo")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)

        st.markdown("### SKU fantasma (cualitativo de control)")
        if "SKU_Fantasma" in df.columns:
            sku_f = int(df["SKU_Fantasma"].sum())
            pct = (sku_f / len(df) * 100) if len(df) else 0
            st.write(f"Ventas con SKU fantasma: **{sku_f:,}** (**{pct:.2f}%**)")

            if {"Canal_Venta_norm", "Ingreso_Total"}.issubset(df.columns):
                tab = (df.assign(es_fantasma=df["SKU_Fantasma"])
                         .groupby(["Canal_Venta_norm", "es_fantasma"])["Ingreso_Total"]
                         .sum()
                         .unstack(fill_value=0))
                st.dataframe(tab.sort_values(by=True if True in tab.columns else tab.columns[-1], ascending=False).head(15),
                             use_container_width=True)

    # ---------- VISUALIZACIÓN INTERACTIVA (elige gráfico que “mejor se ajuste”) ----------
    with tab_viz:
        st.markdown("### Constructor de gráficos (interactivo)")

        num = numeric_cols(df)
        cats = cat_cols(df)

        viz_type = st.selectbox(
            "Tipo de visualización",
            options=[
                "Univariado numérico",
                "Univariado categórico",
                "Bivariado (numérico vs numérico)",
                "Bivariado (categórico vs numérico)",
            ],
            index=0
        )

        if viz_type == "Univariado numérico":
            if not num:
                st.info("No hay numéricas.")
            else:
                x = st.selectbox("Variable", options=num, index=0)
                g = st.selectbox("Gráfico", options=["Histograma", "Boxplot"], index=0)
                if g == "Histograma":
                    fig = plt.figure()
                    plt.hist(pd.to_numeric(df[x], errors="coerce").dropna(), bins=30)
                    plt.title(f"Histograma: {x}")
                    plt.xlabel(x); plt.ylabel("Frecuencia")
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    fig = plt.figure()
                    plt.boxplot(pd.to_numeric(df[x], errors="coerce").dropna())
                    plt.title(f"Boxplot: {x}")
                    plt.ylabel(x)
                    plt.tight_layout()
                    st.pyplot(fig)

        elif viz_type == "Univariado categórico":
            if not cats:
                st.info("No hay categóricas.")
            else:
                x = st.selectbox("Variable", options=cats, index=0)
                topn = st.slider("Top N", 5, 40, 15)
                vc = df[x].astype(str).value_counts(dropna=False).head(topn)
                fig = plt.figure()
                vc.plot(kind="bar")
                plt.title(f"Top {topn}: {x}")
                plt.xlabel(x); plt.ylabel("Conteo")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st.pyplot(fig)

        elif viz_type == "Bivariado (numérico vs numérico)":
            if len(num) < 2:
                st.info("Se requieren al menos 2 numéricas.")
            else:
                x = st.selectbox("X", options=num, index=0)
                y = st.selectbox("Y", options=[c for c in num if c != x], index=0)
                fig = plt.figure()
                xx = pd.to_numeric(df[x], errors="coerce")
                yy = pd.to_numeric(df[y], errors="coerce")
                mask = xx.notna() & yy.notna()
                plt.scatter(xx[mask], yy[mask], s=10)
                plt.title(f"Scatter: {x} vs {y}")
                plt.xlabel(x); plt.ylabel(y)
                plt.tight_layout()
                st.pyplot(fig)

        else:  # categórico vs numérico
            if not cats or not num:
                st.info("Se requiere 1 categórica y 1 numérica.")
            else:
                x = st.selectbox("Categoría (X)", options=cats, index=0)
                y = st.selectbox("Numérica (Y)", options=num, index=0)
                agg = st.selectbox("Agregación", options=["mean", "sum", "count"], index=0)

                tmp = df[[x, y]].copy()
                tmp[x] = tmp[x].astype(str)
                tmp[y] = pd.to_numeric(tmp[y], errors="coerce")

                if agg == "count":
                    s = tmp.groupby(x)[y].apply(lambda s: s.notna().sum()).sort_values(ascending=False).head(20)
                    ylabel = f"{y} (count)"
                elif agg == "sum":
                    s = tmp.groupby(x)[y].sum().sort_values(ascending=False).head(20)
                    ylabel = f"{y} (sum)"
                else:
                    s = tmp.groupby(x)[y].mean().sort_values(ascending=False).head(20)
                    ylabel = f"{y} (mean)"

                fig = plt.figure()
                s.plot(kind="bar")
                plt.title(f"{ylabel} por {x} (Top 20)")
                plt.xlabel(x); plt.ylabel(ylabel)
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st.pyplot(fig)
