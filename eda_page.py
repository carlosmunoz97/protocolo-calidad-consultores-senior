# eda_page.py
import re
import unicodedata
from typing import Dict, Tuple, Any, Optional

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


def safe_mean(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return float("nan")
    return float(pd.to_numeric(df[col], errors="coerce").mean())


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


# =========================
# BUILD UNIFIED DATASET (JOIN + FE)
# =========================
@st.cache_data(show_spinner=False)
def build_unified_dataset(inv: pd.DataFrame, trx: pd.DataFrame, fb: pd.DataFrame,
                          join_trx_fb: str = "inner",
                          join_inv: str = "left") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    - Normaliza llaves
    - Agrega feedback a nivel Transaccion_ID
    - Join trx + fb_agg (inner/left)
    - Join con inventario (left/inner)
    - Feature engineering
    """
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
        # validate: muchas transacciones -> 1 fila de fb_agg por id
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
    # Ingreso total
    if {"Cantidad_Vendida", "Precio_Venta_Final"}.issubset(dataset.columns):
        dataset["Ingreso_Total"] = (
            pd.to_numeric(dataset["Cantidad_Vendida"], errors="coerce")
            * pd.to_numeric(dataset["Precio_Venta_Final"], errors="coerce")
        )
    else:
        dataset["Ingreso_Total"] = np.nan

    # Costo producto
    if {"Cantidad_Vendida", "Costo_Unitario_USD"}.issubset(dataset.columns):
        dataset["Costo_Producto"] = (
            pd.to_numeric(dataset["Cantidad_Vendida"], errors="coerce")
            * pd.to_numeric(dataset["Costo_Unitario_USD"], errors="coerce")
        )
    else:
        dataset["Costo_Producto"] = np.nan

    # Costo logístico
    if "Costo_Envio" in dataset.columns:
        dataset["Costo_Logistico"] = pd.to_numeric(dataset["Costo_Envio"], errors="coerce")
    else:
        dataset["Costo_Logistico"] = np.nan

    # Margen
    dataset["Margen_Utilidad"] = dataset["Ingreso_Total"] - dataset["Costo_Producto"] - dataset["Costo_Logistico"]

    dataset["Margen_%"] = np.where(
        pd.to_numeric(dataset["Ingreso_Total"], errors="coerce") > 0,
        dataset["Margen_Utilidad"] / dataset["Ingreso_Total"],
        np.nan
    )

    # SKU fantasma
    if "Costo_Unitario_USD" in dataset.columns:
        dataset["SKU_Fantasma"] = dataset["Costo_Unitario_USD"].isna()
    else:
        dataset["SKU_Fantasma"] = False

    # Stock insuficiente
    if {"Stock_Actual", "Cantidad_Vendida"}.issubset(dataset.columns):
        dataset["Stock_Insuficiente"] = (
            pd.to_numeric(dataset["Stock_Actual"], errors="coerce")
            < pd.to_numeric(dataset["Cantidad_Vendida"], errors="coerce")
        )
    else:
        dataset["Stock_Insuficiente"] = False

    # Entrega tardía
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

    # Normalizaciones de texto (si existen)
    for col in ["Ciudad_Destino", "Canal_Venta"]:
        if col in dataset.columns:
            dataset[f"{col}_norm"] = dataset[col].apply(normalize_text_full)

    # Fechas
    ensure_dt(dataset, "Fecha_Venta", "Fecha_Venta_dt")
    ensure_dt(dataset, "Ultima_Revision", "Ultima_Revision_dt")

    # Ticket
    if "Ticket_Soporte_Abierto" in dataset.columns:
        ensure_bool_ticket(dataset, "Ticket_Soporte_Abierto")
        dataset["Ticket_Indicador"] = dataset["Ticket_Soporte_Abierto"].astype(int)
    else:
        dataset["Ticket_Indicador"] = 0

    # Meta
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
# MAIN EDA RENDER
# =========================
def render_eda(inv_df: Optional[pd.DataFrame],
               trx_df: Optional[pd.DataFrame],
               fb_df: Optional[pd.DataFrame],
               inv_name: str = "Inventario",
               trx_name: str = "Transacciones",
               fb_name: str = "Feedback"):
    st.subheader("📈 EDA dinámico")

    if inv_df is None or trx_df is None or fb_df is None:
        st.info("Para el EDA consolidado se requieren los 3 datasets.")
        st.caption("Cargue Inventario + Transacciones + Feedback.")
        return

    # =========================
    # CONTROLES INTERACTIVOS
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

        show_only_controlados = st.checkbox(
            "Filtrar solo SKUs controlados (no fantasma)",
            value=False
        )

        # Filtros dinámicos (se aplican luego de construir dataset)
        enable_filters = st.checkbox("Activar filtros (canal/categoría/fechas)", value=True)

    # Construir dataset unificado (cacheado)
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

    # Warnings
    if meta.get("warnings"):
        for w in meta["warnings"]:
            st.warning(w)

    # Chequeos de unión
    with st.expander("🔎 Unión estratégica (conteos y control)", expanded=False):
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric(f"Filas {inv_name}", meta["counts"]["inv_rows"])
        with c2:
            st.metric(f"Filas {trx_name}", meta["counts"]["trx_rows"])
        with c3:
            st.metric(f"Filas {fb_name}", meta["counts"]["fb_rows"])
        with c4:
            st.metric("Feedback agregado", meta["counts"]["fb_agg_rows"])
        with c5:
            st.metric("Dataset unificado", meta["counts"]["dataset_rows"])

        if "SKU_Fantasma" in dataset.columns:
            sku_f = int(dataset["SKU_Fantasma"].sum())
            pct = (sku_f / len(dataset) * 100) if len(dataset) else 0
            st.write(f"**SKU fantasma (sin match en inventario):** {sku_f} (**{pct:.2f}%**)")

    # =========================
    # FILTROS INTERACTIVOS
    # =========================
    df = dataset.copy()

    if show_only_controlados and "SKU_Fantasma" in df.columns:
        df = df[df["SKU_Fantasma"] == False].copy()

    if enable_filters:
        cols = st.columns(3)

        # Canal
        with cols[0]:
            if "Canal_Venta_norm" in df.columns:
                canales = ["(Todos)"] + sorted([c for c in df["Canal_Venta_norm"].dropna().unique().tolist() if str(c).strip() != ""])
                sel_canal = st.selectbox("Canal (norm)", options=canales, index=0)
                if sel_canal != "(Todos)":
                    df = df[df["Canal_Venta_norm"] == sel_canal].copy()
            else:
                st.caption("Sin Canal_Venta_norm")

        # Categoría
        with cols[1]:
            if "Categoria" in df.columns:
                cats = ["(Todas)"] + sorted([c for c in df["Categoria"].dropna().unique().tolist() if str(c).strip() != ""])
                sel_cat = st.selectbox("Categoría", options=cats, index=0)
                if sel_cat != "(Todas)":
                    df = df[df["Categoria"] == sel_cat].copy()
            else:
                st.caption("Sin Categoria")

        # Fechas
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

    st.caption(f"Filas tras filtros: {len(df):,}")

    # Particiones (para KPIs / comparaciones)
    controlados = df[df["SKU_Fantasma"] == False].copy() if "SKU_Fantasma" in df.columns else df.copy()
    fantasmas = df[df["SKU_Fantasma"] == True].copy() if "SKU_Fantasma" in df.columns else df.iloc[0:0].copy()

    # =========================
    # DESCARGA DATASET CONSOLIDADO
    # =========================
    st.markdown("### ⬇️ Descargar dataset consolidado")
    st.caption("Incluye joins + variables derivadas. Respeta los filtros aplicados arriba.")
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Descargar dataset_consolidado.csv",
        data=csv_bytes,
        file_name="dataset_consolidado.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.divider()

    # =========================
    # KPIs dinámicos
    # =========================
    st.markdown("### 📌 KPIs (según filtros)")

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Ingreso total", f"{safe_sum(df,'Ingreso_Total'):,.2f}")
    with k2:
        st.metric("Ingreso controlados", f"{safe_sum(controlados,'Ingreso_Total'):,.2f}")
    with k3:
        st.metric("Ingreso fantasmas", f"{safe_sum(fantasmas,'Ingreso_Total'):,.2f}")
    with k4:
        st.metric("Margen controlados", f"{safe_sum(controlados,'Margen_Utilidad'):,.2f}")

    k5, k6, k7, k8 = st.columns(4)
    with k5:
        st.metric("Margen % medio (ctrl)", f"{safe_mean(controlados,'Margen_%')*100:,.2f}%")
    with k6:
        st.metric("Entrega tardía %", f"{float(df['Entrega_Tardia'].mean()*100) if 'Entrega_Tardia' in df.columns else np.nan:,.2f}%")
    with k7:
        st.metric("Stock insuficiente %", f"{float(df['Stock_Insuficiente'].mean()*100) if 'Stock_Insuficiente' in df.columns else np.nan:,.2f}%")
    with k8:
        st.metric("Tickets (suma)", f"{int(df['Ticket_Indicador'].sum()) if 'Ticket_Indicador' in df.columns else 0}")

    st.divider()

    # =========================
    # GRÁFICOS (dinámicos)
    # =========================
    st.markdown("### 📊 Visualizaciones")

    chart = st.selectbox(
        "Seleccione análisis",
        options=[
            "Distribución Margen (controlados)",
            "Ingreso por Categoría (Top 10, controlados)",
            "Entregas tardías (conteo)",
            "Margen vs Costo Envío (controlados)",
            "Margen % por Canal (controlados)",
            "NPS vs Entrega tardía",
            "Riesgo por Categoría (Top 10, controlados)",
            "SKU fantasma: ciudades Top 10",
            "Vista previa / describe"
        ],
        index=0
    )

    if chart == "Distribución Margen (controlados)":
        if "Margen_Utilidad" in controlados.columns:
            fig = plt.figure()
            plt.hist(controlados["Margen_Utilidad"].dropna(), bins=30)
            plt.title("Distribución del Margen de Utilidad (SKUs controlados)")
            plt.xlabel("Margen_Utilidad")
            plt.ylabel("Número de ventas")
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No existe Margen_Utilidad en el dataset consolidado.")

    elif chart == "Ingreso por Categoría (Top 10, controlados)":
        if {"Categoria", "Ingreso_Total"}.issubset(controlados.columns):
            top = (controlados.groupby("Categoria")["Ingreso_Total"]
                   .sum()
                   .sort_values(ascending=False)
                   .head(10))
            fig = plt.figure()
            top.plot(kind="bar")
            plt.title("Ingreso Total por Categoría (Top 10, SKUs controlados)")
            plt.xlabel("Categoría")
            plt.ylabel("Ingreso_Total")
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Faltan columnas Categoria/Ingreso_Total.")

    elif chart == "Entregas tardías (conteo)":
        if "Entrega_Tardia" in df.columns:
            vc = df["Entrega_Tardia"].value_counts(dropna=False)
            fig = plt.figure()
            vc.plot(kind="bar")
            plt.title("Entregas tardías vs normales (conteo)")
            plt.xlabel("Entrega_Tardia")
            plt.ylabel("Número de ventas")
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No existe Entrega_Tardia.")

    elif chart == "Margen vs Costo Envío (controlados)":
        if {"Margen_Utilidad", "Costo_Envio"}.issubset(controlados.columns):
            fig = plt.figure()
            plt.scatter(controlados["Costo_Envio"], controlados["Margen_Utilidad"], s=10)
            plt.title("Margen vs Costo de Envío (SKUs controlados)")
            plt.xlabel("Costo_Envio")
            plt.ylabel("Margen_Utilidad")
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Faltan columnas Margen_Utilidad/Costo_Envio.")

    elif chart == "Margen % por Canal (controlados)":
        if {"Canal_Venta_norm", "Margen_%"}.issubset(controlados.columns):
            by = (controlados.groupby("Canal_Venta_norm")["Margen_%"]
                  .mean()
                  .sort_values(ascending=False))
            fig = plt.figure()
            by.plot(kind="bar")
            plt.title("Margen % promedio por Canal (SKUs controlados)")
            plt.xlabel("Canal_Venta_norm")
            plt.ylabel("Margen_% promedio")
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Faltan columnas Canal_Venta_norm/Margen_%.")

    elif chart == "NPS vs Entrega tardía":
        if {"Satisfaccion_NPS", "Entrega_Tardia"}.issubset(df.columns):
            grp = df.groupby("Entrega_Tardia")["Satisfaccion_NPS"].mean()
            fig = plt.figure()
            grp.plot(kind="bar")
            plt.title("NPS promedio: Entrega tardía vs normal")
            plt.xlabel("Entrega_Tardia")
            plt.ylabel("Satisfaccion_NPS promedio")
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Faltan columnas Satisfaccion_NPS/Entrega_Tardia.")

    elif chart == "Riesgo por Categoría (Top 10, controlados)":
        if {"Categoria", "Riesgo_Operacion"}.issubset(controlados.columns):
            vol = controlados["Categoria"].value_counts().head(10).index
            sub = controlados[controlados["Categoria"].isin(vol)].copy()

            risk_rate = (sub.assign(Riesgo_Alto=sub["Riesgo_Operacion"].eq("Alto"))
                           .groupby("Categoria")["Riesgo_Alto"]
                           .mean()
                           .sort_values(ascending=False))
            fig = plt.figure()
            risk_rate.plot(kind="bar")
            plt.title("Tasa de Riesgo Alto por Categoría")
            plt.xlabel("Categoría")
            plt.ylabel("Proporción Riesgo Alto")
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Faltan columnas Categoria/Riesgo_Operacion.")

    elif chart == "SKU fantasma: ciudades Top 10":
        if "SKU_Fantasma" in df.columns and "Ciudad_Destino_norm" in df.columns:
            top_c = df[df["SKU_Fantasma"] == True]["Ciudad_Destino_norm"].value_counts().head(10)
            if len(top_c) > 0:
                fig = plt.figure()
                top_c.plot(kind="bar")
                plt.title("Top 10 ciudades con más ventas SKU fantasma")
                plt.xlabel("Ciudad_Destino_norm")
                plt.ylabel("Número de ventas fantasma")
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("No hay registros fantasma (con los filtros actuales) o no hay ciudad normalizada.")
        else:
            st.info("Se requiere SKU_Fantasma y Ciudad_Destino_norm.")

    elif chart == "Vista previa / describe":
        st.markdown("#### Vista previa")
        st.dataframe(df.head(30), use_container_width=True)
        st.markdown("#### Describe")
        st.dataframe(df.describe(include="all").T, use_container_width=True)
