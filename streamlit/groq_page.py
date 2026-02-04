# groq_page.py
from __future__ import annotations

from typing import Optional, Dict, Any
import pandas as pd
import streamlit as st

try:
    from groq import Groq
except Exception:
    Groq = None


def _safe_rate(series: pd.Series) -> Optional[float]:
    """Convierte a bool y devuelve tasa (%). None si no aplicable."""
    if series is None or len(series) == 0:
        return None
    s = series.copy()
    # aceptar bool, 0/1, strings comunes
    if s.dtype != bool:
        s = s.astype(str).str.strip().str.lower().isin(["true", "1", "si", "sí", "s", "yes"])
    return float(s.mean() * 100)


def _build_stat_payload(df: pd.DataFrame) -> Dict[str, Any]:
    """Crea un payload resumido (sin PII ni datos fila-a-fila)."""
    n_rows, n_cols = df.shape
    dup = int(df.duplicated().sum())
    null_cells = int(df.isna().sum().sum())
    total_cells = int(n_rows * n_cols) if n_rows and n_cols else 0
    null_pct = float((null_cells / total_cells * 100) if total_cells else 0.0)

    dtypes = df.dtypes.astype(str).value_counts().to_dict()
    describe_all = df.describe(include="all").T

    # Para texto muy largo, recorte seguro
    describe_str = describe_all.to_string(max_rows=80, max_cols=30)

    # Top nulos por columna
    null_prof = (df.isna().mean() * 100).sort_values(ascending=False).head(20)
    null_str = null_prof.to_string()

    # KPIs agregados (si existen) - NO son filas, son métricas generales
    kpis: Dict[str, Any] = {}
    if "Ingreso_Total" in df.columns:
        kpis["Ingreso_Total_sum"] = float(pd.to_numeric(df["Ingreso_Total"], errors="coerce").fillna(0).sum())
    if "Margen_Utilidad" in df.columns:
        kpis["Margen_Utilidad_sum"] = float(pd.to_numeric(df["Margen_Utilidad"], errors="coerce").fillna(0).sum())
    if "Margen_%" in df.columns:
        # promedio robusto (ignora NaN)
        kpis["Margen_pct_mean"] = float(pd.to_numeric(df["Margen_%"], errors="coerce").dropna().mean()) if df["Margen_%"].notna().any() else None

    if "SKU_Fantasma" in df.columns and "Ingreso_Total" in df.columns:
        fant = df[df["SKU_Fantasma"] == True]
        ctrl = df[df["SKU_Fantasma"] == False]
        total_ing = float(pd.to_numeric(df["Ingreso_Total"], errors="coerce").fillna(0).sum())
        ing_fant = float(pd.to_numeric(fant["Ingreso_Total"], errors="coerce").fillna(0).sum()) if len(fant) else 0.0
        ing_ctrl = float(pd.to_numeric(ctrl["Ingreso_Total"], errors="coerce").fillna(0).sum()) if len(ctrl) else 0.0
        kpis["Ingreso_Fantasma_sum"] = ing_fant
        kpis["Ingreso_Controlado_sum"] = ing_ctrl
        kpis["Ingreso_Fantasma_pct"] = (ing_fant / total_ing * 100) if total_ing > 0 else None

    if "Entrega_Tardia" in df.columns:
        kpis["Entrega_Tardia_pct"] = _safe_rate(df["Entrega_Tardia"])
    if "Stock_Insuficiente" in df.columns:
        kpis["Stock_Insuficiente_pct"] = _safe_rate(df["Stock_Insuficiente"])
    if "Ticket_Indicador" in df.columns:
        kpis["Tickets_sum"] = int(pd.to_numeric(df["Ticket_Indicador"], errors="coerce").fillna(0).sum())

    return {
        "shape": {"rows": n_rows, "cols": n_cols},
        "duplicates_rows": dup,
        "null_cells": null_cells,
        "null_pct": round(null_pct, 2),
        "dtypes_counts": dtypes,
        "describe": describe_str,
        "top_nulls_pct": null_str,
        "columns": df.columns.tolist(),
        "kpis": kpis,
    }


def _groq_recommendations(api_key: str, payload: Dict[str, Any], model: str) -> str:
    """Llama Groq Chat Completions y devuelve recomendaciones."""
    if Groq is None:
        raise RuntimeError("No está instalado el paquete 'groq'. Haga pip install groq")

    client = Groq(api_key=api_key)

    system = (
        "Usted es un consultor senior de analítica y operación. "
        "Debe generar recomendaciones estratégicas accionables basadas en un resumen estadístico agregado. "
        "No invente datos. No solicite datos adicionales. "
        "Salida: mínimo 3 párrafos (pueden ser más), cada párrafo con 4-6 líneas. "
        "Enfoque: calidad de datos, control operacional, rentabilidad/logística y próximos pasos. "
        "Si hay KPIs agregados, interprételos y conviértalos en acciones concretas.\n\n"
        "Además, al final debe incluir una sección titulada exactamente: 'Plan de acción propuesto', "
        "con 4 a 6 viñetas. Cada viñeta debe tener: (Prioridad P0/P1/P2) + (Horizonte 0-30, 31-60 o 61-90 días) "
        "+ una acción concreta + el resultado esperado (en una línea)."
    )


    user = (
        "Analice el siguiente RESUMEN ESTADÍSTICO (sin datos fila-a-fila) de un dataset filtrado por el usuario "
        "y genere recomendaciones estratégicas en tiempo real. "
        "Incluya señales de alerta si detecta: alta nulidad, duplicados, sesgos por faltantes, o variables críticas incompletas. "
        "Al final incluya la sección 'Plan de acción propuesto' según el formato solicitado.\n\n"
        f"SHAPE: {payload['shape']}\n"
        f"DUPLICATES_ROWS: {payload['duplicates_rows']}\n"
        f"NULL_CELLS: {payload['null_cells']} ({payload['null_pct']}%)\n"
        f"DTYPES_COUNTS: {payload['dtypes_counts']}\n\n"
        "KPIS (agregados, si existen):\n"
        f"{payload.get('kpis', {})}\n\n"
        "TOP NULLS (%):\n"
        f"{payload['top_nulls_pct']}\n\n"
        "DESCRIBE (include=all):\n"
        f"{payload['describe']}\n"
    )

    resp = client.chat.completions.create(
        model=model,
        temperature=0.25,
        max_tokens=1500,  # un poco más para permitir 3+ párrafos
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    return resp.choices[0].message.content


def render_groq_assistant(df_filtered: Optional[pd.DataFrame]):
    st.subheader("🤖 Asistente de análisis (Groq)")

    st.caption(
        "Este asistente analiza un resumen estadístico agregado del dataset filtrado por usted "
        "(no se envían filas crudas) y genera recomendaciones estratégicas en tiempo real."
    )

    if df_filtered is None or len(df_filtered) == 0:
        st.info("No hay dataset consolidado/filtrado disponible desde el EDA.")
        st.caption("Vaya a la pestaña EDA, aplique filtros y luego vuelva aquí.")
        return

    api_key = st.text_input(
        "Ingrese su Groq API Key",
        type="password",
        help="Se usa solo durante la sesión."
    )

    model = st.selectbox(
        "Modelo Llama (Groq)",
        options=[
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
        ],
        index=0,
        help="70B suele dar mejor análisis; 8B es más rápido."
    )

    st.divider()

    run = st.button("Generar recomendaciones", use_container_width=True, type="primary")

    if run:
        if not api_key or len(api_key.strip()) < 10:
            st.error("Ingrese una API key válida (no vacía).")
            return

        payload = _build_stat_payload(df_filtered)

        with st.spinner("Consultando Groq..."):
            try:
                text = _groq_recommendations(api_key=api_key.strip(), payload=payload, model=model)
            except Exception as e:
                st.error(f"Error llamando Groq: {e}")
                return

        st.markdown("### Recomendación estratégica")
        st.write(text)

        st.session_state["groq_last_response"] = text
        st.session_state["groq_last_model"] = model
