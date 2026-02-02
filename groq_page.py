# groq_page.py
from __future__ import annotations

from typing import Optional, Dict, Any
import pandas as pd
import streamlit as st

try:
    from groq import Groq
except Exception:
    Groq = None


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
    null_profile = (df.isna().mean() * 100).sort_values(ascending=False).head(20)
    null_str = null_profile.to_string()

    return {
        "shape": {"rows": n_rows, "cols": n_cols},
        "duplicates_rows": dup,
        "null_cells": null_cells,
        "null_pct": round(null_pct, 2),
        "dtypes_counts": dtypes,
        "describe": describe_str,
        "top_nulls_pct": null_str,
        "columns": df.columns.tolist(),
    }


def _groq_recommendations(api_key: str, payload: Dict[str, Any], model: str) -> str:
    """Llama Groq Chat Completions y devuelve 3 párrafos."""
    if Groq is None:
        raise RuntimeError("No está instalado el paquete 'groq'. Haga pip install groq")

    client = Groq(api_key=api_key)

    system = (
        "Usted es un consultor senior de analítica y operación. "
        "Debe generar recomendaciones estratégicas accionables basadas en un resumen estadístico. "
        "No invente datos. No pida datos adicionales. "
        "Entregue exactamente 3 párrafos, cada uno con 4-6 líneas. "
        "Enfoque: calidad de datos, control operacional, rentabilidad/logística, y próximos pasos."
    )

    user = (
        "Analice el siguiente RESUMEN ESTADÍSTICO (sin datos fila-a-fila) de un dataset filtrado por el usuario. "
        "Genere 3 párrafos de recomendación estratégica. "
        "Incluya señales de alerta si detecta: alta nulidad, duplicados, sesgos por faltantes, o variables críticas incompletas.\n\n"
        f"SHAPE: {payload['shape']}\n"
        f"DUPLICATES_ROWS: {payload['duplicates_rows']}\n"
        f"NULL_CELLS: {payload['null_cells']} ({payload['null_pct']}%)\n"
        f"DTYPES_COUNTS: {payload['dtypes_counts']}\n\n"
        "TOP NULLS (%):\n"
        f"{payload['top_nulls_pct']}\n\n"
        "DESCRIBE (include=all):\n"
        f"{payload['describe']}\n"
    )

    resp = client.chat.completions.create(
        model=model,
        temperature=0.25,
        max_tokens=600,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    return resp.choices[0].message.content


def render_groq_assistant(df_filtered: Optional[pd.DataFrame]):
    st.subheader("🤖 Asistente de análisis (Groq)")

    st.caption(
        "Este asistente analiza **solo el resumen estadístico** del dataset filtrado (describe + métricas agregadas). "
        "No se envían filas crudas."
    )

    if df_filtered is None or len(df_filtered) == 0:
        st.info("No hay dataset consolidado/filtrado disponible desde el EDA.")
        st.caption("Vaya a la pestaña EDA, aplique filtros y luego vuelva aquí.")
        return

    # --- UI: API KEY ---
    api_key = st.text_input(
        "Ingrese su Groq API Key",
        type="password",
        help="Se usa solo durante la sesión. Recomendado: manejarla como secreto/variable de entorno en producción."
    )

    # Modelo (según docs Groq; puede cambiar con el tiempo)
    model = st.selectbox(
        "Modelo Llama (Groq)",
        options=[
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "llama3-70b-8192",
        ],
        index=0,
        help="Seleccione el modelo. 70B suele dar mejor análisis; 8B es más rápido."
    )

    with st.expander("Ver resumen que se enviará al modelo (solo estadísticos)", expanded=False):
        payload = _build_stat_payload(df_filtered)
        st.json({k: payload[k] for k in ["shape", "duplicates_rows", "null_cells", "null_pct", "dtypes_counts"]})
        st.text("TOP NULLS (%):\n" + payload["top_nulls_pct"])
        st.text("DESCRIBE:\n" + payload["describe"])

    st.divider()

    colA, colB = st.columns([1, 1])
    with colA:
        st.write("**Dataset filtrado actual:**")
        st.write(f"- Filas: **{len(df_filtered):,}**")
        st.write(f"- Columnas: **{df_filtered.shape[1]:,}**")

    with colB:
        st.write("**Salida esperada:**")
        st.write("- 3 párrafos")
        st.write("- Recomendaciones accionables")
        st.write("- Alertas si aplica")

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

        st.markdown("### Recomendación estratégica (3 párrafos)")
        st.write(text)

        # Guardar en session_state por conveniencia
        st.session_state["groq_last_response"] = text
        st.session_state["groq_last_model"] = model
