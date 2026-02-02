from eda_page import render_eda
import io
import json
import re
import unicodedata
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import ftfy

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.ensemble import RandomForestClassifier


# =========================
# UI CONFIG
# =========================
st.set_page_config(
    page_title="Auditoría de Calidad y Transparencia",
    page_icon="🧾",
    layout="wide",
)

st.title("🧾 Auditoría de Calidad y Transparencia")
st.caption(
    "Carga hasta 3 datasets (Inventario, Transacciones, Feedback). "
    "La app calcula Health Score antes/después, métricas de calidad, "
    "bitácora ética (eliminación vs imputación) y genera CSV corregido descargable."
)

# =========================
# HELPERS: AUDIT + HEALTH SCORE
# =========================
def audit_nulls_types(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "nulos_%": (df.isna().mean() * 100).round(2),
        "tipo": df.dtypes.astype(str)
    }).sort_values("nulos_%", ascending=False)

def duplicate_count(df: pd.DataFrame) -> int:
    return int(df.duplicated().sum())

def numeric_outlier_report(df: pd.DataFrame, iqr_factor: float = 1.5) -> Dict[str, Any]:
    """
    Detecta outliers por IQR en columnas numéricas.
    Retorna: conteo por columna, porcentaje, y magnitud (máx distancia en 'IQR units').
    """
    num_cols = df.select_dtypes(include="number").columns.tolist()
    out = {
        "iqr_factor": iqr_factor,
        "columns": {},
        "total_outlier_cells": 0,
        "total_numeric_cells": int(df[num_cols].size) if num_cols else 0,
        "outlier_cell_%": 0.0,
    }
    if not num_cols:
        return out

    total_outliers = 0
    total_cells = df[num_cols].size

    for col in num_cols:
        s = df[col].dropna()
        if s.empty:
            out["columns"][col] = {"outliers": 0, "outliers_%_over_rows": 0.0, "max_iqr_units": 0.0}
            continue

        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            mask = pd.Series(False, index=df.index)
            max_units = 0.0
        else:
            lower, upper = q1 - iqr_factor * iqr, q3 + iqr_factor * iqr
            mask = (df[col] < lower) | (df[col] > upper)

            dist = pd.Series(0.0, index=df.index)
            dist.loc[df[col] < lower] = (lower - df.loc[df[col] < lower, col]).abs() / iqr
            dist.loc[df[col] > upper] = (df.loc[df[col] > upper, col] - upper).abs() / iqr
            max_units = float(dist.max()) if not dist.empty else 0.0

        c = int(mask.sum())
        total_outliers += c
        out["columns"][col] = {
            "outliers": c,
            "outliers_%_over_rows": round((c / len(df) * 100) if len(df) else 0.0, 2),
            "max_iqr_units": round(max_units, 3)
        }

    out["total_outlier_cells"] = int(total_outliers)
    out["outlier_cell_%"] = round((total_outliers / total_cells * 100) if total_cells else 0.0, 2)
    return out

def health_score(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Health Score simple y explicable (0-100):
    - penaliza nulidad promedio
    - penaliza % duplicados
    - penaliza % outlier_cells (en numéricas)
    """
    if len(df) == 0:
        return {"score": 0.0, "components": {"null_avg_%": 100.0, "dup_%": 100.0, "outlier_cell_%": 100.0}}

    null_avg = float(df.isna().mean().mean() * 100)
    dup_pct = float(df.duplicated().mean() * 100)

    out_rep = numeric_outlier_report(df)
    outlier_cell_pct = float(out_rep.get("outlier_cell_%", 0.0))

    w_null, w_dup, w_out = 0.55, 0.20, 0.25
    penalty = w_null * null_avg + w_dup * dup_pct + w_out * outlier_cell_pct
    score = max(0.0, 100.0 - penalty)

    return {
        "score": round(score, 2),
        "components": {
            "null_avg_%": round(null_avg, 2),
            "dup_%": round(dup_pct, 2),
            "outlier_cell_%": round(outlier_cell_pct, 2),
        }
    }

def justify_imputation_strategy(series: pd.Series) -> str:
    """
    Justificación automática:
    - Si es numérica: usa mediana si |skew|>1, si no media.
    - Si es categórica: moda.
    """
    if series.dropna().empty:
        return "Sin datos suficientes para estimar distribución (columna vacía tras NaN)."

    if pd.api.types.is_numeric_dtype(series):
        s = pd.to_numeric(series.dropna(), errors="coerce").dropna()
        skew = float(s.skew()) if len(s) >= 3 else 0.0
        if abs(skew) > 1:
            return f"Se usa **mediana** por sesgo alto (skew={skew:.2f}), robusta ante asimetría/outliers."
        return f"Se usa **media** por distribución relativamente simétrica (skew={skew:.2f})."
    else:
        return "Se usa **moda** (variable categórica), preserva la categoría más frecuente."

def df_to_download_bytes(df: pd.DataFrame, encoding="utf-8") -> bytes:
    return df.to_csv(index=False, encoding=encoding).encode(encoding)


# =========================
# CLEANING LOG / AUDIT STRUCTURES
# =========================
@dataclass
class EthicalLog:
    dataset_name: str
    dropped_rows: int = 0
    dropped_rows_reason: str = ""
    dropped_row_indices_sample: List[Any] = None
    duplicates_removed: int = 0
    imputations: List[Dict[str, Any]] = None
    outlier_handling: List[Dict[str, Any]] = None
    notes: List[str] = None

    def __post_init__(self):
        self.dropped_row_indices_sample = self.dropped_row_indices_sample or []
        self.imputations = self.imputations or []
        self.outlier_handling = self.outlier_handling or []
        self.notes = self.notes or []

def drop_rows_with_many_nulls(df: pd.DataFrame, k: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    null_count = df.isna().sum(axis=1)
    mask_drop = null_count >= k

    cols_nulas = (
        df[mask_drop]
        .isna()
        .apply(lambda row: list(df.columns[row]), axis=1)
    )

    df_eliminados = df[mask_drop].copy()
    df_eliminados["Columnas_Nulas"] = cols_nulas
    df_eliminados["N_Nulos"] = null_count[mask_drop]

    df_limpio = df[~mask_drop].copy()
    return df_limpio, df_eliminados


# =========================
# INVENTARIO CLEANING
# =========================
def parse_lead_time(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in {"inmediato", "inmediate", "immediate"}:
        return 1.0
    m = re.search(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", s)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return max(a, b)
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if m:
        return float(m.group(1))
    return np.nan

def parse_categoria(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    s = re.sub(r"[_\-\/]+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    if s == "":
        return np.nan
    if "smart" in s and "phone" in s:
        return "Smartphones"
    if s in {"laptop", "laptops"}:
        return "Laptops"
    return s.title()

def clean_inventario(inv: pd.DataFrame) -> Tuple[pd.DataFrame, EthicalLog]:
    log = EthicalLog(dataset_name="Inventario")
    df = inv.copy()

    if "Lead_Time_Dias" in df.columns:
        df["Lead_Time_Dias"] = df["Lead_Time_Dias"].apply(parse_lead_time)
        log.notes.append("Normalización Lead_Time_Dias: 'Inmediato'->1, rangos -> máximo del rango, strings -> primer número.")
    if "Categoria" in df.columns:
        df["Categoria"] = df["Categoria"].apply(parse_categoria)
        log.notes.append("Normalización Categoria: limpieza de separadores, caracteres y canonicalización (Smartphones/Laptops).")

    # Eliminar filas con >=2 nulos
    df2, df_elim = drop_rows_with_many_nulls(df, k=2)
    log.dropped_rows = int(len(df_elim))
    log.dropped_rows_reason = "Eliminación de registros con >=2 nulos (imputación extensiva aumenta incertidumbre)."
    # FIX: no forzar astype(int)
    log.dropped_row_indices_sample = df_elim.index[:20].tolist()

    # Duplicados
    dup_before = duplicate_count(df2)
    if dup_before > 0:
        df2 = df2.drop_duplicates().copy()
        log.duplicates_removed = int(dup_before)

    # Imputación Lead_Time_Dias (mediana)
    if "Lead_Time_Dias" in df2.columns:
        n_before = int(df2["Lead_Time_Dias"].isna().sum())
        if n_before > 0:
            med = float(df2["Lead_Time_Dias"].median())
            df2["Lead_Time_Dias"] = df2["Lead_Time_Dias"].fillna(med)
            log.imputations.append({
                "column": "Lead_Time_Dias",
                "strategy": "median",
                "value_used": med,
                "n_imputed": n_before,
                "ethical_justification": justify_imputation_strategy(inv["Lead_Time_Dias"] if "Lead_Time_Dias" in inv.columns else df2["Lead_Time_Dias"])
            })

    # Normalizar Bodega_Origen
    if "Bodega_Origen" in df2.columns:
        df2["Bodega_Origen"] = df2["Bodega_Origen"].replace({"norte": "Norte", "ZONA_FRANCA": "Zona Franca"})
        df2["Bodega_Origen"] = df2["Bodega_Origen"].replace({"BOD-EXT-99": "Externa"})
        log.notes.append("Normalización Bodega_Origen: casos norte/ZONA_FRANCA/BOD-EXT-99->Externa.")

    # Imputación Stock_Actual por mediana dentro de Categoria
    if "Stock_Actual" in df2.columns and "Categoria" in df2.columns:
        n_before = int(df2["Stock_Actual"].isna().sum())
        if n_before > 0:
            med_by_cat = df2.groupby("Categoria")["Stock_Actual"].median()

            def _imp(row):
                if pd.isna(row["Stock_Actual"]):
                    return med_by_cat.get(row["Categoria"], np.nan)
                return row["Stock_Actual"]

            df2["Stock_Actual"] = df2.apply(_imp, axis=1)
            log.imputations.append({
                "column": "Stock_Actual",
                "strategy": "median_by_category",
                "value_used": "median per Categoria",
                "n_imputed": n_before,
                "ethical_justification": "Se imputa por **mediana por categoría** para preservar escala operativa por familia de producto y evitar sesgos de un valor global."
            })

    # Imputación Categoria por perfil de stock por bodega
    if "Categoria" in df2.columns and "Bodega_Origen" in df2.columns and "Stock_Actual" in df2.columns:
        n_before = int(df2["Categoria"].isna().sum())
        if n_before > 0:
            perfil = (df2.dropna(subset=["Categoria"])
                      .groupby(["Bodega_Origen", "Categoria"])
                      .agg(stock_mediana=("Stock_Actual", "median"), n=("Categoria", "count"))
                      .reset_index())

            def inferir_categoria_por_stock(row):
                zona = row["Bodega_Origen"]
                stock = row["Stock_Actual"]
                sub = perfil[perfil["Bodega_Origen"] == zona]
                if len(sub) == 0:
                    sub = perfil
                dist = (sub["stock_mediana"] - stock).abs()
                return sub.loc[dist.idxmin(), "Categoria"]

            df2.loc[df2["Categoria"].isna(), "Categoria"] = (
                df2[df2["Categoria"].isna()].apply(inferir_categoria_por_stock, axis=1)
            )

            log.imputations.append({
                "column": "Categoria",
                "strategy": "profile_match_by_stock_and_warehouse",
                "value_used": "closest median stock within Bodega_Origen",
                "n_imputed": n_before,
                "ethical_justification": (
                    "Se infiere categoría por **perfil de inventario (mediana de Stock_Actual) dentro de la misma bodega**; "
                    "preserva lógica logística y evita asignaciones arbitrarias."
                )
            })

    # Correcciones puntuales Costo_Unitario_USD
    if "Costo_Unitario_USD" in df2.columns:
        rep_before = numeric_outlier_report(df2[["Costo_Unitario_USD"]].copy())

        n_fix_005 = int((df2["Costo_Unitario_USD"] == 0.05).sum())
        if n_fix_005 > 0:
            df2.loc[df2["Costo_Unitario_USD"] == 0.05, "Costo_Unitario_USD"] = 500
            log.outlier_handling.append({
                "column": "Costo_Unitario_USD",
                "rule": "0.05 -> 500",
                "n_affected": n_fix_005,
                "ethical_justification": "0.05 USD para smartphone es implausible (error de captura/formato)."
            })

        n_fix_850k = int((df2["Costo_Unitario_USD"] == 850000).sum())
        if n_fix_850k > 0:
            tasa_cop_usd = 3668.80
            df2.loc[df2["Costo_Unitario_USD"] == 850000, "Costo_Unitario_USD"] = 850000 / tasa_cop_usd
            log.outlier_handling.append({
                "column": "Costo_Unitario_USD",
                "rule": "850000 (asumido COP) -> USD usando tasa 3668.80",
                "n_affected": n_fix_850k,
                "ethical_justification": "850,000 USD es extremo; consistente con valor en COP; se convierte a USD con tasa definida."
            })

        log.notes.append(f"Outliers Costo_Unitario_USD (pre): outlier_cell_%={rep_before.get('outlier_cell_%', 0)}")

    return df2, log


# =========================
# TRANSACCIONES CLEANING
# =========================
def normalize_text_full(s):
    if pd.isna(s):
        return s
    s = ftfy.fix_text(str(s))
    s = s.lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_fuzzy_map(values: pd.Series, threshold=0.9):
    values = sorted(set(values.dropna().astype(str)))
    canonical, mapping = [], {}
    import difflib
    for v in values:
        match = difflib.get_close_matches(v, canonical, n=1, cutoff=threshold)
        if match:
            mapping[v] = match[0]
        else:
            canonical.append(v)
            mapping[v] = v
    return mapping

def clean_numeric_outliers_winsorize(df: pd.DataFrame, numeric_cols: List[str], iqr_factor=1.5) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df_clean = df.copy()
    report = {"iqr_factor": iqr_factor, "columns": {}}
    for col in numeric_cols:
        series = df_clean[col]
        s = series.dropna()
        if s.empty:
            report["columns"][col] = {"clipped": 0}
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            report["columns"][col] = {"clipped": 0}
            continue
        lower, upper = q1 - iqr_factor * iqr, q3 + iqr_factor * iqr
        before = series.copy()
        df_clean[col] = series.clip(lower, upper)
        clipped = int((before != df_clean[col]).sum())
        df_clean[col] = df_clean[col].fillna(s.median())
        report["columns"][col] = {"clipped": clipped, "lower": float(lower), "upper": float(upper)}
    return df_clean, report

def clean_transacciones(trx: pd.DataFrame) -> Tuple[pd.DataFrame, EthicalLog]:
    log = EthicalLog(dataset_name="Transacciones")
    df = trx.copy()

    if "Cantidad_Vendida" in df.columns:
        n_neg5 = int((df["Cantidad_Vendida"] == -5).sum())
        df["Cantidad_Vendida"] = df["Cantidad_Vendida"].replace(-5, 5)
        if n_neg5:
            log.outlier_handling.append({
                "column": "Cantidad_Vendida", "rule": "-5 -> 5", "n_affected": n_neg5,
                "ethical_justification": "Corrección de codificación; cantidad negativa específica mapeada a su magnitud."
            })

    if "Tiempo_Entrega_Real" in df.columns:
        n_999 = int((df["Tiempo_Entrega_Real"] == 999).sum())
        df["Tiempo_Entrega_Real"] = df["Tiempo_Entrega_Real"].replace(999, np.nan)
        if n_999:
            log.outlier_handling.append({
                "column": "Tiempo_Entrega_Real", "rule": "999 -> NaN", "n_affected": n_999,
                "ethical_justification": "999 usado como centinela; se convierte a faltante para imputación."
            })

    if "Fecha_Venta" in df.columns:
        TODAY = pd.Timestamp("2026-01-31")
        df["Fecha_Venta"] = pd.to_datetime(df["Fecha_Venta"], errors="coerce")
        n_future = int((df["Fecha_Venta"] > TODAY).sum())
        df.loc[df["Fecha_Venta"] > TODAY, "Fecha_Venta"] = pd.NaT
        if n_future:
            log.outlier_handling.append({
                "column": "Fecha_Venta", "rule": f"> {TODAY.date()} -> NaT", "n_affected": n_future,
                "ethical_justification": "Fechas futuras son inconsistentes con registro histórico; se anulan para consistencia."
            })

    EMPTY_VALUES = ["", " ", "nan", "NaN", "null", "NULL", "none", "None", "?", "-", "--"]
    df = df.replace(EMPTY_VALUES, np.nan).replace(r"^\s*$", np.nan, regex=True)
    log.notes.append("Estandarización de vacíos: ['', 'nan', 'null', '?', '--', espacios] -> NaN.")

    city_aliases = {"med": "medellin", "mde": "medellin", "medell": "medellin",
                    "bog": "bogota", "bta": "bogota", "bgta": "bogota"}
    for col in ["Ciudad_Destino", "Canal_Venta"]:
        if col in df.columns:
            df[f"{col}_norm"] = df[col].apply(normalize_text_full)

    if "Ciudad_Destino_norm" in df.columns:
        df["Ciudad_Destino_norm"] = df["Ciudad_Destino_norm"].replace(city_aliases)
        city_map = build_fuzzy_map(df["Ciudad_Destino_norm"], threshold=0.9)
        df["Ciudad_Destino_norm"] = df["Ciudad_Destino_norm"].map(city_map)
        log.notes.append("Normalización Ciudad_Destino: aliases + fuzzy matching (threshold=0.9).")

    # Eliminar filas con >=2 nulos
    before_idx = df.index
    before_len = len(df)
    df = df[df.isna().sum(axis=1) < 2].copy()
    dropped = before_len - len(df)
    if dropped > 0:
        dropped_idx = before_idx.difference(df.index)[:20].tolist()
        log.dropped_rows = int(dropped)
        log.dropped_rows_reason = "Eliminación de registros con >=2 nulos (imputación extensiva aumenta incertidumbre)."
        # FIX: no astype(int)
        log.dropped_row_indices_sample = dropped_idx

    dup_before = duplicate_count(df)
    if dup_before > 0:
        df = df.drop_duplicates().copy()
        log.duplicates_removed = int(dup_before)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        n_missing_num = int(df[numeric_cols].isna().sum().sum())
        if n_missing_num > 0:
            it_imputer = IterativeImputer(max_iter=10, random_state=42)
            df[numeric_cols] = it_imputer.fit_transform(df[numeric_cols])
            log.imputations.append({
                "column": "NUMERIC (multiple)",
                "strategy": "IterativeImputer",
                "value_used": "model-based iterative",
                "n_imputed": n_missing_num,
                "ethical_justification": "Imputación multivariante para preservar correlaciones entre variables numéricas y reducir sesgo."
            })

        df, clip_report = clean_numeric_outliers_winsorize(df, numeric_cols, iqr_factor=1.5)
        log.outlier_handling.append({
            "column": "NUMERIC (multiple)",
            "rule": "winsorize (IQR clipping) + fill median",
            "n_affected": int(sum(v.get("clipped", 0) for v in clip_report["columns"].values())),
            "ethical_justification": "Se limita influencia de extremos preservando registros y reduciendo distorsión."
        })

    # RandomForest para Estado_Envio
    col_obj = "Estado_Envio"
    cols_base = ["Cantidad_Vendida", "Precio_Venta_Final", "Costo_Envio", "Tiempo_Entrega_Real"]
    if col_obj in df.columns and all(c in df.columns for c in cols_base):
        if df[col_obj].isna().any():
            imp_simple = SimpleImputer(strategy="median")
            X_val = pd.DataFrame(imp_simple.fit_transform(df[cols_base]), index=df.index, columns=cols_base)

            train_idx = df[df[col_obj].notna()].index
            predict_idx = df[df[col_obj].isna()].index

            if len(train_idx) >= 10 and len(predict_idx) > 0:
                rf_model = RandomForestClassifier(n_estimators=150, random_state=42)
                rf_model.fit(X_val.loc[train_idx], df.loc[train_idx, col_obj].astype(str))
                df.loc[predict_idx, col_obj] = rf_model.predict(X_val.loc[predict_idx])

                log.imputations.append({
                    "column": col_obj,
                    "strategy": "RandomForestClassifier",
                    "value_used": "predicted class",
                    "n_imputed": int(len(predict_idx)),
                    "ethical_justification": "Se infiere categoría con modelo supervisado usando señales logísticas numéricas."
                })
            else:
                mode_val = df[col_obj].mode(dropna=True)
                fill = mode_val.iloc[0] if len(mode_val) else "Desconocido"
                n_pred = int(df[col_obj].isna().sum())
                df[col_obj] = df[col_obj].fillna(fill)
                log.imputations.append({
                    "column": col_obj,
                    "strategy": "mode_fallback",
                    "value_used": str(fill),
                    "n_imputed": n_pred,
                    "ethical_justification": "Datos insuficientes para entrenar modelo; se usa moda."
                })

    return df, log


# =========================
# FEEDBACK CLEANING
# =========================
def clean_feedback(fb: pd.DataFrame) -> Tuple[pd.DataFrame, EthicalLog]:
    log = EthicalLog(dataset_name="Feedback")
    df = fb.copy()

    # Ticket_Soporte_Abierto: 1/0 -> Sí/No
    if "Ticket_Soporte_Abierto" in df.columns:
        before = df["Ticket_Soporte_Abierto"].copy()
        df["Ticket_Soporte_Abierto"] = (
            df["Ticket_Soporte_Abierto"]
            .astype(str).str.strip()
            .replace({"1": "Sí", "0": "No", "Si": "Sí", "Sí": "Sí", "No": "No"})
        )
        changed = int((before.astype(str) != df["Ticket_Soporte_Abierto"].astype(str)).sum())
        if changed:
            log.outlier_handling.append({
                "column": "Ticket_Soporte_Abierto",
                "rule": "binario -> Sí/No",
                "n_affected": changed,
                "ethical_justification": "Estandarización semántica para análisis consistente."
            })

    # Edad_Cliente > 90 -> eliminar (FIX: índices desde df, antes del filtro)
    if "Edad_Cliente" in df.columns:
        mask = df["Edad_Cliente"] > 90
        dropped = int(mask.sum())
        if dropped:
            idx_drop = df.index[mask][:20].tolist()  # <- FIX
            df = df[~mask].copy()
            log.dropped_rows += dropped
            log.dropped_rows_reason += (" | " if log.dropped_rows_reason else "") + \
                "Eliminación de Edad_Cliente>90 (fuera de rango definido)."
            log.dropped_row_indices_sample.extend(idx_drop)

    # Recomienda_Marca
    if "Recomienda_Marca" in df.columns:
        df["Recomienda_Marca"] = (
            df["Recomienda_Marca"]
            .astype(str).str.strip()
            .replace({"SI": "Sí", "NO": "No", "Maybe": "Tal vez", "nan": "Sin respuesta"})
        )
        log.notes.append("Normalización Recomienda_Marca: SI/NO/Maybe y faltantes -> Sin respuesta.")

    # Satisfaccion_NPS norm
    if "Satisfaccion_NPS" in df.columns:
        df["Satisfaccion_NPS_norm"] = (df["Satisfaccion_NPS"] / 100).round(2)
        log.notes.append("Satisfaccion_NPS_norm = Satisfaccion_NPS / 100 (redondeo 2 decimales).")

    # Comentario_Texto: '---' y NaN -> Sin comentarios
    if "Comentario_Texto" in df.columns:
        base = fb["Comentario_Texto"] if "Comentario_Texto" in fb.columns else df["Comentario_Texto"]
        n_imp = int((base.isna() | (base == "---")).sum())
        df["Comentario_Texto"] = df["Comentario_Texto"].replace("---", pd.NA).fillna("Sin comentarios")
        log.imputations.append({
            "column": "Comentario_Texto",
            "strategy": "fillna_constant",
            "value_used": "Sin comentarios",
            "n_imputed": n_imp,
            "ethical_justification": "Se preserva registro sin inventar contenido; constante neutral cuando no hay comentario."
        })

    # Rating_Producto fuera de 1..5 -> eliminar (FIX: índices desde df, antes del filtro)
    if "Rating_Producto" in df.columns:
        mask = ~df["Rating_Producto"].between(1, 5)
        dropped = int(mask.sum())
        if dropped:
            idx_drop = df.index[mask][:20].tolist()  # <- FIX
            df = df[~mask].copy()
            log.dropped_rows += dropped
            log.dropped_rows_reason += (" | " if log.dropped_rows_reason else "") + \
                "Eliminación Rating_Producto fuera de [1,5] (subjetivo; imputar puede sesgar)."
            log.dropped_row_indices_sample.extend(idx_drop)

    dup_before = duplicate_count(df)
    if dup_before > 0:
        log.notes.append(
            f"Duplicados detectados (total filas duplicadas exactas): {dup_before}. "
            "No se elimina automáticamente si corresponde a granularidad legítima (ej. múltiples productos por feedback)."
        )

    if "Feedback_ID" in df.columns and "Transaccion_ID" in df.columns:
        df["Cantidad_Productos_Feedback"] = df.groupby("Feedback_ID")["Transaccion_ID"].transform("nunique")
        log.notes.append("Se agrega Cantidad_Productos_Feedback = nunique(Transaccion_ID) por Feedback_ID.")

    return df, log


# =========================
# DATASET ROUTING
# =========================
def load_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)

def process_dataset(df: pd.DataFrame, dtype: str) -> Tuple[pd.DataFrame, EthicalLog]:
    if dtype == "Inventario":
        return clean_inventario(df)
    if dtype == "Transacciones":
        return clean_transacciones(df)
    if dtype == "Feedback":
        return clean_feedback(df)

    log = EthicalLog(dataset_name="Genérico")
    df2, elim = drop_rows_with_many_nulls(df, k=2)
    log.dropped_rows = int(len(elim))
    log.dropped_rows_reason = "Fallback: eliminación de filas con >=2 nulos."
    log.dropped_row_indices_sample = elim.index[:20].tolist()

    dup_before = duplicate_count(df2)
    if dup_before:
        df2 = df2.drop_duplicates().copy()
        log.duplicates_removed = int(dup_before)

    for col in df2.columns:
        n_before = int(df2[col].isna().sum())
        if n_before == 0:
            continue

        if pd.api.types.is_numeric_dtype(df2[col]):
            val = float(pd.to_numeric(df2[col], errors="coerce").median())
            df2[col] = df2[col].fillna(val)
            strat = "median"
        else:
            mode = df2[col].mode(dropna=True)
            val = mode.iloc[0] if len(mode) else "Sin dato"
            df2[col] = df2[col].fillna(val)
            strat = "mode"

        log.imputations.append({
            "column": col,
            "strategy": strat,
            "value_used": str(val),
            "n_imputed": n_before,
            "ethical_justification": justify_imputation_strategy(df[col])
        })

    return df2, log


# =========================
# UI: FILE UPLOAD
# =========================
with st.sidebar:
    st.header("📤 Carga de archivos")
    st.write("Suba 1, 2 o 3 archivos CSV. Si sube uno, solo se procesa ese dataset.")
    inv_file = st.file_uploader("Inventario (CSV)", type=["csv"], key="inv")
    trx_file = st.file_uploader("Transacciones (CSV)", type=["csv"], key="trx")
    fb_file  = st.file_uploader("Feedback (CSV)", type=["csv"], key="fb")
    st.divider()
    st.subheader("Parámetros")
    iqr_factor = st.slider("Factor IQR para outliers (auditoría)", 1.0, 3.0, 1.5, 0.1)
    st.caption("El factor IQR afecta el conteo de outliers reportados.")

uploaded = [(inv_file, "Inventario"), (trx_file, "Transacciones"), (fb_file, "Feedback")]
uploaded = [(f, t) for (f, t) in uploaded if f is not None]

# =========================
# MAIN TABS: AUDITORÍA / EDA
# =========================
tab_audit, tab_eda = st.tabs(["🧾 Auditoría de Calidad y Transparencia", "📈 EDA"])

with tab_eda:
    st.caption("EDA interactivo basado en datasets cargados. Por defecto se usa la versión limpia (recomendada).")

    fuente = st.radio(
        "Fuente para EDA",
        options=["Limpios (recomendado)", "Crudos"],
        index=0,
        horizontal=True
    )

    source_map = clean_dfs if fuente.startswith("Limpios") else raw_dfs

    inv_df = source_map.get("Inventario")
    trx_df = source_map.get("Transacciones")
    fb_df  = source_map.get("Feedback")

    render_eda(inv_df, trx_df, fb_df)

with tab_audit:
    if not uploaded:
        st.info("Suba al menos un archivo para comenzar.")
        st.stop()

    st.markdown("## 📊 Resultados por dataset")

    sub_tabs = st.tabs([f"📁 {dtype}" for _, dtype in uploaded])

    all_reports = {}
    cleaned_files = {}

    for (up, dtype), sub_tab in zip(uploaded, sub_tabs):
        with sub_tab:
            st.subheader(dtype)
            st.caption(f"Archivo: {up.name}")

            try:
                df_raw = load_csv(up)
            except Exception as e:
                st.error(f"No se pudo leer el CSV: {e}")
                continue

            # Pre metrics
            hs_before = health_score(df_raw)
            nulls_before = audit_nulls_types(df_raw)
            dup_before = duplicate_count(df_raw)
            out_before = numeric_outlier_report(df_raw, iqr_factor=iqr_factor)

            # Clean
            df_clean, ethic_log = process_dataset(df_raw, dtype)

            # Post metrics
            hs_after = health_score(df_clean)
            nulls_after = audit_nulls_types(df_clean)
            dup_after = duplicate_count(df_clean)
            out_after = numeric_outlier_report(df_clean, iqr_factor=iqr_factor)

            report = {
                "dataset": dtype,
                "file_name": up.name,
                "shape_before": list(df_raw.shape),
                "shape_after": list(df_clean.shape),
                "health_before": hs_before,
                "health_after": hs_after,
                "duplicates_before": dup_before,
                "duplicates_after": dup_after,
                "outliers_before": out_before,
                "outliers_after": out_after,
                "ethical_log": asdict(ethic_log),
                "nulls_before_top": nulls_before.head(15).to_dict(orient="index"),
                "nulls_after_top": nulls_after.head(15).to_dict(orient="index"),
            }
            all_reports[dtype] = report

            cleaned_csv_bytes = df_to_download_bytes(df_clean, encoding="utf-8")
            cleaned_files[dtype] = cleaned_csv_bytes

            # --- UI LAYOUT ---
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Health Score (antes)", hs_before["score"])
            with c2:
                st.metric("Health Score (después)", hs_after["score"])
            with c3:
                st.metric("Duplicados (antes)", dup_before)
            with c4:
                st.metric("Duplicados (después)", dup_after)

            st.divider()

            colA, colB = st.columns([1, 1])
            with colA:
                st.markdown("### Nulidad por columna (antes)")
                st.dataframe(nulls_before, use_container_width=True, height=360)
            with colB:
                st.markdown("### Nulidad por columna (después)")
                st.dataframe(nulls_after, use_container_width=True, height=360)

            st.divider()

            colC, colD = st.columns([1, 1])
            with colC:
                st.markdown("### Outliers detectados (antes)")
                st.write(f"Outlier cells % (numéricas): **{out_before.get('outlier_cell_%', 0)}%**")
                st.dataframe(
                    pd.DataFrame(out_before.get("columns", {})).T.sort_values("outliers", ascending=False),
                    use_container_width=True,
                    height=280
                )
            with colD:
                st.markdown("### Outliers detectados (después)")
                st.write(f"Outlier cells % (numéricas): **{out_after.get('outlier_cell_%', 0)}%**")
                st.dataframe(
                    pd.DataFrame(out_after.get("columns", {})).T.sort_values("outliers", ascending=False),
                    use_container_width=True,
                    height=280
                )

            st.divider()

            st.markdown("### 🧭 Decisión ética y trazabilidad")
            st.write("**Filas eliminadas**:", ethic_log.dropped_rows)
            if ethic_log.dropped_rows_reason:
                st.info(ethic_log.dropped_rows_reason)
            if ethic_log.dropped_row_indices_sample:
                st.caption(f"Muestra de índices eliminados (hasta 20): {ethic_log.dropped_row_indices_sample}")

            st.write("**Duplicados eliminados**:", ethic_log.duplicates_removed)

            if ethic_log.imputations:
                st.markdown("#### Imputaciones realizadas (qué y por qué)")
                st.dataframe(pd.DataFrame(ethic_log.imputations), use_container_width=True)
            else:
                st.markdown("#### Imputaciones realizadas")
                st.caption("No se realizaron imputaciones en este dataset (o no aplicaron reglas).")

            if ethic_log.outlier_handling:
                st.markdown("#### Manejo/corrección de outliers (reglas aplicadas)")
                st.dataframe(pd.DataFrame(ethic_log.outlier_handling), use_container_width=True)
            else:
                st.markdown("#### Manejo/corrección de outliers")
                st.caption("No se aplicaron reglas específicas de corrección de outliers.")

            if ethic_log.notes:
                st.markdown("#### Notas")
                for n in ethic_log.notes:
                    st.write(f"- {n}")

            st.divider()

            st.markdown("### ⬇️ Descargar dataset corregido")
            out_name = up.name.replace(".csv", "") + "_limpio.csv"
            st.download_button(
                label=f"Descargar {out_name}",
                data=cleaned_csv_bytes,
                file_name=out_name,
                mime="text/csv",
                use_container_width=True
            )

            st.markdown("### Vista previa (antes / después)")
            p1, p2 = st.columns(2)
            with p1:
                st.caption("Antes")
                st.dataframe(df_raw.head(20), use_container_width=True, height=280)
            with p2:
                st.caption("Después")
                st.dataframe(df_clean.head(20), use_container_width=True, height=280)

    st.divider()

