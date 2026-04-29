import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import re
from html import escape
from io import BytesIO
from pathlib import Path
from zipfile import BadZipFile, ZipFile
import xml.etree.ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_PATH = str(PROJECT_ROOT / "backend")

if BACKEND_PATH not in sys.path:
    sys.path.append(BACKEND_PATH)

from api.ml.ml_models import compare_models, compare_history_models, forecast_period_returns
from api.ml.research_context import RESEARCH_CONTEXT
from api.ml.sentiment import sentiment_breakdown
from api.sentiment.pipeline import get_news_sentiment

def _resolve_optional_path(secret_name, default_path):
    configured_path = st.secrets.get(secret_name) or os.getenv(secret_name)
    if not configured_path:
        return default_path

    candidate = Path(configured_path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


CUSTOM_ASSET_WORKBOOK = _resolve_optional_path(
    "CUSTOM_ASSET_WORKBOOK",
    PROJECT_ROOT / "data" / "custom_assets.xlsx",
)
CUSTOM_SUMMARY_WORKBOOK = _resolve_optional_path(
    "CUSTOM_SUMMARY_WORKBOOK",
    PROJECT_ROOT / "data" / "custom_summary.xlsx",
)
EXCEL_NS = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024
MAX_WORKBOOK_UNCOMPRESSED_BYTES = 40 * 1024 * 1024
MAX_WORKBOOK_ENTRIES = 2000
SAFE_MARKET_SYMBOL = re.compile(r"^[A-Z0-9.\-_=^]{1,20}$")
SAFE_FRED_SERIES = re.compile(r"^[A-Z0-9_]{1,32}$")
DATA_SOURCE_HISTORICAL = "Dissertation Models"
DATA_SOURCE_MARKET_TOOLS = "Market Tools"


def _sanitize_market_symbol(symbol):
    cleaned_symbol = clean_symbol(symbol).upper()
    if not SAFE_MARKET_SYMBOL.fullmatch(cleaned_symbol):
        raise ValueError(f"Unsupported symbol format: {symbol}")
    return cleaned_symbol


def _validate_uploaded_workbook(uploaded_bytes, label):
    if not uploaded_bytes:
        raise ValueError(f"{label} is empty.")
    if len(uploaded_bytes) > MAX_UPLOAD_SIZE_BYTES:
        raise ValueError(f"{label} exceeds the 10 MB upload limit.")


def _validate_workbook_archive(workbook):
    members = workbook.infolist()
    if len(members) > MAX_WORKBOOK_ENTRIES:
        raise ValueError("Workbook contains too many internal files.")

    total_size = sum(member.file_size for member in members)
    if total_size > MAX_WORKBOOK_UNCOMPRESSED_BYTES:
        raise ValueError("Workbook expands to an unsafe size.")


def _safe_api_get(url, *, params=None, timeout=8):
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response


def _excel_column_to_index(cell_ref):
    letters = "".join(ch for ch in str(cell_ref) if ch.isalpha()).upper()
    total = 0
    for char in letters:
        total = total * 26 + (ord(char) - ord("A") + 1)
    return max(total - 1, 0)


def _excel_serial_to_datetime(value):
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return pd.to_datetime(value, errors="coerce")

    if numeric_value > 1000:
        return pd.Timestamp("1899-12-30") + pd.to_timedelta(numeric_value, unit="D")

    return pd.to_datetime(value, errors="coerce")


def _parse_number(value):
    if value is None:
        return np.nan

    cleaned = str(value).replace(",", "").strip()
    if cleaned in {"", "-", "—", "nan", "None"}:
        return np.nan

    try:
        return float(cleaned)
    except ValueError:
        return np.nan


def _normalize_header(value):
    return " ".join(str(value).strip().lower().replace(".", " ").split())


def _looks_like_asset_workbook(sheets):
    for rows in sheets.values():
        header_row = _find_header_row(rows)
        if header_row is not None:
            return True
    return False


def _looks_like_summary_workbook(sheets):
    for rows in sheets.values():
        for row in rows[:10]:
            normalized = {_normalize_header(value) for value in row if str(value).strip()}
            if {"company", "year", "actual return"}.issubset(normalized):
                return True
    return False


def _discover_workbook_path(workbook_kind):
    data_dir = PROJECT_ROOT / "data"
    if not data_dir.exists():
        return None

    for candidate in sorted(data_dir.glob("*.xlsx")):
        try:
            sheets = _read_xlsx_sheet_rows(candidate)
        except ValueError:
            continue

        if workbook_kind == "asset" and _looks_like_asset_workbook(sheets):
            return candidate
        if workbook_kind == "summary" and _looks_like_summary_workbook(sheets):
            return candidate

    return None


def _get_workbook_source(session_key, configured_path, workbook_kind):
    uploaded_bytes = st.session_state.get(session_key)
    if uploaded_bytes:
        return uploaded_bytes

    if configured_path.exists():
        return configured_path

    discovered_path = _discover_workbook_path(workbook_kind)
    if discovered_path is not None:
        return discovered_path

    return configured_path


def _read_xlsx_sheet_rows(workbook_source):
    if workbook_source is None:
        return {}

    workbook_bytes = None
    workbook_path = None

    if isinstance(workbook_source, (str, Path)):
        workbook_path = Path(workbook_source)
        if not workbook_path.exists():
            return {}
    elif hasattr(workbook_source, "getvalue"):
        workbook_bytes = workbook_source.getvalue()
    elif isinstance(workbook_source, (bytes, bytearray)):
        workbook_bytes = bytes(workbook_source)
    else:
        return {}

    if workbook_bytes is not None and not workbook_bytes:
        return {}

    if workbook_bytes is not None:
        _validate_uploaded_workbook(workbook_bytes, "Workbook upload")

    shared_strings = []
    sheets = {}

    try:
        workbook_handle = ZipFile(BytesIO(workbook_bytes)) if workbook_bytes is not None else ZipFile(workbook_path)
    except BadZipFile:
        raise ValueError("Workbook is not a valid .xlsx file.")

    with workbook_handle as workbook:
        _validate_workbook_archive(workbook)
        if "xl/sharedStrings.xml" in workbook.namelist():
            shared_root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
            for shared_item in shared_root.findall("main:si", EXCEL_NS):
                text = "".join(node.text or "" for node in shared_item.iterfind(".//main:t", EXCEL_NS))
                shared_strings.append(text)

        workbook_root = ET.fromstring(workbook.read("xl/workbook.xml"))
        rels_root = ET.fromstring(workbook.read("xl/_rels/workbook.xml.rels"))
        rel_namespace = "{http://schemas.openxmlformats.org/package/2006/relationships}"
        sheet_namespace = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"

        rel_map = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rels_root.findall(f"{rel_namespace}Relationship")
        }

        for sheet in workbook_root.find("main:sheets", EXCEL_NS):
            sheet_name = sheet.attrib["name"]
            target = rel_map.get(sheet.attrib[sheet_namespace])
            if not target:
                continue

            target_path = target if target.startswith("xl/") else f"xl/{target}"
            sheet_root = ET.fromstring(workbook.read(target_path))
            sheet_rows = []

            for row in sheet_root.findall(".//main:sheetData/main:row", EXCEL_NS):
                values = {}
                for cell in row.findall("main:c", EXCEL_NS):
                    ref = cell.attrib.get("r", "")
                    index = _excel_column_to_index(ref)
                    cell_type = cell.attrib.get("t")
                    value_node = cell.find("main:v", EXCEL_NS)

                    if cell_type == "s" and value_node is not None:
                        shared_index = int(value_node.text)
                        value = shared_strings[shared_index] if shared_index < len(shared_strings) else ""
                    elif cell_type == "inlineStr":
                        inline = cell.find("main:is", EXCEL_NS)
                        value = "".join(node.text or "" for node in inline.iterfind(".//main:t", EXCEL_NS)) if inline is not None else ""
                    else:
                        value = value_node.text if value_node is not None else ""

                    values[index] = value

                if not values:
                    sheet_rows.append([])
                    continue

                width = max(values) + 1
                sheet_rows.append([values.get(i, "") for i in range(width)])

            sheets[sheet_name] = sheet_rows

    return sheets


def _find_header_row(rows):
    for index, row in enumerate(rows[:10]):
        normalized = {_normalize_header(value) for value in row if str(value).strip()}
        if "date" in normalized and ("close" in normalized or "close price" in normalized):
            return index
    return None


def _match_column(columns, aliases):
    normalized_map = {_normalize_header(name): name for name in columns}
    for alias in aliases:
        if alias in normalized_map:
            return normalized_map[alias]
    return None


def _normalize_custom_asset_sheet(rows):
    header_row = _find_header_row(rows)
    if header_row is None:
        return pd.DataFrame()

    header = rows[header_row]
    width = len(header)
    records = []
    for row in rows[header_row + 1:]:
        padded = row + [""] * (width - len(row))
        if any(str(value).strip() for value in padded):
            records.append(padded[:width])

    raw_df = pd.DataFrame(records, columns=header)
    raw_df.columns = [str(col).strip() for col in raw_df.columns]

    date_col = _match_column(raw_df.columns, ["date"])
    open_col = _match_column(raw_df.columns, ["open", "open price"])
    high_col = _match_column(raw_df.columns, ["high", "high price"])
    low_col = _match_column(raw_df.columns, ["low", "low price"])
    close_col = _match_column(raw_df.columns, ["close", "close price"])
    volume_col = _match_column(raw_df.columns, ["volume", "total traded quantity"])
    return_col = _match_column(raw_df.columns, ["return"])

    required = [date_col, open_col, high_col, low_col, close_col]
    if any(col is None for col in required):
        return pd.DataFrame()

    normalized = pd.DataFrame(
        {
            "Date": raw_df[date_col].map(_excel_serial_to_datetime),
            "Open": raw_df[open_col].map(_parse_number),
            "High": raw_df[high_col].map(_parse_number),
            "Low": raw_df[low_col].map(_parse_number),
            "Close": raw_df[close_col].map(_parse_number),
            "Volume": raw_df[volume_col].map(_parse_number) if volume_col else 0,
        }
    )

    if return_col:
        normalized["Returns"] = raw_df[return_col].map(_parse_number)

    normalized = normalized.dropna(subset=["Date", "Open", "High", "Low", "Close"])
    normalized = normalized.sort_values("Date")
    normalized["Volume"] = normalized["Volume"].fillna(0)
    normalized = normalized.set_index("Date")
    return normalized


def get_custom_asset_data_map():
    workbook_source = _get_workbook_source("custom_asset_workbook_bytes", CUSTOM_ASSET_WORKBOOK, "asset")
    try:
        sheets = _read_xlsx_sheet_rows(workbook_source)
    except ValueError as exc:
        st.error(f"Asset workbook error: {exc}")
        return {}
    asset_map = {}

    for sheet_name, rows in sheets.items():
        df = _normalize_custom_asset_sheet(rows)
        if not df.empty:
            asset_map[sheet_name] = df

    return asset_map


def load_custom_asset_summary():
    workbook_source = _get_workbook_source("custom_summary_workbook_bytes", CUSTOM_SUMMARY_WORKBOOK, "summary")
    try:
        sheets = _read_xlsx_sheet_rows(workbook_source)
    except ValueError as exc:
        st.error(f"Summary workbook error: {exc}")
        return pd.DataFrame()
    if not sheets:
        return pd.DataFrame()

    first_sheet = next(iter(sheets.values()))
    if not first_sheet:
        return pd.DataFrame()

    header_row = None
    for index, row in enumerate(first_sheet[:10]):
        normalized = [_normalize_header(value) for value in row]
        if "company" in normalized and "year" in normalized and "actual return" in normalized:
            header_row = index
            break

    if header_row is None:
        return pd.DataFrame()

    header = [str(value).strip() for value in first_sheet[header_row]]
    width = len(header)
    records = []
    current_company = ""

    for row in first_sheet[header_row + 1:]:
        padded = row + [""] * (width - len(row))
        if not any(str(value).strip() for value in padded):
            continue

        row_company = str(padded[0]).strip()
        if row_company:
            current_company = row_company
        padded[0] = current_company
        records.append(padded[:width])

    summary_df = pd.DataFrame(records, columns=header)
    if "Company" in summary_df.columns:
        summary_df["Company"] = summary_df["Company"].astype(str).str.strip().str.upper()

    return summary_df.replace("", np.nan)


def _parse_financial_year_range(year_label):
    match = re.match(r"^\s*(\d{4})\s*-\s*(\d{4})\s*$", str(year_label))
    if not match:
        return None, None

    start_year = int(match.group(1))
    end_year = int(match.group(2))
    return pd.Timestamp(start_year, 4, 1), pd.Timestamp(end_year, 3, 31)


def _filter_data_by_financial_year(df, year_label):
    if df is None or df.empty or not year_label or year_label == "All Years":
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()

    start_date, end_date = _parse_financial_year_range(year_label)
    if start_date is None:
        return df.copy()

    filtered = df.loc[(df.index >= start_date) & (df.index <= end_date)].copy()
    return filtered


def _get_financial_year_options(asset_key, asset_df):
    summary_df = load_custom_asset_summary()
    company_key = str(asset_key).strip().upper()
    years = []

    if not summary_df.empty and {"Company", "Year"}.issubset(summary_df.columns):
        years = (
            summary_df.loc[summary_df["Company"] == company_key, "Year"]
            .dropna()
            .astype(str)
            .str.strip()
            .tolist()
        )

    if not years and asset_df is not None and not asset_df.empty:
        start_year = int(asset_df.index.min().year)
        end_year = int(asset_df.index.max().year)
        for year in range(start_year, end_year + 1):
            fy_label = f"{year}-{year + 1}"
            period_df = _filter_data_by_financial_year(asset_df, fy_label)
            if not period_df.empty:
                years.append(fy_label)

    ordered_years = []
    seen = set()
    for year_label in years:
        if year_label not in seen:
            ordered_years.append(year_label)
            seen.add(year_label)

    return ["All Years"] + ordered_years if ordered_years else ["All Years"]


def _prepare_summary_display(df):
    if df is None or df.empty:
        return pd.DataFrame()

    display_df = df.copy()
    display_df = display_df.loc[
        :,
        [
            column_name
            for column_name in display_df.columns
            if str(column_name).strip() and not display_df[column_name].isna().all()
        ],
    ]

    placeholder_values = {"": np.nan, "-": np.nan, "—": np.nan, "nan": np.nan, "None": np.nan}
    numeric_keywords = (
        "forecast",
        "return",
        "score",
        "price",
        "pct",
        "percent",
        "probability",
        "value",
        "amount",
        "ratio",
        "volatility",
    )

    for column_name in display_df.columns:
        column = display_df[column_name]

        if pd.api.types.is_numeric_dtype(column):
            continue

        normalized = column.replace(placeholder_values)
        numeric_candidate = pd.to_numeric(normalized, errors="coerce")
        normalized_name = str(column_name).strip().lower()
        should_cast_numeric = any(keyword in normalized_name for keyword in numeric_keywords)

        if (
            should_cast_numeric
            and normalized.notna().sum() > 0
            and numeric_candidate.notna().sum() == normalized.notna().sum()
        ):
            display_df[column_name] = numeric_candidate
        else:
            display_df[column_name] = normalized.fillna("—")

    return display_df


def render_three_market_scene(asset_name, year_label, trend_label):
    asset_name = escape(str(asset_name))
    year_label = escape(str(year_label))
    trend_label = escape(str(trend_label))

    components.html(
        f"""
        <div id="r3f-root" style="width:100%;height:500px;border-radius:26px;overflow:hidden;position:relative;background:
        radial-gradient(circle at 18% 18%, rgba(16,255,163,.18), transparent 28%),
        radial-gradient(circle at 82% 16%, rgba(52,211,153,.12), transparent 24%),
        linear-gradient(135deg, #020706 0%, #03110d 52%, #061917 100%);">
          <canvas id="market-canvas" style="position:absolute;inset:0;width:100%;height:100%;z-index:0;display:block;"></canvas>
          <div style="position:absolute;inset:0;z-index:1;background:
            linear-gradient(180deg, rgba(2,12,10,0.06) 0%, rgba(2,12,10,0.32) 100%),
            radial-gradient(circle at top left, rgba(16, 255, 163, 0.08), transparent 32%);"></div>
          <div id="hero-overlay" style="position:absolute;left:22px;right:22px;top:18px;z-index:3;"></div>
        </div>

        <script>
          const overlayNode = document.getElementById("hero-overlay");
          const canvas = document.getElementById("market-canvas");
          const ctx = canvas.getContext("2d");
          const pulses = [
            "Scanning live motion field",
            "Balancing collision particles",
            "Refreshing dashboard energy"
          ];
          let pulseIndex = 0;

          overlayNode.innerHTML = `
            <div style="color:#dcfce7;font-family:'Space Grotesk',Arial,sans-serif;">
              <div style="font-size:30px;font-weight:700;margin-top:8px;line-height:1;text-shadow:0 0 18px rgba(16,255,163,.24);">{asset_name}</div>
              <div style="font-size:13px;opacity:.88;margin-top:8px;color:rgba(187,247,208,.92);letter-spacing:.08em;text-transform:uppercase;">{year_label} • {trend_label}</div>
              <div id="pulse-chip" style="margin-top:12px;display:inline-flex;align-items:center;gap:8px;border-radius:999px;padding:7px 11px;background:rgba(3,18,14,.58);border:1px solid rgba(52,211,153,.22);font-size:12px;color:rgba(220,252,231,.92);">
                <span style="width:8px;height:8px;border-radius:999px;background:#22c55e;box-shadow:0 0 12px rgba(34,197,94,.75);"></span>
                <span id="pulse-label">${{pulses[0]}}</span>
              </div>
            </div>
          `;

          const pulseLabel = document.getElementById("pulse-label");
          window.setInterval(() => {{
            pulseIndex = (pulseIndex + 1) % pulses.length;
            if (pulseLabel) pulseLabel.textContent = pulses[pulseIndex];
          }}, 1800);

          const state = {{
            bars: Array.from({{ length: 14 }}, (_, i) => ({{
              x: 80 + i * 34,
              width: 18,
              base: 0.8 + (i % 5) * 0.12,
              phase: i * 0.45,
              color: i % 2 === 0 ? "#46d7ff" : "#ff7ac6"
            }})),
            orbs: Array.from({{ length: 7 }}, (_, i) => ({{
              x: 180 + i * 44,
              y: 120 + (i % 3) * 28,
              z: 0.5 + (i % 4) * 0.18,
              vx: (Math.random() - 0.5) * 1.7,
              vy: 0.8 + Math.random() * 1.1,
              radius: 10 + (i % 3) * 4,
              color: ["#7dd3fc", "#f9a8d4", "#fde68a", "#93c5fd", "#86efac", "#c4b5fd", "#67e8f9"][i]
            }})),
            particles: Array.from({{ length: 48 }}, () => ({{
              angle: Math.random() * Math.PI * 2,
              radius: 120 + Math.random() * 180,
              speed: 0.0015 + Math.random() * 0.0025,
              y: -40 + Math.random() * 220,
              size: 1 + Math.random() * 2.5
            }}))
          }};

          function resize() {{
            const dpr = Math.min(window.devicePixelRatio || 1, 2);
            const rect = canvas.getBoundingClientRect();
            canvas.width = Math.max(1, Math.floor(rect.width * dpr));
            canvas.height = Math.max(1, Math.floor(rect.height * dpr));
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
          }}

          function animate(now) {{
            const t = now * 0.001;
            const width = canvas.getBoundingClientRect().width;
            const height = canvas.getBoundingClientRect().height;
            const floorY = height * 0.78;
            const centerX = width * 0.67;
            const centerY = height * 0.5;

            ctx.clearRect(0, 0, width, height);

            const bg = ctx.createLinearGradient(0, 0, 0, height);
            bg.addColorStop(0, "rgba(0, 18, 12, 0.24)");
            bg.addColorStop(1, "rgba(0, 8, 6, 0.04)");
            ctx.fillStyle = bg;
            ctx.fillRect(0, 0, width, height);

            const haze = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, width * 0.38);
            haze.addColorStop(0, "rgba(16, 255, 163, 0.12)");
            haze.addColorStop(1, "rgba(0,0,0,0)");
            ctx.fillStyle = haze;
            ctx.fillRect(0, 0, width, height);

            for (const particle of state.particles) {{
              particle.angle += particle.speed * 18;
              const px = centerX + Math.cos(particle.angle) * particle.radius;
              const py = centerY + Math.sin(particle.angle * 1.15) * 52 + particle.y;
              ctx.fillStyle = "rgba(110, 231, 183, 0.38)";
              ctx.beginPath();
              ctx.arc(px, py % height, particle.size, 0, Math.PI * 2);
              ctx.fill();
            }}

            ctx.save();
            ctx.translate(centerX, centerY);
            ctx.rotate(0.95 + Math.sin(t * 0.55) * 0.06);
            ctx.shadowBlur = 20;
            ctx.shadowColor = "rgba(16,255,163,.35)";
            ctx.strokeStyle = "rgba(16, 255, 163, 0.62)";
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.ellipse(0, 0, 126, 62, 0, 0, Math.PI * 2);
            ctx.stroke();
            ctx.shadowBlur = 0;
            ctx.strokeStyle = "rgba(187, 247, 208, 0.22)";
            ctx.lineWidth = 1.2;
            ctx.beginPath();
            ctx.ellipse(0, 0, 150, 74, 0, 0, Math.PI * 2);
            ctx.stroke();
            ctx.restore();

            for (const bar of state.bars) {{
              const barHeight = 60 + Math.sin(t * 1.8 + bar.phase) * 34 + bar.base * 48;
              const x = width * 0.5 + (bar.x - 290);
              const y = floorY - barHeight;
              const neonBar = ctx.createLinearGradient(x, y, x, floorY);
              neonBar.addColorStop(0, "rgba(110, 231, 183, 0.98)");
              neonBar.addColorStop(1, "rgba(16, 185, 129, 0.62)");
              ctx.fillStyle = neonBar;
              ctx.fillRect(x, y, bar.width, barHeight);
              ctx.fillStyle = "rgba(220,252,231,0.24)";
              ctx.fillRect(x, y, bar.width, 6);
              ctx.fillStyle = "rgba(0, 24, 17, 0.46)";
              ctx.beginPath();
              ctx.moveTo(x + bar.width, y);
              ctx.lineTo(x + bar.width + 10, y - 8);
              ctx.lineTo(x + bar.width + 10, floorY - 8);
              ctx.lineTo(x + bar.width, floorY);
              ctx.closePath();
              ctx.fill();
            }}

            ctx.strokeStyle = "rgba(16, 255, 163, 0.16)";
            ctx.lineWidth = 1;
            for (let i = 0; i < 9; i += 1) {{
              const gy = floorY - i * 24;
              ctx.beginPath();
              ctx.moveTo(width * 0.4, gy);
              ctx.lineTo(width * 0.9, gy);
              ctx.stroke();
            }}

            for (const orb of state.orbs) {{
              orb.vy += 0.055;
              orb.x += orb.vx;
              orb.y += orb.vy;

              if (orb.y + orb.radius > floorY) {{
                orb.y = floorY - orb.radius;
                orb.vy *= -0.92;
              }}
              if (orb.x - orb.radius < width * 0.42 || orb.x + orb.radius > width * 0.92) {{
                orb.vx *= -1;
              }}

              const glow = ctx.createRadialGradient(orb.x, orb.y, 0, orb.x, orb.y, orb.radius * 2.7);
              glow.addColorStop(0, "rgba(16,255,163,0.9)");
              glow.addColorStop(1, "rgba(0,0,0,0)");
              ctx.fillStyle = glow;
              ctx.beginPath();
              ctx.arc(orb.x, orb.y, orb.radius * 2.7, 0, Math.PI * 2);
              ctx.fill();

              ctx.fillStyle = "rgba(110, 231, 183, 0.96)";
              ctx.beginPath();
              ctx.arc(orb.x, orb.y, orb.radius, 0, Math.PI * 2);
              ctx.fill();

              ctx.fillStyle = "rgba(220,252,231,0.32)";
              ctx.beginPath();
              ctx.arc(orb.x - orb.radius * 0.25, orb.y - orb.radius * 0.25, orb.radius * 0.3, 0, Math.PI * 2);
              ctx.fill();
            }}

            const horizon = ctx.createLinearGradient(0, floorY - 18, 0, floorY + 42);
            horizon.addColorStop(0, "rgba(16, 255, 163, 0.10)");
            horizon.addColorStop(1, "rgba(0, 14, 10, 0)");
            ctx.fillStyle = horizon;
            ctx.fillRect(width * 0.36, floorY - 18, width * 0.6, 60);

            window.requestAnimationFrame(animate);
          }}

          window.addEventListener("resize", resize);
          resize();
          window.requestAnimationFrame(animate);
        </script>
        """,
        height=500,
    )


def render_terminal_background():
    components.html(
        """
        <div id="terminal-bg-root" style="position:fixed;inset:0;z-index:0;pointer-events:none;overflow:hidden;"></div>
        <script type="module">
          import * as THREE from "https://esm.sh/three@0.169.0";

          const mount = document.getElementById("terminal-bg-root");
          const scene = new THREE.Scene();
          scene.fog = new THREE.FogExp2("#03110d", 0.045);

          const camera = new THREE.PerspectiveCamera(48, window.innerWidth / window.innerHeight, 0.1, 100);
          camera.position.set(0, 0.45, 8.8);

          const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
          renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.8));
          renderer.setSize(window.innerWidth, window.innerHeight);
          renderer.outputColorSpace = THREE.SRGBColorSpace;
          mount.appendChild(renderer.domElement);

          const ambient = new THREE.AmbientLight("#a7f3d0", 0.88);
          const keyLight = new THREE.DirectionalLight("#34d399", 1.55);
          keyLight.position.set(5, 6, 7);
          const rimLight = new THREE.DirectionalLight("#22c55e", 1.3);
          rimLight.position.set(-4, 2, 5);
          scene.add(ambient, keyLight, rimLight);

          const uniforms = {
            uTime: { value: 0 },
            uResolution: { value: new THREE.Vector2(window.innerWidth, window.innerHeight) }
          };

          const backgroundPlane = new THREE.Mesh(
            new THREE.PlaneGeometry(24, 14, 1, 1),
            new THREE.ShaderMaterial({
              uniforms,
              transparent: true,
              depthWrite: false,
              vertexShader: `
                varying vec2 vUv;
                void main() {
                  vUv = uv;
                  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
              `,
              fragmentShader: `
                precision highp float;
                varying vec2 vUv;
                uniform float uTime;
                uniform vec2 uResolution;

                float hash(vec2 p) {
                  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
                }

                float noise(vec2 p) {
                  vec2 i = floor(p);
                  vec2 f = fract(p);
                  vec2 u = f * f * (3.0 - 2.0 * f);
                  return mix(
                    mix(hash(i + vec2(0.0, 0.0)), hash(i + vec2(1.0, 0.0)), u.x),
                    mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x),
                    u.y
                  );
                }

                void main() {
                  vec2 uv = vUv;
                  vec2 centered = uv - 0.5;
                  centered.x *= uResolution.x / max(uResolution.y, 1.0);

                  float t = uTime * 0.08;
                  float waveA = sin(centered.x * 8.0 + t * 8.0) * 0.08;
                  float waveB = sin(centered.y * 10.0 - t * 6.5) * 0.06;
                  float field = noise(centered * 5.5 + vec2(t * 1.7, -t * 1.2));
                  float grid = abs(sin((centered.x + waveB) * 20.0)) * 0.06 + abs(sin((centered.y + waveA) * 24.0)) * 0.045;
                  float energy = smoothstep(0.24, 0.95, field + grid);

                  vec3 base = vec3(0.01, 0.05, 0.04);
                  vec3 neonA = vec3(0.06, 1.0, 0.68);
                  vec3 neonB = vec3(0.20, 0.96, 0.48);
                  vec3 neonC = vec3(0.50, 1.0, 0.78);
                  vec3 glow = mix(neonA, neonB, smoothstep(-0.45, 0.55, centered.x + waveA));
                  float radial = 1.0 - smoothstep(0.15, 1.1, length(centered) * 1.15);
                  vec3 color = base + glow * energy * 0.16 + neonC * radial * 0.12;
                  color += vec3(0.62, 1.0, 0.84) * pow(max(0.0, 1.0 - length(centered * vec2(1.0, 1.4))), 3.0) * 0.08;
                  float alpha = 0.74 + radial * 0.14;
                  gl_FragColor = vec4(color, alpha);
                }
              `
            })
          );
          backgroundPlane.position.set(0, 0.15, -6.5);
          scene.add(backgroundPlane);

          const orb = new THREE.Mesh(
            new THREE.IcosahedronGeometry(1.42, 24),
            new THREE.ShaderMaterial({
              uniforms,
              transparent: true,
              wireframe: false,
              vertexShader: `
                uniform float uTime;
                varying vec3 vNormal;
                varying vec3 vPosition;
                void main() {
                  vNormal = normalize(normalMatrix * normal);
                  vec3 transformed = position + normal * sin(position.y * 4.2 + uTime * 1.6) * 0.08;
                  vec4 worldPos = modelMatrix * vec4(transformed, 1.0);
                  vPosition = worldPos.xyz;
                  gl_Position = projectionMatrix * viewMatrix * worldPos;
                }
              `,
              fragmentShader: `
                precision highp float;
                uniform float uTime;
                varying vec3 vNormal;
                varying vec3 vPosition;
                void main() {
                  float fresnel = pow(1.0 - abs(dot(normalize(vNormal), vec3(0.0, 0.0, 1.0))), 2.4);
                  float pulse = 0.55 + 0.45 * sin(uTime * 1.5 + vPosition.y * 1.3);
                  vec3 neonA = vec3(0.10, 1.0, 0.66);
                  vec3 neonB = vec3(0.32, 0.96, 0.58);
                  vec3 mint = vec3(0.72, 1.0, 0.86);
                  vec3 color = mix(neonA, neonB, smoothstep(-1.1, 1.1, vPosition.y * 0.7 + sin(uTime * 0.9)));
                  color += fresnel * mint;
                  gl_FragColor = vec4(color * (0.72 + pulse * 0.35), 0.92);
                }
              `
            })
          );
          orb.position.set(0.45, 0.8, -0.65);
          scene.add(orb);

          const haloA = new THREE.Mesh(
            new THREE.TorusGeometry(3.45, 0.055, 32, 220),
            new THREE.MeshPhysicalMaterial({
              color: "#34d399",
              emissive: "#10b981",
              emissiveIntensity: 0.9,
              transparent: true,
              opacity: 0.82,
              roughness: 0.12,
              metalness: 0.92
            })
          );
          haloA.rotation.set(1.05, 0.15, 0.28);
          haloA.position.set(0.35, 0.55, -0.55);
          scene.add(haloA);

          const haloB = new THREE.Mesh(
            new THREE.TorusGeometry(2.4, 0.03, 24, 180),
            new THREE.MeshPhysicalMaterial({
              color: "#86efac",
              emissive: "#22c55e",
              emissiveIntensity: 1.0,
              transparent: true,
              opacity: 0.78,
              roughness: 0.16,
              metalness: 0.94
            })
          );
          haloB.rotation.set(1.12, 0.3, 1.08);
          haloB.position.set(0.35, 0.55, -0.55);
          scene.add(haloB);

          const floorMaterial = new THREE.ShaderMaterial({
            uniforms,
            transparent: true,
            side: THREE.DoubleSide,
            vertexShader: `
              varying vec2 vUv;
              uniform float uTime;
              void main() {
                vUv = uv;
                vec3 transformed = position;
                transformed.z += sin(position.x * 0.65 + uTime * 0.65) * 0.08;
                transformed.z += cos(position.y * 0.7 - uTime * 0.45) * 0.06;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(transformed, 1.0);
              }
            `,
            fragmentShader: `
              precision highp float;
              varying vec2 vUv;
              uniform float uTime;
              void main() {
                vec2 grid = abs(fract(vUv * vec2(22.0, 16.0) - 0.5) - 0.5) / fwidth(vUv * vec2(22.0, 16.0));
                float line = 1.0 - min(min(grid.x, grid.y), 1.0);
                float fade = smoothstep(1.1, 0.0, vUv.y);
                vec3 color = mix(vec3(0.01, 0.06, 0.04), vec3(0.10, 0.92, 0.60), line * 0.58);
                color += vec3(0.72, 1.0, 0.84) * pow(1.0 - vUv.y, 2.4) * 0.18;
                gl_FragColor = vec4(color, (0.18 + line * 0.22) * fade);
              }
            `
          });

          const floor = new THREE.Mesh(
            new THREE.PlaneGeometry(30, 18, 100, 100),
            floorMaterial
          );
          floor.rotation.x = -Math.PI / 2;
          floor.position.set(0, -2.55, -0.8);
          scene.add(floor);

          const particleCount = 1500;
          const positions = new Float32Array(particleCount * 3);
          const scales = new Float32Array(particleCount);
          for (let i = 0; i < particleCount; i += 1) {
            const i3 = i * 3;
            const radius = 2.4 + Math.random() * 5.8;
            const angle = Math.random() * Math.PI * 2.0;
            positions[i3] = Math.cos(angle) * radius;
            positions[i3 + 1] = (Math.random() - 0.5) * 6.5;
            positions[i3 + 2] = Math.sin(angle) * radius - 1.4;
            scales[i] = 0.45 + Math.random() * 1.3;
          }

          const particlesGeometry = new THREE.BufferGeometry();
          particlesGeometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
          particlesGeometry.setAttribute("aScale", new THREE.BufferAttribute(scales, 1));

          const particles = new THREE.Points(
            particlesGeometry,
            new THREE.ShaderMaterial({
              uniforms,
              transparent: true,
              depthWrite: false,
              blending: THREE.AdditiveBlending,
              vertexShader: `
                attribute float aScale;
                uniform float uTime;
                varying float vMix;
                void main() {
                  vec3 transformed = position;
                  transformed.y += sin(uTime * 0.45 + position.x * 0.8 + position.z * 0.25) * 0.18;
                  transformed.x += cos(uTime * 0.35 + position.y * 0.5) * 0.08;
                  vec4 mvPosition = modelViewMatrix * vec4(transformed, 1.0);
                  gl_Position = projectionMatrix * mvPosition;
                  gl_PointSize = aScale * (18.0 / -mvPosition.z);
                  vMix = 0.5 + 0.5 * sin(uTime * 0.8 + position.x);
                }
              `,
              fragmentShader: `
                precision highp float;
                varying float vMix;
                void main() {
                  vec2 uv = gl_PointCoord - 0.5;
                  float d = length(uv);
                  float alpha = smoothstep(0.48, 0.0, d);
                  vec3 neonA = vec3(0.10, 1.0, 0.68);
                  vec3 neonB = vec3(0.34, 0.94, 0.52);
                  vec3 color = mix(neonA, neonB, vMix);
                  gl_FragColor = vec4(color, alpha * 0.72);
                }
              `
            })
          );
          scene.add(particles);

          const petalCount = 42;
          const petalGeometry = new THREE.PlaneGeometry(0.12, 0.18, 1, 1);
          const petalMaterial = new THREE.MeshBasicMaterial({
            color: "#bbf7d0",
            transparent: true,
            opacity: 0.32,
            side: THREE.DoubleSide,
            depthWrite: false
          });
          const petals = [];
          for (let i = 0; i < petalCount; i += 1) {
            const petal = new THREE.Mesh(petalGeometry, petalMaterial.clone());
            petal.position.set(
              (Math.random() - 0.5) * 13,
              Math.random() * 6 - 1,
              -2.5 - Math.random() * 4
            );
            petal.rotation.set(Math.random() * Math.PI, Math.random() * Math.PI, Math.random() * Math.PI);
            petal.userData = {
              speed: 0.15 + Math.random() * 0.22,
              drift: 0.2 + Math.random() * 0.35,
              spin: 0.003 + Math.random() * 0.01
            };
            petals.push(petal);
            scene.add(petal);
          }

          const bars = [];
          for (let i = 0; i < 18; i += 1) {
            const height = 0.85 + ((i * 17) % 7) * 0.22;
            const geometry = new THREE.BoxGeometry(0.18, height, 0.18);
            const material = new THREE.MeshPhysicalMaterial({
              color: i % 2 === 0 ? "#34d399" : "#86efac",
              emissive: i % 2 === 0 ? "#10b981" : "#22c55e",
              emissiveIntensity: 0.58,
              roughness: 0.28,
              metalness: 0.86,
              transparent: true,
              opacity: 0.84
            });
            const bar = new THREE.Mesh(geometry, material);
            bar.position.set(-4.8 + i * 0.58, -1.8 + Math.sin(i * 0.72) * 0.15, -1.1 + (i % 4) * 0.12);
            bars.push(bar);
            scene.add(bar);
          }

          const clock = new THREE.Clock();
          let rafId = null;

          function animate() {
            const t = clock.getElapsedTime();
            uniforms.uTime.value = t;

            orb.rotation.x = Math.sin(t * 0.42) * 0.28;
            orb.rotation.y += 0.0065;
            haloA.rotation.z += 0.0018;
            haloB.rotation.z -= 0.0026;
            haloA.rotation.x = 1.02 + Math.sin(t * 0.35) * 0.09;
            haloB.rotation.x = 1.08 + Math.cos(t * 0.28) * 0.08;
            particles.rotation.y = t * 0.035;
            backgroundPlane.rotation.z = Math.sin(t * 0.08) * 0.04;

            bars.forEach((bar, index) => {
              bar.scale.y = 0.85 + Math.sin(t * 1.35 + index * 0.48) * 0.34;
            });

            petals.forEach((petal, index) => {
              petal.position.y -= petal.userData.speed * 0.01;
              petal.position.x += Math.sin(t * petal.userData.drift + index) * 0.0028;
              petal.rotation.z += petal.userData.spin;
              petal.rotation.x += petal.userData.spin * 0.55;
              if (petal.position.y < -3.2) {
                petal.position.y = 4.6;
                petal.position.x = (Math.random() - 0.5) * 13;
              }
            });

            camera.position.x = Math.sin(t * 0.12) * 0.35;
            camera.position.y = 0.45 + Math.cos(t * 0.16) * 0.18;
            camera.lookAt(0.2, 0.15, -0.8);

            renderer.render(scene, camera);
            rafId = requestAnimationFrame(animate);
          }

          function onResize() {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.8));
            uniforms.uResolution.value.set(window.innerWidth, window.innerHeight);
          }

          window.addEventListener("resize", onResize);
          animate();

          window.addEventListener("beforeunload", () => {
            if (rafId) cancelAnimationFrame(rafId);
            window.removeEventListener("resize", onResize);
            renderer.dispose();
          });
        </script>
        """,
        height=0,
    )


def build_custom_asset_summary(asset_key):
    summary_df = load_custom_asset_summary()
    asset_df = get_custom_asset_data_map().get(asset_key, pd.DataFrame())
    company_key = str(asset_key).strip().upper()

    if summary_df.empty or "Company" not in summary_df.columns:
        return pd.DataFrame()

    asset_summary = summary_df[summary_df["Company"] == company_key].copy()
    if asset_summary.empty or asset_df.empty or "Year" not in asset_summary.columns:
        return asset_summary

    periods = []
    for year_label in asset_summary["Year"]:
        start_date, end_date = _parse_financial_year_range(year_label)
        if start_date is None:
            continue
        periods.append(
            {
                "label": str(year_label).strip(),
                "start": start_date,
                "end": end_date,
            }
        )

    forecast_map = forecast_period_returns(asset_df, periods)

    if "ARIMA Forecast" not in asset_summary.columns:
        asset_summary["ARIMA Forecast"] = np.nan
    else:
        asset_summary["ARIMA Forecast"] = pd.to_numeric(
            asset_summary["ARIMA Forecast"],
            errors="coerce",
        )

    if "Random Forest Forecast" not in asset_summary.columns:
        asset_summary["Random Forest Forecast"] = np.nan
    else:
        asset_summary["Random Forest Forecast"] = pd.to_numeric(
            asset_summary["Random Forest Forecast"],
            errors="coerce",
        )

    for row_index, row in asset_summary.iterrows():
        year_label = str(row.get("Year", "")).strip()
        forecast_values = forecast_map.get(year_label, {})
        for column_name in ["ARIMA Forecast", "Random Forest Forecast"]:
            value = forecast_values.get(column_name)
            if value is not None and not pd.isna(value):
                asset_summary.at[row_index, column_name] = round(float(value), 4)

    return asset_summary

# ================= CLEAN SYMBOL HELPER =================
def clean_symbol(symbol):
    symbol = str(symbol).strip()

    # ONLY remove junk AFTER X
    if "=X" in symbol:
        return symbol

    return symbol.split("^")[0].strip()

st.set_page_config(
    page_title="Rupatchi Model",
    layout="wide",
    initial_sidebar_state="collapsed"
)

render_terminal_background()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Space+Grotesk:wght@500;700&display=swap');

:root {
    --terminal-bg: rgba(4, 18, 14, 0.54);
    --terminal-border: rgba(52, 211, 153, 0.22);
    --terminal-accent: #22c55e;
    --terminal-cyan: #34d399;
    --terminal-text: #ecfdf5;
    --terminal-dim: #9dd6b6;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background:
        radial-gradient(circle at top right, rgba(16, 255, 163, 0.16), transparent 20%),
        radial-gradient(circle at 18% 16%, rgba(110, 231, 183, 0.11), transparent 16%),
        radial-gradient(circle at 50% 120%, rgba(34, 197, 94, 0.10), transparent 32%),
        linear-gradient(180deg, #020706 0%, #04120e 48%, #030a08 100%);
    color: var(--terminal-text);
    font-family: "IBM Plex Mono", monospace;
}

[data-testid="stAppViewContainer"] > .main {
    position: relative;
    z-index: 1;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(4, 18, 14, 0.96) 0%, rgba(2, 10, 8, 0.92) 100%);
    border-right: 1px solid rgba(52, 211, 153, 0.16);
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] [data-testid="stSidebarHeader"],
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] select,
[data-testid="stSidebar"] [data-baseweb="select"] {
    font-family: "IBM Plex Mono", monospace !important;
}

.stAppHeader {
    background: rgba(8, 6, 18, 0.22);
    backdrop-filter: blur(10px);
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 3rem;
    max-width: 1600px;
}

h1, h2, h3 {
    font-family: "Space Grotesk", sans-serif !important;
    letter-spacing: 0.02em;
}

h1 {
    font-size: 2.35rem !important;
    text-transform: uppercase;
    text-shadow: 0 0 22px rgba(255, 122, 198, 0.16);
    margin-bottom: 0.2rem !important;
}

[data-testid="stMetric"],
[data-testid="stExpander"],
[data-testid="stDataFrame"],
[data-testid="stPlotlyChart"],
[data-testid="stMarkdownContainer"]:has(.terminal-banner) {
    background: var(--terminal-bg);
    border: 1px solid var(--terminal-border);
    box-shadow:
        0 18px 50px rgba(0, 0, 0, 0.28),
        inset 0 1px 0 rgba(255, 255, 255, 0.04);
    backdrop-filter: blur(18px) saturate(145%);
    border-radius: 20px;
}

[data-testid="stMetric"] {
    padding: 0.95rem 1.05rem;
}

[data-testid="stMetricLabel"] {
    color: var(--terminal-dim);
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

[data-testid="stMetricValue"] {
    color: var(--terminal-text);
    font-family: "Space Grotesk", sans-serif !important;
}

.stSelectbox label,
.stCaption,
.stMarkdown,
.stText,
p, li, label, div {
    color: var(--terminal-text);
}

.stSelectbox [data-baseweb="select"] > div,
.stNumberInput input,
.stTextInput input {
    background: rgba(16, 10, 28, 0.78);
    border: 1px solid rgba(158, 116, 255, 0.22);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
    border-radius: 14px;
}

[data-baseweb="tag"] {
    background: rgba(56, 189, 248, 0.14) !important;
}

[data-baseweb="tab-list"] {
    gap: 0.55rem;
    background: rgba(10, 8, 20, 0.36);
    border: 1px solid rgba(166, 127, 255, 0.18);
    border-radius: 18px;
    padding: 0.35rem;
    margin: 0.5rem 0 1rem;
    backdrop-filter: blur(14px);
}

button[data-baseweb="tab"] {
    border-radius: 14px !important;
    color: var(--terminal-dim) !important;
    font-family: "Space Grotesk", sans-serif !important;
    letter-spacing: 0.04em;
    min-height: 44px !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, rgba(70, 215, 255, 0.18), rgba(255, 122, 198, 0.16)) !important;
    color: var(--terminal-text) !important;
    border: 1px solid rgba(198, 160, 255, 0.26) !important;
}

.ui-section-kicker {
    color: var(--terminal-dim);
    text-transform: uppercase;
    letter-spacing: 0.16em;
    font-size: 0.72rem;
    margin-bottom: 0.25rem;
}

.ui-section-title {
    font-family: "Space Grotesk", sans-serif;
    font-size: 1.15rem;
    margin-bottom: 0.8rem;
}

.sidebar-panel {
    border: 1px solid rgba(166, 127, 255, 0.18);
    border-radius: 18px;
    padding: 0.9rem 0.95rem 0.7rem;
    margin-bottom: 1rem;
    background:
        linear-gradient(160deg, rgba(18, 10, 31, 0.92), rgba(10, 10, 20, 0.82)),
        radial-gradient(circle at top right, rgba(70, 215, 255, 0.12), transparent 34%);
    box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.04),
        0 12px 28px rgba(0, 0, 0, 0.22);
}

.sidebar-panel-title {
    font-family: "Space Grotesk", sans-serif;
    font-size: 1rem;
    color: var(--terminal-text);
    margin-bottom: 0.2rem;
}

.sidebar-panel-copy {
    color: var(--terminal-dim);
    font-size: 0.78rem;
    line-height: 1.45;
    margin-bottom: 0.65rem;
}

.sidebar-status-list {
    display: grid;
    gap: 0.45rem;
    margin: 0.5rem 0 0.65rem;
}

.sidebar-status-chip {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.6rem;
    padding: 0.48rem 0.65rem;
    border-radius: 999px;
    background: rgba(8, 11, 20, 0.72);
    border: 1px solid rgba(148, 163, 184, 0.16);
    color: var(--terminal-text);
    font-size: 0.76rem;
}

.sidebar-status-chip strong {
    color: var(--terminal-text);
    font-weight: 600;
}

.sidebar-status-chip span {
    color: #9cf6d0;
}

.sidebar-status-chip.is-missing span {
    color: #f9caca;
}

.terminal-metrics-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.85rem;
    margin: 0.3rem 0 1rem;
}

.control-panel-grid {
    display: grid;
    grid-template-columns: repeat(6, minmax(0, 1fr));
    gap: 0.7rem;
    margin: 0.15rem 0 1rem;
}

.control-panel-card {
    min-width: 0;
    padding: 0.72rem 0.78rem;
    margin-bottom: 0.7rem;
    border-radius: 16px;
    background:
        linear-gradient(160deg, rgba(5, 20, 15, 0.90), rgba(5, 14, 12, 0.76)),
        radial-gradient(circle at top right, rgba(52, 211, 153, 0.16), transparent 34%);
    border: 1px solid rgba(52, 211, 153, 0.16);
    box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.04),
        0 10px 24px rgba(0, 0, 0, 0.18);
}

.control-panel-label {
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--terminal-dim);
    margin-bottom: 0.38rem;
}

.control-panel-value {
    font-family: "Space Grotesk", sans-serif;
    font-size: 0.9rem;
    line-height: 1.2;
    color: var(--terminal-text);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.comparison-summary-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.75rem;
    margin: 0.35rem 0 0.8rem;
}

.comparison-summary-card {
    min-width: 0;
    padding: 0.82rem 0.9rem;
    margin-bottom: 0.7rem;
    border-radius: 18px;
    background:
        linear-gradient(160deg, rgba(5, 20, 15, 0.90), rgba(5, 14, 12, 0.76)),
        radial-gradient(circle at top right, rgba(52, 211, 153, 0.16), transparent 34%);
    border: 1px solid rgba(52, 211, 153, 0.16);
    box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.04),
        0 10px 24px rgba(0, 0, 0, 0.18);
}

.comparison-summary-label {
    font-size: 0.64rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--terminal-dim);
    margin-bottom: 0.42rem;
}

.comparison-summary-value {
    font-family: "Space Grotesk", sans-serif;
    font-size: 0.92rem;
    line-height: 1.2;
    color: var(--terminal-text);
    white-space: normal;
    word-break: break-word;
}

.snapshot-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.75rem;
    margin: 0.2rem 0 0.9rem;
}

.snapshot-card {
    min-width: 0;
    padding: 0.78rem 0.85rem;
    border-radius: 16px;
    background:
        linear-gradient(160deg, rgba(5, 20, 15, 0.90), rgba(5, 14, 12, 0.76)),
        radial-gradient(circle at top right, rgba(52, 211, 153, 0.14), transparent 34%);
    border: 1px solid rgba(52, 211, 153, 0.16);
    box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.04),
        0 10px 24px rgba(0, 0, 0, 0.18);
}

.snapshot-label {
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--terminal-dim);
    margin-bottom: 0.4rem;
}

.snapshot-value {
    font-family: "Space Grotesk", sans-serif;
    font-size: 0.9rem;
    line-height: 1.2;
    color: var(--terminal-text);
    white-space: normal;
    word-break: break-word;
}

.tab-3d-hero {
    position: relative;
    overflow: hidden;
    margin: 0.1rem 0 1rem;
    padding: 1rem 1.05rem 0.95rem;
    border-radius: 22px;
    background:
        linear-gradient(155deg, rgba(6, 24, 18, 0.92), rgba(4, 14, 11, 0.82)),
        radial-gradient(circle at top right, rgba(52, 211, 153, 0.22), transparent 34%),
        radial-gradient(circle at bottom left, rgba(110, 231, 183, 0.10), transparent 30%);
    border: 1px solid rgba(52, 211, 153, 0.18);
    box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.05),
        0 22px 38px rgba(0, 0, 0, 0.22);
    backdrop-filter: blur(18px) saturate(150%);
    transform: perspective(1400px) rotateX(5deg) translateY(0);
}

.tab-3d-hero::before {
    content: "";
    position: absolute;
    inset: 0;
    background:
        linear-gradient(135deg, rgba(255,255,255,0.05), transparent 36%),
        repeating-linear-gradient(
            90deg,
            transparent 0,
            transparent 18px,
            rgba(52, 211, 153, 0.035) 18px,
            rgba(52, 211, 153, 0.035) 19px
        );
    pointer-events: none;
}

.tab-3d-hero::after {
    content: "";
    position: absolute;
    inset: auto 18px 10px 18px;
    height: 28px;
    border-radius: 999px;
    background: radial-gradient(circle, rgba(16,255,163,0.18), transparent 72%);
    filter: blur(16px);
    pointer-events: none;
}

.tab-3d-kicker {
    position: relative;
    z-index: 1;
    color: var(--terminal-dim);
    text-transform: uppercase;
    letter-spacing: 0.16em;
    font-size: 0.68rem;
    margin-bottom: 0.3rem;
}

.tab-3d-title {
    position: relative;
    z-index: 1;
    font-family: "Space Grotesk", sans-serif;
    font-size: 1.3rem;
    color: var(--terminal-text);
    margin-bottom: 0.3rem;
    text-shadow: 0 0 16px rgba(16,255,163,0.12);
}

.tab-3d-copy {
    position: relative;
    z-index: 1;
    max-width: 58rem;
    color: #cdeedd;
    font-size: 0.82rem;
    line-height: 1.55;
}

[data-testid="stTabs"] [role="tabpanel"] [data-testid="stPlotlyChart"],
[data-testid="stTabs"] [role="tabpanel"] [data-testid="stDataFrame"],
[data-testid="stTabs"] [role="tabpanel"] [data-testid="stMetric"],
[data-testid="stTabs"] [role="tabpanel"] [data-testid="stExpander"] {
    transform: perspective(1200px) rotateX(3deg) translateY(0);
    transition: transform 220ms ease, box-shadow 220ms ease, border-color 220ms ease;
}

[data-testid="stTabs"] [role="tabpanel"] [data-testid="stPlotlyChart"]:hover,
[data-testid="stTabs"] [role="tabpanel"] [data-testid="stDataFrame"]:hover,
[data-testid="stTabs"] [role="tabpanel"] [data-testid="stMetric"]:hover,
[data-testid="stTabs"] [role="tabpanel"] [data-testid="stExpander"]:hover {
    transform: perspective(1200px) rotateX(1deg) translateY(-3px);
}

[data-testid="stTabs"] [role="tabpanel"] [data-testid="stMarkdownContainer"]:has(.tab-3d-hero) {
    background: transparent;
    border: 0;
    box-shadow: none;
    backdrop-filter: none;
}

.terminal-metric-card {
    position: relative;
    overflow: hidden;
    min-height: 154px;
    border-radius: 20px;
    padding: 0.82rem 0.88rem 0.8rem;
    background:
        linear-gradient(155deg, rgba(20, 13, 35, 0.90), rgba(9, 10, 20, 0.74)),
        radial-gradient(circle at top right, rgba(70, 215, 255, 0.22), transparent 34%),
        radial-gradient(circle at bottom left, rgba(255, 122, 198, 0.14), transparent 26%);
    border: 1px solid rgba(166, 127, 255, 0.24);
    box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.06),
        0 22px 44px rgba(0, 0, 0, 0.30),
        0 0 0 1px rgba(255, 255, 255, 0.02);
    backdrop-filter: blur(20px) saturate(150%);
    transform: perspective(1200px) rotateX(7deg) translateY(0);
    transition: transform 220ms ease, border-color 220ms ease, box-shadow 220ms ease;
}

.terminal-metric-card::before {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.06), transparent 38%);
    pointer-events: none;
}

.terminal-metric-card::after {
    content: "";
    position: absolute;
    inset: auto 12px 10px 12px;
    height: 26px;
    border-radius: 999px;
    background: radial-gradient(circle, rgba(255, 122, 198, 0.14), transparent 72%);
    filter: blur(18px);
    pointer-events: none;
}

.terminal-metric-card:hover {
    transform: perspective(1200px) rotateX(4deg) translateY(-4px);
    border-color: rgba(198, 160, 255, 0.38);
    box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.06),
        0 28px 52px rgba(0, 0, 0, 0.34);
}

.terminal-metric-card.is-bullish {
    border-color: rgba(34, 197, 94, 0.28);
    box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.06),
        0 16px 34px rgba(8, 95, 41, 0.16);
}

.terminal-metric-card.is-bearish {
    border-color: rgba(248, 113, 113, 0.26);
    box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.06),
        0 16px 34px rgba(127, 29, 29, 0.16);
}

.terminal-metric-card.is-neutral {
    border-color: rgba(250, 204, 21, 0.24);
}

.terminal-metric-label {
    color: var(--terminal-dim);
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-size: 0.64rem;
    margin-bottom: 0.6rem;
}

.terminal-metric-value {
    font-family: "Space Grotesk", sans-serif;
    font-size: 1.62rem;
    line-height: 1;
    color: #f8fbff;
}

.terminal-metric-delta {
    margin-top: 0.45rem;
    font-size: 0.82rem;
    color: #9dd7ff;
}

.terminal-metric-delta.is-up {
    color: #4ade80;
}

.terminal-metric-delta.is-down {
    color: #f87171;
}

.terminal-metric-subtext {
    margin-top: 0.5rem;
    color: #c6d4e1;
    font-size: 0.76rem;
    line-height: 1.45;
    min-height: 2.1rem;
}

.terminal-action-cluster {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
    margin-top: 0.7rem;
}

.terminal-action-chip {
    padding: 0.28rem 0.52rem;
    border-radius: 999px;
    font-size: 0.68rem;
    letter-spacing: 0.06em;
    color: #91a6bb;
    background: rgba(12, 19, 29, 0.72);
    border: 1px solid rgba(138, 160, 181, 0.16);
}

.terminal-action-chip.active-buy {
    color: #d8ffe4;
    background: rgba(20, 83, 45, 0.52);
    border-color: rgba(74, 222, 128, 0.38);
}

.terminal-action-chip.active-sell {
    color: #ffe0e0;
    background: rgba(127, 29, 29, 0.45);
    border-color: rgba(248, 113, 113, 0.38);
}

.terminal-action-chip.active-hold {
    color: #fff3c4;
    background: rgba(120, 53, 15, 0.42);
    border-color: rgba(250, 204, 21, 0.34);
}

.terminal-action-chip.active-exit {
    color: #dbeafe;
    background: rgba(30, 41, 59, 0.72);
    border-color: rgba(148, 163, 184, 0.34);
}

.terminal-sentiment-row {
    display: flex;
    justify-content: space-between;
    gap: 0.75rem;
    margin-top: 0.65rem;
    font-size: 0.76rem;
}

.terminal-sentiment-stat strong {
    display: block;
    font-family: "Space Grotesk", sans-serif;
    font-size: 1rem;
    color: #f8fbff;
}

.terminal-progress {
    margin-top: 0.7rem;
    height: 7px;
    border-radius: 999px;
    background: rgba(148, 163, 184, 0.15);
    overflow: hidden;
}

.terminal-progress-bar {
    height: 100%;
    border-radius: inherit;
    background: linear-gradient(90deg, #38bdf8 0%, #22c55e 100%);
}

@media (max-width: 1100px) {
    .terminal-metrics-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .control-panel-grid {
        grid-template-columns: repeat(3, minmax(0, 1fr));
    }

    .comparison-summary-grid {
        grid-template-columns: 1fr;
    }

    .snapshot-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}

@media (max-width: 700px) {
    .terminal-metrics-grid {
        grid-template-columns: 1fr;
    }

    .control-panel-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .snapshot-grid {
        grid-template-columns: 1fr;
    }
}

[data-testid="stDataFrame"] [role="grid"] {
    background: transparent;
}

[data-testid="stPlotlyChart"] {
    padding: 0.45rem;
}

[data-testid="stExpander"] details summary p {
    font-family: "Space Grotesk", sans-serif !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stSelectbox"] {
    position: relative;
    z-index: 2;
}

</style>
""", unsafe_allow_html=True)

def _get_required_secret(name):
    value = st.secrets.get(name) or os.getenv(name)
    if value:
        return value

    st.error(
        f"Missing required secret `{name}`. Add it to `.streamlit/secrets.toml` locally "
        "or to your Streamlit Cloud app secrets."
    )
    st.stop()


TWELVE_API_KEY = _get_required_secret("TWELVE_API_KEY")
ALPHA_API_KEY = _get_required_secret("ALPHA_API_KEY")
FRED_API_KEY = _get_required_secret("FRED_API_KEY")

# ---------------- LIVE PRICE FUNCTION ----------------
def get_live_price(symbol):
    symbol = _sanitize_market_symbol(symbol)
    ticker = yf.Ticker(symbol)

    try:
        data = ticker.history(period="1d", interval="1m")

        if data.empty:
            data = ticker.history(period="5d", interval="1d")

        if data.empty:
            return None

        price = float(data["Close"].iloc[-1])

        # Currency detection
        if ".NS" in symbol or ".BO" in symbol:
            currency = "₹"
        elif "=X" in symbol:
            currency = "$"
        elif "-USD" in symbol:
            currency = "$"
        else:
            currency = "$"

        return price, currency
    
    except Exception:
        return None
    
@st.cache_data(ttl=5)
def cached_price(sym):
        return get_live_price(sym)

# ---------------- Django API ----------------
def fetch_news(symbol):
    safe_symbol = _sanitize_market_symbol(symbol)
    url = f"http://127.0.0.1:8000/api/news/{safe_symbol}/"

    try:
        response = _safe_api_get(url, timeout=5)
        return response.json()
    except (requests.RequestException, ValueError):
        return {"news": []}


@st.cache_data(ttl=300)
def get_cached_sentiment(symbol):
    return get_news_sentiment(symbol)


@st.cache_data(ttl=900)
def get_cached_model_results(asset_key, source, year_label="All Years"):
    if source == DATA_SOURCE_HISTORICAL:
        asset_df = get_custom_asset_data_map().get(asset_key, pd.DataFrame())
        asset_df = _filter_data_by_financial_year(asset_df, year_label)
        return compare_history_models(asset_df, asset_key)

    return compare_models(asset_key)


@st.cache_data(ttl=3600)
def get_research_context():
    return RESEARCH_CONTEXT

# ---------------- TWELVE DATA API ----------------
def get_twelve_price(symbol):
    safe_symbol = _sanitize_market_symbol(symbol)
    response = _safe_api_get(
        "https://api.twelvedata.com/price",
        params={"symbol": safe_symbol, "apikey": TWELVE_API_KEY},
    )
    data = response.json()

    if "price" in data:
        return float(data["price"])

    return None

# ---------------- MACRO / MARKET SIGNALS ----------------

def get_sp500_returns(index):
    df = yf.download("^GSPC", period="1y", interval="1d", progress=False)
    if df.empty:
        return pd.Series(0, index=index)

    returns = df["Close"].pct_change()
    return returns.reindex(index).fillna(0)


def get_vix(index):
    df = yf.download("^VIX", period="1y", interval="1d", progress=False)
    if df.empty:
        return pd.Series(0, index=index)

    vix = df["Close"]
    return vix.reindex(index).ffill().fillna(0)


def get_tnx_yield(index):
    df = yf.download("^TNX", period="1y", interval="1d", progress=False)
    if df.empty:
        return pd.Series(0, index=index)

    rate = df["Close"]
    return rate.reindex(index).ffill().fillna(0)


def build_macro_features(index):
    sp500 = get_sp500_returns(index)
    vix = get_vix(index)
    rates = get_tnx_yield(index)

    return sp500, vix, rates


# ---------------- ALPHA VANTAGE API ----------------
def get_alpha_price(symbol):
    safe_symbol = _sanitize_market_symbol(symbol)
    response = _safe_api_get(
        "https://www.alphavantage.co/query",
        params={"function": "GLOBAL_QUOTE", "symbol": safe_symbol, "apikey": ALPHA_API_KEY},
    )
    data = response.json()

    if "Global Quote" in data:
        return float(data["Global Quote"]["05. price"])

    return None

# ---------------- FOREX LIVE DATA  ----------------
def get_live_forex(symbol):
    try:
        symbol = _sanitize_market_symbol(symbol)
        # TRY intraday first
        df = yf.download(symbol, period="7d", interval="1h", progress=False)

        # fallback to daily if empty
        if df is None or df.empty:
            df = yf.download(symbol, period="1mo", interval="1d", progress=False)

        if df is None or df.empty:
            return pd.DataFrame()

        return df

    except Exception:
        return pd.DataFrame()


# ---------------- FRED API ----------------
def get_fred_data(series):
    safe_series = str(series).strip().upper()
    if not SAFE_FRED_SERIES.fullmatch(safe_series):
        return None

    response = _safe_api_get(
        "https://api.stlouisfed.org/fred/series/observations",
        params={"series_id": safe_series, "api_key": FRED_API_KEY, "file_type": "json"},
    )
    data = response.json()

    if "observations" in data:
        return data["observations"][-1]["value"]

    return None

# -------------------- EQUITY ASSETS --------------------
ASSETS = {
    "🇺🇸 Stocks": [
        # NASDAQ
        "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL", "GOOG",
        "AMD", "INTC", "ADBE", "NFLX", "AVGO", "QCOM", "CSCO", "ORCL",
        "PEP", "COST", "TXN", "AMAT",

        # NYSE
        "IBM", "JNJ", "V", "PG", "DIS", "KO", "WMT", "MA", "BAC",
        "XOM", "CVX", "MCD", "GS", "UNH", "HD", "CAT", "NKE",

        # S&P 500 ETFs 
        "SPY", "VOO", "IVV", "SPLG", "SPYG", "SPYV"

    ],

    "🇮🇳 Stocks": [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
        "SBIN.NS", "LT.NS", "ITC.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
        "AXISBANK.NS", "WIPRO.NS", "HCLTECH.NS", "TITAN.NS", "TATASTEEL.NS", "ASIANPAINT.NS"
    ],
}

st.sidebar.title("Market Tools")
asset_workbook_available = _get_workbook_source("custom_asset_workbook_bytes", CUSTOM_ASSET_WORKBOOK, "asset")
summary_workbook_available = _get_workbook_source("custom_summary_workbook_bytes", CUSTOM_SUMMARY_WORKBOOK, "summary")

has_asset_workbook = bool(st.session_state.get("custom_asset_workbook_bytes")) or isinstance(asset_workbook_available, Path)
has_summary_workbook = bool(st.session_state.get("custom_summary_workbook_bytes")) or isinstance(summary_workbook_available, Path)

if not has_asset_workbook or not has_summary_workbook:
    st.sidebar.markdown(
        f"""
        <div class="sidebar-panel">
            <div class="sidebar-panel-title">Historical Data</div>
            <div class="sidebar-panel-copy">Dissertation mode needs the workbook files below before it can load historical assets.</div>
            <div class="sidebar-status-list">
                <div class="sidebar-status-chip {'is-missing' if not has_asset_workbook else ''}">
                    <strong>Assets workbook</strong>
                    <span>{'Connected' if has_asset_workbook else 'Missing'}</span>
                </div>
                <div class="sidebar-status-chip {'is-missing' if not has_summary_workbook else ''}">
                    <strong>Summary workbook</strong>
                    <span>{'Connected' if has_summary_workbook else 'Missing'}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar.expander("Upload Files", expanded=False):
        asset_workbook_upload = st.file_uploader("Asset Workbook (.xlsx)", type="xlsx", key="custom_asset_workbook_upload")
        summary_workbook_upload = st.file_uploader("Summary Workbook (.xlsx)", type="xlsx", key="custom_summary_workbook_upload")

        if asset_workbook_upload is not None:
            try:
                asset_bytes = asset_workbook_upload.getvalue()
                _validate_uploaded_workbook(asset_bytes, "Asset workbook")
                st.session_state["custom_asset_workbook_bytes"] = asset_bytes
            except ValueError as exc:
                st.error(str(exc))
        if summary_workbook_upload is not None:
            try:
                summary_bytes = summary_workbook_upload.getvalue()
                _validate_uploaded_workbook(summary_bytes, "Summary workbook")
                st.session_state["custom_summary_workbook_bytes"] = summary_bytes
            except ValueError as exc:
                st.error(str(exc))

data_source = st.sidebar.selectbox("Workspace Mode", [DATA_SOURCE_HISTORICAL, DATA_SOURCE_MARKET_TOOLS])

custom_assets = get_custom_asset_data_map()

if data_source == DATA_SOURCE_HISTORICAL:
    custom_asset_names = sorted(custom_assets.keys())
    if not custom_asset_names:
        st.warning("Dissertation Models needs your workbook files before it can load any asset sheets.")
        st.markdown(
            f"""
            Upload the dissertation files from the sidebar:

            - `Asset Workbook (.xlsx)` is required
            - `Summary Workbook (.xlsx)` is recommended for yearly forecast summaries

            Current status:

            - Asset workbook: {"ready" if has_asset_workbook else "missing"}
            - Summary workbook: {"ready" if has_summary_workbook else "missing"}

            If you want the app to load them automatically on Streamlit Cloud, add these files to your repo:

            - `{CUSTOM_ASSET_WORKBOOK}`
            - `{CUSTOM_SUMMARY_WORKBOOK}`
            """
        )
        st.stop()

    category = "Dissertation Equity Models"
    symbol = st.sidebar.selectbox("Assets", custom_asset_names)
    display_symbol = symbol
    asset_key = symbol
    is_custom_asset = True
    available_years = _get_financial_year_options(asset_key, custom_assets.get(asset_key, pd.DataFrame()))
    selected_year = st.sidebar.selectbox("Forecast Window", available_years)
else:
    category = st.sidebar.selectbox("Equity Market", list(ASSETS.keys()))
    symbol = st.sidebar.selectbox("Equity Ticker", ASSETS[category])
    symbol = clean_symbol(symbol)
    display_symbol = symbol
    asset_key = symbol
    is_custom_asset = False
    selected_year = "All Years"

chart_type = st.sidebar.selectbox(
    "Chart Type",
    ["Candlestick", "Heikin Ashi", "OHLC Bars", "Line Chart"]
)

# -------------------- DATA --------------------
@st.cache_data(ttl=300)
def load_data(asset_key, source):
    if source == DATA_SOURCE_HISTORICAL:
        return get_custom_asset_data_map().get(asset_key, pd.DataFrame()).copy()

    symbol = clean_symbol(asset_key)

    if not symbol:
        return pd.DataFrame()

    try:
        # FOREX FIX
        if "=X" in symbol:
            return get_live_forex(symbol)

        df = yf.download(symbol, period="6mo", interval="1d", progress=False)

    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df

# -------------------- INDICATORS --------------------

@st.cache_data(ttl=300)
def compute_indicators(df):

    if df is None or df.empty:
        return df

    if "Close" not in df.columns:
        return df

    df = df.copy()
    close = df["Close"]

    # ===== MOVING AVERAGES =====
    df["MA20"] = close.rolling(20).mean()
    df["MA50"] = close.rolling(50).mean()

    # ===== RSI =====
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()

    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # ===== MACD =====
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()

    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9).mean()

    # ===== BOLLINGER BANDS =====
    mid = close.rolling(20).mean()
    std = close.rolling(20).std()

    df["BB_UP"] = mid + (2 * std)
    df["BB_LOW"] = mid - (2 * std)

    df["BB_Width"] = (df["BB_UP"] - df["BB_LOW"]) / mid

    # ===== VWAP =====
    tp = (df["High"] + df["Low"] + close) / 3
    df["VWAP"] = (tp * df["Volume"]).cumsum() / df["Volume"].cumsum()

    # ===== VOLUME MA =====
    df["Volume_MA20"] = df["Volume"].rolling(20).mean()

    # ===== SUPPORT / RESISTANCE =====
    df["Support"] = df["Low"].rolling(20).min()
    df["Resistance"] = df["High"].rolling(20).max()

    # ===== ATR =====
    df["ATR"] = (df["High"] - df["Low"]).rolling(14).mean()

    # ===== OBV =====
    df["OBV"] = (np.sign(close.diff()) * df["Volume"]).fillna(0).cumsum()

    # ===== STOCHASTIC =====
    low14 = df["Low"].rolling(14).min()
    high14 = df["High"].rolling(14).max()

    df["STOCH_K"] = 100 * (close - low14) / (high14 - low14 + 1e-9)
    df["STOCH_D"] = df["STOCH_K"].rolling(3).mean()

    # ===== CCI =====
    tp2 = (df["High"] + df["Low"] + close) / 3
    mean_tp = tp2.rolling(20).mean()
    std_tp = tp2.rolling(20).std()

    df["CCI"] = (tp2 - mean_tp) / (0.015 * std_tp)

    return df


raw_data = load_data(asset_key, data_source)
data = _filter_data_by_financial_year(raw_data, selected_year) if is_custom_asset else raw_data.copy()

if data.empty:
    if is_custom_asset and selected_year != "All Years":
        st.error(f"No data available for {display_symbol} in {selected_year}")
    else:
        st.error(f"Market data unavailable for {symbol}")
    st.stop()

if "=X" in symbol:
    data["Volume"] = data.get("Volume", 0).fillna(0)

# fix MultiIndex early
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# 🔥 convert safely BEFORE indicators
data = data.apply(pd.to_numeric, errors="coerce")

if "Close" not in data.columns:
    st.error("Missing Close column — data invalid")
    st.stop()

# compute indicators ONLY ONCE
data = compute_indicators(data)

# ---------------- SENTIMENT ----------------
def get_sentiment_series(symbol, index):
    score = 0 if is_custom_asset else get_cached_sentiment(symbol)

    # spread same sentiment across chart timeline
    return pd.Series([score] * len(index), index=index)

sentiment_score = 0 if is_custom_asset else get_cached_sentiment(symbol)

data["sentiment"] = np.linspace(
    sentiment_score - 0.1,
    sentiment_score + 0.1,
    len(data)
)

sp500, vix, rates = build_macro_features(data.index)

data["macro_market"] = sp500
data["macro_volatility"] = vix
data["macro_rates"] = rates

# ---------------- CORRELATION MATRIX ----------------
corr_cols = [c for c in ["Close","sentiment","macro_market","macro_volatility","macro_rates","RSI","MACD"] if c in data.columns]
corr_data = data[corr_cols]
corr_matrix = corr_data.corr()

# final cleanup AFTER indicators
data = data.replace([np.inf, -np.inf], np.nan)

# 🔥 FIX: only drop critical OHLC columns (NOT full dataframe)
data = data.dropna(subset=["Close", "Open", "High", "Low"])

if data.empty:
    st.error("No usable data after processing")
    st.stop()

close = data["Close"].copy()


# -------------------- PRICE ACTION FUNCTION --------------------
def price_action(df):
    signals = []

    for i in range(1, len(df)):
        prev_open = df["Open"].iloc[i-1]
        prev_close = df["Close"].iloc[i-1]
        curr_open = df["Open"].iloc[i]
        curr_close = df["Close"].iloc[i]

        # Bullish Engulfing
        if curr_close > curr_open and prev_close < prev_open:
            signals.append((df.index[i], "Bullish Engulfing"))

        # Bearish Engulfing
        elif curr_close < curr_open and prev_close > prev_open:
            signals.append((df.index[i], "Bearish Engulfing"))

    return signals

# -------------------- PARABOLIC SAR --------------------

# 🛡 Safety check (prevents crash if dataframe empty)
if len(close) < 2:
    st.warning("Not enough data for indicators")
    st.stop()

af = 0.02
max_af = 0.2

psar = [close.iloc[0]]
trend = 1
ep = data["High"].iloc[0]

for i in range(1, len(data)):
    prev_sar = psar[-1]

    if trend == 1:
        sar = prev_sar + af * (ep - prev_sar)
        sar = min(sar, data["Low"].iloc[i-1])

        if data["Low"].iloc[i] < sar:
            trend = -1
            sar = ep
            ep = data["Low"].iloc[i]
            af = 0.02
    else:
        sar = prev_sar + af * (ep - prev_sar)
        sar = max(sar, data["High"].iloc[i-1])

        if data["High"].iloc[i] > sar:
            trend = 1
            sar = ep
            ep = data["High"].iloc[i]
            af = 0.02

    af = min(af, max_af)

    psar.append(sar)

data["PSAR"] = psar

patterns = price_action(data)

# -------------------- MAIN CHART --------------------
fig = go.Figure()

if chart_type == "Candlestick":
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data["Open"],
        high=data["High"],
        low=data["Low"],
        close=data["Close"]
    ))

elif chart_type == "Line Chart":
    fig.add_trace(go.Scatter(x=data.index, y=close, mode="lines"))

elif chart_type == "OHLC Bars":
    fig.add_trace(go.Ohlc(
        x=data.index,
        open=data["Open"],
        high=data["High"],
        low=data["Low"],
        close=data["Close"]
    ))

elif chart_type == "Heikin Ashi":
    ha = data.copy()
    ha["Close"] = (ha["Open"] + ha["High"] + ha["Low"] + ha["Close"]) / 4
    ha["Open"] = ha["Open"].shift(1).fillna(ha["Open"])

    fig.add_trace(go.Candlestick(
        x=ha.index,
        open=ha["Open"],
        high=ha["High"],
        low=ha["Low"],
        close=ha["Close"]
    ))

# -------------------- PRICE OVERLAYS --------------------
fig.add_trace(go.Scatter(x=data.index, y=data["MA20"], name="MA20"))
fig.add_trace(go.Scatter(x=data.index, y=data["MA50"], name="MA50"))
fig.add_trace(go.Scatter(x=data.index, y=data["VWAP"], name="VWAP"))

fig.add_trace(go.Scatter(x=data.index, y=data["BB_UP"], name="BB Upper"))
fig.add_trace(go.Scatter(x=data.index, y=data["BB_LOW"], name="BB Lower"))

fig.add_trace(go.Scatter(
    x=data.index,
    y=data["Support"],
    name="Support",
    line=dict(dash="dot")
))

fig.add_trace(go.Scatter(
    x=data.index,
    y=data["Resistance"],
    name="Resistance",
    line=dict(dash="dot")
))

fig.update_layout(template="plotly_dark", height=650)

def analyze_asset(df):
    close = df["Close"]
    ma20 = df["MA20"].iloc[-1]
    ma50 = df["MA50"].iloc[-1]
    rsi = df["RSI"].iloc[-1]

    # Trend
    trend = "Bullish 🟢" if ma20 > ma50 else "Bearish 🔴"

    # Signal logic
    if rsi > 70:
        signal = "SELL 🔴"
        sentiment = 25
    elif rsi < 30:
        signal = "BUY 🟢"
        sentiment = 80
    elif ma20 > ma50:
        signal = "HOLD 🟡"
        sentiment = 65
    else:
        signal = "EXIT ⚪"
        sentiment = 35

    return signal, trend, sentiment, rsi


def _safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def build_header_metrics(df, signal, trend, sentiment_score, currency):
    latest_price = _safe_float(df["Close"].iloc[-1])
    previous_price = latest_price
    if len(df) > 1:
        previous_price = _safe_float(df["Close"].iloc[-2], latest_price)

    price_change = latest_price - previous_price
    price_change_pct = (price_change / previous_price * 100) if previous_price else 0
    people_positive = max(0.0, min(100.0, _safe_float(sentiment_score, 50.0)))
    people_negative = max(0.0, 100.0 - people_positive)
    market_view = sentiment_breakdown(people_positive)

    signal_text = str(signal).upper()
    active_action = "EXIT"
    for action_name in ["BUY", "SELL", "HOLD", "EXIT"]:
        if action_name in signal_text:
            active_action = action_name
            break

    signal_class = {
        "BUY": "is-bullish",
        "SELL": "is-bearish",
        "HOLD": "is-neutral",
        "EXIT": "is-neutral",
    }.get(active_action, "is-neutral")

    delta_class = "is-up" if price_change >= 0 else "is-down"

    return {
        "price": f"{currency}{latest_price:,.2f}",
        "price_change": price_change,
        "price_change_pct": price_change_pct,
        "price_delta_text": f"{price_change:+.2f} ({price_change_pct:+.2f}%)",
        "delta_class": delta_class,
        "signal": active_action,
        "signal_class": signal_class,
        "trend": trend,
        "people_positive": people_positive,
        "people_negative": people_negative,
        "bullish": market_view["bullish_percent"],
        "bearish": market_view["bearish_percent"],
        "neutral": market_view["neutral_percent"],
    }


def render_terminal_header_metrics(metrics):
    signal = metrics["signal"]
    active_classes = {
        action: f"terminal-action-chip active-{action.lower()}" if action == signal else "terminal-action-chip"
        for action in ["BUY", "SELL", "HOLD", "EXIT"]
    }

    st.markdown(
        f"""
        <div class="terminal-metrics-grid">
            <div class="terminal-metric-card {metrics['signal_class']}">
                <div class="terminal-metric-label">Live Price</div>
                <div class="terminal-metric-value">{metrics['price']}</div>
                <div class="terminal-metric-delta {metrics['delta_class']}">{metrics['price_delta_text']}</div>
                <div class="terminal-metric-subtext">Realtime equity move supporting the current forecast view.</div>
            </div>
            <div class="terminal-metric-card {metrics['signal_class']}">
                <div class="terminal-metric-label">Action Matrix</div>
                <div class="terminal-metric-value">{metrics['signal']}</div>
                <div class="terminal-metric-subtext">Current desk stance: {metrics['trend']}</div>
                <div class="terminal-action-cluster">
                    <span class="{active_classes['BUY']}">BUY</span>
                    <span class="{active_classes['SELL']}">SELL</span>
                    <span class="{active_classes['HOLD']}">HOLD</span>
                    <span class="{active_classes['EXIT']}">EXIT</span>
                </div>
            </div>
            <div class="terminal-metric-card is-bullish">
                <div class="terminal-metric-label">People Sentiment</div>
                <div class="terminal-metric-value">{metrics['people_positive']:.1f}% +ve</div>
                <div class="terminal-metric-subtext">Audience tone split from news and terminal sentiment scoring.</div>
                <div class="terminal-sentiment-row">
                    <div class="terminal-sentiment-stat"><strong>{metrics['people_positive']:.1f}%</strong>Positive</div>
                    <div class="terminal-sentiment-stat"><strong>{metrics['people_negative']:.1f}%</strong>Negative</div>
                </div>
            </div>
            <div class="terminal-metric-card {'is-bullish' if metrics['bullish'] >= metrics['bearish'] else 'is-bearish'}">
                <div class="terminal-metric-label">Equity Sentiment</div>
                <div class="terminal-metric-value">{metrics['bullish']:.1f}% Bullish</div>
                <div class="terminal-metric-subtext">Bearish pressure: {metrics['bearish']:.1f}% • Neutral: {metrics['neutral']:.1f}%</div>
                <div class="terminal-progress">
                    <div class="terminal-progress-bar" style="width:{metrics['bullish']:.1f}%"></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _find_summary_column(df, aliases):
    if df is None or df.empty:
        return None

    normalized_map = {_normalize_header(column): column for column in df.columns}
    for alias in aliases:
        matched = normalized_map.get(_normalize_header(alias))
        if matched:
            return matched
    return None


def build_historical_header_metrics(df, asset_summary, selected_year, currency):
    period_label = selected_year if selected_year != "All Years" else "Full Archive"
    start_price = _safe_float(df["Close"].iloc[0])
    end_price = _safe_float(df["Close"].iloc[-1], start_price)
    absolute_move = end_price - start_price
    calculated_return = (absolute_move / start_price * 100) if start_price else 0.0
    high_price = _safe_float(df["High"].max(), end_price)
    low_price = _safe_float(df["Low"].min(), start_price)

    summary_row = pd.DataFrame()
    if asset_summary is not None and not asset_summary.empty:
        summary_row = asset_summary.copy()
        if selected_year != "All Years" and "Year" in summary_row.columns:
            summary_row = summary_row[summary_row["Year"].astype(str).str.strip() == selected_year]
        if not summary_row.empty:
            summary_row = summary_row.head(1)

    actual_return_pct = calculated_return
    if not summary_row.empty:
        actual_return_col = _find_summary_column(summary_row, ["Actual Return"])
        if actual_return_col:
            raw_value = pd.to_numeric(summary_row.iloc[0][actual_return_col], errors="coerce")
            if not pd.isna(raw_value):
                actual_return_pct = float(raw_value) * 100 if abs(float(raw_value)) <= 2 else float(raw_value)

    trend = "Bullish" if end_price >= start_price else "Bearish"
    sentiment_proxy = max(0.0, min(100.0, 50 + (actual_return_pct * 0.9) + (6 if trend == "Bullish" else -6)))
    people_negative = max(0.0, 100.0 - sentiment_proxy)
    market_view = sentiment_breakdown(sentiment_proxy)
    regime_class = "is-bullish" if actual_return_pct >= 0 else "is-bearish"
    delta_class = "is-up" if actual_return_pct >= 0 else "is-down"

    return {
        "period_label": period_label,
        "start_price": f"{currency}{start_price:,.2f}",
        "end_price": f"{currency}{end_price:,.2f}",
        "range_text": f"Low {currency}{low_price:,.2f} • High {currency}{high_price:,.2f}",
        "return_text": f"{actual_return_pct:+.2f}%",
        "move_text": f"{absolute_move:+.2f}",
        "trend": trend,
        "sentiment_proxy": sentiment_proxy,
        "people_negative": people_negative,
        "bullish": market_view["bullish_percent"],
        "bearish": market_view["bearish_percent"],
        "neutral": market_view["neutral_percent"],
        "regime_class": regime_class,
        "delta_class": delta_class,
    }


def render_historical_terminal_header_metrics(metrics):
    st.markdown(
        f"""
        <div class="terminal-metrics-grid">
            <div class="terminal-metric-card {metrics['regime_class']}">
                <div class="terminal-metric-label">Back Then Price</div>
                <div class="terminal-metric-value">{metrics['end_price']}</div>
                <div class="terminal-metric-delta {metrics['delta_class']}">From {metrics['start_price']} to {metrics['end_price']}</div>
                <div class="terminal-metric-subtext">{metrics['period_label']} snapshot • {metrics['range_text']}</div>
            </div>
            <div class="terminal-metric-card {metrics['regime_class']}">
                <div class="terminal-metric-label">Year Performance</div>
                <div class="terminal-metric-value">{metrics['return_text']}</div>
                <div class="terminal-metric-delta {metrics['delta_class']}">Net price move {metrics['move_text']}</div>
                <div class="terminal-metric-subtext">Historical market bias for this selected dissertation year was {metrics['trend']}.</div>
            </div>
            <div class="terminal-metric-card {'is-bullish' if metrics['sentiment_proxy'] >= 50 else 'is-bearish'}">
                <div class="terminal-metric-label">Back Then Sentiment</div>
                <div class="terminal-metric-value">{metrics['sentiment_proxy']:.1f}% +ve</div>
                <div class="terminal-metric-subtext">Historical sentiment proxy reconstructed from the selected year's realized price regime.</div>
                <div class="terminal-sentiment-row">
                    <div class="terminal-sentiment-stat"><strong>{metrics['sentiment_proxy']:.1f}%</strong>Positive</div>
                    <div class="terminal-sentiment-stat"><strong>{metrics['people_negative']:.1f}%</strong>Negative</div>
                </div>
            </div>
            <div class="terminal-metric-card {'is-bullish' if metrics['bullish'] >= metrics['bearish'] else 'is-bearish'}">
                <div class="terminal-metric-label">Historical Mood</div>
                <div class="terminal-metric-value">{metrics['bullish']:.1f}% Bullish</div>
                <div class="terminal-metric-subtext">Bearish pressure: {metrics['bearish']:.1f}% • Neutral: {metrics['neutral']:.1f}%</div>
                <div class="terminal-progress">
                    <div class="terminal-progress-bar" style="width:{metrics['bullish']:.1f}%"></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_compact_control_panel(items):
    first_row = items[:3]
    second_row = items[3:6]

    for row_items in [first_row, second_row]:
        if not row_items:
            continue
        columns = st.columns(len(row_items))
        for column, (label, value) in zip(columns, row_items):
            with column:
                st.markdown(
                    f"""
                    <div class="control-panel-card">
                        <div class="control-panel-label">{escape(str(label))}</div>
                        <div class="control-panel-value" title="{escape(str(value))}">{escape(str(value))}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_comparison_summary(items):
    columns = st.columns(len(items))
    for column, (label, value) in zip(columns, items):
        with column:
            st.markdown(
                f"""
                <div class="comparison-summary-card">
                    <div class="comparison-summary-label">{escape(str(label))}</div>
                    <div class="comparison-summary-value" title="{escape(str(value))}">{escape(str(value))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_snapshot_cards(items):
    cards = []
    for label, value in items:
        cards.append(
            f"""
            <div class="snapshot-card">
                <div class="snapshot-label">{escape(str(label))}</div>
                <div class="snapshot-value" title="{escape(str(value))}">{escape(str(value))}</div>
            </div>
            """
        )

    st.markdown(
        f'<div class="snapshot-grid">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


def render_tab_3d_hero(kicker, title, copy):
    st.markdown(
        f"""
        <div class="tab-3d-hero">
            <div class="tab-3d-kicker">{escape(str(kicker))}</div>
            <div class="tab-3d-title">{escape(str(title))}</div>
            <div class="tab-3d-copy">{escape(str(copy))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

title_suffix = f" ({selected_year})" if is_custom_asset and selected_year != "All Years" else ""
st.title(f"📈 RUPATCHI MODEL — {display_symbol}{title_suffix}")
st.caption(
    "Market tools workspace blending compact signals, price structure, sentiment, macro context, and model comparison."
)

research_context = get_research_context()

signal, trend, sentiment, rsi = analyze_asset(data.copy())
currency = "₹" if is_custom_asset or ".NS" in symbol or ".BO" in symbol else "$"
hero_window = selected_year if is_custom_asset else DATA_SOURCE_MARKET_TOOLS
render_three_market_scene(display_symbol, hero_window, trend)
if is_custom_asset:
    custom_asset_summary = build_custom_asset_summary(asset_key)
    historical_header_metrics = build_historical_header_metrics(
        data,
        custom_asset_summary,
        selected_year,
        currency,
    )
    render_historical_terminal_header_metrics(historical_header_metrics)
else:
    header_metrics = build_header_metrics(data, signal, trend, sentiment, currency)
    render_terminal_header_metrics(header_metrics)

with st.expander("📘 Forecasting Framework", expanded=False):
    st.markdown(f"**Problem Statement**: {research_context['problem']}")
    st.markdown("**Objectives**")
    for item in research_context["objectives"]:
        st.write("•", item)
    st.markdown("**Hypotheses**")
    for item in research_context["hypotheses"]:
        st.write("•", item)
    st.markdown("**Methodology**")
    for item in research_context["methodology"]:
        st.write("•", item)
    st.markdown("**Data Sources**")
    for item in research_context["data_sources"]:
        st.write("•", item)

st.plotly_chart(
    fig,
    width="stretch",
    config={"displayModeBar": False}
)

if is_custom_asset and selected_year != "All Years":
    st.caption(f"Showing workbook data and forecasts for financial year **{selected_year}**.")

live_df = data.copy()
price = live_df["Close"].iloc[-1]

dashboard_trend = "Bullish 🟢" if live_df["MA20"].iloc[-1] > live_df["MA50"].iloc[-1] else "Bearish 🔴"
rsi = live_df["RSI"].iloc[-1]
macd = live_df["MACD"].iloc[-1]
macd_signal = live_df["MACD_SIGNAL"].iloc[-1]
ma20 = live_df["MA20"].iloc[-1]
ma50 = live_df["MA50"].iloc[-1]
volume = live_df["Volume"].iloc[-1]
vol_ma = live_df["Volume_MA20"].iloc[-1]

score = 0

# RSI
if rsi < 30:
    score += 35
elif rsi > 70:
    score -= 35
else:
    score += 10

# Trend
if ma20 > ma50:
    score += 30
else:
    score -= 30

# MACD
if macd > macd_signal:
    score += 20
else:
    score -= 20

# Volume
if volume > vol_ma:
    score += 10

# Normalize
bull = max(0, min(100, 50 + score))
bear = max(0, min(100, 100 - bull))
neu = max(0, 100 - (bull + bear))

overview_tab, factors_tab, models_tab, indicators_tab = st.tabs(
    ["Overview", "Factors", "Models", "Indicators"]
)

with overview_tab:
    st.markdown('<div class="ui-section-kicker">Workspace</div><div class="ui-section-title">Control Panel</div>', unsafe_allow_html=True)
    render_compact_control_panel(
        [
            ("Asset", display_symbol),
            ("Price", f"{currency}{price:.2f}"),
            ("Signal", signal),
            ("Trend", trend),
            ("Sentiment", f"{sentiment}%"),
            ("RSI", f"{rsi:.2f}"),
        ]
    )

    st.markdown('<div class="ui-section-kicker">Market Read</div><div class="ui-section-title">Dashboard</div>', unsafe_allow_html=True)
    dash_a, dash_b, dash_c = st.columns(3)
    dash_a.metric("Trend", dashboard_trend)
    dash_b.metric("Volatility", round(live_df["BB_Width"].iloc[-1], 3))
    dash_c.metric(
        "RSI Zone",
        "Overbought 🔴" if rsi > 70 else "Oversold 🟢" if rsi < 30 else "Neutral 🟡"
    )

    sent_a, sent_b, sent_c = st.columns(3)
    sent_a.metric("Bullish 🟢", f"{round(bull,1)}%")
    sent_b.metric("Bearish 🔴", f"{round(bear,1)}%")
    sent_c.metric("Neutral ⚪", f"{round(neu,1)}%")

    st.markdown('<div class="ui-section-kicker">Model Inputs</div><div class="ui-section-title">Input Snapshot</div>', unsafe_allow_html=True)
    render_snapshot_cards(
        [
            ("RSI", f"{live_df['RSI'].iloc[-1]:.2f}"),
            ("MACD", f"{live_df['MACD'].iloc[-1]:.2f}"),
            ("ATR", f"{live_df['ATR'].iloc[-1]:.2f}"),
            ("OBV", f"{live_df['OBV'].iloc[-1]:,.2f}"),
        ]
    )

    if is_custom_asset:
        st.markdown(
            '<div class="ui-section-kicker">Historical Context</div><div class="ui-section-title">Dissertation Snapshot</div>',
            unsafe_allow_html=True,
        )
        context_a, context_b, context_c = st.columns(3)
        context_a.metric("Window", selected_year)
        context_b.metric("Historical Bias", historical_header_metrics["trend"])
        context_c.metric("Back Then Mood", f'{historical_header_metrics["bullish"]:.1f}% Bullish')
        st.caption(
            "Dissertation mode uses your workbook history and forecast summaries for the selected year instead of live news."
        )
    else:
        st.markdown('<div class="ui-section-kicker">News</div><div class="ui-section-title">News Flow</div>', unsafe_allow_html=True)
        news_data = {"news": []}
        news_data = fetch_news(symbol)
        news_list = news_data.get("news", [])
        if news_list:
            for n in news_list:
                st.write("•", n)
        else:
            st.info("No market news available for this asset right now")

with factors_tab:
    st.markdown('<div class="ui-section-kicker">Cross Signals</div><div class="ui-section-title">Correlation Map</div>', unsafe_allow_html=True)
    heatmap_fig, ax = plt.subplots(figsize=(7.5, 5.2))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(heatmap_fig)

    factor_left, factor_right = st.columns(2)
    with factor_left:
        st.subheader("📊 Price vs Sentiment")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Price"))
        fig.add_trace(go.Scatter(x=data.index, y=data["sentiment"], name="Sentiment"))
        st.plotly_chart(fig, width="stretch")

        st.subheader("📦 Price + Volume Confirmation")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Price"))
        fig.add_trace(go.Bar(x=data.index, y=data["Volume"], name="Volume"))
        st.plotly_chart(fig, width="stretch")

    with factor_right:
        st.subheader("🏦 Price vs Macro Drivers")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Price"))
        fig.add_trace(go.Scatter(x=data.index, y=data["macro_market"], name="S&P500"))
        fig.add_trace(go.Scatter(x=data.index, y=data["macro_volatility"], name="VIX"))
        st.plotly_chart(fig, width="stretch")

        st.subheader("📉 Volatility (Bollinger Width)")
        st.line_chart(data["BB_Width"])

with models_tab:
    st.markdown('<div class="ui-section-kicker">Forecast Engine</div><div class="ui-section-title">Model Comparison</div>', unsafe_allow_html=True)
    results = get_cached_model_results(asset_key, data_source, selected_year)
    if "Error" in results:
        st.warning(results["Error"])
    else:
        st.caption(
            f"Test window: {results['test_period_start']} to {results['test_period_end']} "
            f"across {results['test_points']} points"
        )

        metric_rows = []
        for model_name, values in results["metrics"].items():
            metric_rows.append(
                {
                    "Model": model_name,
                    "MAE": values["MAE"],
                    "RMSE": values["RMSE"],
                    "R2": values["R2"],
                }
            )

        metric_df = pd.DataFrame(metric_rows)
        best_rmse = metric_df.loc[metric_df["Model"] == results["best_model"], "RMSE"].iloc[0]

        better_ml_model = results.get("best_forecasting_model")
        comparison_label = "N/A"
        if better_ml_model:
            better_rmse = metric_df.loc[metric_df["Model"] == better_ml_model, "RMSE"].iloc[0]
            comparison_label = better_ml_model
            render_comparison_summary(
                [
                    ("Best Overall", results["best_model"]),
                    ("Best RMSE", f"{best_rmse:.4f}"),
                    ("ARIMA vs RF", comparison_label),
                ]
            )
            st.caption(f"Between ARIMA and Random Forest, **{better_ml_model}** has the lower RMSE of **{better_rmse}**.")
        else:
            render_comparison_summary(
                [
                    ("Best Overall", results["best_model"]),
                    ("Best RMSE", f"{best_rmse:.4f}"),
                    ("ARIMA vs RF", comparison_label),
                ]
            )

        st.dataframe(metric_df, width="stretch", hide_index=True)

        compare_fig = go.Figure()
        compare_fig.add_trace(go.Bar(x=metric_df["Model"], y=metric_df["MAE"], name="MAE"))
        compare_fig.add_trace(go.Bar(x=metric_df["Model"], y=metric_df["RMSE"], name="RMSE"))
        compare_fig.update_layout(
            barmode="group",
            template="plotly_dark",
            title="ARIMA vs Random Forest vs Benchmark",
            height=420,
        )
        st.plotly_chart(compare_fig, width="stretch")

        comparison_df = pd.DataFrame(results["comparison_frame"])
        forecast_fig = go.Figure()
        forecast_fig.add_trace(go.Scatter(x=comparison_df["Date"], y=comparison_df["Actual"], name="Actual"))
        forecast_fig.add_trace(go.Scatter(x=comparison_df["Date"], y=comparison_df["ARIMA"], name="ARIMA"))
        forecast_fig.add_trace(
            go.Scatter(
                x=comparison_df["Date"],
                y=comparison_df["Random Forest"],
                name="Random Forest",
            )
        )
        forecast_fig.update_layout(
            template="plotly_dark",
            title="Out-of-Sample Forecast Comparison",
            height=460,
        )
        st.plotly_chart(forecast_fig, width="stretch")

        if is_custom_asset:
            asset_summary = build_custom_asset_summary(asset_key)
            if not asset_summary.empty:
                if selected_year != "All Years" and "Year" in asset_summary.columns:
                    asset_summary = asset_summary[
                        asset_summary["Year"].astype(str).str.strip() == selected_year
                    ].copy()
                asset_summary = _prepare_summary_display(asset_summary)
                st.subheader("📘 Historical Forecast Summary")
                st.dataframe(asset_summary, width="stretch", hide_index=True)

with indicators_tab:
    ind_left, ind_right = st.columns(2)
    with ind_left:
        st.subheader("📈 RSI")
        st.line_chart(data["RSI"])

        st.subheader("📊 MACD")
        st.line_chart(data[["MACD", "MACD_SIGNAL"]])

        st.subheader("📉 Stochastic")
        st.line_chart(data[["STOCH_K", "STOCH_D"]])

        st.subheader("📦 Volume Analysis")
        st.bar_chart(data["Volume"])
        st.line_chart(data["Volume_MA20"])

    with ind_right:
        st.subheader("📊 CCI")
        st.line_chart(data["CCI"])

        st.subheader("🟢 Parabolic SAR")
        st.line_chart(data["PSAR"])

        st.subheader("🕯 Price Action Signals")
        if patterns:
            for p in patterns[-10:]:
                st.write(f"{p[0].date()} → {p[1]}")
        else:
            st.caption("No recent price action signals detected.")
