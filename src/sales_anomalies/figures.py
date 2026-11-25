"""
Plotly figures shared by the app and the README charts, in the portfolio look: teal for the data and the totals,
one copper mark for what matters (the selected anomaly, the main cause), mist for the rest, value labels at the bars.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from sales_anomalies import i18n
from sales_anomalies import theme as t
from sales_anomalies.decompose import Breakdown

CONFIG = {"displayModeBar": False}
PART_LABELS = {
    "en": {"volume_traffic": "Volume: visits", "volume_conversion": "Volume: conversion",
           "volume_share": "Volume: share of orders", "volume_basket": "Volume: lines per order",
           "prices": "Prices and mix", "discount": "Discount", "shipping": "Shipping", "payment": "Payment fees",
           "commission": "Commission", "single_account": "One customer account",
           "returns_cost_each": "Cost per return", "other": "Other parts"},
    "pl": {"volume_traffic": "Wolumen: odwiedziny", "volume_conversion": "Wolumen: konwersja",
           "volume_share": "Wolumen: udział w zamówieniach", "volume_basket": "Wolumen: linie na zamówienie",
           "prices": "Ceny i asortyment", "discount": "Rabat", "shipping": "Wysyłka", "payment": "Opłaty za płatności",
           "commission": "Prowizja", "single_account": "Jedno konto klienta",
           "returns_cost_each": "Koszt jednego zwrotu", "other": "Pozostałe części"},
}
TOTALS = {"en": ("Expected", "Actual"), "pl": ("Oczekiwane", "Rzeczywiste")}


def _num(size: int, color) -> dict:
    """Font of numbers on a chart (value labels, ticks)."""
    return {"family": t.NUM_STACK, "size": size, "color": color}


def _layout(fig: go.Figure, height: int, **extra) -> go.Figure:
    margin = extra.pop("margin", {"l": 8, "r": 8, "t": 12, "b": 8})
    fig.update_layout(
        height=height, margin=margin, paper_bgcolor=t.PAGE, plot_bgcolor=t.PAGE,
        font={"family": t.SANS_STACK, "size": 13, "color": t.TEXT}, showlegend=False,
        hoverlabel={"bgcolor": t.PAGE, "bordercolor": t.LINE,
                    "font": {"family": t.SANS_STACK, "color": t.INK, "size": 13}},
        barcornerradius=4, dragmode=False, **extra,
    )
    return fig


def part_label(name: str, lang: str = "en") -> str:
    labels = PART_LABELS[lang]
    if name in labels:
        return labels[name]
    if name.startswith("returns_"):
        reason = name.removeprefix("returns_")
        reason = i18n.REASONS_PL.get(reason, reason) if lang == "pl" else reason.replace("_", " ")
        return f"Zwroty: {reason}" if lang == "pl" else f"Returns: {reason}"
    if name.startswith("conversion_"):
        device = name.removeprefix("conversion_")
        device = i18n.DEVICES_PL.get(device, device) if lang == "pl" else device
        return f"Zamówienia: {device}" if lang == "pl" else f"Orders: {device}"
    return name


def bridge_data(b: Breakdown, lang: str = "en", small: float = 0.03) -> pd.DataFrame:
    """Rows of the bridge: expected, the parts (those under `small` of the change joined as other), actual."""
    change = abs(b.change_grosz) or 1
    rows, other = [], 0
    for part in b.parts:
        if abs(part.grosz) < small * change:
            other += part.grosz
        else:
            rows.append({"label": part_label(part.name, lang), "amount": part.grosz / 100, "cause": part.cause,
                         "kind": "part"})
    if other:
        rows.append({"label": PART_LABELS[lang]["other"], "amount": other / 100, "cause": "", "kind": "part"})
    expected = round(b.expected, 2)
    frame = pd.DataFrame(rows)
    frame["end"] = expected + frame["amount"].cumsum()
    frame["start"] = frame["end"] - frame["amount"]
    first, last = TOTALS[lang]
    head = pd.DataFrame([{"label": first, "amount": expected, "cause": "", "kind": "total", "start": 0.0,
                          "end": expected}])
    tail = pd.DataFrame([{"label": last, "amount": expected + b.change, "cause": "", "kind": "total", "start": 0.0,
                          "end": expected + b.change}])
    return pd.concat([head, frame, tail], ignore_index=True)


def bridge(b: Breakdown, lang: str = "en", height: int | None = None) -> go.Figure:
    """Expected value, the parts as floating bars (the main cause in copper), the actual value."""
    data = bridge_data(b, lang)
    low = data[["start", "end"]].min(axis=1)
    size = (data["end"] - data["start"]).abs()
    colors = [t.PRIMARY if kind == "total" else t.ACCENT if cause == b.main_cause else t.MIST
              for kind, cause in zip(data["kind"], data["cause"], strict=True)]
    labels = [i18n.money(a, lang) if kind == "total" else i18n.money(a, lang, sign=True)
              for kind, a in zip(data["kind"], data["amount"], strict=True)]
    fig = go.Figure()
    fig.add_bar(y=data["label"], x=size, base=low, orientation="h", marker={"color": colors}, text=labels,
                textposition="outside", cliponaxis=False,
                textfont=_num(13, [t.INK if c != t.MIST else t.TEXT for c in colors]),
                hovertemplate="%{y}<br><b>%{text}</b><extra></extra>")
    for index in range(len(data) - 1):
        level = float(data["end"].iloc[index])
        fig.add_shape(type="line", xref="x", yref="y", x0=level, x1=level, y0=index + 0.34, y1=index + 0.66,
                      line={"color": t.MUTED, "width": 1})
    left = min(0.0, float(data[["start", "end"]].min().min()))
    right = max(0.0, float(data[["start", "end"]].max().max()))
    span = (right - left) or 1.0
    fig.update_yaxes(autorange="reversed", showgrid=False, showline=False, ticks="", tickfont={"color": t.TEXT},
                     fixedrange=True, ticklabelstandoff=8)
    fig.update_xaxes(visible=False, range=[left - span * 0.02, right + span * 0.04], fixedrange=True)
    return _layout(fig, height or 40 * len(data) + 24, bargap=0.32, margin={"l": 8, "r": 96, "t": 12, "b": 8})


def timeline(table: pd.DataFrame, selected: int | None = None, lang: str = "en", height: int = 340) -> go.Figure:
    """Every anomaly on the calendar: how far it moved its metric (in % of expected), the dot sized by PLN."""
    data = table.copy()
    data["y"] = (data["change_percent"] * 100).clip(-100, 150)
    scale = np.sqrt(data["change"].abs() / max(data["change"].abs().max(), 1))
    data["size"] = 9 + 22 * scale
    others = data[data["anomaly_id"] != selected]
    chosen = data[data["anomaly_id"] == selected]
    fig = go.Figure()
    for frame, color, line in ((others, t.MIST, t.PAGE), (chosen, t.ACCENT, t.PAGE)):
        if frame.empty:
            continue
        hover = [f"{d}<br>{s} · {m}<br><b>{i18n.money(c, lang, sign=True)}</b> ({i18n.percent(p, lang)})<br>{cause}"
                 for d, s, m, c, p, cause in zip(frame["days"], frame["segment"], frame["metric"], frame["change"],
                                                 frame["change_percent"], frame["cause"], strict=True)]
        fig.add_scatter(x=frame["start"], y=frame["y"], mode="markers", customdata=frame["anomaly_id"],
                        marker={"size": frame["size"], "color": color, "line": {"color": line, "width": 1.5},
                                "opacity": 0.95},
                        hovertext=hover, hovertemplate="%{hovertext}<extra></extra>")
    fig.add_hline(y=0, line={"color": t.MUTED, "width": 1})
    fig.update_yaxes(ticksuffix="%", showgrid=True, gridcolor=t.LINE, zeroline=False, tickfont=_num(12, t.MUTED),
                     fixedrange=True, range=[-112, 162])
    fig.update_xaxes(showgrid=False, tickfont=_num(12, t.MUTED), fixedrange=True, showline=False, ticks="",
                     tickformat="%b %Y", dtick="M3")
    return _layout(fig, height, margin={"l": 8, "r": 8, "t": 12, "b": 8})


def series(window: pd.DataFrame, metric: str, lang: str = "en", height: int = 260) -> go.Figure:
    """The segment's metric around the anomaly: what was expected (dashed) and what happened, the days in copper."""
    fig = go.Figure()
    inside = window[window["inside"]]
    if not inside.empty:
        fig.add_vrect(x0=inside["date"].min() - pd.Timedelta(hours=12), x1=inside["date"].max() + pd.Timedelta(
            hours=12), fillcolor=t.COPPER[1], line_width=0, layer="below")
    unit = "" if metric == "conversion" else (" zł" if lang == "pl" else " PLN")
    fig.add_scatter(x=window["date"], y=window["expected"], mode="lines", line={"color": t.MUTED, "width": 1.5,
                                                                                 "dash": "dot"},
                    name=TOTALS[lang][0], hovertemplate="%{x|%d %b}<br>%{y:,.0f}" + unit + "<extra></extra>")
    fig.add_scatter(x=window["date"], y=window["observed"], mode="lines", line={"color": t.PRIMARY, "width": 2},
                    name=TOTALS[lang][1], hovertemplate="%{x|%d %b}<br><b>%{y:,.0f}</b>" + unit + "<extra></extra>")
    fig.update_yaxes(showgrid=True, gridcolor=t.LINE, zeroline=False, tickfont=_num(12, t.MUTED), fixedrange=True,
                     rangemode="tozero")
    fig.update_xaxes(showgrid=False, tickfont=_num(12, t.MUTED), fixedrange=True, tickformat="%d %b")
    return _layout(fig, height, margin={"l": 8, "r": 8, "t": 8, "b": 8})
