"""Pump Failure Early Warning: Streamlit front end for the model.

Run `python pump_risk.py` once to build outputs/, then `streamlit run app.py`.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

import pump_risk as pr

OUT = Path(__file__).resolve().parent / "outputs"
BLUE, LIGHT_BLUE, ORANGE, RED, INK, MUTED, GRID = "#2a78d6", "#a9c8ef", "#eb6834", "#b42318", "#101828", "#667085", "#eaecf0"
SOFT_GRID = "rgba(128,128,128,0.22)"        # readable on light and dark backgrounds
AREA_PALETTE = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189), (140, 86, 75),
    (227, 119, 194), (23, 190, 207), (188, 189, 34), (127, 127, 127), (0, 92, 175), (255, 187, 120),
    (152, 223, 138), (197, 176, 213),
]

st.set_page_config(page_title="Pump Failure Early Warning", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
  #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {visibility: hidden; height: 0;}
  [data-testid="stHeader"] {background: transparent; height: 0;}
  .block-container {padding-top: 1.2rem; padding-bottom: 1rem; max-width: 1500px;}
  h1 {font-size: 1.55rem !important; font-weight: 700 !important; margin: 0 !important;}
  .sub {color: #667085; font-size: 0.92rem; margin: 2px 0 10px;}
  [data-testid="stMetricValue"] {font-size: 1.3rem; font-weight: 650;}
  [data-testid="stMetric"] {padding: 8px 14px !important;}
  [data-testid="stMetricLabel"] p {font-size: 0.8rem; color: #667085;}
  .stTabs [data-baseweb="tab"] {font-size: 0.98rem; font-weight: 600; padding: 6px 4px;}
  .chip {display:inline-block; padding:2px 9px; border-radius:99px; font-size:0.78rem; font-weight:650; color:white;}
  .sign {margin: 2px 0; font-size: 0.9rem;}
  .muted {color: #667085; font-size: 0.85rem;}
  .legend {display:flex; align-items:center; gap:8px; font-size:0.8rem; color:#667085; margin:4px 0 0;}
  .bar {width:170px; height:9px; border-radius:5px;
        background:linear-gradient(90deg,rgb(200,245,200),rgb(245,160,50),rgb(130,0,0));}
  h3 {font-size: 1.1rem !important; margin-top: 0.4rem !important;}
  [data-testid="stVerticalBlock"] .stButton button {justify-content: flex-start; text-align: left;}
  .stButton button p {font-size: 0.9rem;}
  .step {color: #2a78d6; font-size: 0.78rem; font-weight: 700; letter-spacing: .06em; margin-top: 18px;}
  .q {font-size: 1.25rem; font-weight: 700; margin: 2px 0 4px;}
  .a {font-size: 1.02rem; opacity: 0.85; margin: 0 0 10px; max-width: 900px; line-height: 1.5;}
  .big {font-size: 2.1rem; font-weight: 750; line-height: 1.1;}
  .tile {border: 1px solid rgba(128,128,128,0.25); border-radius: 12px; padding: 14px 16px;
         background: rgba(128,128,128,0.06); height: 100%;}
  .tile .lbl {color: #667085; font-size: 0.85rem; margin-top: 4px;}
</style>
""", unsafe_allow_html=True)


def style(fig, height=320):
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=40, b=10),
                      font=dict(size=13), hoverlabel=dict(bgcolor="white", font_color=INK),
                      legend=dict(orientation="h", y=1.1, x=0))
    fig.update_xaxes(showgrid=False, linecolor=SOFT_GRID)
    fig.update_yaxes(gridcolor=SOFT_GRID, zeroline=False)
    return fig


@st.cache_data
def load():
    scores = pd.read_parquet(OUT / "scores.parquet")
    scores["d"] = pd.to_datetime(scores["d"])
    scores["next_event"] = pd.to_datetime(scores["next_event"])
    scores["area"] = scores["area"].astype(str).str.title()
    events = pd.read_parquet(OUT / "events.parquet")
    events["d"] = pd.to_datetime(events["d"])
    return dict(scores=scores, events=events,
                ev_costs=pd.read_parquet(OUT / "event_costs.parquet"),
                comp=pd.read_csv(OUT / "model_comparison.csv", index_col=0),
                imp=pd.read_csv(OUT / "feature_importance.csv", index_col=0),
                wf=pd.read_csv(OUT / "walk_forward.csv"),
                areas=pd.read_csv(OUT / "area_metrics.csv"),
                cal=pd.read_csv(OUT / "calibration.csv"),
                head=json.loads((OUT / "headline.json").read_text()))


POLICY_KS = (5, 10, 15, 20, 25, 30, 40, 50, 75, 100)


@st.cache_data(show_spinner=False)
def costs_for(netback):
    return pr.cost_table(load()["ev_costs"], netback)


@st.cache_data(show_spinner=False)
def picks(k, area=None):
    """Replay Jan-Sep 2026: each week visit the top-k wells not visited in the last 30 days (same rules as
    pr.simulate_policy). Which wells get visited doesn't depend on costs, so this runs once per k and the
    cost sliders only redo the arithmetic in policy()."""
    a, b = pr.SPLITS["test"]
    s = load()["scores"]
    test = s[(s["d"] >= a) & (s["d"] <= b) & s["y30"].notna()]
    if area:
        test = test[test["area"] == area]
    last, caught, actions = {}, {}, 0
    for day, wk in test.sort_values("score", ascending=False).groupby("d", sort=True):
        n = 0
        for row in wk.itertuples():
            if n >= k:
                break
            prev = last.get(row.uwi)
            if prev is not None and (day - prev).days < 30:
                continue
            last[row.uwi] = day
            n += 1
            actions += 1
            if row.y30 == 1:
                caught.setdefault((row.uwi, row.next_event), row.next_type)
    return {"actions": actions, "weeks": test["d"].nunique(),
            "events": test.loc[test["y30"] == 1].drop_duplicates(["uwi", "next_event"]).shape[0],
            "by_type": pd.Series(list(caught.values()), dtype=object).value_counts().to_dict()}


def policy(k, eff, check, netback, area=None):
    """Back-test on Jan-Sep 2026. With `area`, only that area's wells compete for its k visits."""
    p, saving = picks(k, area), costs_for(netback)["Saving if caught"]
    warned = sum(p["by_type"].values())
    net = eff * sum(n * saving.get(t, 0.0) for t, n in p["by_type"].items()) - p["actions"] * check
    return {"k": k, "weeks": p["weeks"], "actions": p["actions"], "events": p["events"], "events_warned": warned,
            "event_recall": warned / max(p["events"], 1), "action_precision": warned / max(p["actions"], 1),
            "net_saving_per_year": net * 52 / p["weeks"]}


def policy_curve(eff, check, netback):
    return pd.DataFrame([policy(k, eff, check, netback) for k in POLICY_KS])


def risk_rgb(p):
    """Very light green (won't fail) -> orange -> dark red (likely to fail in the next month); 45%+ is full dark red."""
    t = min(1.0, p / 0.45)
    lo, mid, hi = np.array([200, 245, 200]), np.array([245, 160, 50]), np.array([130, 0, 0])
    c = lo + (mid - lo) * (t * 2) if t < 0.5 else mid + (hi - mid) * ((t - 0.5) * 2)
    return [int(x) for x in c]


def hull(points):
    """Convex hull (monotone chain), pushed outward a little so the outline sits around the wells."""
    pts = sorted(set(map(tuple, points)))
    if len(pts) < 3:
        x, y = pts[0]
        return [[x - .02, y - .012], [x + .02, y - .012], [x + .02, y + .012], [x - .02, y + .012]]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    h = np.array(lower[:-1] + upper[:-1])
    c = h.mean(axis=0)
    v = h - c
    n = np.linalg.norm(v * [0.62, 1], axis=1, keepdims=True) + 1e-9
    return (h + v / n * [0.016, 0.010]).round(5).tolist()


def view_for(df):
    lat0, lat1 = df["lat"].quantile([.03, .97]) if len(df) > 30 else (df["lat"].min(), df["lat"].max())
    lon0, lon1 = df["lon"].quantile([.03, .97]) if len(df) > 30 else (df["lon"].min(), df["lon"].max())
    span = max(lat1 - lat0, (lon1 - lon0) * 0.62, 0.04)
    zoom = float(np.clip(np.log2(1.0 / span) + 6.9, 7.6, 12.0))
    return pdk.ViewState(latitude=(lat0 + lat1) / 2, longitude=(lon0 + lon1) / 2, zoom=zoom, pitch=0,
                         min_zoom=7.5, max_zoom=15, bearing=0)


def ratio(a, b):
    return a / b if pd.notna(a) and pd.notna(b) and b > 0 else np.nan


def warning_signs(r):
    """Plain-English warning signs: this week vs the well's own 90-day normal. Signs, not a diagnosis."""
    out = []
    hd, hd_n, hd3 = r["hd7"], r["hd90"], r["hd3"]
    if pd.notna(hd) and hd >= 1 and (pd.isna(hd_n) or hd >= 1.5 * max(hd_n, 0.1)):
        times = f", {hd / hd_n:.1f}× its normal" if pd.notna(hd_n) and hd_n >= 0.1 else ""
        out.append(f"Down **{hd:.1f} h/day** this week{times}")
    if pd.notna(hd3) and pd.notna(hd) and hd3 >= 2 and hd3 >= 1.3 * max(hd, 0.1):
        out.append(f"Downtime rising in the last 3 days (**{hd3:.1f} h/day**)")
    e = ratio(r["eff7"], r["eff90"])
    if pd.notna(e) and e <= 0.93:
        out.append(f"Pump efficiency **down {(1 - e):.0%}** vs normal")
    o = ratio(r["oil7"], r["oil90"])
    if pd.notna(o) and o <= 0.9:
        out.append(f"Oil **down {(1 - o):.0%}** vs normal ({r['oil7']:.1f} vs {r['oil90']:.1f} m³/d)")
    t = ratio(r["tq7"], r["tq90"])
    if pd.notna(t) and t >= 1.1:
        out.append(f"Rod torque **up {(t - 1):.0%}** vs normal")
    s = ratio(r["rpm7"], r["rpm90"])
    ups = int(r["rpm_up30"]) if pd.notna(r["rpm_up30"]) else 0
    if (pd.notna(s) and s >= 1.03) or ups >= 2:
        pct = f" {(s - 1):.0%}" if pd.notna(s) and s >= 1.03 else ""
        out.append(f"Pump speed **raised{pct}** ({ups} increase{'s' if ups != 1 else ''} in 30 days), "
                   "often a sign the pump is struggling")
    dh = int(r["dhdays30"]) if pd.notna(r["dhdays30"]) else 0
    if dh:
        out.append(f"**{dh} day{'s' if dh != 1 else ''}** with a downhole downtime code in the last 30 days")
    return out


def history_line(r):
    n = int(r["fails365"])
    dsf = r["days_since_fail"]
    last = {"RODS": "a rod", "PUMP": "a pump", "TUBING": "a tubing"}.get(str(r["last_ftype"]).upper(), "")
    if n == 0:
        return "No failures in the last 12 months"
    when = f", the last one {last} failure {int(dsf)} days ago" if pd.notna(dsf) and dsf < 3000 and last else ""
    return f"Failed **{n} time{'s' if n != 1 else ''}** in the last 12 months{when}"


def action(r):
    last = str(r["last_ftype"]).upper()
    if r["fails365"] >= 6:
        return "Root-cause review: this is a chronic well. Check the whole rod string, not just the last break."
    if last in ("PUMP", "TUBING"):
        return "Book a workover rig: its last failure needed one."
    return "Send a rod truck: check the rods and the pump speed."


D = load()
scores, events, comp, head = D["scores"], D["events"], D["comp"], D["head"]
latest_day = scores["d"].max()
latest = scores[scores["d"] == latest_day].sort_values("score", ascending=False).reset_index(drop=True)
latest = latest.dropna(subset=["lat", "lon"])
area_counts = latest["area"].value_counts()
areas = area_counts.index.tolist()
area_color = {a: AREA_PALETTE[i % len(AREA_PALETTE)] for i, a in enumerate(areas)}

st.markdown("<h1>Pump Failure Early Warning</h1>", unsafe_allow_html=True)
st.markdown(f'<div class="sub">Strathcona heavy-oil PCP wells · week of {latest_day:%d %b %Y} · '
            f'{len(latest):,} wells scored · risk = chance of a rod, pump or tubing failure in the next 4–30 days</div>',
            unsafe_allow_html=True)

# Only the open view is drawn (st.tabs would build both on every click).
TAB1, TAB2 = "This Monday", "Does it work, and is it worth it?"
view = st.segmented_control("View", [TAB1, TAB2], default=TAB1, key="view", label_visibility="collapsed")
if view is None:                       # clicking the open view again deselects it: stay where we were
    view = st.session_state.get("last_view", TAB1)
st.session_state["last_view"] = view

# ====================================================================== tab 1
if view == TAB1:
    c1, c2, c3 = st.columns([2, 2, 3])
    c3.markdown('<div style="height:28px"></div>', unsafe_allow_html=True)
    with c3.expander("ℹ️  How the model works"):
        sc = scores[(scores["d"] >= pr.SPLITS["test"][0]) & (scores["d"] <= pr.SPLITS["test"][1])
                    & scores["y30"].notna()]
        b1, b2 = sc[(sc["score"] >= 0.4) & (sc["score"] < 0.5)], sc[sc["score"] >= 0.5]
        st.markdown(f"""
The model works like a mechanic who has studied five years of records from **about 1,400 historical PCP wells** (2021–2026).

1. **Every Monday it checks each active well** against about 50 clues from the last few weeks: hours down, pump speed, rod torque, efficiency, oil, how often it has failed, and when its rods and pump were last replaced.
2. **It compares each well to its own normal.** "Down 6 hours a day" means little on its own; "down 6 hours a day when it's normally down 1" is a warning sign.
3. **It learned from history which patterns came before a failure**, using about 270,000 past well-weeks where the answer is known.
4. **It turns the clues into a probability** of a rod, pump or tubing failure 4–30 days from now. (Method: LightGBM, hundreds of small yes/no checklists that each nudge the risk up or down.)
5. **It ranks all ~{len(latest):,} active wells each Monday by that risk.** The top of the ranking is the visit list; the second tab shows how many wells are worth visiting once visit costs are counted.

**What a score means.** Checked on 2026, data the model never learned from:

| Model said | Well-weeks | Actually broke within 4–30 days |
|---|---|---|
| 40–50% | {len(b1):,} | {b1['y30'].mean():.0%} |
| 50% or more | {len(b2):,} | {b2['y30'].mean():.0%} |
| *Any well (average)* | {len(sc):,} | {sc['y30'].mean():.0%} |

So a well scored 55% really does break about half the time or more, roughly **4–5× the average well**. It tells you *which* wells are likely to break this month, not the exact day, and the warning signs are signs, not a confirmed broken part.
""")

    area_pick = c1.selectbox("Operating area", ["All areas"] + areas,
                             format_func=lambda a: a if a == "All areas" else f"{a} ({area_counts[a]} wells)")
    k = c2.slider("Wells visited this Monday, all areas combined" if area_pick == "All areas"
                  else f"Wells visited this Monday in {area_pick} only", 5, 50, 20, 5,
                  help="One shared list across every area, or, when an area is picked, that area's own crew list.")

    shown = (latest if area_pick == "All areas" else latest[latest["area"] == area_pick]).copy()
    shown["rank"] = np.arange(1, len(shown) + 1)
    shown["on_list"] = shown["rank"] <= k
    top = shown[shown["on_list"]]

    pol = policy(k, 0.5, 1500, pr.DEFAULT_NETBACK_BBL, None if area_pick == "All areas" else area_pick)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Wells to visit" + (", all areas" if area_pick == "All areas" else f", {area_pick}"), f"{len(top)}",
              border=True, help="The highest-risk wells: bigger dots with a dark outline on the map.")
    m2.metric("Expected to fail among them", f"≈ {top['score'].sum():.0f}", border=True,
              help="Sum of the model's failure probabilities. On the 2026 test these were well calibrated.")
    m3.metric("Chronic wells on the list", f"{int((top['fails365'] >= 6).sum())}", border=True,
              help="Failed 6+ times in the last 12 months: candidates for a root-cause review.")
    ny = pol["net_saving_per_year"]
    m4.metric("Back-tested net saving" + ("" if area_pick == "All areas" else f", {area_pick}"),
              f"{'' if ny >= 0 else '−'}${abs(ny) / 1e6:,.2f}M / yr", border=True,
              help=f"Jan–Sep 2026 back-test: {k} visits a week {'across all areas' if area_pick == 'All areas' else 'in ' + area_pick + ' only'}, "
                   "assuming $1,500 per visit and half of warned failures become planned work. "
                   "Negative = the visits cost more than they save at this list size.")

    # ---- map data
    shown["fill"] = [risk_rgb(p) + [235 if on else 175] for p, on in zip(shown["score"], shown["on_list"])]
    shown["line"] = [[20, 20, 20, 255] if on else [110, 110, 110, 150] for on in shown["on_list"]]
    shown["radius"] = np.where(shown["on_list"], 300.0, 110.0)
    shown["tip_risk"] = (shown["score"] * 100).round(0).astype(int).astype(str) + "%"
    shown["tip_list"] = np.where(shown["on_list"], "Visit #" + shown["rank"].astype(str) + " this Monday", "")
    shown["tip_ll"] = shown["lat"].map("{:.5f}°N".format) + ", " + shown["lon"].abs().map("{:.5f}°W".format)
    wells_df = shown.sort_values("score")[["uwi", "well_name", "area", "lon", "lat", "fill", "line", "radius",
                                           "tip_risk", "tip_list", "tip_ll"]]


    polys = []
    for a in areas:
        g = latest[latest["area"] == a]
        if len(g) >= 8:
            dist = np.hypot((g["lon"] - g["lon"].median()) * 0.62, g["lat"] - g["lat"].median())
            g = g[dist <= max(dist.quantile(0.85) * 1.4, 0.02)]
        if len(g) < 3 and area_pick != a:
            continue
        r, gg, b = area_color[a]
        sel = area_pick in ("All areas", a)
        poly = hull(g[["lon", "lat"]].values)
        top_pt = max(poly, key=lambda q: q[1])
        show_name = (a == area_pick) or (area_pick == "All areas" and area_counts[a] >= 70)
        polys.append({"poly": poly, "fill": [r, gg, b, 30 if sel else 8], "line": [r, gg, b, 200 if sel else 60],
                      "pos": [top_pt[0], top_pt[1] + 0.006], "name": a.upper() if show_name else ""})
    polys_df = pd.DataFrame(polys)

    # Selected well (clicked on the map or in the list): the map flies to it, like Google Maps.
    sel_uwi = st.session_state.get("well_uwi")
    sel_here = shown[shown["uwi"] == sel_uwi]
    if len(sel_here):
        s0 = sel_here.iloc[0]
        view = pdk.ViewState(latitude=float(s0["lat"]), longitude=float(s0["lon"]), zoom=12.3, pitch=0, bearing=0,
                             min_zoom=7.5, max_zoom=16, transitionDuration=900)
        sel_df = sel_here[["lon", "lat"]]
    else:
        view = view_for(shown if len(shown) else latest)
        sel_df = shown.iloc[0:0][["lon", "lat"]]

    layers = [
        pdk.Layer("PolygonLayer", polys_df, id="areas", get_polygon="poly", get_fill_color="fill",
                  get_line_color="line", line_width_min_pixels=1.5, stroked=True, filled=True, pickable=False),
        pdk.Layer("ScatterplotLayer", wells_df, id="wells", get_position=["lon", "lat"],
                  get_fill_color="fill", get_line_color="line", get_radius="radius",
                  radius_min_pixels=3, radius_max_pixels=12, stroked=True, line_width_min_pixels=1,
                  pickable=True, auto_highlight=True),
        pdk.Layer("TextLayer", polys_df[polys_df["name"] != ""], id="area_names", get_position="pos",
                  get_text="name", get_size=12, background=True, get_background_color=[255, 255, 255, 220],
                  background_padding=[4, 2], get_color=[52, 64, 84, 255], font_weight=700,
                  get_text_anchor='"middle"', get_alignment_baseline='"bottom"'),
        pdk.Layer("ScatterplotLayer", sel_df, id="selected", get_position=["lon", "lat"], get_radius=1,
                  radius_min_pixels=17, filled=False, stroked=True, get_line_color=[42, 120, 214, 255],
                  line_width_min_pixels=3, pickable=False),
    ]
    deck = pdk.Deck(
        layers=layers, initial_view_state=view,
        map_provider="carto", map_style="light",
        tooltip={"html": "<div style='font-family:monospace;font-size:14px;font-weight:700'>{uwi}</div>"
                         "{well_name}<br/>{area} · {tip_ll}<br/>Risk next 30 days: <b>{tip_risk}</b><br/>"
                         "<i>{tip_list}</i>",
                 "style": {"backgroundColor": "white", "color": "#101828", "fontSize": "12px",
                           "border": "1px solid #d0d5dd", "borderRadius": "8px"}},
    )

    left, right = st.columns([3, 2], gap="medium")
    with left:
        # A new key per selection makes the map re-centre on the selected well.
        event = st.pydeck_chart(deck, height=620, on_select="rerun", selection_mode="single-object",
                                key=f"map_{area_pick}_{sel_uwi if len(sel_here) else 'all'}")
        lg1, lg2 = st.columns([4, 1])
        lg1.markdown('<div class="legend"><span>Won\'t fail</span><span class="bar"></span>'
                     '<span>Likely to fail in the next month</span><span style="margin-left:12px">'
                     '<b>Big outlined dots</b> = this week\'s visits · click a dot or a well in the list to zoom to it'
                     '</span></div>',
                     unsafe_allow_html=True)
        if len(sel_here) and lg2.button("Show whole area", width="stretch"):
            st.session_state.pop("well_uwi", None)
            st.rerun()

        # ---- selected well history, under the map
        if sel_uwi in set(latest["uwi"]):
            w = scores[scores["uwi"] == sel_uwi].sort_values("d")
            ev = events[(events["uwi"] == sel_uwi) & (events["d"] >= w["d"].min())]
            st.markdown(f"### {sel_uwi} · history")
            h1, h2 = st.columns(2)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=w["d"], y=w["score"] * 100, mode="lines", name="Risk score",
                                     line=dict(color=BLUE, width=2), hovertemplate="%{x|%d %b %Y}: %{y:.0f}%<extra></extra>"))
            fig.add_trace(go.Scatter(x=ev["d"], y=[0] * len(ev), mode="markers", name="Actual failure",
                                     marker=dict(color=RED, size=10, symbol="x"), text=ev["ftype"],
                                     hovertemplate="%{x|%d %b %Y}: %{text} failure<extra></extra>"))
            fig.update_yaxes(title="Risk (%)", rangemode="tozero")
            fig.update_layout(title=dict(text="Weekly risk score vs actual failures", font=dict(size=14)))
            h1.plotly_chart(style(fig, 280), width="stretch")
            sig = {"Hours down per day": "hd7", "Oil rate (m³/d)": "oil7", "Rod torque (ft·lb)": "tq7",
                   "Pump speed (rpm)": "rpm7", "Pump efficiency": "eff7"}
            with h2:
                pick = st.selectbox("Pump signal", list(sig), label_visibility="collapsed")
                if w[sig[pick]].notna().sum():
                    f = go.Figure(go.Scatter(x=w["d"], y=w[sig[pick]], mode="lines", line=dict(color=BLUE, width=2),
                                             hovertemplate="%{x|%d %b %Y}: %{y:.2f}<extra></extra>"))
                    for d in ev["d"]:
                        f.add_vline(x=d, line=dict(color="rgba(180,35,24,0.35)", width=1))
                    f.update_layout(title=dict(text=f"{pick}, 7-day average · red lines = failures", font=dict(size=14)),
                                    showlegend=False)
                    st.plotly_chart(style(f, 240), width="stretch")
                else:
                    st.info(f"No {pick.lower()} recorded for this well.")
            st.caption("Each week is scored by a model trained only on data from before it (2025 by the 2021–24 "
                       "model, 2026 by a model retrained up to Nov 2025).")
    try:
        objs = event.selection.objects.get("wells", []) if event and event.selection else []
        if objs and objs[0]["uwi"] != sel_uwi:
            st.session_state["well_uwi"] = objs[0]["uwi"]
            st.rerun()
    except (AttributeError, KeyError, TypeError):
        pass

    costs = pr.cost_table(D["ev_costs"])
    cost_key = {"RODS": "Rods", "PUMP": "Pump", "TUBING": "Tubing"}

    def well_panel(r):
        rgb = risk_rgb(r["score"])
        st.markdown(f'<span class="chip" style="background:rgb({rgb[0]},{rgb[1]},{rgb[2]})">'
                    f'{r["score"]:.0%} risk, next 30 days</span> <span class="muted">&nbsp;{r["well_name"]} · '
                    f'{r["make"]} pump</span>', unsafe_allow_html=True)
        signs = warning_signs(r)
        st.markdown("**Warning signs**")
        st.markdown("\n".join(f"- {s}" for s in signs + [history_line(r)]))
        if not signs:
            st.caption("No unusual pump signals this week: the risk comes mostly from its failure history.")
        st.markdown(f"**Suggested action:** {action(r)}")
        t = cost_key.get(str(r["last_ftype"]).upper(), "Rods")
        if t in costs.index:
            st.markdown(f"**If it breaks** (typical {t.lower().rstrip('s')} failure): **${costs.loc[t, 'Reactive cost']:,.0f}** "
                        f"· **if fixed early:** ${costs.loc[t, 'Planned cost']:,.0f}")

    with right:
        st.markdown(f"### Monday visit list{'' if area_pick == 'All areas' else ' · ' + area_pick}")
        st.caption("Click a well to zoom the map to it and see what's wrong.")
        if len(sel_here) and not sel_here.iloc[0]["on_list"]:
            r = sel_here.iloc[0]
            with st.container(border=True):
                st.markdown(f"**Selected: {r['uwi']}** · {r['area']} · not on this week's list")
                well_panel(r)
        off_list = len(sel_here) and not sel_here.iloc[0]["on_list"]
        box = st.container(height=420 if off_list else 600, border=False)
        groups = [
            ("🔁 Chronic wells → engineering review",
             "Failed 6+ times in 12 months. Another quick repair won't hold: find the root cause.",
             top[top["fails365"] >= 6]),
            ("⚠️ New warnings → field check",
             "Not chronic, but at risk now. This is where the model beats the most-failed-wells shortcut.",
             top[top["fails365"] < 6]),
        ]
        with box:
            for title, note, g in groups:
                if not len(g):
                    continue
                st.markdown(f"**{title}** · {len(g)}")
                st.caption(note)
                for _, r in g.iterrows():
                    is_sel = r["uwi"] == sel_uwi
                    label = f"{'▾' if is_sel else '▸'}  #{r['rank']}  {r['uwi']} · {r['area']} · {r['score']:.0%}"
                    if st.button(label, key=f"pick_{r['uwi']}", width="stretch",
                                 type="primary" if is_sel else "secondary"):
                        if is_sel:
                            st.session_state.pop("well_uwi", None)     # click again to close
                        else:
                            st.session_state["well_uwi"] = r["uwi"]
                        st.rerun()
                    if is_sel:
                        with st.container(border=True):
                            well_panel(r)
        st.download_button("Download the list (CSV)",
                           top[["rank", "uwi", "well_name", "area", "score", "fails365", "days_since_fail",
                                "last_ftype", "make", "lat", "lon"]].to_csv(index=False),
                           f"visit_list_{latest_day:%Y%m%d}.csv", "text/csv")

# ====================================================================== tab 2
def tile(col, big, label):
    col.markdown(f'<div class="tile"><div class="big">{big}</div><div class="lbl">{label}</div></div>',
                 unsafe_allow_html=True)


def section(step, question, answer):
    st.markdown(f'<div class="step">{step}</div><div class="q">{question}</div><div class="a">{answer}</div>',
                unsafe_allow_html=True)


if view == TAB2:
    base_costs = pr.cost_table(D["ev_costs"])
    rod_after, rod_before = base_costs.loc["Rods", "Reactive cost"], base_costs.loc["Rods", "Planned cost"]
    n_model = head["model_top25_precision"] * 25
    n_rule = head["rule_top25_precision"] * 25
    n_rand = head["base_rate"] * 25
    wf = D["wf"]

    # ---- 1. the problem
    section("1 · THE PROBLEM", "Broken pumps are expensive, and they keep breaking.",
            f"In one year ({head['period']}) Strathcona's heavy-oil wells had <b>{head['failures']:,}</b> rod, pump "
            f"or tubing failures. Each one means a repair crew and days of lost oil. More than a quarter of them "
            f"broke again within a month of being fixed.")
    t1, t2, t3 = st.columns(3)
    tile(t1, f"${head['total_cost'] / 1e6:,.0f}M", "cost of failures in one year (repairs + lost oil)")
    tile(t2, f"{head['failures']:,}", "failures in one year")
    tile(t3, f"{head['repeat_30d_share']:.0%}", "broke again within 30 days of a repair")

    # ---- 2. can we see it coming
    section("2 · THE IDEA", "Can we tell which wells will break next month?",
            f"Every Monday the model ranks all ~1,000 wells by how likely they are to fail in the next 30 days. "
            f"We tested it on 2026, data it had never seen. <b>Of the 25 wells it flagged each week, about "
            f"{n_model:.0f} really did fail.</b> Picking 25 wells at random catches about {n_rand:.0f}; "
            f"the obvious shortcut, picking the wells that failed most last year, catches about {n_rule:.0f}.")
    c1, c2 = st.columns([3, 2])
    f = go.Figure(go.Bar(
        x=[n_rand, n_rule, n_model], y=["Pick 25 at random", "Pick the most-failed wells", "Our model"],
        orientation="h", marker=dict(color=[GRID, LIGHT_BLUE, BLUE], cornerradius=6),
        text=[f"{v:.0f} of 25" for v in (n_rand, n_rule, n_model)], textposition="outside",
        hovertemplate="%{y}: %{x:.1f} of 25<extra></extra>"))
    f.update_xaxes(range=[0, 25], title="Flagged wells that failed within a month (out of 25)")
    f.update_layout(title=dict(text="Out of 25 wells flagged each week, how many really failed? (2026)",
                               font=dict(size=14)), showlegend=False)
    c1.plotly_chart(style(f, 260), width="stretch")
    lo, hi = head["gap_vs_rule_ci95"]
    c2.markdown(f"""
<div class="a" style="margin-top:28px">
<b>About {n_model / n_rand:.1f}× better than guessing.</b><br/><br/>
It also beats the shortcut by {lo * 100:.0f}–{hi * 100:.0f} percentage points. We checked this isn't luck by
re-running the test on random mixes of the 38 test weeks 1,000 times.
</div>""", unsafe_allow_html=True)

    # ---- 3. not a fluke
    section("3 · IS IT RELIABLE?", "Does it work every year, or did we get lucky once?",
            "We replayed history: for each year, the model only learned from the years before it, then had to "
            "predict that year. <b>It beat the shortcut in every year</b>, and it gets better as it has more "
            "history to learn from.")
    f = go.Figure()
    f.add_trace(go.Bar(x=wf["year"].astype(str), y=wf["model_top25"] * 25, name="Our model", marker_color=BLUE,
                       text=[f"{v * 25:.0f}" for v in wf["model_top25"]], textposition="outside",
                       hovertemplate="%{x}: %{y:.1f} of 25<extra>Our model</extra>"))
    f.add_trace(go.Bar(x=wf["year"].astype(str), y=wf["rule_top25"] * 25, name="Most-failed wells shortcut",
                       marker_color=LIGHT_BLUE, text=[f"{v * 25:.0f}" for v in wf["rule_top25"]],
                       textposition="outside", hovertemplate="%{x}: %{y:.1f} of 25<extra>Shortcut</extra>"))
    f.add_trace(go.Scatter(x=wf["year"].astype(str), y=wf["base_rate"] * 25, name="Random pick",
                           mode="lines+markers", line=dict(color=MUTED, dash="dot")))
    f.update_yaxes(title="Flagged wells that failed (of 25)", range=[0, 18])
    f.update_layout(title=dict(text="Each year predicted using only the years before it", font=dict(size=14)),
                    barmode="group")
    st.plotly_chart(style(f, 300), width="stretch")
    am = D["areas"][D["areas"]["rows"] >= 2500].sort_values("model_auc")
    st.markdown(f'<div class="a">It works best in <b>{am.iloc[-1]["area"]}</b>, where failures are most common, '
                f'and is weakest in <b>{am.iloc[0]["area"]}</b>. Its percentages are honest too: when it says a '
                f'well has a 30% chance, about 30% of those wells fail.</div>', unsafe_allow_html=True)

    # ---- 4. worth it
    section("4 · IS IT WORTH IT?", "Is fixing a well early cheaper than waiting for it to break?",
            f"Yes. A typical rod failure costs about <b>${rod_after:,.0f}</b> when it breaks (emergency repair plus "
            f"lost oil), but about <b>${rod_before:,.0f}</b> if it's fixed as planned work. The catch: some visits are "
            f"to wells that weren't going to fail, and every visit costs money. Move the sliders to see when it pays.")
    s1, s2, s3 = st.columns(3)
    kk = s1.slider("Wells visited each week", 5, 100, 20, 5)
    check = s2.slider("Cost of one visit ($)", 0, 5000, 1500, 250,
                      help="A field check, speed change or diagnostic on a flagged well")
    eff = s3.slider("How many warned failures the visit prevents", 0.1, 1.0, 0.5, 0.05, format="%.2f",
                    help="0.5 = half of the failures we warn about get fixed before they break")
    with st.expander("Oil price assumption"):
        netback = st.slider("Netback: profit lost per barrel not produced ($/bbl)", 30.0, 90.0,
                            pr.DEFAULT_NETBACK_BBL, 0.5,
                            help="Strathcona Q3 2026: sales $99.47 − royalties $22.97 − energy $5.77 − transport $9.13")
    costs, one, curve = costs_for(netback), policy(kk, eff, check, netback), policy_curve(eff, check, netback)
    yearly = one["net_saving_per_year"]
    best = curve.loc[curve["net_saving_per_year"].idxmax()]
    w1, w2, w3 = st.columns(3)
    tile(w1, f"{'' if yearly >= 0 else '−'}${abs(yearly) / 1e6:,.2f}M",
         "saved per year, after paying for the visits" if yearly >= 0 else "lost per year: the visits cost more")
    tile(w2, f"{one['event_recall']:.0%}", "of failures get a warning at least 4 days ahead")
    tile(w3, f"{one['action_precision']:.0%}", "of visits are to a well that really was about to fail")
    f = go.Figure(go.Scatter(x=curve["k"], y=curve["net_saving_per_year"] / 1e6, mode="lines+markers",
                             line=dict(color=BLUE, width=2), marker=dict(size=7), fill="tozeroy",
                             fillcolor="rgba(42,120,214,0.08)",
                             hovertemplate="%{x} wells a week: $%{y:.2f}M a year<extra></extra>"))
    f.add_hline(y=0, line=dict(color=MUTED, width=1))
    f.add_vline(x=kk, line=dict(color=ORANGE, width=1, dash="dot"))
    f.update_xaxes(title="Wells visited each week")
    f.update_yaxes(title="Saved per year ($M)")
    f.update_layout(title=dict(text="How much is saved, depending on how many wells the team visits",
                               font=dict(size=14)))
    st.plotly_chart(style(f, 280), width="stretch")
    if best["net_saving_per_year"] > 0:
        st.markdown(f'<div class="a">With these assumptions the sweet spot is about <b>{int(best["k"])} wells a '
                    f'week</b> (${best["net_saving_per_year"] / 1e6:,.2f}M a year). Visit more and you start paying '
                    f'for trips to healthy wells. Tested on Jan–Sep 2026.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="a">With these assumptions visits cost more than they save: they need to be '
                    'cheaper or prevent more failures.</div>', unsafe_allow_html=True)

    # ---- right, wasted, missed: counted once per failure, per week
    wk = one["weeks"]
    caught = one["events_warned"] / wk
    wasted = (one["actions"] - one["events_warned"]) / wk
    missed = (one["events"] - one["events_warned"]) / wk
    st.markdown(f'<div class="q" style="font-size:1.05rem;margin-top:14px">What happens in an average week '
                f'with {kk} visits</div>', unsafe_allow_html=True)
    o1, o2, o3 = st.columns(3)
    tile(o1, f"{caught:.0f}", "✅ visits that catch a failure in time (true positives)")
    tile(o2, f"{wasted:.0f}", "⚠️ visits to a well that didn't fail in the next month (false alarms)")
    tile(o3, f"{missed:.0f}", f"❌ failures with no warning (missed), out of about {one['events'] / wk:.0f} a week")
    tst = scores[(scores["d"] >= pr.SPLITS["test"][0]) & (scores["d"] <= pr.SPLITS["test"][1]) & scores["y30"].notna()]
    healthy_wk = (tst["y30"] == 0).sum() / tst["d"].nunique()
    fp_share = wasted / healthy_wk
    st.markdown(f'<div class="a" style="margin-top:8px">Only about <b>{fp_share:.1%}</b> of healthy wells get a visit '
                f'(about 1 in {max(1, round(1 / fp_share)) if fp_share else "∞"}), and many "false alarms" are wells that '
                f'broke a few days too early to count or a few weeks later. What limits us is how many wells a crew can afford to visit: about '
                f'{one["events"] / wk:.0f} pumps fail a week, and {kk} truck visits can\'t reach them all.</div>',
                unsafe_allow_html=True)

    # ---- cheaper screening
    truck = policy(20, eff, 1500, netback)
    remote = policy(100, eff, 200, netback)
    st.markdown('<div class="q" style="font-size:1.05rem;margin-top:14px">A cheaper way to check catches far more'
                '</div><div class="a">If an engineer first reviews the top 100 wells\' data remotely (about $200 '
                'each, an assumption to confirm), only the worrying ones need a truck. Same model, same data:</div>',
                unsafe_allow_html=True)
    cmp_df = pd.DataFrame({
        "": ["Truck visits only", "Remote screen first"],
        "Wells checked a week": ["20 at $1,500", "100 at $200"],
        "Failures warned in time": [f"{truck['event_recall']:.0%}", f"{remote['event_recall']:.0%}"],
        "Missed a week": [f"{(truck['events'] - truck['events_warned']) / truck['weeks']:.0f}",
                          f"{(remote['events'] - remote['events_warned']) / remote['weeks']:.0f}"],
        "Saved a year": [f"${truck['net_saving_per_year'] / 1e6:,.1f}M", f"${remote['net_saving_per_year'] / 1e6:,.1f}M"],
    })
    st.dataframe(cmp_df, hide_index=True, width="stretch")
    st.caption("Uses the prevention rate and oil price set above. Back-test on Jan–Sep 2026.")

    # ---- 5. bigger lever
    section("5 · THE BIGGER PRIZE", "The real money is in the wells that keep breaking.",
            f"Repeat failures within a month cost about <b>${head['repeat_30d_cost'] / 1e6:,.1f}M a year</b>. "
            f"On this week's list, {int((latest.head(20)['fails365'] >= 6).sum())} of the top 20 wells have failed "
            f"6 or more times in a year. For those, another quick repair isn't enough: they need a root-cause "
            f"review. The weekly list is how those wells get found.")

    # ---- 6. fair test
    section("6 · CAN WE TRUST THE TEST?", "How we made sure the results aren't too good to be true.", "")
    st.markdown("""
- **No peeking at the future.** The model only ever sees data from before the Monday it's predicting.
- **Tested on data it never saw.** It learned from data up to Nov 2025 and was graded on Jan–Sep 2026.
- **A warning only counts if it's early.** The failure has to happen at least 4 days after the warning, so spotting a well that's already broken earns nothing.
- **Checked against the official failure log**, not just our own reconstruction of past failures, and the results hold.
""")
    with st.expander("Technical details (for the judges)"):
        st.markdown("**All methods on the 2026 test**")
        show = comp[["ROC AUC", "PR AUC", "Precision top 25/wk", "Recall top 25/wk", "Precision top 50/wk"]]
        st.dataframe(show.style.format("{:.2f}"), width="stretch")
        st.caption(f"Scored only against the official failure log, the final model's ROC AUC is "
                   f"{comp['ROC AUC vs official log only'].dropna().iloc[0]:.2f}. AUC: 0.5 = random, 1.0 = perfect.")
        st.markdown("**By operating area (2026 test)**")
        st.dataframe(D["areas"].rename(columns={"area": "Area", "rows": "Well-weeks", "base_rate": "Failure rate",
                                                "model_auc": "Model AUC", "rule_auc": "Shortcut AUC"})
                     .style.format({"Failure rate": "{:.0%}", "Model AUC": "{:.2f}", "Shortcut AUC": "{:.2f}",
                                    "Well-weeks": "{:,}"}), hide_index=True, width="stretch")
        st.markdown("**Cost per failure type**")
        st.dataframe(costs[["Events", "Share needing a rig", "Reactive cost", "Planned cost", "Saving if caught"]]
                     .style.format({"Share needing a rig": "{:.0%}", "Reactive cost": "${:,.0f}",
                                    "Planned cost": "${:,.0f}", "Saving if caught": "${:,.0f}"}), width="stretch")
        st.markdown("**What the model relies on most**")
        st.dataframe(D["imp"]["share_of_gain"].head(10).rename(index=lambda i: pr.FEATURE_LABELS.get(i, i))
                     .to_frame("Share of model gain").style.format("{:.1%}"), width="stretch")
        st.caption("Walk-forward: 2025 scored by the 2021–24 model; 2026 by a model retrained to Nov 2025. Weeks with "
                   "a failure in the next 1–3 days are removed from training only, never from the test.")
