"""Pump failure early warning for heavy-oil PCP wells.

Data preparation, features, model, evaluation and cost logic. The Streamlit app
reads the outputs this file writes, so every number comes from one place.

Data files expected in DATA_DIR (names as in the Dataset Handbook):
    daily_production.parquet, well_jobs.parquet, production_failures.parquet
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"

# --------------------------------------------------------------------------
# 1. Failure events
# --------------------------------------------------------------------------
# The failure log (production_failures) only starts in mid-2025. 99% of logged
# failures match a repair job in well_jobs on the same day, so repair jobs give
# us failure history back to 2021. These job types matched a logged failure in
# 64-96% of cases from June 2025 on.
REPAIR_JOB_TYPES = {
    "Corod Rod Repair": "Rods", "Stick Rod Repair": "Rods", "Rod Repair": "Rods",
    "Polish Rod Repair": "Rods", "Rod Break - Corod": "Rods", "Rod Break - Stick": "Rods",
    "Corod Rod  Replacement - New String": "Rods", "Stick Rod Replacement - New String": "Rods",
    "Pump Repair/Replacement": "Pump", "Pump Repair": "Pump", "PCP Replacement": "Pump",
    "Tubing Repair": "Tubing", "Packer Repair": "Tubing",
}
PLANNED_OBJECTIVE = "as per program|upgrade"  # planned pump changes are not failures
MERGE_DAYS = 3        # log rows and job rows within 3 days on one well = one event
LEAD_DAYS = 4         # a warning must come at least 4 days before the event
HORIZON_DAYS = 30     # predict failures 4-30 days ahead (also 4-14)


def connect(data_dir: Path | str = DATA_DIR, db_path: str = ":memory:") -> duckdb.DuckDBPyConnection:
    data_dir = Path(data_dir)
    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=2")
    con.execute(f"create or replace view d as select * from read_parquet('{data_dir / 'daily_production.parquet'}')")
    con.execute(f"create or replace view j as select * from read_parquet('{data_dir / 'well_jobs.parquet'}')")
    con.execute(f"create or replace view f as select * from read_parquet('{data_dir / 'production_failures.parquet'}')")
    return con


def build_events(con) -> pd.DataFrame:
    """One row per failure event: official log rows plus repair jobs, merged."""
    con.execute("create or replace table repair_map(job_type varchar, ftype varchar)")
    con.executemany("insert into repair_map values (?, ?)", list(REPAIR_JOB_TYPES.items()))
    con.execute(f"""
    create or replace table ev_raw as
    select uwi, failure_start_date as d,
      case when failure_type ilike '%rod%' or failure_type ilike '%broken%' then 'Rods'
           when failure_type ilike '%pump%' or failure_type ilike '%pcp%' then 'Pump'
           when failure_type ilike '%tub%' or failure_type ilike '%tbg%' or failure_type ilike '%packer%' then 'Tubing'
           else 'Other' end as ftype,
      cause, 'log' as src, null::double as job_cost, null::varchar as job_category
    from f where failure_start_date is not null
    union all
    select j.uwi, j.start_date, m.ftype, null, 'job', j.field_est_total, j.job_category
    from j join repair_map m using (job_type)
    where j.start_date is not null
      and not regexp_matches(lower(coalesce(j.objective, '')), '{PLANNED_OBJECTIVE}')""")
    con.execute(f"""
    create or replace table events as
    with o as (select *, row_number() over (partition by uwi order by d, src desc) rn
               from ev_raw where ftype <> 'Other'),
    g as (select *, case when datediff('day', lag(d) over (partition by uwi order by d, rn), d) <= {MERGE_DAYS}
                         then 0 else 1 end as newgrp from o),
    h as (select *, sum(newgrp) over (partition by uwi order by d, rn rows unbounded preceding) gid from g)
    select uwi, min(d) as d,
           coalesce(arg_min(ftype, d) filter (where src = 'log'), arg_min(ftype, d)) as ftype,
           max((src = 'log')::int) as in_log,
           any_value(cause) filter (where cause is not null) as cause,
           sum(job_cost) as repair_cost,
           max(case when job_category = 'Workover' then 1 else 0 end) as needed_rig
    from h group by uwi, gid""")
    return con.execute("select * from events order by uwi, d").df()


# --------------------------------------------------------------------------
# 2. Daily signals and weekly snapshots
# --------------------------------------------------------------------------
def _window(n: int) -> str:
    return f"over (partition by uwi order by d range between interval {n - 1} days preceding and current row)"


def build_snapshots(con, start: str = "2021-04-05", end: str | None = None) -> pd.DataFrame:
    """One row per producing PCP well per Monday, with trailing features and labels.

    Features use only data up to and including the snapshot day.
    Labels: y30 = a failure event 4-30 days later, y14 = 4-14 days later.
    Labels are left empty when the window runs past the end of the data.
    """
    con.execute("create or replace table pcp_wells as select distinct uwi from d where pcp_pump_rpm > 0")
    con.execute("""
    create or replace table dd as
    select uwi, production_date d, well_daily_status st, field_operating_area area,
      nullif(pcp_pump_rpm, 0) rpm,
      case when pcp_pump_rod_torque_ftlb > 0 and pcp_pump_rod_torque_ftlb < 20000 then pcp_pump_rod_torque_ftlb end tq,
      case when pump_efficiency_pct > 0 and pump_efficiency_pct < 2 then pump_efficiency_pct end eff,
      nullif(pcp_pump_electric_current_a, 0) amps,
      case when hours_on_prod > 0 then oil_produced_m3 end oil_on,
      oil_produced_m3 oil, water_produced_m3 wat,
      case when casing_pressure_kpa between 0 and 20000 then casing_pressure_kpa end csg,
      case when tubing_pressure_kpa between 0 and 20000 then tubing_pressure_kpa end tbg,  -- 999999 = placeholder
      coalesce(hours_down, 0) hd, coalesce(lost_oil_m3, 0) lost,
      regexp_matches(coalesce(downtime_code_1, ''), '(?i)downhole|rod|pump failure|tubing')::int dh_code,
      regexp_matches(coalesce(downtime_code_1, ''), '(?i)torqued')::int torqued,
      pcp_pump_displacement_m3d100rpm disp, pcp_pump_make make, on_production_date onprod,
      well_name, surface_latitude lat, surface_longitude lon
    from d where uwi in (select uwi from pcp_wells)
    -- a few well-days appear twice in the raw file: keep one row per well per day
    qualify row_number() over (partition by uwi, production_date order by hours_on_prod desc nulls last) = 1""")
    # Operator speed changes (raising speed is a sign of a struggling pump) and a day number for trend slopes.
    con.execute("""
    create or replace table dd as
    select *, coalesce(rpm <> lag_rpm, false)::int rpm_chg, coalesce(rpm > lag_rpm, false)::int rpm_up,
           datediff('day', date '2020-01-01', d)::double dn
    from (select *, lag(rpm) over (partition by uwi order by d) lag_rpm from dd)""")
    w = _window
    con.execute(f"""
    create or replace table roll as
    select uwi, d, st, area, make, disp, onprod, well_name, lat, lon,
      avg(rpm) {w(7)} rpm7, avg(rpm) {w(90)} rpm90, max(rpm) {w(30)} rpmmax30, count(rpm) {w(7)} nrpm7,
      avg(tq) {w(7)} tq7, avg(tq) {w(90)} tq90, max(tq) {w(7)} tqmax7, stddev(tq) {w(30)} tqsd30,
      avg(eff) {w(7)} eff7, avg(eff) {w(90)} eff90,
      avg(amps) {w(7)} amps7, avg(amps) {w(90)} amps90,
      avg(oil_on) {w(7)} oil7, avg(oil_on) {w(90)} oil90,
      sum(wat) {w(30)} wat30, sum(oil) {w(30)} oilsum30,
      avg(csg) {w(7)} csg7, avg(csg) {w(90)} csg90, avg(tbg) {w(7)} tbg7,
      avg(hd) {w(7)} hd7, avg(hd) {w(30)} hd30, sum((hd >= 1)::int) {w(30)} downdays30,
      sum(lost) {w(30)} lost30, sum(dh_code) {w(30)} dhdays30, sum(torqued) {w(90)} torq90,
      -- short-term: last 3 days, 14-day trends, speed changes
      avg(rpm) {w(3)} rpm3, avg(tq) {w(3)} tq3, avg(eff) {w(3)} eff3, avg(oil_on) {w(3)} oil3,
      avg(hd) {w(3)} hd3, avg(hd) {w(90)} hd90,
      regr_slope(hd, dn) {w(14)} hd_slope14, regr_slope(eff, dn) {w(14)} eff_slope14,
      regr_slope(tq, dn) {w(14)} tq_slope14, regr_slope(oil_on, dn) {w(14)} oil_slope14,
      sum(rpm_chg) {w(30)} rpm_chg30, sum(rpm_up) {w(30)} rpm_up30
    from dd""")
    end_clause = f"and d <= '{end}'" if end else ""
    con.execute(f"""
    create or replace table snap as select * from roll
    where dayofweek(d) = 1 and st = 'Producing' and nrpm7 > 0 and d >= '{start}' {end_clause}""")
    last_day = con.execute("select max(production_date) from d").fetchone()[0]
    con.execute(f"""
    create or replace table snap2 as
    select s.*,
      case when s.d + {HORIZON_DAYS} <= date '{last_day}' then
        exists(select 1 from events e where e.uwi = s.uwi and e.d between s.d + {LEAD_DAYS} and s.d + {HORIZON_DAYS})::int end y30,
      case when s.d + 14 <= date '{last_day}' then
        exists(select 1 from events e where e.uwi = s.uwi and e.d between s.d + {LEAD_DAYS} and s.d + 14)::int end y14,
      case when s.d + {HORIZON_DAYS} <= date '{last_day}' then
        exists(select 1 from events e where e.uwi = s.uwi and e.in_log = 1
               and e.d between s.d + {LEAD_DAYS} and s.d + {HORIZON_DAYS})::int end y30_log,
      (select min(e.d) from events e where e.uwi = s.uwi and e.d >= s.d + {LEAD_DAYS} and e.d <= s.d + {HORIZON_DAYS}) next_event,
      (select arg_min(e.ftype, e.d) from events e where e.uwi = s.uwi and e.d >= s.d + {LEAD_DAYS} and e.d <= s.d + {HORIZON_DAYS}) next_type,
      -- in_event_past is known on the Monday; in_event_next looks 1-3 days ahead and is used ONLY to clean training rows
      exists(select 1 from events e where e.uwi = s.uwi and e.d between s.d - 3 and s.d)::int in_event_past,
      exists(select 1 from events e where e.uwi = s.uwi and e.d between s.d + 1 and s.d + 3)::int in_event_next,
      (select datediff('day', max(j.start_date), s.d) from j where j.uwi = s.uwi and j.start_date < s.d
         and j.job_type ilike '%new string%') days_since_string,
      (select datediff('day', max(j.start_date), s.d) from j where j.uwi = s.uwi and j.start_date < s.d
         and j.job_type in ('PCP Replacement', 'Pump Repair/Replacement', 'Pump Repair')) days_since_pump,
      (select datediff('day', max(e.d), s.d) from events e where e.uwi = s.uwi and e.d < s.d - 3) days_since_fail,
      (select count(*) from events e where e.uwi = s.uwi and e.d between s.d - 365 and s.d - 4) fails365,
      (select count(*) from events e where e.uwi = s.uwi and e.d between s.d - 90 and s.d - 4) fails90,
      (select count(*) from events e where e.uwi = s.uwi and e.d < s.d - 3) fails_all,
      (select arg_max(e.ftype, e.d) from events e where e.uwi = s.uwi and e.d < s.d - 3) last_ftype,
      (select datediff('day', max(j.start_date), s.d) from j where j.uwi = s.uwi and j.start_date < s.d) days_since_job,
      (select coalesce(sum(j.field_est_total), 0) from j where j.uwi = s.uwi and j.start_date between s.d - 365 and s.d - 1) cost365
    from snap s""")
    # Drop weeks where the well failed in the last 3 days (known on the Monday). Weeks with a failure in the
    # next 1-3 days stay in: in live use we can't know that. split() removes them from training only.
    # Fixed row order so training is reproducible (DuckDB returns rows in parallel order).
    df = con.execute("select * from snap2 where in_event_past = 0 order by d, uwi").df()
    return add_features(df)


FEATURES = [
    # pump signals (this week vs the well's own 90-day baseline)
    "rpm7", "rpm_ratio", "rpmmax30", "tq7", "tq_ratio", "tqmax7", "tq_cv30", "tq_per_rpm",
    "eff7", "eff_ratio", "amps7", "amps_ratio",
    # production and pressures
    "oil7", "oil90", "oil_ratio", "wc30", "csg7", "csg_ratio", "tbg7",
    # downtime
    "hd7", "hd30", "downdays30", "lost30", "dhdays30", "torq90",
    # equipment and history
    "disp", "well_age_days", "days_since_fail", "fails365", "fails90", "last_ftype",
    "days_since_job", "cost365", "area", "make", "month",
]
FEATURES_V1 = list(FEATURES)   # the original feature set, kept for the before/after comparison
# New feature groups, each tested on the walk-forward back-test before being kept.
NEW_EQUIPMENT = ["days_since_string", "days_since_pump"]
NEW_SHORT = ["hd3", "hd_jump", "rpm3_ratio", "tq3_ratio", "eff3_ratio", "oil3_ratio", "hd90",
             "hd_slope14", "eff_slope14", "tq_slope14", "oil_slope14"]
NEW_SPEED = ["rpm_chg30", "rpm_up30"]
NEW_CHRONIC = ["fail_rate_all"]   # tested, no consistent gain: not used
# Final set: equipment age + short-term signals + speed changes gave a small, consistent gain on the
# 2023-2026 walk-forward test (top-25 precision 46.2% -> 47.0%), and they power the plain-English reasons.
FEATURES = FEATURES_V1 + NEW_EQUIPMENT + NEW_SHORT + NEW_SPEED
CATEGORICAL = ["area", "make", "last_ftype"]
HISTORY = ["days_since_fail", "fails365", "fails90", "last_ftype", "days_since_job", "cost365",
           "days_since_string", "days_since_pump", "fail_rate_all"]

FEATURE_LABELS = {
    "fails365": "Failures in last 12 months", "last_ftype": "Type of last failure",
    "days_since_fail": "Days since last failure", "oil90": "Oil rate, 90-day avg", "area": "Operating area",
    "wc30": "Water cut, 30 days", "days_since_job": "Days since last well job", "cost365": "Job spend, last 12 months",
    "dhdays30": "Days with downhole downtime code, 30 d", "well_age_days": "Well age", "rpmmax30": "Max pump speed, 30 d",
    "eff7": "Pump efficiency, 7 d", "disp": "Pump displacement", "hd7": "Hours down per day, 7 d", "oil7": "Oil rate, 7 d",
    "csg7": "Casing pressure, 7 d", "rpm7": "Pump speed, 7 d", "tbg7": "Tubing pressure, 7 d",
    "eff_ratio": "Efficiency vs own baseline", "hd30": "Hours down per day, 30 d", "tq7": "Rod torque, 7 d",
    "tq_ratio": "Torque vs own baseline", "oil_ratio": "Oil rate vs own baseline", "make": "Pump make",
    "fails90": "Failures in last 90 days", "lost30": "Lost oil, 30 d", "amps7": "Motor current, 7 d",
    "tq_per_rpm": "Torque per rpm", "tq_cv30": "Torque variability", "rpm_ratio": "Speed vs own baseline",
    "downdays30": "Days with any downtime, 30 d", "torq90": "Days 'well torqued', 90 d", "month": "Month",
    "csg_ratio": "Casing pressure vs baseline", "amps_ratio": "Current vs baseline", "tqmax7": "Max torque, 7 d",
    "days_since_string": "Days since new rod string", "days_since_pump": "Days since pump change",
    "hd3": "Hours down per day, 3 d", "hd_jump": "Downtime jump, last 3 days", "rpm3_ratio": "Speed, 3 d vs baseline",
    "tq3_ratio": "Torque, 3 d vs baseline", "eff3_ratio": "Efficiency, 3 d vs baseline",
    "oil3_ratio": "Oil, 3 d vs baseline", "hd90": "Hours down per day, 90 d", "hd_slope14": "Downtime trend, 14 d",
    "eff_slope14": "Efficiency trend, 14 d", "tq_slope14": "Torque trend, 14 d", "oil_slope14": "Oil trend, 14 d",
    "rpm_chg30": "Speed changes, 30 d", "rpm_up30": "Speed increases, 30 d", "fail_rate_all": "Long-run failure rate",
}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["d"] = pd.to_datetime(df["d"])

    def ratio(a, b):
        return df[a] / df[b].replace(0, np.nan)

    df["rpm_ratio"] = ratio("rpm7", "rpm90")
    df["tq_ratio"] = ratio("tq7", "tq90")
    df["eff_ratio"] = ratio("eff7", "eff90")
    df["amps_ratio"] = ratio("amps7", "amps90")
    df["oil_ratio"] = ratio("oil7", "oil90")
    df["csg_ratio"] = ratio("csg7", "csg90")
    df["tq_per_rpm"] = ratio("tq7", "rpm7")
    df["tq_cv30"] = ratio("tqsd30", "tq90")
    df["wc30"] = df["wat30"] / (df["wat30"] + df["oilsum30"]).replace(0, np.nan)
    df["well_age_days"] = (df["d"] - pd.to_datetime(df["onprod"])).dt.days
    df["days_since_fail"] = df["days_since_fail"].fillna(3000)
    df["days_since_job"] = df["days_since_job"].fillna(3000)
    df["days_since_string"] = df["days_since_string"].fillna(3000)
    df["days_since_pump"] = df["days_since_pump"].fillna(3000)
    # long-run failure rate per year, over the well's observed history (data starts 2021-01-01)
    seen_from = pd.to_datetime(df["onprod"]).clip(lower=pd.Timestamp("2021-01-01")).fillna(pd.Timestamp("2021-01-01"))
    years = ((df["d"] - seen_from).dt.days / 365.25).clip(lower=0.5)
    df["fail_rate_all"] = df["fails_all"] / years
    df["hd_jump"] = df["hd3"] - df["hd30"]          # hours down: last 3 days vs last 30
    df["rpm3_ratio"] = ratio("rpm3", "rpm90")
    df["tq3_ratio"] = ratio("tq3", "tq90")
    df["eff3_ratio"] = ratio("eff3", "eff90")
    df["oil3_ratio"] = ratio("oil3", "oil90")
    df["month"] = df["d"].dt.month
    for c in CATEGORICAL:
        df[c] = df[c].fillna("NA").astype(str).str.upper().astype("category")
    return df


# --------------------------------------------------------------------------
# 3. Model and evaluation
# --------------------------------------------------------------------------
# Time-based split with a gap longer than the label window, so no training
# label overlaps the validation or test period.
SPLITS = {
    "train": ("2021-04-05", "2024-11-30"),
    "val": ("2025-01-06", "2025-11-30"),
    "test": ("2026-01-05", "2026-09-28"),
}
LGB_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=200,
                  feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5,
                  verbose=-1, seed=7, deterministic=True, force_col_wise=True, num_threads=2)


def split(df: pd.DataFrame, target: str = "y30") -> dict[str, pd.DataFrame]:
    out = {}
    for name, (a, b) in SPLITS.items():
        part = df[(df["d"] >= a) & (df["d"] <= b)]
        if name == "train" and "in_event_next" in part:
            part = part[part["in_event_next"] == 0]   # clean training labels; never applied to val/test
        out[name] = part[part[target].notna()]
    return out


def train_lgb(tr: pd.DataFrame, va: pd.DataFrame, features=FEATURES, target="y30"):
    import lightgbm as lgb
    dtr = lgb.Dataset(tr[features], tr[target].astype(int))
    dva = lgb.Dataset(va[features], va[target].astype(int))
    model = lgb.train(LGB_PARAMS, dtr, 3000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
    return model


def predict(model, df: pd.DataFrame, features=FEATURES) -> np.ndarray:
    return model.predict(df[features], num_iteration=model.best_iteration)


def topk_by_week(df: pd.DataFrame, score: np.ndarray, k: int, target: str = "y30"):
    d = df.assign(_s=score)
    top = d.sort_values("_s", ascending=False).groupby("d").head(k)
    return top[target].mean(), top[target].sum() / d[target].sum()


def metrics(df: pd.DataFrame, score: np.ndarray, target: str = "y30") -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = df[target].astype(int)
    out = {"ROC AUC": roc_auc_score(y, score), "PR AUC": average_precision_score(y, score),
           "Base rate": y.mean()}
    for k in (25, 50, 100):
        p, r = topk_by_week(df, score, k, target)
        out[f"Precision top {k}/wk"] = p
        out[f"Recall top {k}/wk"] = r
    return out


# --------------------------------------------------------------------------
# 4. Costs from the data
# --------------------------------------------------------------------------
M3_TO_BBL = 6.2898
# Q3 2026 Strathcona: sales net of blending $99.47/boe, royalties $22.97, energy opex $5.77,
# transport $9.13 -> about $61.60/bbl of margin is lost when a barrel is not produced.
DEFAULT_NETBACK_BBL = 61.60


def event_costs(con) -> pd.DataFrame:
    """Repair cost, downtime and lost oil per failure event (2025-06 onward, logged events)."""
    return con.execute("""
    with e as (select * from events where in_log = 1 and d >= '2025-06-01'),
    x as (
      select e.uwi, e.d, e.ftype, e.repair_cost, e.needed_rig,
        (select sum(coalesce(lost_oil_m3, 0)) from d where d.uwi = e.uwi and d.production_date between e.d - 3 and e.d + 10) lost_m3,
        (select sum(coalesce(hours_down, 0)) from d where d.uwi = e.uwi and d.production_date between e.d - 3 and e.d + 10) down_hrs,
        (select avg(oil_produced_m3) from d where d.uwi = e.uwi and d.production_date between e.d - 37 and e.d - 8 and hours_on_prod > 0) pre_rate
      from e)
    select * from x""").df()


def cost_table(ev: pd.DataFrame, netback_bbl: float = DEFAULT_NETBACK_BBL) -> pd.DataFrame:
    """Reactive vs planned cost per failure type, built from the data."""
    nb_m3 = netback_bbl * M3_TO_BBL
    rows = []
    for t, g in ev.groupby("ftype"):
        planned_job = g.loc[g["needed_rig"] == (0 if t == "Rods" else 1), "repair_cost"].median()
        planned_days = 0.23 if t == "Rods" else 1.40  # median job duration: rod truck / workover
        rows.append({
            "Failure type": t,
            "Events": len(g),
            "Share needing a rig": g["needed_rig"].mean(),
            "Repair cost (mean)": g["repair_cost"].mean(),
            "Lost oil m3 (mean)": g["lost_m3"].mean(),
            "Hours down (mean)": g["down_hrs"].mean(),
            "Pre-failure rate m3/d": g["pre_rate"].median(),
            "Reactive cost": g["repair_cost"].mean() + g["lost_m3"].mean() * nb_m3,
            "Planned job cost": planned_job,
            "Planned lost oil m3": planned_days * g["pre_rate"].median(),
        })
    t = pd.DataFrame(rows).set_index("Failure type")
    t["Planned cost"] = t["Planned job cost"] + t["Planned lost oil m3"] * nb_m3
    t["Saving if caught"] = t["Reactive cost"] - t["Planned cost"]
    return t


def simulate_policy(scored: pd.DataFrame, costs: pd.DataFrame, k: int = 50, effectiveness: float = 0.5,
                    check_cost: float = 1500.0, cooldown_days: int = 30) -> dict:
    """Walk forward week by week: each Monday, act on the top-k wells not acted on in the last 30 days.

    An action costs `check_cost` (field check / optimisation visit). If the well would have failed
    4-30 days later, that failure becomes planned work with probability `effectiveness`, saving
    the reactive-minus-planned cost for its failure type. Each failure is counted once.
    """
    saving = costs["Saving if caught"].to_dict()
    last_action: dict[str, pd.Timestamp] = {}
    caught: set = set()
    actions = 0
    expected_saving = 0.0
    for day, wk in scored.sort_values("score", ascending=False).groupby("d", sort=True):
        picked = 0
        for row in wk.itertuples():
            if picked >= k:
                break
            prev = last_action.get(row.uwi)
            if prev is not None and (day - prev).days < cooldown_days:
                continue
            last_action[row.uwi] = day
            picked += 1
            actions += 1
            if row.y30 == 1 and (row.uwi, row.next_event) not in caught:
                caught.add((row.uwi, row.next_event))
                expected_saving += effectiveness * saving.get(row.next_type, 0.0)
    n_events = scored.loc[scored["y30"] == 1].drop_duplicates(["uwi", "next_event"]).shape[0]
    weeks = scored["d"].nunique()
    net = expected_saving - actions * check_cost
    return {"k": k, "weeks": weeks, "actions": actions, "events": n_events, "events_warned": len(caught),
            "event_recall": len(caught) / max(n_events, 1), "action_precision": len(caught) / max(actions, 1),
            "gross_saving": expected_saving, "action_cost": actions * check_cost, "net_saving": net,
            "net_saving_per_year": net * 52 / weeks}


def policy_curve(scored, costs, ks=(5, 10, 15, 20, 25, 30, 40, 50, 75, 100), **kw) -> pd.DataFrame:
    return pd.DataFrame([simulate_policy(scored, costs, k=k, **kw) for k in ks])


def repeat_failures(events: pd.DataFrame) -> pd.DataFrame:
    """Days since the previous failure on the same well, for every event."""
    ev = events.copy()
    ev["d"] = pd.to_datetime(ev["d"])
    ev = ev.sort_values(["uwi", "d"])
    ev["gap_days"] = ev.groupby("uwi")["d"].diff().dt.days
    return ev


def explain(model, df: pd.DataFrame, features=FEATURES, top_n: int = 3) -> pd.Series:
    """Top drivers per row from LightGBM's built-in SHAP values (pred_contrib)."""
    contrib = model.predict(df[features], num_iteration=model.best_iteration, pred_contrib=True)[:, :-1]
    names = np.array([FEATURE_LABELS.get(f, f) for f in features])
    order = np.argsort(-contrib, axis=1)[:, :top_n]
    return pd.Series(["; ".join(names[o]) for o in order], index=df.index)


# --------------------------------------------------------------------------
# 5. Run everything and save what the app needs
# --------------------------------------------------------------------------
KEEP_COLS = ["uwi", "well_name", "area", "make", "lat", "lon", "d", "y30", "y14", "y30_log", "next_event",
             "next_type", "fails365", "fails90", "days_since_fail", "last_ftype", "cost365",
             "rpm7", "tq7", "eff7", "oil7", "hd7", "wc30", "dhdays30",
             # baselines and short-term signals, used for the plain-English "what's wrong" in the app
             "rpm90", "tq90", "eff90", "oil90", "hd30", "hd90", "hd3", "rpm3", "tq3", "eff3", "oil3",
             "hd_slope14", "eff_slope14", "rpm_chg30", "rpm_up30", "days_since_string", "days_since_pump"]


def rule_score(df: pd.DataFrame) -> np.ndarray:
    """The simple rule to beat: wells with the most failures in the last 12 months (ties: most recent)."""
    return (df["fails365"] + 1 / (1 + df["days_since_fail"])).values


def fit_rounds(df: pd.DataFrame, features, n_rounds: int):
    import lightgbm as lgb
    return lgb.train(LGB_PARAMS, lgb.Dataset(df[features], df["y30"].astype(int)), n_rounds)


def walk_forward_table(labelled: pd.DataFrame, snaps: pd.DataFrame, n_rounds: int,
                       years=(2023, 2024, 2025, 2026), features=None) -> pd.DataFrame:
    """Retrain on everything before each year (1-month gap), test on that year."""
    from sklearn.metrics import roc_auc_score
    features = features or FEATURES
    rows = []
    for y in years:
        m = fit_rounds(labelled[labelled["d"] <= f"{y - 1}-11-30"], features, n_rounds)
        ts = snaps[(snaps["d"] >= f"{y}-01-01") & (snaps["d"] <= f"{y}-12-01") & snaps["y30"].notna()]
        s, r = m.predict(ts[features]), rule_score(ts)
        rows.append({"year": y, "weeks": ts["d"].nunique(), "base_rate": ts["y30"].mean(),
                     "model_top25": topk_by_week(ts, s, 25)[0], "rule_top25": topk_by_week(ts, r, 25)[0],
                     "model_auc": roc_auc_score(ts["y30"].astype(int), s),
                     "rule_auc": roc_auc_score(ts["y30"].astype(int), r)})
    return pd.DataFrame(rows)


def area_metrics(df: pd.DataFrame, score: np.ndarray, min_rows: int = 1000) -> pd.DataFrame:
    from sklearn.metrics import roc_auc_score
    d = df.assign(_s=score)
    rows = []
    for a, g in d.groupby("area", observed=True):
        if len(g) >= min_rows and g["y30"].nunique() > 1:
            rows.append({"area": str(a).title(), "rows": len(g), "base_rate": g["y30"].mean(),
                         "model_auc": roc_auc_score(g["y30"].astype(int), g["_s"]),
                         "rule_auc": roc_auc_score(g["y30"].astype(int), rule_score(g))})
    return pd.DataFrame(rows).sort_values("rows", ascending=False)


def calibration(df: pd.DataFrame, score: np.ndarray, bins: int = 10) -> pd.DataFrame:
    d = df.assign(_s=score, _b=pd.qcut(score, bins, labels=False, duplicates="drop"))
    return d.groupby("_b").agg(mean_score=("_s", "mean"), actual_rate=("y30", "mean"), rows=("y30", "size")).reset_index(drop=True)


def bootstrap_gap(df: pd.DataFrame, score: np.ndarray, base: np.ndarray, k: int = 25, n: int = 1000):
    """95% CI of (model - rule) top-k precision, resampling whole weeks (rows within a week are correlated)."""
    d = df.assign(_m=score, _r=base)
    wk = np.array([(g.nlargest(k, "_m")["y30"].mean(), g.nlargest(k, "_r")["y30"].mean()) for _, g in d.groupby("d")])
    rng = np.random.RandomState(0)
    diffs = [wk[rng.randint(0, len(wk), len(wk))].mean(axis=0) for _ in range(n)]
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs[:, 0] - diffs[:, 1], [2.5, 97.5])
    return float(lo), float(hi)


def build_all(data_dir: Path | str = DATA_DIR, out_dir: Path | str = OUT_DIR, verbose: bool = True) -> dict:
    import json
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = print if verbose else (lambda *a, **k: None)

    con = connect(data_dir)
    events = build_events(con)
    log(f"events: {len(events):,}")
    snaps = build_snapshots(con)
    log(f"snapshots: {len(snaps):,} rows, {snaps['uwi'].nunique():,} wells")
    ev_costs = event_costs(con)
    costs = cost_table(ev_costs)

    # Walk-forward: every week is scored by a model trained only on labels that were known before it.
    #   2025 weeks          <- trained on 2021-2024 (early stopping on 2025 sets the number of rounds)
    #   2026 test weeks     <- retrained on everything up to Nov 2025
    #   weeks after the test <- the live model, retrained on every labelled week (this is the watch list)
    parts = split(snaps)
    m_tune = train_lgb(parts["train"], parts["val"])
    n_rounds = m_tune.best_iteration
    labelled = snaps[snaps["y30"].notna() & (snaps["in_event_next"] == 0)]
    m_test = fit_rounds(labelled[labelled["d"] <= "2025-11-30"], FEATURES, n_rounds)
    m_live = fit_rounds(labelled, FEATURES, n_rounds)
    m_live.save_model(str(out_dir / "model_y30.txt"))
    log(f"rounds: {n_rounds}")

    latest_day = snaps["d"].max()
    scored = snaps[(snaps["d"] >= SPLITS["val"][0])].copy()
    in_val = scored["d"] <= SPLITS["val"][1]
    in_test = (scored["d"] >= SPLITS["test"][0]) & (scored["d"] <= SPLITS["test"][1])
    after = scored["d"] > SPLITS["test"][1]
    scored["score"] = np.nan
    scored.loc[in_val, "score"] = predict(m_tune, scored[in_val])
    scored.loc[in_test, "score"] = predict(m_test, scored[in_test])
    scored.loc[after, "score"] = predict(m_live, scored[after])
    scored = scored[scored["score"].notna()]
    latest = scored[scored["d"] == latest_day].copy()
    latest["drivers"] = explain(m_live, latest)
    scored = scored.merge(latest[["uwi", "d", "drivers"]], on=["uwi", "d"], how="left")
    scored[KEEP_COLS + ["score", "drivers"]].to_parquet(out_dir / "scores.parquet", index=False)

    # Model comparison on the 2026 test period: every model trained on data up to Nov 2025.
    test = scored[(scored["d"] >= SPLITS["test"][0]) & (scored["d"] <= SPLITS["test"][1]) & scored["y30"].notna()]
    test_full = parts["test"]
    train_to_2025 = labelled[labelled["d"] <= "2025-11-30"]
    sig = [f for f in FEATURES if f not in HISTORY]
    m_sig = fit_rounds(train_to_2025, sig, train_lgb(parts["train"], parts["val"], features=sig).best_iteration)
    m_v1 = fit_rounds(train_to_2025, FEATURES_V1,
                      train_lgb(parts["train"], parts["val"], features=FEATURES_V1).best_iteration)
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    num = [f for f in FEATURES if f not in CATEGORICAL]
    lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                       LogisticRegression(max_iter=2000, C=0.5)).fit(train_to_2025[num], train_to_2025["y30"].astype(int))
    rng = np.random.RandomState(0)
    final = predict(m_test, test_full)
    comp = pd.DataFrame({
        "Random ranking": metrics(test_full, rng.rand(len(test_full))),
        "Rule: most failures in last 12 months": metrics(test_full, rule_score(test_full)),
        "Logistic regression": metrics(test_full, lr.predict_proba(test_full[num])[:, 1]),
        "LightGBM, pump signals only": metrics(test_full, predict(m_sig, test_full, sig)),
        "LightGBM, original features": metrics(test_full, predict(m_v1, test_full, FEATURES_V1)),
        "LightGBM, signals + history (final)": metrics(test_full, final),
    }).T
    comp["ROC AUC vs official log only"] = [np.nan] * 5 + [metrics(test_full, final, "y30_log")["ROC AUC"]]
    comp.to_csv(out_dir / "model_comparison.csv")
    log(comp.round(3).to_string())

    # Walk-forward by year, by area, calibration and a block bootstrap of the gap over the rule.
    wf = walk_forward_table(labelled, snaps, n_rounds)
    wf.to_csv(out_dir / "walk_forward.csv", index=False)
    log(wf.round(3).to_string())
    area_metrics(test_full, final).to_csv(out_dir / "area_metrics.csv", index=False)
    calibration(test_full, final).to_csv(out_dir / "calibration.csv", index=False)
    boot = bootstrap_gap(test_full, final, rule_score(test_full))

    imp = pd.Series(m_live.feature_importance("gain"), index=FEATURES)
    (imp / imp.sum()).sort_values(ascending=False).rename("share_of_gain").to_csv(out_dir / "feature_importance.csv")
    costs.to_csv(out_dir / "cost_table.csv")
    ev_costs.to_parquet(out_dir / "event_costs.parquet", index=False)
    events.to_parquet(out_dir / "events.parquet", index=False)
    curve = policy_curve(test, costs)
    curve.to_csv(out_dir / "policy_curve.csv", index=False)

    # Headline numbers: 12 months of logged failures (Jun 2025 - May 2026)
    nb_m3 = DEFAULT_NETBACK_BBL * M3_TO_BBL
    ec = ev_costs.assign(d=pd.to_datetime(ev_costs["d"]))
    yr = ec[(ec["d"] >= "2025-06-01") & (ec["d"] < "2026-06-01")]
    rep = repeat_failures(events)
    yr = yr.merge(rep[["uwi", "d", "gap_days"]], on=["uwi", "d"], how="left")
    repeats = yr[yr["gap_days"] <= 30]
    head = {
        "period": "Jun 2025 - May 2026",
        "failures": int(len(yr)),
        "repair_cost": float(yr["repair_cost"].sum()),
        "lost_oil_m3": float(yr["lost_m3"].sum()),
        "lost_oil_value": float(yr["lost_m3"].sum() * nb_m3),
        "total_cost": float(yr["repair_cost"].sum() + yr["lost_m3"].sum() * nb_m3),
        "repeat_30d_share": float(len(repeats) / len(yr)),
        "repeat_30d_cost": float(repeats["repair_cost"].sum() + repeats["lost_m3"].sum() * nb_m3),
        "netback_bbl": DEFAULT_NETBACK_BBL,
        "latest_week": str(latest_day.date()),
        "wells_scored_latest": int(len(latest)),
        "model_top25_precision": float(comp.loc["LightGBM, signals + history (final)", "Precision top 25/wk"]),
        "rule_top25_precision": float(comp.loc["Rule: most failures in last 12 months", "Precision top 25/wk"]),
        "model_auc": float(comp.loc["LightGBM, signals + history (final)", "ROC AUC"]),
        "base_rate": float(comp.loc["LightGBM, signals + history (final)", "Base rate"]),
        "gap_vs_rule_ci95": [boot[0], boot[1]],
        "walk_forward_top25_range": [float(wf["model_top25"].min()), float(wf["model_top25"].max())],
    }
    (out_dir / "headline.json").write_text(json.dumps(head, indent=2))
    log(json.dumps(head, indent=2))
    return {"events": events, "snapshots": snaps, "model": m_live, "costs": costs, "comparison": comp,
            "curve": curve, "headline": head, "scored": scored}


if __name__ == "__main__":
    build_all()
