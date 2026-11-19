/**
 * gta6-trophy-rarity-forecast-01 — forecast every GTA VI trophy's Exophase
 * earn rate at the case's target, as five quantiles per trophy.
 *
 * Node 22, standard library only, no network, deterministic.
 *
 * Model v3 (rank defense: v2 ranked 858/4536, only 142 inside the
 * top 1,000; v3 beats v2 on a cohort-simulator backtest, 6 tuned + 3
 * held-out scenarios, on the final-shaped case and on the mean):
 *
 * What v3 changed vs v2, and why (cohort-simulator backtest, 6 tuned
 * scenarios x 4 cases + 3 held-out scenarios; mean pinball loss):
 *   - recent-slope tangent (was the quadratic's slope-now): the
 *     backtest's median logit error fell by roughly half at every
 *     checkpoint, because the quadratic averages the stale
 *     launch-night stretch and the dilution dip into today's slope.
 *   - tangent weight 0.9 (was 0.5): the tangent is the better
 *     estimator at both ends of the history; the global line is
 *     only a 10% anchor against a noisy recent window.
 *   - pre-refresh reads at 25% weight (was full weight): the stale
 *     launch-night page bends every fitted slope while the history
 *     is short, and the effect washes out only once many refreshes
 *     have landed.
 *   - spread horizon term 0.05*sqrt(days) (was 0.15): the optimal
 *     spread grows far slower than a random walk once the history
 *     holds several refreshes; the 0.2 floor and the residual /
 *     disagreement / slope terms carry the real uncertainty.
 * Backtest (v2 -> v3 mean pinball; lower is better):
 *   scenario   f00 (final-shaped)   mean (4 cases)
 *   central    0.122 -> 0.065        0.158 -> 0.117
 *   bigN       0.107 -> 0.059        0.179 -> 0.132
 *   smallN     0.121 -> 0.077        0.155 -> 0.120
 *   fast       0.142 -> 0.065        0.144 -> 0.090
 *   slow       0.095 -> 0.062        0.319 -> 0.246
 *   lagworld   0.163 -> 0.087        0.198 -> 0.119
 *   held-out A/B/C f00: 0.120/0.201/0.110 -> 0.069/0.075/0.066.
 * v3 wins the final-shaped case and the mean in all 9 scenarios.
 *
 *
 * A trophy's earn rate is a ratio, rate = earned / players_tracked,
 * and the two sides move in opposite directions: new players joining
 * dilute every rate down, while the players already tracked keep
 * earning trophies up. Fitting a line to the rate alone confounds the
 * two forces — on a short history it extrapolates the launch-night
 * dilution dip and under-predicts the later rise by ~1 logit (measured
 * on a cohort-simulator backtest). So v2 forecasts the two sides
 * separately and divides:
 *  - players_tracked N(t): a recency-weighted line of ln N on
 *    x = ln(1 + hours since release). Join growth is sublinear in
 *    log-time, so this extrapolates the dilution honestly.
 *  - earned counts E_i(t) = rate_i(t) * N(t) / 100: monotone (a count
 *    only grows), so a recency-weighted line of ln E_i on x is stable
 *    where the rate line bends. Slopes are shrunk toward the tier
 *    median; noisy small-count reads are down-weighted.
 *  - the rate forecast is E_i(target) / N(target), ensembled in logit
 *    space with v1's rate-line forecast. Each member is mostly a
 *    tangent forecast: the curve bends (it steepens while new cohorts
 *    pile in, then flattens as earned counts saturate), so the slope
 *    *now* is read off a short linear fit on the last few post-refresh
 *    reads — the recent slope, not the quadratic's slope-now, which
 *    the flat launch-night segment bends the wrong way — and the
 *    tangent extends that slope from the last rate, damped as the
 *    growth saturates. The global log-time line is only a 10% anchor:
 *    on the backtest the tangent's median error is roughly half the
 *    line's at every checkpoint. The two members disagree exactly when
 *    the extrapolation is fragile, so their disagreement also sets
 *    the spread.
 *  - spread: the honest uncertainty is far smaller than v1's
 *    0.45*sqrt(days) random walk once the history holds several
 *    refreshes (v1's f00 spread ~1.0 logit vs a measured median error
 *    of ~0.16 across backtest worlds, most of its pinball loss was
 *    width, not bias). v3's spread combines a small base term, a
 *    horizon term (0.05*sqrt(days): the backtest's optimal horizon
 *    growth is far below the random walk's, and the 0.2 floor plus
 *    the residual/disagreement/slope terms carry the real
 *    uncertainty), the fits' residuals, the ensemble disagreement, a
 *    staleness term and a thin-data term.
 *  - the flat-history refresh prior from v1 is kept unchanged: when
 *    the page has never refreshed, a flat forecast would trust rates
 *    Exophase has not refreshed yet, so the first refresh dilutes
 *    every rate to about 0.75x as the tracked-player count jumps
 *    (638 -> 3868 on the first refresh), then rates resume their
 *    log-time climb.
 *  - a trophy at 100% in the last read stays 100%.
 *
 *  - the reads before Exophase's first refresh show a stale page
 *    (launch-night rates on 638 players, before the tracked count
 *    jumped to 3,868): they still anchor the level, but they bend
 *    every slope the wrong way while the history is short, so the
 *    line and count fits down-weight them to 25%, and the recent
 *    slope reads only post-refresh reads.
 *
 * Usage: MAKEFUN_INPUT_DIR=<dir> MAKEFUN_OUTPUT_DIR=<dir> node main.ts
 * Reads every *.in in the input dir, writes <id>.out per case.
 */

import { readFileSync, writeFileSync, readdirSync, mkdirSync } from "node:fs";
import { join } from "node:path";

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? "/input";
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? "/output";

// Release day begins 2026-11-19T00:00:00Z; x measures log-hours from it.
const RELEASE_MS = Date.parse("2026-11-19T00:00:00Z");
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
const Z = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];
const HALF_LIFE_HOURS = 18;
// v3: the tangent carries the forecast (see header).
const TANGENT_W = 0.9;
const RECENT_N = 6;
const PRE_REFRESH_W = 0.25;
const SPREAD_HORIZ_COEF = 0.05;

// First-refresh dilution: on the first refresh Exophase's tracked
// players jumped 638 -> 3868 while total trophies earned went 943 ->
// 4276. The one trophy at 100% stayed there, so the other 54 rates
// summed 47.78 -> 10.55: a x0.221 scale. (The total-sum ratio 0.748
// is wrong here: the 100% trophy takes 100 of the 110.55 total.)
const REFRESH_SCALE = 0.221;

// Easiness tilt for the refresh: within the refresh, easy/early
// trophies keep more of their rate than hard/late ones, and a
// trophy's launch-night rate is the case's own signal for that
// (fishing 2.04% kept ~0.36x, the platinum 0.63% ~0.19x). The
// medians follow a power law in the last rate, normalised so the
// non-100 medians still sum to REFRESH_SCALE x the launch sum.
const REFRESH_POWER = 1.35;

// Prior log-time slopes per tier, used only when a trophy has no usable
// fit of its own (flat history). Early story bronze climbs; platinum
// and gold barely move in week one.
const TIER_PRIOR_SLOPE: Record<string, number> = {
  platinum: -0.1,
  gold: 0.0,
  silver: 0.08,
  bronze: 0.22,
};

function clampRate(pct: number): number {
  return Math.min(99.99, Math.max(0.01, pct));
}

function logit(pct: number): number {
  const p = clampRate(pct) / 100;
  return Math.log(p / (1 - p));
}

function invLogit(l: number): number {
  if (l >= 20) return 99.99;
  if (l <= -20) return 0.01;
  return clampRate(100 / (1 + Math.exp(-l)));
}

function round4(x: number): number {
  const v = Math.floor(x * 10000 + 0.5) / 10000;
  return v;
}

function hoursSinceRelease(iso: string): number {
  return Math.max(0, (Date.parse(iso) - RELEASE_MS) / 3600000);
}

function logHours(iso: string): number {
  return Math.log1p(hoursSinceRelease(iso));
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const s = [...values].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, x));
}

interface Obs {
  x: number;
  y: number;
  w: number;
  t: number; // ms since release, for recency weighting
  dj?: number; // index in `distinct`
}

interface LineFit {
  slope: number;
  intercept: number;
  rms: number;
  seSlope: number;
  n: number;
  xLast: number;
  lastRate: number;
  quad: { a: number; b: number; c: number; rms: number; u0: number } | null;
  u0: number;
  recentSlope: number | null;
}

/** Weighted least squares of y on x; returns slope, intercept, rms. */
function wls(obs: Obs[]): { slope: number; intercept: number; rms: number; seSlope: number } {
  let sw = 0;
  let sx = 0;
  let sy = 0;
  for (const o of obs) {
    sw += o.w;
    sx += o.w * o.x;
    sy += o.w * o.y;
  }
  const mx = sx / sw;
  const my = sy / sw;
  let sxx = 0;
  let sxy = 0;
  for (const o of obs) {
    sxx += o.w * (o.x - mx) * (o.x - mx);
    sxy += o.w * (o.x - mx) * (o.y - my);
  }
  const slope = sxx > 1e-9 ? sxy / sxx : 0;
  const intercept = my - slope * mx;
  let se = 0;
  for (const o of obs) {
    const r = o.y - (intercept + slope * o.x);
    se += o.w * r * r;
  }
  const rms = Math.sqrt(se / sw);
  return { slope, intercept, rms, seSlope: sxx > 1e-9 ? rms / Math.sqrt(sxx) : 1 };
}

/**
 * Weighted least squares of y on u = x - u0 with a quadratic term:
 * y = a + b*u + c*u^2. The curve a trophy follows in log-time bends —
 * it steepens while new cohorts pile in, then flattens as the earned
 * counts saturate and the tracked-player count keeps growing — and a
 * straight line extrapolated 4 days either under- or over-shoots
 * depending on where the window sits on the bend. The quadratic term
 * is shrunk toward 0 (a straight line) and clamped, and is only used
 * with enough points and x-spread; with too little data it falls back
 * to the linear fit. Solves the 3x3 normal equations by Cramer.
 *
 * v3: the quadratic is now only the fallback for the tangent slope
 * (used when the recent slope is null, i.e. fewer than 3 post-refresh
 * reads or too little x-spread). Its slope-now was v2's tangent
 * slope; the backtest showed the recent slope reads the phase better,
 * because the quadratic averages the stale launch-night stretch and
 * the dilution dip into today's slope.
 */
function wlsQuad(obs: Obs[]): { a: number; b: number; c: number; rms: number; u0: number } | null {
  const n = obs.length;
  if (n < 5) return null;
  const xs = obs.map((o) => o.x);
  if (Math.max(...xs) - Math.min(...xs) < 0.8) return null;
  let sw = 0;
  let sx = 0;
  for (const o of obs) {
    sw += o.w;
    sx += o.w * o.x;
  }
  const u0 = sx / sw;
  // Moments of u and u^2.
  let s0 = 0;
  let s1 = 0;
  let s2 = 0;
  let s3 = 0;
  let s4 = 0;
  let t0 = 0;
  let t1 = 0;
  let t2 = 0;
  for (const o of obs) {
    const u = o.x - u0;
    const u2 = u * u;
    s0 += o.w;
    s1 += o.w * u;
    s2 += o.w * u2;
    s3 += o.w * u2 * u;
    s4 += o.w * u2 * u2;
    t0 += o.w * o.y;
    t1 += o.w * u * o.y;
    t2 += o.w * u2 * o.y;
  }
  // Normal equations M * [a,b,c] = [t0,t1,t2]; det via expansion.
  const det =
    s0 * (s2 * s4 - s3 * s3) -
    s1 * (s1 * s4 - s3 * s2) +
    s2 * (s1 * s3 - s2 * s2);
  if (Math.abs(det) < 1e-12) return null;
  const da =
    t0 * (s2 * s4 - s3 * s3) -
    s1 * (t1 * s4 - s3 * t2) +
    s2 * (t1 * s3 - s2 * t2);
  const db =
    s0 * (t1 * s4 - s3 * t2) -
    t0 * (s1 * s4 - s3 * s2) +
    s2 * (s1 * t2 - t1 * s2);
  const dc =
    s0 * (s2 * t2 - t1 * s3) -
    s1 * (s1 * t2 - t1 * s2) +
    t0 * (s1 * s3 - s2 * s2);
  const a = da / det;
  const b = db / det;
  // Shrink the bend toward a straight line; allow a little positive
  // curvature (the accelerating phase) and more negative (saturation).
  const cRaw = dc / det;
  const c = clamp(cRaw * (n / (n + 3)), -0.6, 0.15);
  let se = 0;
  for (const o of obs) {
    const u = o.x - u0;
    const r = o.y - (a + b * u + c * u * u);
    se += o.w * r * r;
  }
  return { a, b, c, rms: Math.sqrt(se / sw), u0 };
}

/**
 * The slope *now*: a straight line through the last RECENT_N
 * post-first-refresh observations. The global line and the quadratic
 * both average the whole history, whose early stretch is the stale
 * launch-night page and the dilution dip; the recent slope reads the
 * phase the curve is in today — steepening while new cohorts pile in,
 * flattening as earned counts saturate — which is what a 4-day
 * extrapolation has to get right. Null with too few post-refresh
 * points or too little x-spread (the caller falls back to the
 * quadratic's slope-now, then the global slope).
 */
function recentSlopeOf(obs: Obs[], fromIdx: number): number | null {
  const pool = obs.filter((o) => (o.dj ?? 0) >= fromIdx);
  const recent = pool.slice(-RECENT_N);
  if (recent.length < 3) return null;
  const xs = recent.map((o) => o.x);
  if (Math.max(...xs) - Math.min(...xs) < 0.05) return null;
  return wls(recent).slope;
}

/** Root-sum-of-squares of spread terms (each already in logit units). */
function quadSpread(terms: number[]): number {
  let s = 0;
  for (const t of terms) s += t * t;
  return Math.sqrt(s);
}

/**
 * Combine the two members in logit space. The counts member compounds
 * two forecasts (earned and tracked), so its edge fades as the horizon
 * lengthens; the rate line is the steadier long-horizon member, so the
 * counts weight falls from 0.7 at a 0-day horizon to 0.3 at 5 days.
 * The members' disagreement is returned too: it feeds the spread,
 * because the members disagree exactly when the extrapolation is
 * fragile. With only one member, the other contributes nothing and
 * there is nothing to disagree about.
 */
function ensemblePred(
  linePred: number | null,
  countPred: number | null,
  lineConf: number,
  countConf: number,
  horizonDays: number,
): { pred: number; disagree: number } {
  if (linePred !== null && countPred !== null) {
    const wCounts = clamp(0.7 - 0.08 * horizonDays, 0.3, 0.7);
    const lw = lineConf * (1 - wCounts);
    const cw = countConf * wCounts;
    return {
      pred: (lw * linePred + cw * countPred) / (lw + cw),
      disagree: Math.abs(linePred - countPred),
    };
  }
  if (linePred !== null) return { pred: linePred, disagree: 0 };
  return { pred: countPred as number, disagree: 0 };
}

/**
 * Flat history: the page has never refreshed, so the last rates are
 * the stale launch-night page. The refresh prior (from v1) dilutes
 * every rate to ~0.22x with an easiness tilt (easy trophies keep more
 * of their rate), then the tier's log-time climb resumes from there.
 * Returns the median logit and the spread.
 */
function flatHistoryForecast(
  lastRate: number,
  lastLogit: number,
  capMove: number,
  flatNorm: number,
  tier: string,
  targetX: number,
  xLast: number,
  horizonDays: number,
): { medLogit: number; spread: number } {
  const g = TIER_PRIOR_SLOPE[tier] ?? 0.15;
  const climb = g * Math.max(0, targetX - xLast);
  const powered = Math.pow(lastRate, REFRESH_POWER);
  let medLogit = logit(flatNorm * powered) + climb;
  medLogit = clamp(medLogit, lastLogit - capMove, lastLogit + capMove);
  const horizonTerm = 0.45 * Math.sqrt(horizonDays);
  return { medLogit, spread: quadSpread([0.3, horizonTerm, 0.6]) };
}

interface CaseTrophy {
  id: number;
  tier: string;
}

function forecastCase(kase: any): { trophies: { id: number; q: number[] }[] } {
  const trophies: CaseTrophy[] = kase.trophies;
  const history: any[] = kase.history ?? [];
  const targetX = logHours(kase.target);
  const horizonDays = (kase.horizon_minutes ?? 0) / 1440;
  const checkpointMs = Date.parse(kase.at);

  // Collapse to distinct reads: a read counts as new only when some
  // trophy's rate or the tracked-player count differs from the
  // previous distinct read. Exophase refreshes every few hours while
  // the recorder reads every 10-30 minutes, so most reads repeat the
  // last refreshed page and would otherwise swamp the fits with
  // dozens of copies of one point. Keep the first occurrence time
  // of each distinct read. A null rate means "this read lacked the
  // trophy", not "the rate changed", so nulls are forward-filled for
  // the comparison only.
  const distinct: any[] = [];
  let prevKey = "";
  let prevRates: (number | null)[] = [];
  let prevN: number | null = null;
  for (const read of history) {
    const filled = (read.rates as (number | null)[]).map((r, i) =>
      r === null || r === undefined ? prevRates[i] ?? null : r,
    );
    prevRates = filled;
    if (typeof read.players_tracked === "number" && read.players_tracked > 0) {
      prevN = read.players_tracked;
    }
    const key = JSON.stringify([prevN, filled]);
    if (key !== prevKey) {
      distinct.push(read);
      prevKey = key;
    }
  }
  // Always keep the very last read as the anchor even if identical.
  const lastRead = history[history.length - 1];
  if (distinct.length === 0 && lastRead) distinct.push(lastRead);

  // Last non-null rate per trophy, and the tier median of those.
  const lastRates: (number | null)[] = trophies.map((_, i) => {
    for (let j = history.length - 1; j >= 0; j--) {
      const r = history[j].rates[i];
      if (r !== null && r !== undefined) return r;
    }
    return null;
  });
  const knownLast = lastRates.filter((r): r is number => r !== null);
  const globalLastMedian = knownLast.length ? median(knownLast) : 1.0;

  // Players tracked at each distinct read (forward-filled; a read
  // without the count borrows the previous one).
  const nSeries: number[] = [];
  let fillN = 0;
  for (const read of distinct) {
    if (typeof read.players_tracked === "number" && read.players_tracked > 0) {
      fillN = read.players_tracked;
    }
    nSeries.push(fillN);
  }
  const lastN = nSeries.length ? nSeries[nSeries.length - 1] : 0;

  // First-refresh index: the first distinct read whose tracked-player
  // count jumps >1.5x the first read's (Exophase's first refresh, 638
  // -> 3,868). Reads before it show the stale launch-night page.
  let fitStartIdx = 0;
  {
    const n0 = nSeries[0] || 0;
    for (let j = 1; j < nSeries.length; j++) {
      if (n0 > 0 && nSeries[j] > 1.5 * n0) { fitStartIdx = j; break; }
    }
  }

  // Forecast players tracked at the target: ln N on log-hours.
  // Join growth is sublinear, so the slope is clamped to [0, 1.5];
  // the count only grows, so the forecast never falls below the last.
  let nPred = lastN;
  if (distinct.length >= 2 && lastN > 0) {
    const obs: Obs[] = [];
    for (let j = 0; j < distinct.length; j++) {
      if (nSeries[j] <= 0) continue;
      const ageH = Math.max(0, (checkpointMs - Date.parse(distinct[j].at)) / 3600000);
      obs.push({
        x: logHours(distinct[j].at),
        y: Math.log(nSeries[j]),
        w: Math.pow(0.5, ageH / HALF_LIFE_HOURS),
        t: Date.parse(distinct[j].at) - RELEASE_MS,
      });
    }
    const xs = obs.map((o) => o.x);
    if (obs.length >= 2 && Math.max(...xs) - Math.min(...xs) >= 0.05) {
      const { slope, intercept } = wls(obs);
      const s = clamp(slope, 0, 1.5);
      nPred = Math.exp(intercept + s * targetX);
      nPred = clamp(nPred, lastN, lastN * 5);
    }
  }

  // Per-trophy rate-line fits (v1's member) on the whole history.
  // The fits keep every distinct read: the launch-night reads still
  // anchor the level, but the pre-first-refresh ones are down-weighted
  // (PRE_REFRESH_W) because their stale rates bend the slope, and the
  // tangent slope is read off the recent post-refresh window instead
  // (see recentSlopeOf). An earlier variant that fit only post-refresh
  // reads fixed short histories but over-shot the final's long
  // horizon, so the full history stays, weighted.
  const lineFits: (LineFit | null)[] = trophies.map((t, i) => {
    const obs: Obs[] = [];
    for (let dj = 0; dj < distinct.length; dj++) {
      const read = distinct[dj];
      const r = read.rates[i];
      if (r === null || r === undefined) continue;
      const ageH = Math.max(0, (checkpointMs - Date.parse(read.at)) / 3600000);
      const w = Math.pow(0.5, ageH / HALF_LIFE_HOURS);
      obs.push({ x: logHours(read.at), y: logit(r), w: dj < fitStartIdx ? w * PRE_REFRESH_W : w, t: Date.parse(read.at) - RELEASE_MS, dj });
    }
    if (obs.length < 2) return null;
    const xs = obs.map((o) => o.x);
    if (Math.max(...xs) - Math.min(...xs) < 0.05) return null;
    const { slope, intercept, rms, seSlope } = wls(obs);
    const lastRate = lastRates[i] ?? globalLastMedian;
    return {
      slope,
      intercept,
      rms,
      seSlope,
      n: obs.length,
      xLast: lastRead ? logHours(lastRead.at) : 0,
      lastRate,
      quad: wlsQuad(obs),
      u0: obs.reduce((s, o) => s + o.w * o.x, 0) / obs.reduce((s, o) => s + o.w, 0),
      recentSlope: recentSlopeOf(obs, fitStartIdx),
    };
  });

  // Per-trophy earned-count fits: ln E on log-hours, E monotone
  // (running max over distinct reads) and small counts down-weighted
  // (a 2-decimal rate on few players quantises the count hard).
  interface CountFit {
    slope: number;
    intercept: number;
    rms: number;
    seSlope: number;
    n: number;
    eLast: number;
    xLast: number;
    quad: { a: number; b: number; c: number; rms: number; u0: number } | null;
    recentSlope: number | null;
  }
  const countFits: (CountFit | null)[] = trophies.map((t, i) => {
    const obs: Obs[] = [];
    let runMax = 0;
    for (let j = 0; j < distinct.length; j++) {
      const read = distinct[j];
      const r = read.rates[i];
      if (r === null || r === undefined) continue;
      const n = nSeries[j];
      if (n <= 0) continue;
      const e = (r / 100) * n;
      runMax = Math.max(runMax, e);
      const ageH = Math.max(0, (checkpointMs - Date.parse(read.at)) / 3600000);
      const w =
        Math.pow(0.5, ageH / HALF_LIFE_HOURS) * Math.min(1, runMax / 30);
      obs.push({
        x: logHours(read.at),
        y: Math.log(Math.max(runMax, 0.5)),
        w: j < fitStartIdx ? w * PRE_REFRESH_W : w,
        t: Date.parse(read.at) - RELEASE_MS,
        dj: j,
      });
    }
    if (obs.length < 2) return null;
    const xs = obs.map((o) => o.x);
    if (Math.max(...xs) - Math.min(...xs) < 0.05) return null;
    const { slope, intercept, rms, seSlope } = wls(obs);
    return {
      slope,
      intercept,
      rms,
      seSlope,
      n: obs.length,
      eLast: runMax,
      xLast: lastRead ? logHours(lastRead.at) : 0,
      quad: wlsQuad(obs),
      recentSlope: recentSlopeOf(obs, fitStartIdx),
    };
  });

  // Tier-group median slopes for hierarchical shrinkage, for both
  // members, plus a global fallback.
  function tierMedian(
    fits: (LineFit | CountFit | null)[],
    get: (f: LineFit | CountFit) => number,
    prior: Record<string, number>,
    fallback: number,
  ): Map<string, number> {
    const m = new Map<string, number>();
    for (const tier of ["platinum", "gold", "silver", "bronze"]) {
      const slopes = fits
        .map((f, i) => (f && trophies[i].tier === tier ? get(f) : null))
        .filter((s): s is number => s !== null);
      m.set(tier, slopes.length ? median(slopes) : (prior[tier] ?? fallback));
    }
    return m;
  }
  const lineGroup = tierMedian(lineFits, (f) => f.slope, TIER_PRIOR_SLOPE, 0.1);
  const countGroup = tierMedian(countFits, (f) => f.slope, {}, 0.5);
  const allCountSlopes = countFits
    .map((f) => (f ? f.slope : null))
    .filter((s): s is number => s !== null);
  const globalCountSlope = allCountSlopes.length ? median(allCountSlopes) : 0.5;

  // Staleness: hours since the last distinct read changed.
  const lastChangeMs =
    distinct.length > 1
      ? Date.parse(distinct[distinct.length - 1].at)
      : history.length
        ? Date.parse(history[0].at)
        : checkpointMs;
  const staleHours = Math.max(0, (checkpointMs - lastChangeMs) / 3600000);

  // Normaliser for the flat-history power law: the non-100 medians
  // sum to REFRESH_SCALE x the non-100 last rates.
  let flatNum = 0;
  let flatDen = 0;
  for (const r of lastRates) {
    if (r !== null && r < 99.99) {
      flatNum += REFRESH_SCALE * r;
      flatDen += Math.pow(r, REFRESH_POWER);
    }
  }
  const flatNorm = flatDen > 0 ? flatNum / flatDen : 0;

  // The total move from the last rate is capped; the cap grows with
  // the horizon (a week of play moves a trophy further than an hour).
  const capMove = 3.5 + 0.35 * Math.log1p(horizonDays);

  const rows = trophies.map((t, i) => {
    const lastRate = lastRates[i] ?? globalLastMedian;
    // A trophy every tracked player holds stays held.
    if (lastRate >= 99.99) {
      return { id: t.id, q: [100, 100, 100, 100, 100] };
    }
    const fit = lineFits[i];
    const cfit = countFits[i];
    const lastLogit = logit(lastRate);
    let medLogit: number;
    let spread: number;
    if (fit || cfit) {
      // --- rate-line member (v1, tightened) ---
      let linePred: number | null = null;
      let lineConf = 0;
      if (fit) {
        const g = lineGroup.get(t.tier) ?? 0.1;
        const k = 3; // pseudo-count of group observations
        const slope = (fit.n * fit.slope + k * g) / (fit.n + k);
        let pred = fit.intercept + slope * targetX;
        // Tangent prediction: the recent slope (see recentSlopeOf)
        // is the slope *now*, at the last read, and the forecast
        // extends that slope linearly from the last rate. The global
        // line averages the whole history, so it over-shoots once the
        // curve flattens and under-shoots while it steepens; the
        // tangent corrects both, and the 10% line weight hedges a
        // noisy recent window.
        const xLast = fit.xLast;
        let slopeNow = fit.slope;
        if (fit.recentSlope !== null) {
          slopeNow = fit.recentSlope;
        } else if (fit.quad) {
          slopeNow = fit.quad.b + 2 * fit.quad.c * (xLast - fit.quad.u0);
        }
        slopeNow = (fit.n * clamp(slopeNow, -0.3, 1.2) + k * g) / (fit.n + k);
        // The tangent slope is today's slope; it decays over the
        // extrapolation (growth saturates), so the tangent move is
        // damped by ln(1 + L*d)/(L*d) with L = 0.6 per log-hour.
        const dx = Math.max(0, targetX - xLast);
        const damp = dx > 0 ? Math.log1p(0.6 * dx) / (0.6 * dx) : 1;
        const tangentPred = lastLogit + slopeNow * dx * damp;
        pred = (1 - TANGENT_W) * pred + TANGENT_W * tangentPred;
        const wFit =
          Math.max(0.35, 1 - fit.rms / 2) * (0.6 + 0.1 * Math.min(fit.n, 4));
        pred = wFit * pred + (1 - wFit) * lastLogit;
        linePred = pred;
        lineConf = clamp(0.3 + 0.1 * Math.min(fit.n, 5) - fit.rms, 0.2, 1);
      }
      // --- earned-counts member ---
      let countPred: number | null = null;
      let countConf = 0;
      if (cfit && lastN > 0 && nPred > 0) {
        const g = countGroup.get(t.tier) ?? globalCountSlope;
        const k = 2;
        const slope = (cfit.n * cfit.slope + k * g) / (cfit.n + k);
        let ePred = Math.exp(cfit.intercept + slope * targetX);
        // Tangent counts: extend the *current* earned-count growth
        // rate (the recent slope) linearly in log-time from the last
        // count. Same idea as the rate line's tangent: today's slope
        // sets the move, not the shape of the whole extrapolation.
        let slopeNow = cfit.slope;
        if (cfit.recentSlope !== null) {
          slopeNow = cfit.recentSlope;
        } else if (cfit.quad) {
          slopeNow = cfit.quad.b + 2 * cfit.quad.c * (cfit.xLast - cfit.quad.u0);
        }
        slopeNow = (cfit.n * clamp(slopeNow, 0, 2.0) + k * g) / (cfit.n + k);
        const dx = Math.max(0, targetX - cfit.xLast);
        const damp = dx > 0 ? Math.log1p(0.6 * dx) / (0.6 * dx) : 1;
        const tangentE =
          cfit.eLast * Math.exp(slopeNow * dx * damp);
        ePred = Math.exp((1 - TANGENT_W) * Math.log(Math.max(ePred, 0.5)) + TANGENT_W * Math.log(Math.max(tangentE, 0.5)));
        // Counts only grow; never predict fewer earned than today.
        ePred = Math.max(ePred, cfit.eLast);
        const ratePred = (ePred / nPred) * 100;
        countPred = logit(ratePred);
        countConf = clamp(0.3 + 0.1 * Math.min(cfit.n, 5) - cfit.rms, 0.2, 1);
      }
      // --- ensemble in logit space ---
      const ens = ensemblePred(linePred, countPred, lineConf, countConf, horizonDays);
      let pred = ens.pred;
      const disagree = ens.disagree;
      pred = clamp(pred, lastLogit - capMove, lastLogit + capMove);
      medLogit = pred;
      const horizonTerm = SPREAD_HORIZ_COEF * Math.sqrt(horizonDays);
      const rmsTerm = fit ? 0.6 * fit.rms : 0;
      const disagreeTerm = 0.5 * disagree;
      // Slope uncertainty over the extrapolation distance: the
      // standard error of each member's slope, times how far past
      // the last read the target sits. Short histories have a short
      // x-range, so their slopes are poorly pinned and this term
      // widens the spread exactly where the median is least sure.
      const extrap = Math.max(0, targetX - (fit?.xLast ?? cfit?.xLast ?? targetX));
      const slopeUnc = Math.min(
        0.8,
        Math.max(
          fit ? fit.seSlope * extrap : 0,
          cfit ? cfit.seSlope * extrap : 0,
        ),
      );
      const staleTerm = staleHours >= 6 ? 0.15 : 0;
      const thinTerm = fit && fit.n === 2 ? 0.25 : 0;
      spread = quadSpread([
        0.15,
        horizonTerm,
        rmsTerm,
        disagreeTerm,
        slopeUnc,
        staleTerm,
        thinTerm,
      ]);
    } else if (distinct.length <= 1) {
      // Flat history: refresh prior, then the tier's log-time climb.
      const f = flatHistoryForecast(
        lastRate,
        lastLogit,
        capMove,
        flatNorm,
        t.tier,
        targetX,
        lastRead ? logHours(lastRead.at) : 0,
        horizonDays,
      );
      medLogit = f.medLogit;
      spread = f.spread;
    } else {
      // The page has refreshed at least once but this trophy has no
      // usable line (too few points, or no x-spread): persist the last
      // rate with a wide spread. Never re-apply the refresh prior
      // here — the dilution it prices has already happened.
      medLogit = lastLogit;
      const horizonTerm = 0.15 * Math.sqrt(horizonDays);
      spread = quadSpread([0.15, horizonTerm, 0.6]);
    }
    spread = clamp(spread, 0.2, 1.3);
    let prev = 0;
    const q = Z.map((z) => {
      const v = round4(invLogit(medLogit + z * spread));
      const clamped = Math.min(99.99, Math.max(0.01, Math.max(prev, v)));
      prev = clamped;
      return clamped;
    });
    return { id: t.id, q };
  });

  return { trophies: rows };
}

function main(): void {
  mkdirSync(OUTPUT_DIR, { recursive: true });
  const files = readdirSync(INPUT_DIR)
    .filter((f) => f.endsWith(".in"))
    .sort();
  for (const file of files) {
    const kase = JSON.parse(readFileSync(join(INPUT_DIR, file), "utf-8"));
    const forecast = forecastCase(kase);
    const outName = file.replace(/\.in$/, ".out");
    writeFileSync(join(OUTPUT_DIR, outName), JSON.stringify(forecast) + "\n", "utf-8");
  }
}

main();
