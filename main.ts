/**
 * gta6-trophy-rarity-forecast-01 — forecast every GTA VI trophy's Exophase
 * earn rate at the case's target, as five quantiles per trophy.
 *
 * Node 22, standard library only, no network, deterministic.
 *
 * Model (written from scratch for this bounty):
 *  - Work on the logit scale, against x = ln(1 + hours since release
 *    2026-11-19T00:00:00Z). Log-time flattens growth the way a launch
 *    cohort's progression does: fast early gains, slower later ones.
 *  - Collapse the history to its distinct reads (Exophase refreshes a
 *    game's counts every few hours, so most reads repeat the last page
 *    verbatim; counting them would fake confidence).
 *  - Per trophy, a recency-weighted least-squares line of logit(rate)
 *    on x over the distinct reads (two points already give a line,
 *    trusted less). Slopes are shrunk toward the median slope of the
 *    story trophies climb fast, completion and grind trophies barely
 *    move, and a noisy per-trophy slope on a handful of refreshes is
 *    not trusted on its own.
 *  - The fitted median is blended back toward the last observed rate
 *    in proportion to the fit's residual, and the total move from the
 *    last rate is capped at 3.5 logit units.
 *  - When the history holds fewer than three distinct reads the page
 *    is stale (launch night: one flat page). A flat forecast would
 *    trust rates Exophase has not refreshed yet, so a refresh prior
 *    applies instead: the first refresh diluted every rate to about
 *    0.75x as the tracked-player count jumped (638 -> 3868 on the
 *    first refresh), then rates resume their log-time climb.
 *  - Spread (logit sd) combines a base term, a term growing with the
 *    square root of the horizon in days, the fit's residual, a
 *    staleness term for flat histories, and slope uncertainty over
 *    the extrapolation distance. Pinball loss punishes both
 *    too-narrow and too-wide, so the spread is honest, not minimal.
 *  - A trophy at 100% in the last read stays 100%: every refresh so
 *    far kept it there.
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

interface Obs {
  x: number;
  y: number;
  w: number;
  t: number; // ms since release, for recency weighting
}

interface TrophyFit {
  slope: number;
  intercept: number;
  rms: number;
  n: number;
  xLast: number;
  lastRate: number;
  lastT: number;
}

/** Weighted least squares of y on x; returns slope, intercept, rms. */
function wls(obs: Obs[]): { slope: number; intercept: number; rms: number } {
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
  return { slope, intercept, rms: Math.sqrt(se / sw) };
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
  // previous distinct read. Keep the first occurrence time of each.
  const distinct: any[] = [];
  let prevKey = "";
  for (const read of history) {
    const key = JSON.stringify([read.players_tracked, read.rates]);
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

  // Per-trophy fits on distinct reads.
  const fits: (TrophyFit | null)[] = trophies.map((t, i) => {
    const obs: Obs[] = [];
    for (const read of distinct) {
      const r = read.rates[i];
      if (r === null || r === undefined) continue;
      const tMs = Date.parse(read.at) - RELEASE_MS;
      // Recency weight: reads near the checkpoint count most.
      const ageH = Math.max(0, (checkpointMs - Date.parse(read.at)) / 3600000);
      const w = Math.pow(0.5, ageH / HALF_LIFE_HOURS);
      obs.push({ x: logHours(read.at), y: logit(r), w, t: tMs });
    }
    if (obs.length < 2) return null;
    const xs = obs.map((o) => o.x);
    if (Math.max(...xs) - Math.min(...xs) < 0.05) return null;
    const { slope, intercept, rms } = wls(obs);
    const lastRate = lastRates[i] ?? globalLastMedian;
    return {
      slope,
      intercept,
      rms,
      n: obs.length,
      xLast: logHours(lastRead.at),
      lastRate,
      lastT: Date.parse(lastRead.at),
    };
  });

  // Tier-group median slopes for hierarchical shrinkage.
  const groupSlope = new Map<string, number>();
  for (const tier of ["platinum", "gold", "silver", "bronze"]) {
    const slopes = fits
      .map((f, i) => (f && trophies[i].tier === tier ? f.slope : null))
      .filter((s): s is number => s !== null);
    groupSlope.set(tier, slopes.length ? median(slopes) : TIER_PRIOR_SLOPE[tier]);
  }
  const allSlopes = fits
    .map((f) => (f ? f.slope : null))
    .filter((s): s is number => s !== null);
  const globalSlope = allSlopes.length ? median(allSlopes) : 0.1;

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

  const rows = trophies.map((t, i) => {
    const lastRate = lastRates[i] ?? globalLastMedian;
    // A trophy every tracked player holds stays held.
    if (lastRate >= 99.99) {
      return { id: t.id, q: [100, 100, 100, 100, 100] };
    }
    const fit = fits[i];
    const lastLogit = logit(lastRate);
    let medLogit: number;
    let spread: number;
    if (fit) {
      // Shrink the slope toward its tier group: weight grows with the
      // number of distinct reads behind the fit.
      const g = groupSlope.get(t.tier) ?? globalSlope;
      const k = 2; // pseudo-count of group observations
      const slope = (fit.n * fit.slope + k * g) / (fit.n + k);
      let pred = fit.intercept + slope * targetX;
      // Blend toward the last rate when the fit is noisy, and trust
      // a two-point line less than a many-read fit.
      const wFit =
        Math.max(0.35, 1 - fit.rms / 2) * (0.6 + 0.1 * Math.min(fit.n, 4));
      pred = wFit * pred + (1 - wFit) * lastLogit;
      // Cap the total move from the last observed rate.
      pred = Math.min(lastLogit + 3.5, Math.max(lastLogit - 3.5, pred));
      medLogit = pred;
      const horizonTerm = 0.45 * Math.sqrt(horizonDays);
      const slopeUnc = 0.25 * Math.abs(slope) * Math.max(0, targetX - fit.xLast);
      const staleTerm = staleHours >= 6 ? 0.3 : 0;
      const thinTerm = fit.n === 2 ? 0.4 : 0;
      spread = Math.sqrt(
        0.3 * 0.3 +
          horizonTerm * horizonTerm +
          (0.5 * fit.rms) * (0.5 * fit.rms) +
          staleTerm * staleTerm +
          slopeUnc * slopeUnc +
          thinTerm * thinTerm,
      );
    } else if (distinct.length <= 1) {
      // Flat history: refresh prior, then the tier's log-time climb.
      // Power-law medians: easy trophies keep more of their rate.
      const g = TIER_PRIOR_SLOPE[t.tier] ?? 0.15;
      const xLast = lastRead ? logHours(lastRead.at) : 0;
      const climb = g * Math.max(0, targetX - xLast);
      const norm = flatNorm;
      const powered = Math.pow(lastRate, REFRESH_POWER);
      medLogit = logit(norm * powered) + climb;
      medLogit = Math.min(lastLogit + 3.5, Math.max(lastLogit - 3.5, medLogit));
      const horizonTerm = 0.45 * Math.sqrt(horizonDays);
      spread = Math.sqrt(0.3 * 0.3 + horizonTerm * horizonTerm + 0.6 * 0.6);
    } else {
      // The page has refreshed at least once but this trophy has no
      // usable line (too few points, or no x-spread): persist the last
      // rate with a wide spread. Never re-apply the refresh prior
      // here — the dilution it prices has already happened.
      medLogit = lastLogit;
      const horizonTerm = 0.45 * Math.sqrt(horizonDays);
      spread = Math.sqrt(0.3 * 0.3 + horizonTerm * horizonTerm + 0.8 * 0.8);
    }
    spread = Math.min(1.6, Math.max(0.35, spread));
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
