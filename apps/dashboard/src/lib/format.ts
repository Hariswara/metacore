/** Formatters. Fixed locale so a screenshot reads the same for everyone reviewing it. */

const LOCALE = "en-GB";

export const kwh = (v: number): string =>
  new Intl.NumberFormat(LOCALE, { maximumFractionDigits: 0 }).format(v);

export const litres = kwh;

export const rupees = (v: number): string =>
  new Intl.NumberFormat(LOCALE, { maximumFractionDigits: 0 }).format(v);

/** Large money, in millions — the fleet spends hundreds of millions a year. */
export const rupeesM = (v: number): string =>
  `${new Intl.NumberFormat(LOCALE, { maximumFractionDigits: 1 }).format(v / 1e6)}M`;

export const gwh = (v: number): string =>
  `${new Intl.NumberFormat(LOCALE, { maximumFractionDigits: 2 }).format(v / 1e6)}`;

export const decimal = (v: number, places = 2): string =>
  new Intl.NumberFormat(LOCALE, {
    minimumFractionDigits: places,
    maximumFractionDigits: places,
  }).format(v);

export const percent = (v: number, places = 1): string => `${decimal(v * 100, places)}%`;

/** "Eluvaitivu-Hybrid" reads as a key, not a place. */
export const systemLabel = (system: string): string => system.replace("-", " · ");
