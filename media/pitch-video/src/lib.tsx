import React from 'react';
import {interpolate, Easing, useCurrentFrame, AbsoluteFill} from 'remotion';
import tl from './timeline.json';
import geo from './geo.json';

export const C = {
  bg: '#050B14', bg2: '#0A1626', cyan: '#34E1FF', amber: '#FFB020', red: '#FF3B4F',
  green: '#3DFFA2', violet: '#A78BFA', white: '#EAF6FF', dim: '#5B7A99', grid: '#12304A',
};
export const HEAD = '"Space Grotesk", sans-serif';
export const MONO = '"JetBrains Mono", monospace';
export const W = 1920, H = 1080;

export const outExpo = Easing.bezier(0.16, 1, 0.3, 1);
export const inOut = Easing.bezier(0.65, 0, 0.35, 1);
export const clamp = {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'} as const;
export const ramp = (f: number, a: number, b: number, e = outExpo) => interpolate(f, [a, b], [0, 1], {...clamp, easing: e});
export const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

/** frame offset (scene-relative) at which phrase i of scene s starts */
export const ph = (s: number, i: number) => tl.scenes[s - 1].phrases[i]?.f ?? 0;
export const phEnd = (s: number, i: number) => {
  const p = tl.scenes[s - 1].phrases[i];
  return p ? p.f + p.d : 0;
};
export const sceneDur = (s: number) => tl.scenes[s - 1].dur;

// ---------- projections ----------
export const projLocal = (lon: number, lat: number) => {
  const g = geo.local;
  const c = Math.cos((g.lat0 * Math.PI) / 180);
  return [(lon - g.lon0) * g.k * c + W / 2, -(lat - g.lat0) * g.k + H / 2] as const;
};
export const projRegion = (lon: number, lat: number) => {
  const g = geo.region;
  const c = Math.cos((g.lat0 * Math.PI) / 180);
  return [(lon - g.lon0) * g.k * c + W / 2, -(lat - g.lat0) * g.k + H / 2] as const;
};

export const ISLANDS = [
  {name: 'DELFT', sub: 'NEDUNTIVU PLANT', lon: 79.678, lat: 9.523, kwh: 1126880, dx: -40, dy: 70},
  {name: 'NAINATIVU', sub: '', lon: 79.766, lat: 9.604, kwh: 1198460, dx: -40, dy: 50},
  {name: 'ANALAITIVU', sub: '', lon: 79.773, lat: 9.664, kwh: 426418, dx: -50, dy: -20},
  {name: 'ELUVAITIVU', sub: 'DIESEL + HYBRID', lon: 79.809, lat: 9.698, kwh: 196607, dx: 50, dy: -40},
];

// ---------- shared layers ----------
export const Backdrop: React.FC<{grid?: number; children?: React.ReactNode}> = ({grid = 0.5, children}) => {
  const f = useCurrentFrame();
  return (
    <AbsoluteFill style={{background: `radial-gradient(ellipse at 50% 45%, ${C.bg2} 0%, ${C.bg} 70%)`}}>
      <svg width={W} height={H} style={{position: 'absolute', opacity: grid * 0.35}}>
        {Array.from({length: 33}).map((_, i) => (
          <line key={'v' + i} x1={i * 60} y1={0} x2={i * 60} y2={H} stroke={C.grid} strokeWidth={1} />
        ))}
        {Array.from({length: 19}).map((_, i) => (
          <line key={'h' + i} x1={0} y1={i * 60} x2={W} y2={i * 60} stroke={C.grid} strokeWidth={1} />
        ))}
      </svg>
      {children}
    </AbsoluteFill>
  );
};

export const Overlay: React.FC = () => {
  const f = useCurrentFrame();
  return (
    <AbsoluteFill style={{pointerEvents: 'none'}}>
      <AbsoluteFill style={{background: 'radial-gradient(ellipse at center, rgba(0,0,0,0) 55%, rgba(0,0,0,0.65) 100%)'}} />
      <AbsoluteFill
        style={{
          backgroundImage: 'repeating-linear-gradient(0deg, rgba(255,255,255,0.025) 0px, rgba(255,255,255,0.025) 1px, transparent 1px, transparent 3px)',
          transform: `translateY(${(f * 0.5) % 3}px)`,
        }}
      />
    </AbsoluteFill>
  );
};

export const Mono: React.FC<{children: React.ReactNode; size?: number; color?: string; style?: React.CSSProperties}> = ({children, size = 22, color = C.cyan, style}) => (
  <div style={{fontFamily: MONO, fontSize: size, color, letterSpacing: '0.12em', ...style}}>{children}</div>
);

/** letter-by-letter reveal */
export const Reveal: React.FC<{text: string; at: number; size?: number; color?: string; font?: string; weight?: number; spacing?: string; stagger?: number; style?: React.CSSProperties; glow?: boolean}> = ({
  text, at, size = 64, color = C.white, font = HEAD, weight = 700, spacing = '0.04em', stagger = 1.2, style, glow,
}) => {
  const f = useCurrentFrame();
  return (
    <div style={{fontFamily: font, fontSize: size, fontWeight: weight, color, letterSpacing: spacing, whiteSpace: 'pre', textShadow: glow ? `0 0 24px ${color}88` : undefined, ...style}}>
      {text.split('').map((ch, i) => {
        const t = ramp(f, at + i * stagger, at + i * stagger + 14);
        return (
          <span key={i} style={{display: 'inline-block', opacity: t, transform: `translateY(${(1 - t) * 0.5}em)`, filter: `blur(${(1 - t) * 8}px)`}}>
            {ch === ' ' ? ' ' : ch}
          </span>
        );
      })}
    </div>
  );
};

/** decode / scramble text reveal for readouts */
const GLYPHS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789#%/<>';
export const Decode: React.FC<{text: string; at: number; dur?: number; size?: number; color?: string; style?: React.CSSProperties}> = ({text, at, dur = 18, size = 22, color = C.cyan, style}) => {
  const f = useCurrentFrame();
  const n = text.length;
  const p = (f - at) / dur;
  if (p < 0) return null;
  const shown = text
    .split('')
    .map((ch, i) => {
      if (ch === ' ') return ' ';
      if (p * n * 1.0 > i + 2) return ch;
      if (p * n > i) return GLYPHS[(i * 7 + f * 3) % GLYPHS.length];
      return ' ';
    })
    .join('');
  return <Mono size={size} color={color} style={{whiteSpace: 'pre', ...style}}>{shown}</Mono>;
};

export const fmt = (n: number, d = 0) => n.toLocaleString('en-US', {minimumFractionDigits: d, maximumFractionDigits: d});

/** Caption bar for the narration */
export const Captions: React.FC<{scene: number}> = ({scene}) => {
  const f = useCurrentFrame();
  const s = tl.scenes[scene - 1];
  const cur = s.phrases.find((p) => f >= p.f - 2 && f < p.f + p.d + 8);
  if (!cur) return null;
  const o = Math.min(ramp(f, cur.f - 2, cur.f + 4), 1 - ramp(f, cur.f + cur.d + 2, cur.f + cur.d + 8));
  return (
    <div style={{position: 'absolute', bottom: 54, width: '100%', display: 'flex', justifyContent: 'center', opacity: o}}>
      <div style={{fontFamily: HEAD, fontWeight: 500, fontSize: 30, color: C.white, background: 'rgba(5,11,20,0.62)', padding: '8px 22px', borderRadius: 6, maxWidth: 1400, textAlign: 'center', lineHeight: 1.35}}>
        {cur.text}
      </div>
    </div>
  );
};

/** Glowing node */
export const Node: React.FC<{x: number; y: number; r?: number; color?: string; pulse?: number; o?: number}> = ({x, y, r = 10, color = C.cyan, pulse = 0, o = 1}) => {
  const f = useCurrentFrame();
  const ring = ((f % 45) / 45);
  return (
    <g opacity={o}>
      <circle cx={x} cy={y} r={r * 4} fill={color} opacity={0.08} />
      {pulse > 0 && <circle cx={x} cy={y} r={r + ring * 50 * pulse} fill="none" stroke={color} strokeWidth={2} opacity={(1 - ring) * 0.7} />}
      <circle cx={x} cy={y} r={r * 2} fill={color} opacity={0.25} />
      <circle cx={x} cy={y} r={r} fill={color} />
      <circle cx={x} cy={y} r={r * 0.45} fill="#fff" opacity={0.9} />
    </g>
  );
};

/** Verdict stamp */
export const Stamp: React.FC<{label: string; color: string; at: number; size?: number; style?: React.CSSProperties}> = ({label, color, at, size = 64, style}) => {
  const f = useCurrentFrame();
  if (f < at) return null;
  const t = ramp(f, at, at + 8, Easing.out(Easing.back(2)));
  const s = lerp(1.8, 1, t);
  return (
    <div style={{display: 'inline-block', transform: `scale(${s}) rotate(-4deg)`, opacity: Math.min(1, t * 1.5), border: `5px solid ${color}`, color, padding: '8px 26px', fontFamily: HEAD, fontWeight: 700, fontSize: size, letterSpacing: '0.12em', borderRadius: 8, boxShadow: `0 0 40px ${color}55, inset 0 0 30px ${color}33`, textShadow: `0 0 18px ${color}`, ...style}}>
      {label}
    </div>
  );
};

/** Island map coastline (local view) */
export const LocalLand: React.FC<{draw?: number; fill?: number; stroke?: string; glow?: boolean}> = ({draw = 1, fill = 1, stroke = C.cyan, glow = true}) => (
  <svg width={W} height={H} style={{position: 'absolute'}}>
    <defs>
      <filter id="glow"><feGaussianBlur stdDeviation="4" result="b" /><feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
    </defs>
    <path d={geo.local.path} fill={`rgba(52,225,255,${0.06 * fill})`} stroke="none" />
    {glow && <path d={geo.local.path} fill="none" stroke={stroke} strokeWidth={7} pathLength={1} strokeDasharray="1 1" strokeDashoffset={1 - draw} opacity={0.18} />}
    <path d={geo.local.path} fill="none" stroke={stroke} strokeWidth={1.6} pathLength={1} strokeDasharray="1 1" strokeDashoffset={1 - draw} opacity={0.9} />
  </svg>
);

/** RGB-split glitch wrapper */
export const Glitch: React.FC<{amount: number; children: React.ReactNode}> = ({amount, children}) => {
  const f = useCurrentFrame();
  if (amount <= 0.001) return <AbsoluteFill>{children}</AbsoluteFill>;
  const j = Math.sin(f * 12.9898) * 43758.5453;
  const r = (j - Math.floor(j) - 0.5) * 30 * amount;
  return (
    <AbsoluteFill>
      <AbsoluteFill style={{transform: `translateX(${r}px)`, mixBlendMode: 'screen', opacity: 0.7, filter: 'hue-rotate(-40deg)'}}>{children}</AbsoluteFill>
      <AbsoluteFill style={{transform: `translateX(${-r}px)`, mixBlendMode: 'screen', opacity: 0.7, filter: 'hue-rotate(60deg)'}}>{children}</AbsoluteFill>
      <AbsoluteFill style={{clipPath: `inset(${(f * 37) % 90}% 0 ${100 - ((f * 37) % 90) - 6}% 0)`, transform: `translateX(${r * 2}px)`}}>{children}</AbsoluteFill>
    </AbsoluteFill>
  );
};

/** VHS rewind overlay */
export const Rewind: React.FC<{from: number; to: number}> = ({from, to}) => {
  const f = useCurrentFrame();
  if (f < from || f > to) return null;
  const p = (f - from) / (to - from);
  const o = Math.sin(p * Math.PI);
  return (
    <AbsoluteFill style={{opacity: o, pointerEvents: 'none'}}>
      <AbsoluteFill style={{background: 'rgba(5,11,20,0.4)'}} />
      {Array.from({length: 7}).map((_, i) => {
        const y = ((i * 173 + f * 61) % H);
        return <div key={i} style={{position: 'absolute', left: 0, top: y, width: W, height: 4 + (i % 3) * 6, background: 'rgba(234,246,255,0.25)', filter: 'blur(1px)'}} />;
      })}
      <div style={{position: 'absolute', top: 70, left: 90, fontFamily: MONO, fontSize: 44, color: C.white, letterSpacing: '0.1em'}}>◀◀ REWIND</div>
    </AbsoluteFill>
  );
};

/** ECG / grid-frequency trace */
export const ecg = (x: number) => {
  const p = ((x % 1) + 1) % 1;
  const g = (c: number, w: number, a: number) => a * Math.exp(-((p - c) ** 2) / (2 * w * w));
  return g(0.18, 0.025, 0.12) - g(0.38, 0.008, 0.18) + g(0.41, 0.01, 1) - g(0.445, 0.01, 0.32) + g(0.65, 0.04, 0.22);
};
