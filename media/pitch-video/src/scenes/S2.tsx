import {AbsoluteFill, useCurrentFrame, interpolate} from 'remotion';
import {C, W, H, ramp, inOut, lerp, projRegion, Reveal, Mono, Decode, ph, sceneDur, clamp, MONO} from '../lib';
import geo from '../geo.json';

const D2R = Math.PI / 180;
function globePath(lam0: number, phi0: number, R: number, cx: number, cy: number) {
  let d = '';
  for (const ring of geo.world as number[][][]) {
    let pen = false;
    for (const [lon, lat] of ring) {
      const l = lon * D2R, p = lat * D2R, L0 = lam0 * D2R, P0 = phi0 * D2R;
      const cosc = Math.sin(P0) * Math.sin(p) + Math.cos(P0) * Math.cos(p) * Math.cos(l - L0);
      if (cosc < 0) { pen = false; continue; }
      const x = cx + R * Math.cos(p) * Math.sin(l - L0);
      const y = cy - R * (Math.cos(P0) * Math.sin(p) - Math.sin(P0) * Math.cos(p) * Math.cos(l - L0));
      d += (pen ? 'L' : 'M') + x.toFixed(1) + ',' + y.toFixed(1);
      pen = true;
    }
  }
  return d;
}

const TRACK: [number, number][] = [[83.2, 5.6], [82.3, 6.9], [81.6, 8.2], [81.0, 9.5], [80.6, 10.8], [80.4, 12.0]];
const along = (t: number) => {
  const n = TRACK.length - 1, x = Math.min(n - 0.0001, Math.max(0, t * n)), i = Math.floor(x), u = x - i;
  return [lerp(TRACK[i][0], TRACK[i + 1][0], u), lerp(TRACK[i][1], TRACK[i + 1][1], u)];
};

export const Cyclone: React.FC<{x: number; y: number; s: number; rot: number; o?: number}> = ({x, y, s, rot, o = 1}) => {
  const arms = 5;
  const paths: {d: string; w: number; op: number}[] = [];
  for (let a = 0; a < arms; a++) {
    for (let layer = 0; layer < 3; layer++) {
      let d = '';
      for (let i = 0; i <= 70; i++) {
        const r = 14 + i * 4.2 + layer * 6;
        const th = rot + (a / arms) * Math.PI * 2 + Math.log(r / 10) * 1.6 + layer * 0.12;
        d += (i ? 'L' : 'M') + (Math.cos(th) * r).toFixed(1) + ',' + (Math.sin(th) * r * 0.92).toFixed(1);
      }
      paths.push({d, w: 26 - layer * 7, op: 0.18 + layer * 0.08});
    }
  }
  return (
    <g transform={`translate(${x},${y}) scale(${s})`} opacity={o} filter="url(#cblur)">
      <circle r={300} fill="url(#cyc)" />
      {paths.map((p, i) => (
        <path key={i} d={p.d} fill="none" stroke="#DDEBFF" strokeWidth={p.w} strokeLinecap="round" opacity={p.op} />
      ))}
      <circle r={12} fill="#0A1626" />
    </g>
  );
};

export const S2: React.FC = () => {
  const f = useCurrentFrame();
  const dur = sceneDur(2);
  // globe phase
  const g = ramp(f, 0, 70, inOut);
  const R = lerp(360, 2600, g);
  const lam = lerp(55, 80.6, ramp(f, 0, 60, inOut));
  const phi = lerp(-5, 8, ramp(f, 0, 60, inOut));
  const globeO = 1 - ramp(f, 55, 80);
  const regionO = ramp(f, 55, 80);
  // cyclone along track
  const t = interpolate(f, [30, dur - 10], [0.05, 0.75], clamp);
  const [clon, clat] = along(t);
  const [cx, cy] = projRegion(clon, clat);
  const zoomOut = 1 - ramp(f, dur - 26, dur, inOut);
  const camS = lerp(1, 5, ramp(f, dur - 30, dur, inOut));
  const [jx, jy] = projRegion(79.85, 9.62);
  const fog = ramp(f, dur - 18, dur);
  // track drawn so far
  const trackD = Array.from({length: 40}, (_, i) => projRegion(...(along((i / 39) * t) as [number, number]))).map((p, i) => (i ? 'L' : 'M') + p[0].toFixed(1) + ',' + p[1].toFixed(1)).join('');
  return (
    <AbsoluteFill style={{background: '#02060C'}}>
      {/* stars */}
      <svg width={W} height={H} style={{position: 'absolute', opacity: globeO}}>
        {Array.from({length: 140}).map((_, i) => {
          const x = (i * 9301 + 49297) % 233280 / 233280 * W, y = (i * 4253 + 1931) % 104729 / 104729 * H;
          return <circle key={i} cx={x} cy={y} r={(i % 3) * 0.6 + 0.4} fill="#fff" opacity={0.25 + (i % 5) * 0.1} />;
        })}
      </svg>
      <svg width={W} height={H} style={{position: 'absolute', opacity: globeO}}>
        <defs>
          <radialGradient id="atm"><stop offset="85%" stopColor="#34E1FF" stopOpacity={0} /><stop offset="100%" stopColor="#34E1FF" stopOpacity={0.35} /></radialGradient>
        </defs>
        <circle cx={W / 2} cy={H / 2} r={R * 1.04} fill="url(#atm)" />
        <circle cx={W / 2} cy={H / 2} r={R} fill="#071526" stroke="#34E1FF55" strokeWidth={2} />
        <path d={globePath(lam, phi, R, W / 2, H / 2)} fill="none" stroke={C.cyan} strokeWidth={1.4} opacity={0.8} />
      </svg>
      <AbsoluteFill style={{opacity: regionO, transform: `scale(${camS})`, transformOrigin: `${jx}px ${jy}px`}}>
        <svg width={W} height={H} style={{position: 'absolute'}}>
          <defs>
            <filter id="cblur" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="6" /></filter>
            <radialGradient id="cyc"><stop offset="0%" stopColor="#CFE3FF" stopOpacity={0.25} /><stop offset="100%" stopColor="#CFE3FF" stopOpacity={0} /></radialGradient>
          </defs>
          <path d={geo.region.india} fill="#0C1C30" stroke="none" />
          <path d={geo.region.sl} fill="#0F2A44" stroke={C.cyan} strokeWidth={2} style={{filter: `drop-shadow(0 0 8px ${C.cyan})`}} />
          <path d={trackD} fill="none" stroke={C.amber} strokeWidth={3} strokeDasharray="10 8" opacity={0.9} />
          <circle cx={jx} cy={jy} r={6} fill={C.amber} opacity={ramp(f, 80, 95)} />
          <Cyclone x={cx} y={cy} s={0.9} rot={-f * 0.06} o={ramp(f, 30, 60)} />
        </svg>
        <div style={{position: 'absolute', left: jx - 250, top: jy - 40, opacity: ramp(f, 85, 100) * zoomOut}}>
          <Mono size={20} color={C.amber}>JAFFNA ISLANDS ▸</Mono>
        </div>
      </AbsoluteFill>
      <div style={{position: 'absolute', left: 110, bottom: 170, opacity: zoomOut}}>
        <Decode text="26–29 NOV 2025" at={ph(2, 0)} size={30} color={C.amber} />
        <Reveal text="CYCLONE DITWAH" at={ph(2, 1)} size={120} spacing="0.06em" glow />
      </div>
      <AbsoluteFill style={{background: '#CFE3FF', opacity: fog * 0.9}} />
    </AbsoluteFill>
  );
};
