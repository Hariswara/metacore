import {AbsoluteFill, useCurrentFrame} from 'remotion';
import {C, W, H, ramp, inOut, lerp, Mono, ph, sceneDur, Reveal, Backdrop, Rewind, Stamp, MONO, HEAD} from '../lib';
import {OneLine} from '../OneLine';

const Chip: React.FC<{at: number; x: number; title: string; body: string; col: string}> = ({at, x, title, body, col}) => {
  const f = useCurrentFrame();
  const o = ramp(f, at, at + 10);
  return (
    <div style={{position: 'absolute', left: x, top: 250, width: 380, padding: '18px 20px', borderRadius: 12, border: `2px solid ${col}`, background: 'rgba(10,22,38,0.92)', opacity: o, transform: `translateY(${(1 - o) * 24}px)`, boxShadow: `0 0 ${30 + 10 * Math.sin(f * 0.25)}px ${col}55`}}>
      <Mono size={16} color={C.dim}>{title}</Mono>
      <Mono size={24} color={col} style={{marginTop: 6, whiteSpace: 'nowrap', letterSpacing: '0.06em'}}>{body}</Mono>
    </div>
  );
};

export const S10: React.FC = () => {
  const f = useCurrentFrame();
  const dur = sceneDur(10);
  const p0 = ph(10, 0), p1 = ph(10, 1), p2 = ph(10, 2), p3 = ph(10, 3);
  const arrows = [ramp(f, p1 - 6, p1 + 4), ramp(f, p1 + 18, p1 + 26), ramp(f, p2 - 6, p2 + 4)];
  const pts: string[] = [];
  for (let x = 0; x <= 640; x += 4) pts.push(`${1180 + x},${880 - Math.sin(x / 50 + f * 0.25) * 34}`);
  return (
    <AbsoluteFill style={{background: '#050B14'}}>
      <Backdrop grid={0.5} />
      <div style={{position: 'absolute', left: 110, top: 80}}>
        <Reveal text="SAME STORM. SAME PROPOSAL." at={p0} size={58} stagger={0.6} />
      </div>
      <Chip at={p0 + 8} x={110} title="PROPOSAL · TELEMETRY STALE" body="RESTORE FEEDER F2" col={C.white} />
      <Chip at={p1 - 2} x={560} title="M2 · DOUBT" body="UNCERTAINTY HIGH" col={C.amber} />
      <Chip at={p1 + 20} x={1010} title="M3 · DECIDE" body="→ SYSTEM 2" col={C.amber} />
      <svg width={W} height={H} style={{position: 'absolute'}}>
        {[497, 947, 1397].map((x, i) => (
          <path key={i} d={`M${x} 300 h52 m-14 -12 l14 12 l-14 12`} stroke={C.cyan} strokeWidth={4} fill="none" opacity={arrows[i]} />
        ))}
      </svg>
      <div style={{position: 'absolute', left: 1470, top: 236}}>
        <Stamp label="REJECT" color={C.red} at={p2} size={52} />
        <Mono size={16} color={C.red} style={{marginTop: 14, opacity: ramp(f, p2 + 6, p2 + 14)}}>GENERATION ADEQUACY ✕</Mono>
      </div>
      <div style={{position: 'absolute', left: 1470, top: 400, opacity: ramp(f, p2 + 26, p2 + 36)}}>
        <Mono size={16} color={C.dim}>REVISED · HOLD F2, KEEP F1</Mono>
        <div style={{marginTop: 8}}><Stamp label="APPROVE" color={C.green} at={p2 + 30} size={40} /></div>
      </div>
      <svg width={W} height={H} style={{position: 'absolute'}}>
        <OneLine x={150} y={520} s={0.8} f2On={0} dark={0} label="ISLAND STAYS ENERGISED" />
        <polyline points={pts.join(' ')} fill="none" stroke={C.green} strokeWidth={4} style={{filter: `drop-shadow(0 0 8px ${C.green})`}} opacity={ramp(f, p3 - 6, p3 + 6)} />
      </svg>
      <div style={{position: 'absolute', left: 1180, top: 640, opacity: ramp(f, p3 - 6, p3 + 6)}}>
        <Mono size={20} color={C.dim}>GRID FREQUENCY</Mono>
        <div style={{fontFamily: MONO, fontWeight: 700, fontSize: 88, color: C.green, textShadow: `0 0 24px ${C.green}`}}>50.00 Hz</div>
      </div>
      <div style={{position: 'absolute', right: 110, top: 96, display: 'flex', gap: 26}}>
        {[{t: 'CAUGHT', at: p1, c: C.amber}, {t: 'BLOCKED', at: p2, c: C.red}, {t: 'SAFE', at: p3, c: C.green}].map((x) => (
          <div key={x.t} style={{fontFamily: HEAD, fontWeight: 700, fontSize: 40, letterSpacing: '0.12em', color: x.c, opacity: ramp(f, x.at, x.at + 8)}}>{x.t}</div>
        ))}
      </div>
      <Rewind from={0} to={p0 - 2} />
    </AbsoluteFill>
  );
};
