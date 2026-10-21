import {AbsoluteFill, useCurrentFrame, interpolate} from 'remotion';
import {C, W, H, ramp, inOut, lerp, projLocal, ISLANDS, LocalLand, Node, Reveal, Mono, Decode, ph, sceneDur, fmt, clamp, MONO, HEAD} from '../lib';

export const Genset: React.FC<{x: number; y: number; o: number; color?: string}> = ({x, y, o, color = C.amber}) => (
  <g transform={`translate(${x},${y})`} opacity={o}>
    <rect x={-16} y={-11} width={32} height={22} rx={3} fill="#0A1626" stroke={color} strokeWidth={2} />
    <path d="M-9 5 L-3 -5 L1 2 L6 -5" fill="none" stroke={color} strokeWidth={2} />
  </g>
);

export const Boat: React.FC<{x: number; y: number; o?: number; s?: number}> = ({x, y, o = 1, s = 1}) => (
  <g transform={`translate(${x},${y}) scale(${s})`} opacity={o}>
    <path d="M-34 0 L34 0 L24 14 L-26 14 Z" fill="#0A1626" stroke={C.amber} strokeWidth={2.5} />
    {[-20, -6, 8].map((bx) => <rect key={bx} x={bx} y={-14} width={11} height={14} rx={2} fill={C.amber} opacity={0.85} />)}
  </g>
);

export const S3: React.FC = () => {
  const f = useCurrentFrame();
  const dur = sceneDur(3);
  const clear = 1 - ramp(f, 0, 30);
  const cam = lerp(1.25, 1.0, ramp(f, 0, 90, inOut));
  const mainland = projLocal(79.98, 9.67);
  const boatT = ramp(f, ph(3, 6) - 4, dur, inOut);
  const [kx, ky] = projLocal(79.88, 9.6);
  const [dx, dy] = projLocal(79.70, 9.53);
  const bx = lerp(kx, dx, boatT), by = lerp(ky, dy, boatT);
  const linkAt = ph(3, 5);
  const link = ramp(f, linkAt, linkAt + 14);
  const cut = ramp(f, linkAt + 16, linkAt + 22);
  return (
    <AbsoluteFill style={{background: '#050B14'}}>
      <AbsoluteFill style={{transform: `translate(220px, 70px) scale(${cam})`}}>
        <LocalLand draw={ramp(f, 4, 70, inOut)} fill={ramp(f, 20, 70)} />
        <svg width={W} height={H} style={{position: 'absolute'}}>
          {ISLANDS.map((isl, i) => {
            const at = ph(3, i + 1) - 3;
            const [x, y] = projLocal(isl.lon, isl.lat);
            const o = ramp(f, at, at + 10);
            const lit = 1 - (f > linkAt ? 0 : 0);
            return (
              <g key={isl.name}>
                {/* dashed link to mainland */}
                <line x1={x} y1={y} x2={lerp(x, mainland[0], link)} y2={lerp(y, mainland[1], link)} stroke={cut > 0 ? C.red : C.dim} strokeWidth={2} strokeDasharray="8 8" opacity={link * (1 - cut * 0.6)} />
                {cut > 0 && (
                  <g transform={`translate(${lerp(x, mainland[0], 0.5)},${lerp(y, mainland[1], 0.5)}) scale(${cut})`}>
                    <path d="M-12 -12 L12 12 M12 -12 L-12 12" stroke={C.red} strokeWidth={5} strokeLinecap="round" style={{filter: `drop-shadow(0 0 8px ${C.red})`}} />
                  </g>
                )}
                <Node x={x} y={y} r={9} color={C.cyan} pulse={o > 0.5 ? 1 : 0} o={o * lit} />
                <Genset x={x + 26} y={y - 24} o={o} />
              </g>
            );
          })}
          <Boat x={bx} y={by} o={ramp(f, ph(3, 6) - 4, ph(3, 6) + 6)} s={0.9} />
          {f > ph(3, 6) && Array.from({length: 6}).map((_, i) => (
            <circle key={i} cx={bx + 30 + i * 18} cy={by + 10 - i * 4} r={2.5} fill={C.amber} opacity={0.5 - i * 0.08} />
          ))}
        </svg>
        {ISLANDS.map((isl, i) => {
          const at = ph(3, i + 1) - 3;
          const [x, y] = projLocal(isl.lon, isl.lat);
          const o = ramp(f, at, at + 10);
          const left = isl.dx < 0;
          return (
            <div key={isl.name} style={{position: 'absolute', left: left ? x + isl.dx - 300 : x + isl.dx, top: y + isl.dy - 20, width: 300, textAlign: left ? 'right' : 'left', opacity: o, transform: `translateY(${(1 - o) * 12}px)`}}>
              <div style={{fontFamily: HEAD, fontWeight: 700, fontSize: 34, color: C.white, letterSpacing: '0.08em'}}>{isl.name}</div>
              <Mono size={16} color={C.amber}>{fmt(isl.kwh)} kWh · 2025</Mono>
            </div>
          );
        })}
        <div style={{position: 'absolute', left: mainland[0] - 20, top: mainland[1] - 80, opacity: ramp(f, linkAt, linkAt + 10)}}>
          <Mono size={18} color={C.dim}>JAFFNA · NATIONAL GRID</Mono>
        </div>
      </AbsoluteFill>
      <div style={{position: 'absolute', left: 110, top: 90}}>
        <Reveal text="4 ISLANDED MICROGRIDS" at={ph(3, 0) + 6} size={58} />
        <div style={{marginTop: 14, opacity: ramp(f, linkAt + 18, linkAt + 30)}}>
          <Mono size={30} color={C.red}>0 LINKS TO THE NATIONAL GRID</Mono>
        </div>
        <div style={{marginTop: 10, opacity: ramp(f, ph(3, 6), ph(3, 6) + 12)}}>
          <Mono size={30} color={C.amber}>DIESEL ONLY · DELIVERED BY SEA</Mono>
        </div>
      </div>
      <div style={{position: 'absolute', right: 40, bottom: 18, fontFamily: MONO, fontSize: 12, color: C.dim, opacity: 0.7}}>Boundaries: Survey Dept. of Sri Lanka / OCHA via geoBoundaries, CC BY 3.0 IGO</div>
      <AbsoluteFill style={{background: '#CFE3FF', opacity: clear * 0.9}} />
    </AbsoluteFill>
  );
};
