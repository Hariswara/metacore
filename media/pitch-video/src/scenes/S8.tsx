import {AbsoluteFill, useCurrentFrame, interpolate} from 'remotion';
import {C, W, H, ramp, inOut, lerp, Mono, ph, sceneDur, Reveal, Backdrop, clamp, MONO, HEAD} from '../lib';

const STATIONS = [
  {x: 960, tag: 'M1 · SEE', sub: 'SPATIOTEMPORAL GRAPH NETWORK', col: C.cyan},
  {x: 960 + 1500, tag: 'M2 · DOUBT', sub: 'EVIDENTIAL UNCERTAINTY', col: C.amber},
  {x: 960 + 3000, tag: 'M3 · DECIDE', sub: 'METACOGNITIVE GATING', col: C.green},
  {x: 960 + 4500, tag: 'M4 · VERIFY', sub: 'PHYSICS FIREWALL', col: C.red},
];
const rnd = (i: number) => { const x = Math.sin(i * 91.7 + 13.3) * 43758.5453; return x - Math.floor(x); };

const Layers: React.FC<{cx: number; cy: number; t: number}> = ({cx, cy, t}) => {
  const f = useCurrentFrame();
  const names = ['GENERATION', 'LOAD', 'WEATHER'];
  const nodes = Array.from({length: 14}, (_, i) => [rnd(i) * 420 + 40, rnd(i + 50) * 420 + 40]);
  const edges: [number, number][] = [];
  nodes.forEach((a, i) => nodes.forEach((b, j) => { if (j > i && Math.hypot(a[0] - b[0], a[1] - b[1]) < 150) edges.push([i, j]); }));
  return (
    <div style={{position: 'absolute', left: cx - 250, top: cy - 300, width: 500, height: 600, perspective: 1600}}>
      {names.map((n, li) => {
        const o = ramp(t, li * 12, li * 12 + 20);
        const z = (li - 1) * 150 * o;
        return (
          <div key={n} style={{position: 'absolute', left: 0, top: 50, width: 500, height: 500, transform: `rotateX(58deg) rotateZ(-38deg) translateZ(${z}px)`, opacity: o}}>
            <svg width={500} height={500}>
              <rect x={2} y={2} width={496} height={496} rx={18} fill="rgba(52,225,255,0.06)" stroke={li === 2 ? C.amber : C.cyan} strokeWidth={2} />
              {edges.map(([a, b], k) => <line key={k} x1={nodes[a][0]} y1={nodes[a][1]} x2={nodes[b][0]} y2={nodes[b][1]} stroke={C.cyan} strokeWidth={1.5} opacity={0.35} />)}
              {nodes.map((p, k) => <circle key={k} cx={p[0]} cy={p[1]} r={6 + 3 * Math.sin(f * 0.15 + k + li)} fill={li === 2 ? C.amber : C.cyan} opacity={0.9} />)}
              <text x={24} y={480} fill={C.white} fontFamily="JetBrains Mono" fontSize={26} letterSpacing={4}>{n}</text>
            </svg>
          </div>
        );
      })}
    </div>
  );
};

export const S8: React.FC = () => {
  const f = useCurrentFrame();
  const dur = sceneDur(8);
  const p0 = ph(8, 0), p1 = ph(8, 1), p2 = ph(8, 2), p3 = ph(8, 3);
  const camX = interpolate(f, [0, p1 - 14, p1 + 6, p2 - 12, p2 + 8, dur], [0, 0, 1500, 1500, 3000, 3080], {...clamp, easing: inOut});
  const active = f < p1 - 4 ? 0 : f < p2 - 4 ? 1 : 2;
  const needle = interpolate(f, [p1 + 4, p1 + 34], [-80, 62], {...clamp, easing: inOut}) + Math.sin(f * 0.9) * 2;
  const s3x = STATIONS[2].x;
  const fastT = ramp(f, p2 + 6, p2 + 26, (x) => x);
  const slowT = ramp(f, p3 + 4, dur, inOut);
  return (
    <AbsoluteFill style={{background: '#050B14'}}>
      <Backdrop grid={0.4} />
      <AbsoluteFill style={{transform: `translateX(${-camX}px)`}}>
        {/* rail */}
        <svg width={6000} height={H} style={{position: 'absolute'}}>
          <line x1={0} y1={900} x2={6000} y2={900} stroke={C.grid} strokeWidth={4} />
          {Array.from({length: 30}).map((_, i) => <circle key={i} cx={((i * 200 + f * 18) % 6000)} cy={900} r={4} fill={C.cyan} opacity={0.6} />)}
          {STATIONS.map((s, i) => (
            <g key={i}>
              <circle cx={s.x} cy={900} r={16} fill="#050B14" stroke={s.col} strokeWidth={4} />
              <text x={s.x} y={960} fill={s.col} fontFamily="JetBrains Mono" fontSize={24} textAnchor="middle" letterSpacing={4}>{s.tag}</text>
            </g>
          ))}
        </svg>
        {/* M1 */}
        <Layers cx={STATIONS[0].x} cy={460} t={f - p0} />
        {/* M2 gauge + proposal */}
        <div style={{position: 'absolute', left: STATIONS[1].x - 420, top: 230, width: 840, height: 560}}>
          <div style={{position: 'absolute', left: 230, top: 0, padding: '16px 28px', borderRadius: 12, border: `2px solid ${C.amber}`, background: 'rgba(10,22,38,0.9)', boxShadow: `0 0 ${40 + 20 * Math.sin(f * 0.2)}px ${C.amber}88`}}>
            <Mono size={18} color={C.dim}>PROPOSED ACTION</Mono>
            <Mono size={34} color={C.white}>RESTORE FEEDER F2</Mono>
          </div>
          <svg width={840} height={460} style={{position: 'absolute', top: 120}}>
            <path d="M120 400 A300 300 0 0 1 720 400" fill="none" stroke={C.grid} strokeWidth={30} />
            <path d="M120 400 A300 300 0 0 1 720 400" fill="none" stroke="url(#ug)" strokeWidth={30} pathLength={1} strokeDasharray="1 1" strokeDashoffset={1 - ramp(f, p1 - 10, p1 + 10)} />
            <defs><linearGradient id="ug"><stop offset="0" stopColor={C.green} /><stop offset="0.5" stopColor={C.amber} /><stop offset="1" stopColor={C.red} /></linearGradient></defs>
            <g transform={`translate(420 400) rotate(${needle})`}><line x1={0} y1={0} x2={0} y2={-260} stroke="#fff" strokeWidth={6} strokeLinecap="round" /><circle r={16} fill="#fff" /></g>
            <text x={110} y={450} fill={C.green} fontFamily="JetBrains Mono" fontSize={22}>LOW</text>
            <text x={680} y={450} fill={C.red} fontFamily="JetBrains Mono" fontSize={22}>HIGH</text>
            <text x={420} y={450} fill={C.amber} fontFamily="JetBrains Mono" fontSize={24} textAnchor="middle" letterSpacing={3}>EPISTEMIC UNCERTAINTY</text>
          </svg>
        </div>
        {/* M3 fork */}
        <svg width={6000} height={H} style={{position: 'absolute'}}>
          <path d={`M${s3x - 620} 520 H${s3x - 260} C${s3x - 120} 520 ${s3x - 120} 330 ${s3x + 40} 330 H${s3x + 640}`} fill="none" stroke={C.green} strokeWidth={6} opacity={0.8} />
          <path d={`M${s3x - 260} 520 C${s3x - 120} 520 ${s3x - 120} 690 ${s3x + 40} 690 H${s3x + 640}`} fill="none" stroke={C.amber} strokeWidth={6} opacity={0.8} />
          <text x={s3x + 60} y={300} fill={C.green} fontFamily="JetBrains Mono" fontSize={26} letterSpacing={3}>SYSTEM 1 · FAST</text>
          <text x={s3x + 60} y={750} fill={C.amber} fontFamily="JetBrains Mono" fontSize={26} letterSpacing={3}>SYSTEM 2 · DELIBERATE</text>
          {/* fast packet */}
          {fastT > 0 && fastT < 1 && Array.from({length: 8}).map((_, i) => {
            const u = Math.max(0, fastT - i * 0.02);
            const x = lerp(s3x - 620, s3x + 640, u);
            const y = x < s3x - 260 ? 520 : x > s3x + 40 ? 330 : lerp(520, 330, (x - (s3x - 260)) / 300);
            return <circle key={i} cx={x} cy={y} r={14 - i * 1.5} fill={C.green} opacity={1 - i * 0.12} />;
          })}
          {/* slow packet */}
          {f > p3 && (() => {
            const x = lerp(s3x - 620, s3x + 560, slowT);
            const y = x < s3x - 260 ? 520 : x > s3x + 40 ? 690 : lerp(520, 690, (x - (s3x - 260)) / 300);
            return (
              <g>
                <circle cx={x} cy={y} r={22} fill={C.amber} style={{filter: `drop-shadow(0 0 16px ${C.amber})`}} />
                {[0, 1, 2].map((k) => <circle key={k} cx={x - 24 + k * 24} cy={y - 50} r={6} fill={C.white} opacity={0.3 + 0.7 * ((Math.floor(f / 6) % 3) === k ? 1 : 0)} />)}
              </g>
            );
          })()}
        </svg>
      </AbsoluteFill>
      <div style={{position: 'absolute', left: 110, top: 80}}>
        {STATIONS.slice(0, 3).map((s, i) => (
          <div key={i} style={{position: 'absolute', opacity: i === active ? ramp(f, i === 0 ? 4 : [0, p1, p2][i] - 4, (i === 0 ? 4 : [0, p1, p2][i] - 4) + 10) : 0, whiteSpace: 'nowrap'}}>
            <div style={{fontFamily: HEAD, fontWeight: 700, fontSize: 72, color: s.col, letterSpacing: '0.06em'}}>{s.tag}</div>
            <Mono size={24} color={C.dim}>{s.sub}</Mono>
          </div>
        ))}
      </div>
      <div style={{position: 'absolute', right: 110, top: 100, textAlign: 'right', opacity: ramp(f, p2, p2 + 12)}}>
        <Mono size={30} color={C.green}>FAST WHEN SURE</Mono>
        <Mono size={30} color={C.amber} style={{opacity: ramp(f, p3, p3 + 12), marginTop: 8}}>SLOW WHEN NOT</Mono>
      </div>
    </AbsoluteFill>
  );
};
