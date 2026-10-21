import {AbsoluteFill, useCurrentFrame} from 'remotion';
import {C, W, H, ramp, inOut, lerp, Mono, ph, sceneDur, projRegion, Node, HEAD} from '../lib';
import {IconPeople} from '../icons';
import geo from '../geo.json';

const COAST: [number, number][] = [
  [79.85, 6.9], [79.9, 6.4], [80.2, 6.05], [80.6, 5.95], [81.1, 6.15], [81.7, 6.7], [81.85, 7.3], [81.6, 7.9], [81.3, 8.5], [81.0, 8.9], [80.6, 9.3],
  [79.8, 8.0], [79.85, 7.5], [80.0, 9.6], [79.4, 9.3], [79.1, 9.25], [78.6, 9.0], [78.15, 8.4], [77.6, 8.1], [77.2, 8.3], [76.6, 8.9], [76.2, 9.9],
  [75.9, 10.9], [79.85, 10.3], [79.85, 11.0], [80.15, 12.2], [80.25, 13.1], [73.5, 4.2], [73.4, 3.2], [73.0, 5.6],
];
const Bolt = () => (<svg width={64} height={64} viewBox="0 0 64 64"><path d="M36 4 L12 36 h18 l-4 24 l26 -34 h-18z" fill={C.amber} fillOpacity={0.25} stroke={C.amber} strokeWidth={3.5} strokeLinejoin="round" /></svg>);
const Tower = () => (<svg width={64} height={64} viewBox="0 0 64 64" fill="none" stroke={C.green} strokeWidth={3.5} strokeLinecap="round"><path d="M32 22 L20 60 M32 22 L44 60 M24 46 h16 M27 34 h10" /><circle cx={32} cy={18} r={4} fill={C.green} /><path d="M20 8a18 18 0 0 0 0 20 M44 8a18 18 0 0 1 0 20" /></svg>);

export const S11: React.FC = () => {
  const f = useCurrentFrame();
  const dur = sceneDur(11);
  const [jx, jy] = projRegion(79.84, 9.62);
  const s = lerp(3.2, 1.0, ramp(f, 0, dur - 30, inOut));
  const cards = [
    {at: ph(11, 0), icon: <Bolt />, t: 'CEYLON ELECTRICITY BOARD', sub: 'AUTONOMY WHEN CREWS CANNOT REACH THE ISLANDS', c: C.amber},
    {at: ph(11, 1), icon: <IconPeople s={64} c={C.cyan} />, t: 'ISLAND COMMUNITIES', sub: 'POWER FOR HOMES, THE HOSPITAL AND WATER', c: C.cyan},
    {at: ph(11, 2), icon: <Tower />, t: 'REMOTE & COASTAL MICROGRIDS', sub: 'THE SAME SHIELD FOR ANY ISLANDED GRID', c: C.green},
  ];
  return (
    <AbsoluteFill style={{background: '#050B14'}}>
      <AbsoluteFill style={{transform: `scale(${s})`, transformOrigin: `${jx}px ${jy}px`, opacity: 0.85}}>
        <svg width={W} height={H} style={{position: 'absolute'}}>
          <path d={geo.region.india} fill="#0C1C30" />
          <path d={geo.region.sl} fill="#0F2A44" stroke={C.cyan} strokeWidth={2 / s} />
          {COAST.map(([lo, la], i) => {
            const [x, y] = projRegion(lo, la);
            const o = ramp(f, ph(11, 2) + i * 2, ph(11, 2) + i * 2 + 10);
            return <Node key={i} x={x} y={y} r={5 / s} color={C.green} o={o} />;
          })}
          <Node x={jx} y={jy} r={9 / s} color={C.cyan} pulse={1} />
        </svg>
      </AbsoluteFill>
      <div style={{position: 'absolute', left: 110, top: 80}}>
        <div style={{fontFamily: HEAD, fontWeight: 700, fontSize: 64, color: C.white, letterSpacing: '0.06em', opacity: ramp(f, 0, 12)}}>WHO BENEFITS</div>
      </div>
      <div style={{position: 'absolute', left: 110, right: 110, bottom: 150, display: 'flex', gap: 30}}>
        {cards.map((c) => {
          const o = ramp(f, c.at - 2, c.at + 12);
          return (
            <div key={c.t} style={{flex: 1, padding: '28px 30px', borderRadius: 16, border: `2px solid ${c.c}`, background: 'rgba(10,22,38,0.9)', opacity: o, transform: `translateY(${(1 - o) * 60}px)`, boxShadow: `0 0 40px ${c.c}33`}}>
              {c.icon}
              <div style={{fontFamily: HEAD, fontWeight: 700, fontSize: 34, color: C.white, marginTop: 16, letterSpacing: '0.03em'}}>{c.t}</div>
              <Mono size={17} color={c.c} style={{marginTop: 10, letterSpacing: '0.08em'}}>{c.sub}</Mono>
            </div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};
