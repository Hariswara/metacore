import {useCurrentFrame} from 'remotion';
import {C, ramp, inOut, lerp, projLocal, ISLANDS} from './lib';

/** MetaCore mark: hexagon whose vertices are grid nodes, with a core. t = 0..1 build progress */
export const Mark: React.FC<{cx: number; cy: number; r: number; t: number; fromMap?: boolean; color?: string}> = ({cx, cy, r, t, fromMap = false, color = C.cyan}) => {
  const f = useCurrentFrame();
  const verts = Array.from({length: 6}, (_, i) => {
    const a = -Math.PI / 2 + (i * Math.PI) / 3;
    return [cx + Math.cos(a) * r, cy + Math.sin(a) * r];
  });
  const fly = ramp(t, 0, 0.45, inOut);
  const pts = verts.map((v, i) => {
    if (!fromMap || i >= 4) return v;
    const [mx, my] = projLocal(ISLANDS[i].lon, ISLANDS[i].lat);
    return [lerp(mx, v[0], fly), lerp(my, v[1], fly)];
  });
  const edge = ramp(t, 0.4, 0.8, inOut);
  const core = ramp(t, 0.6, 1);
  const d = pts.map((p, i) => (i ? 'L' : 'M') + p[0] + ',' + p[1]).join('') + 'Z';
  return (
    <g>
      <path d={d} fill="none" stroke={color} strokeWidth={r * 0.05} pathLength={1} strokeDasharray="1 1" strokeDashoffset={1 - edge} strokeLinejoin="round" style={{filter: `drop-shadow(0 0 ${r * 0.08}px ${color})`}} />
      {pts.map((p, i) => (
        <line key={'s' + i} x1={p[0]} y1={p[1]} x2={cx} y2={cy} stroke={color} strokeWidth={r * 0.018} opacity={core * 0.6} />
      ))}
      <circle cx={cx} cy={cy} r={r * 0.3 * core} fill="none" stroke={C.amber} strokeWidth={r * 0.04} style={{filter: `drop-shadow(0 0 ${r * 0.08}px ${C.amber})`}} />
      <circle cx={cx} cy={cy} r={r * 0.12 * core} fill="#fff" />
      {pts.map((p, i) => {
        const o = i < 4 && fromMap ? 1 : ramp(t, 0.35 + i * 0.03, 0.45 + i * 0.03);
        return <circle key={i} cx={p[0]} cy={p[1]} r={r * 0.075} fill={color} opacity={o} style={{filter: `drop-shadow(0 0 ${r * 0.06}px ${color})`}} />;
      })}
    </g>
  );
};
