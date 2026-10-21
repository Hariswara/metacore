import {useCurrentFrame} from 'remotion';
import {C} from './lib';
import {IconHospital, IconWater, IconPeople} from './icons';

/** Island one-line diagram. Coordinates in a 900x560 box. */
export const OneLine: React.FC<{
  x: number; y: number; s?: number;
  f2On?: number;      // 0..1 feeder F2 energised
  dark?: number;      // 0..1 whole island de-energised
  stress?: number;    // 0..1 overload stress (red)
  flow?: boolean;
  label?: string;
}> = ({x, y, s = 1, f2On = 0, dark = 0, stress = 0, flow = true, label = 'ISLAND MICROGRID · ONE-LINE'}) => {
  const f = useCurrentFrame();
  const live = 1 - dark;
  const lineCol = stress > 0.5 ? C.red : C.cyan;
  const dash = flow ? `${10} ${14}` : undefined;
  const off = -f * 3;
  const busY = 200;
  const Line = ({d, on, col = lineCol}: {d: string; on: number; col?: string}) => (
    <g>
      <path d={d} stroke={C.grid} strokeWidth={6} fill="none" />
      <path d={d} stroke={col} strokeWidth={4} fill="none" opacity={on} style={{filter: `drop-shadow(0 0 6px ${col})`}} />
      {flow && on > 0.5 && <path d={d} stroke="#fff" strokeWidth={3} fill="none" strokeDasharray={dash} strokeDashoffset={off} opacity={0.6 * on} />}
    </g>
  );
  return (
    <g transform={`translate(${x},${y}) scale(${s})`}>
      <text x={0} y={-24} fill={C.dim} fontFamily="JetBrains Mono" fontSize={18} letterSpacing={3}>{label}</text>
      {/* gensets */}
      {[{gx: 120, on: 1, n: 'G1'}, {gx: 300, on: 0, n: 'G2'}].map((g) => (
        <g key={g.n}>
          <circle cx={g.gx} cy={70} r={38} fill="#08131F" stroke={g.on ? (stress > 0.5 ? C.red : C.amber) : C.dim} strokeWidth={3} opacity={g.on ? live * 0.8 + 0.2 : 0.6} />
          <path d={`M${g.gx - 18} 70 q9 -18 18 0 t18 0`} fill="none" stroke={g.on ? C.amber : C.dim} strokeWidth={3} opacity={g.on ? live : 0.5} />
          <text x={g.gx} y={136} fill={g.on ? C.amber : C.dim} fontFamily="JetBrains Mono" fontSize={18} textAnchor="middle">{g.n}{g.on ? '' : ' · OFFLINE'}</text>
          <Line d={`M${g.gx} 108 V${busY}`} on={g.on ? live : 0} col={C.amber} />
        </g>
      ))}
      {/* bus */}
      <rect x={60} y={busY - 6} width={780} height={12} rx={4} fill={live > 0.5 ? (stress > 0.5 ? C.red : C.cyan) : C.grid} style={{filter: live > 0.5 ? `drop-shadow(0 0 10px ${lineCol})` : undefined}} />
      <text x={850} y={busY + 6} fill={C.dim} fontFamily="JetBrains Mono" fontSize={16}>400 V BUS</text>
      {/* feeders */}
      <Line d={`M220 ${busY} V330 H140 V420`} on={live} />
      <Line d={`M220 330 H320 V420`} on={live} />
      <text x={240} y={318} fill={C.white} fontFamily="JetBrains Mono" fontSize={18}>F1 · CRITICAL</text>
      <Line d={`M640 ${busY} V420`} on={f2On * live} />
      <text x={660} y={318} fill={C.white} fontFamily="JetBrains Mono" fontSize={18}>F2</text>
      {/* breaker on F2 */}
      <rect x={622} y={250} width={36} height={36} rx={4} fill={f2On > 0.5 ? (stress > 0.5 ? C.red : C.cyan) : '#08131F'} stroke={C.white} strokeWidth={2} />
      {/* loads */}
      <g transform="translate(108,430)" opacity={0.25 + 0.75 * live}><IconHospital s={64} c={live > 0.5 ? C.green : C.dim} /></g>
      <g transform="translate(288,430)" opacity={0.25 + 0.75 * live}><IconWater s={64} c={live > 0.5 ? C.cyan : C.dim} /></g>
      <g transform="translate(608,430)" opacity={0.25 + 0.75 * live * (0.3 + 0.7 * f2On)}><IconPeople s={64} c={live > 0.5 && f2On > 0.5 ? C.cyan : C.dim} /></g>
    </g>
  );
};
