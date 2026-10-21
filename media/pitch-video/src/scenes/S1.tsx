import {AbsoluteFill, useCurrentFrame, interpolate} from 'remotion';
import {C, W, H, ecg, ramp, Mono, Decode, MONO, clamp, sceneDur} from '../lib';

// Cold open: hospital ECG -> grid frequency -> flatline
export const S1: React.FC = () => {
  const f = (useCurrentFrame() * 120) / sceneDur(1);
  const sweep = interpolate(f, [0, 52], [0, W + 40], clamp);
  const morph = ramp(f, 50, 72);               // ecg -> sine
  const kill = ramp(f, 86, 100);               // amplitude to 0
  const stutter = f > 80 && f < 96 ? (Math.sin(f * 7.3) > 0.2 ? 0.25 : 1) : 1;
  const amp = (1 - kill) * stutter;
  const col = kill > 0.3 ? C.red : C.cyan;
  const scroll = f > 52 ? (f - 52) * 0.06 : 0;
  const pts: string[] = [];
  const maxX = f < 52 ? sweep : W;
  for (let x = 0; x <= maxX; x += 4) {
    const u = x / 480 + scroll;
    const v = (1 - morph) * ecg(u) + morph * 0.45 * Math.sin(u * Math.PI * 2);
    pts.push(`${x},${H / 2 - v * 240 * amp}`);
  }
  const fade = 1 - ramp(f, sceneEnd(f) - 8, sceneEnd(f));
  const hz = interpolate(kill, [0, 1], [50, 0]);
  return (
    <AbsoluteFill style={{background: '#000', opacity: f > 110 ? 0 : 1}}>
      <svg width={W} height={H} style={{position: 'absolute'}}>
        <line x1={0} y1={H / 2} x2={W} y2={H / 2} stroke={C.grid} strokeWidth={1} />
        <polyline points={pts.join(' ')} fill="none" stroke={col} strokeWidth={4} style={{filter: `drop-shadow(0 0 10px ${col})`}} />
        {f < 52 && <circle cx={sweep} cy={H / 2 - ((1 - morph) * ecg(sweep / 480)) * 240} r={7} fill="#fff" style={{filter: `drop-shadow(0 0 16px ${C.cyan})`}} />}
      </svg>
      <div style={{position: 'absolute', top: 90, left: 110, opacity: 1 - ramp(f, 48, 60)}}>
        <Decode text="ISLAND HOSPITAL · WARD 2" at={4} size={26} color={C.dim} />
        <Mono size={64} color={C.cyan} style={{marginTop: 10, opacity: ramp(f, 8, 20)}}>♥ 72 BPM</Mono>
      </div>
      <div style={{position: 'absolute', top: 90, left: 110, opacity: ramp(f, 56, 66)}}>
        <Mono size={26} color={C.dim}>GRID FREQUENCY · ISLAND BUS</Mono>
        <div style={{fontFamily: MONO, fontSize: 96, fontWeight: 700, color: col, marginTop: 6, textShadow: `0 0 24px ${col}`}}>{hz.toFixed(2)} Hz</div>
      </div>
    </AbsoluteFill>
  );
};
const sceneEnd = (_: number) => 120;
