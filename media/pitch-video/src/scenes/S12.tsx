import {AbsoluteFill, useCurrentFrame} from 'remotion';
import {C, W, H, ramp, inOut, lerp, Mono, ph, sceneDur, Reveal, ecg, HEAD} from '../lib';
import {Mark} from '../Logo';

export const S12: React.FC = () => {
  const f = useCurrentFrame();
  const dur = sceneDur(12);
  const p0 = ph(12, 0), p1 = ph(12, 1), p2 = ph(12, 2);
  const lineO = 1 - ramp(f, p0, p0 + 16);
  const pts: string[] = [];
  for (let x = 0; x <= W; x += 4) pts.push(`${x},${360 - ecg(x / 480 + f * 0.035) * 200}`);
  const end = ramp(f, dur - 22, dur);
  return (
    <AbsoluteFill style={{background: 'radial-gradient(ellipse at 50% 40%, #0B1D33 0%, #050B14 70%)'}}>
      <svg width={W} height={H} style={{position: 'absolute'}}>
        <polyline points={pts.join(' ')} fill="none" stroke={C.green} strokeWidth={4} opacity={lineO} style={{filter: `drop-shadow(0 0 10px ${C.green})`}} />
        <Mark cx={W / 2} cy={330} r={120} t={ramp(f, p0 - 10, p0 + 30, (x) => x)} />
      </svg>
      <div style={{position: 'absolute', top: 500, width: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center'}}>
        <Reveal text="METACORE" at={p0} size={120} spacing="0.22em" stagger={1.2} glow />
        <div style={{marginTop: 40, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 14}}>
          <Reveal text="KNOWS WHEN IT DOESN'T KNOW." at={p1} size={48} weight={500} color={C.amber} stagger={0.5} />
          <Reveal text="VERIFIES BEFORE IT ACTS." at={p2} size={48} weight={500} color={C.green} stagger={0.5} />
        </div>
      </div>
      <AbsoluteFill style={{background: '#000', opacity: end}} />
    </AbsoluteFill>
  );
};
