import {AbsoluteFill, useCurrentFrame} from 'remotion';
import {C, W, H, ramp, Reveal, Mono, ph, HEAD} from '../lib';
import {Mark} from '../Logo';

export const S7: React.FC = () => {
  const f = useCurrentFrame();
  const p = ph(7, 0);
  const t = ramp(f, 0, p + 4, (x) => x);
  const flash = Math.max(0, 1 - Math.abs(f - (p + 2)) / 6);
  return (
    <AbsoluteFill style={{background: 'radial-gradient(ellipse at 50% 45%, #0B1D33 0%, #050B14 70%)'}}>
      <svg width={W} height={H} style={{position: 'absolute'}}>
        <Mark cx={W / 2} cy={400} r={150} t={t} fromMap />
      </svg>
      <div style={{position: 'absolute', top: 610, width: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center'}}>
        <Reveal text="METACORE" at={p - 2} size={150} spacing="0.22em" stagger={1.5} glow />
        <div style={{opacity: ramp(f, p + 16, p + 30), marginTop: 8}}>
          <Mono size={28} color={C.cyan}>METACOGNITIVE AI FOR ISLANDED MICROGRIDS</Mono>
        </div>
      </div>
      <AbsoluteFill style={{background: '#fff', opacity: flash * 0.5}} />
    </AbsoluteFill>
  );
};
