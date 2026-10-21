import {AbsoluteFill, useCurrentFrame, interpolate} from 'remotion';
import {C, W, H, ramp, inOut, lerp, Mono, ph, phEnd, sceneDur, Reveal, Rewind, Backdrop, clamp, MONO, HEAD, Glitch} from '../lib';
import {OneLine} from '../OneLine';

const Typed: React.FC<{text: string; at: number; color?: string; size?: number}> = ({text, at, color = C.white, size = 28}) => {
  const f = useCurrentFrame();
  const n = Math.max(0, Math.min(text.length, Math.floor((f - at) * 1.6)));
  if (f < at) return null;
  return <div style={{fontFamily: MONO, fontSize: size, color, whiteSpace: 'pre', lineHeight: 1.6}}>{text.slice(0, n)}{n < text.length && f % 10 < 5 ? '▌' : ''}</div>;
};

export const S6: React.FC = () => {
  const f = useCurrentFrame();
  const dur = sceneDur(6);
  const p0 = ph(6, 0), e0 = phEnd(6, 0), p1 = ph(6, 1);
  const exec = p0 + 74;
  const f2On = ramp(f, exec, exec + 6);
  const stress = ramp(f, exec + 18, exec + 24);
  const dark = ramp(f, exec + 70, exec + 80);
  const hz = interpolate(f, [exec + 14, exec + 74, exec + 80], [50, 47.2, 0], {...clamp, easing: inOut});
  const shake = f > exec + 20 && f < exec + 80 ? Math.sin(f * 3.1) * 5 * stress : 0;
  const q = ramp(f, p1 - 6, p1 + 6);
  return (
    <AbsoluteFill style={{background: '#050B14'}}>
      <Glitch amount={f > exec + 60 && f < exec + 82 ? 0.6 : 0}>
        <AbsoluteFill style={{transform: `translate(${shake}px, ${shake * 0.6}px)`}}>
          <Backdrop grid={0.5} />
          {/* AI terminal */}
          <div style={{position: 'absolute', left: 110, top: 200, width: 700, padding: '28px 34px', borderRadius: 14, background: 'rgba(10,22,38,0.85)', border: `1.5px solid ${C.cyan}55`}}>
            <Mono size={18} color={C.dim} style={{marginBottom: 14}}>ORDINARY AI CONTROLLER</Mono>
            <Typed text="> STATE    TELEMETRY STALE" at={p0 + 4} color={C.dim} />
            <Typed text="> ACTION   RESTORE FEEDER F2" at={p0 + 22} />
            <Typed text="> CONFIDENT  ██████████████" at={p0 + 40} color={C.amber} />
            {f > exec && <div style={{fontFamily: MONO, fontSize: 28, color: C.red, lineHeight: 1.6}}>{'> EXECUTED ✓'}</div>}
          </div>
          <svg width={W} height={H} style={{position: 'absolute'}}>
            <OneLine x={940} y={230} s={0.95} f2On={f2On} stress={stress} dark={dark} />
          </svg>
          <div style={{position: 'absolute', left: 110, top: 640}}>
            <Mono size={20} color={C.dim}>GRID FREQUENCY</Mono>
            <div style={{fontFamily: MONO, fontWeight: 700, fontSize: 88, color: stress > 0.5 ? C.red : C.cyan, textShadow: `0 0 24px ${stress > 0.5 ? C.red : C.cyan}`}}>{hz.toFixed(2)} Hz</div>
          </div>
          <div style={{position: 'absolute', left: 110, top: 90}}>
            <Reveal text="ONE CONFIDENT WRONG ANSWER" at={p0 + 70} size={54} color={C.white} stagger={0.7} />
          </div>
        </AbsoluteFill>
      </Glitch>
      <AbsoluteFill style={{background: `rgba(0,0,0,${dark * 0.75})`, justifyContent: 'center', alignItems: 'center', opacity: 1 - q}}>
        <div style={{opacity: ramp(f, exec + 80, exec + 86), fontFamily: HEAD, fontWeight: 700, fontSize: 200, letterSpacing: '0.12em', color: C.red, textShadow: `0 0 60px ${C.red}`}}>BLACKOUT</div>
      </AbsoluteFill>
      <Rewind from={e0 + 2} to={p1 + 2} />
      <AbsoluteFill style={{background: `rgba(5,11,20,${q * 0.92})`, justifyContent: 'center', alignItems: 'center', flexDirection: 'column'}}>
        {q > 0 && (
          <>
            <Reveal text="WHAT IF IT KNEW" at={p1 + 6} size={110} stagger={1} />
            <Reveal text="WHEN IT DIDN'T KNOW?" at={p1 + 32} size={110} stagger={1} color={C.amber} glow />
          </>
        )}
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
