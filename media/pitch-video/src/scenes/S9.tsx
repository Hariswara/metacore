import {AbsoluteFill, useCurrentFrame, interpolate} from 'remotion';
import {C, W, H, ramp, inOut, lerp, Mono, ph, sceneDur, Reveal, Backdrop, Stamp, clamp, MONO, HEAD} from '../lib';
import {OneLine} from '../OneLine';

const CHECKS = [
  {t: 'VOLTAGE  0.95–1.05 pu', ok: true},
  {t: 'LINE LOADING  ≤ RATING', ok: true},
  {t: 'GENERATION ADEQUACY', ok: false},
  {t: 'POWER FLOW CONVERGES', ok: true},
];

const Hex: React.FC<{x: number; y: number; r: number; o: number; col: string}> = ({x, y, r, o, col}) => {
  const d = Array.from({length: 6}, (_, i) => { const a = (i * Math.PI) / 3; return (i ? 'L' : 'M') + (x + r * Math.cos(a)) + ',' + (y + r * Math.sin(a)); }).join('') + 'Z';
  return <path d={d} fill={col} fillOpacity={0.06 * o} stroke={col} strokeWidth={1.5} opacity={o} />;
};

const Verdict: React.FC<{at: number; label: string; col: string; desc: string; risk: number | null; x: number}> = ({at, label, col, desc, risk, x}) => {
  const f = useCurrentFrame();
  const o = ramp(f, at - 4, at + 8);
  const bar = ramp(f, at + 4, at + 24, inOut);
  return (
    <div style={{position: 'absolute', left: x, top: 330, width: 520, height: 400, opacity: 0.25 + 0.75 * o, transform: `translateY(${(1 - o) * 20}px)`, borderRadius: 18, border: `2px solid ${col}${o > 0.5 ? '' : '55'}`, background: 'rgba(10,22,38,0.85)', boxShadow: o > 0.5 ? `0 0 50px ${col}44` : undefined, display: 'flex', flexDirection: 'column', alignItems: 'center', paddingTop: 50}}>
      <Stamp label={label} color={col} at={at} size={56} />
      <Mono size={20} color={C.white} style={{marginTop: 46, opacity: o, textAlign: 'center', letterSpacing: '0.06em', whiteSpace: 'pre-line', lineHeight: 1.6}}>{desc}</Mono>
      <div style={{marginTop: 36, width: 400, opacity: o}}>
        <Mono size={16} color={C.dim}>RISK</Mono>
        <div style={{marginTop: 8, height: 16, borderRadius: 8, background: '#0E2236', overflow: 'hidden'}}>
          {risk === null ? (
            <div style={{width: '100%', height: '100%', backgroundImage: `repeating-linear-gradient(45deg, ${col}aa 0 10px, transparent 10px 20px)`, backgroundPosition: `${f * 2}px 0`}} />
          ) : (
            <div style={{width: `${risk * 100 * bar}%`, height: '100%', background: col, boxShadow: `0 0 12px ${col}`}} />
          )}
        </div>
      </div>
    </div>
  );
};

export const S9: React.FC = () => {
  const f = useCurrentFrame();
  const dur = sceneDur(9);
  const P = (i: number) => ph(9, i);
  const wallX = 720;
  const packetX = interpolate(f, [10, 70], [-200, wallX - 260], {...clamp, easing: inOut});
  const phaseA = 1 - ramp(f, P(5) - 10, P(5) + 6, inOut); // twin + checks
  const phaseB = ramp(f, P(5) - 4, P(5) + 10);            // verdicts
  const solving = f > 80 && f < P(1);
  const iter = Math.min(6, Math.max(1, Math.floor((f - 90) / 14) + 1));
  const finale = ramp(f, P(9) - 4, P(9) + 10);
  const trace = ramp(f, P(9) + 40, P(9) + 80, inOut);
  return (
    <AbsoluteFill style={{background: '#050B14'}}>
      <Backdrop grid={0.5} />
      <div style={{position: 'absolute', left: 110, top: 70}}>
        <Reveal text="M4 · DETERMINISTIC PHYSICS FIREWALL" at={4} size={56} color={C.white} stagger={0.6} />
        <Mono size={24} color={C.red} style={{marginTop: 8, opacity: ramp(f, 30, 44)}}>OPENDSS AC POWER FLOW ON A DIGITAL TWIN · ZERO MACHINE LEARNING</Mono>
      </div>
      {/* phase A */}
      <AbsoluteFill style={{opacity: phaseA}}>
        <svg width={W} height={H} style={{position: 'absolute'}}>
          {Array.from({length: 7}).map((_, r) => Array.from({length: 2}).map((__, c) => {
            const y = 260 + r * 100 + (c % 2) * 50, x = wallX + c * 86;
            return <Hex key={r + '-' + c} x={x} y={y} r={50} o={ramp(f, 20 + r * 3 + c * 4, 34 + r * 3 + c * 4)} col={C.red} />;
          }))}
          {/* scan beam */}
          {f > 70 && f < 200 && <rect x={wallX - 60} y={240 + ((f - 70) * 9) % 720} width={240} height={4} fill={C.red} opacity={0.7} style={{filter: `drop-shadow(0 0 10px ${C.red})`}} />}
          <OneLine x={1010} y={290} s={0.88} f2On={ramp(f, 90, 110)} stress={ramp(f, P(3) - 2, P(3) + 4)} label="DIGITAL TWIN · ISLAND FEEDERS" />
        </svg>
        <div style={{position: 'absolute', left: Math.max(packetX, 110), top: 420, padding: '14px 22px', borderRadius: 10, border: `2px solid ${C.amber}`, background: 'rgba(10,22,38,0.92)', boxShadow: `0 0 30px ${C.amber}66`}}>
          <Mono size={16} color={C.dim}>FROM M3</Mono>
          <Mono size={26} color={C.white}>RESTORE FEEDER F2</Mono>
        </div>
        <div style={{position: 'absolute', left: 1010, top: 830, opacity: ramp(f, 86, 96)}}>
          <Mono size={22} color={solving ? C.cyan : C.dim}>{solving ? `SOLVING · ITERATION ${iter}` : 'SOLVED · CONVERGED'}</Mono>
        </div>
        <div style={{position: 'absolute', left: 110, top: 560}}>
          {CHECKS.map((c, i) => {
            const o = ramp(f, P(i + 1) - 2, P(i + 1) + 6);
            const col = c.ok ? C.green : C.red;
            return (
              <div key={i} style={{display: 'flex', alignItems: 'center', gap: 18, marginBottom: 18, opacity: 0.25 + 0.75 * o}}>
                <div style={{width: 38, height: 38, borderRadius: 8, border: `2px solid ${o > 0.5 ? col : C.dim}`, display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: MONO, fontSize: 26, color: col, transform: `scale(${lerp(1.5, 1, o)})`}}>{o > 0.5 ? (c.ok ? '✓' : '✕') : ''}</div>
                <Mono size={26} color={o > 0.5 ? (c.ok ? C.white : C.red) : C.dim}>{c.t}</Mono>
              </div>
            );
          })}
        </div>
      </AbsoluteFill>
      {/* phase B: verdicts */}
      <AbsoluteFill style={{opacity: phaseB * (1 - finale * 0.55)}}>
        <div style={{position: 'absolute', left: 110, top: 230, opacity: ramp(f, P(5), P(5) + 10) * (1 - finale)}}>
          <Mono size={28} color={C.cyan}>EVERY ACTION GETS A VERDICT</Mono>
        </div>
        <Verdict at={P(6)} label="APPROVE" col={C.green} desc={'ALL LIMITS SATISFIED\nACTION EXECUTES'} risk={0.08} x={110} />
        <Verdict at={P(7)} label="REJECT" col={C.red} desc={'A LIMIT IS VIOLATED\nACTION BLOCKED'} risk={0.86} x={700} />
        <Verdict at={P(8)} label="UNVERIFIED" col={C.violet} desc={'CANNOT BE CERTIFIED\nACTION HELD'} risk={null} x={1290} />
      </AbsoluteFill>
      {/* finale */}
      {finale > 0 && (
        <AbsoluteFill style={{background: `rgba(5,11,20,${finale * 0.5})`}}>
          <svg width={W} height={H} style={{position: 'absolute'}}>
            <path d="M960 330 C 960 200, 800 250, 760 252" fill="none" stroke={C.red} strokeWidth={5} pathLength={1} strokeDasharray="1 1" strokeDashoffset={1 - trace} style={{filter: `drop-shadow(0 0 10px ${C.red})`}} />
            {trace > 0.95 && <path d="M776 238 L756 252 L776 266" fill="none" stroke={C.red} strokeWidth={5} />}
          </svg>
          <div style={{position: 'absolute', left: 110, top: 232, opacity: ramp(f, P(9) + 70, P(9) + 84)}}>
            <Mono size={24} color={C.red}>REJECTION TRACE · BACK TO M2 + M3</Mono>
            <Mono size={18} color={C.dim} style={{marginTop: 6}}>WHICH LIMIT · WHERE · CAUSED BY WHICH ACTION</Mono>
          </div>
          <div style={{position: 'absolute', bottom: 150, width: '100%', display: 'flex', justifyContent: 'center'}}>
            <Reveal text="NOTHING EXECUTES WITHOUT APPROVE" at={P(9)} size={78} stagger={0.6} glow />
          </div>
        </AbsoluteFill>
      )}
    </AbsoluteFill>
  );
};
