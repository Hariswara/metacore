import {AbsoluteFill, useCurrentFrame} from 'remotion';
import {C, W, H, ramp, inOut, lerp, Mono, ph, sceneDur, Glitch, Reveal, Backdrop, ecg} from '../lib';

const rnd = (i: number) => { const x = Math.sin(i * 127.1 + 311.7) * 43758.5453; return x - Math.floor(x); };

const Monitor: React.FC<{x: number; y: number; w: number; h: number; dead: number; seed: number}> = ({x, y, w, h, dead, seed}) => {
  const f = useCurrentFrame();
  const pts: string[] = [];
  for (let i = 0; i <= 60; i++) {
    const u = i / 60;
    const v = seed % 2 ? ecg(u * 3 + f * 0.03) * 0.8 : Math.sin(u * 12 + f * 0.15 + seed) * 0.35;
    pts.push(`${x + 14 + u * (w - 28)},${y + h / 2 - v * h * 0.35 * (1 - dead)}`);
  }
  const col = dead > 0.5 ? C.red : C.cyan;
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={8} fill="#08131F" stroke={col} strokeWidth={2} opacity={0.95} />
      <polyline points={pts.join(' ')} fill="none" stroke={col} strokeWidth={2.5} opacity={dead > 0.5 ? (Math.sin(f) > 0 ? 0.9 : 0.3) : 0.9} />
      {dead > 0.5 && <text x={x + w / 2} y={y + h - 18} fill={C.red} fontFamily="JetBrains Mono" fontSize={18} textAnchor="middle" letterSpacing={3}>NO SIGNAL</text>}
    </g>
  );
};

export const S5: React.FC = () => {
  const f = useCurrentFrame();
  const dur = sceneDur(5);
  const gone = ramp(f, ph(5, 0) + 34, ph(5, 0) + 80, inOut);
  const dead = ramp(f, ph(5, 1) - 2, ph(5, 1) + 6);
  const zoom = lerp(1, 1.35, ramp(f, ph(5, 2) - 6, dur, inOut));
  const glitch = f > ph(5, 1) - 2 && f < ph(5, 1) + 20 ? 0.8 : f > ph(5, 1) + 20 ? 0.12 : 0;
  const cx = W / 2, floor = 820;
  return (
    <AbsoluteFill style={{background: '#050B14'}}>
      <Glitch amount={glitch}>
        <AbsoluteFill style={{transform: `scale(${zoom})`, transformOrigin: '50% 45%'}}>
          <Backdrop grid={0.6} />
          <svg width={W} height={H} style={{position: 'absolute'}}>
            {/* monitors */}
            <Monitor x={cx - 560} y={300} w={340} h={210} dead={dead} seed={1} />
            <Monitor x={cx - 170} y={250} w={340} h={240} dead={dead} seed={2} />
            <Monitor x={cx + 220} y={300} w={340} h={210} dead={dead} seed={3} />
            {/* desk */}
            <path d={`M${cx - 700} ${floor - 180} L${cx + 700} ${floor - 180} L${cx + 760} ${floor - 140} L${cx - 760} ${floor - 140} Z`} fill="#0A1626" stroke={C.cyan} strokeWidth={2} opacity={0.8} />
            {/* chair */}
            <path d={`M${cx - 70} ${floor - 120} h140 v18 h-140z M${cx} ${floor - 102} v60 M${cx - 50} ${floor - 40} h100`} stroke={C.dim} strokeWidth={4} fill="none" />
            <path d={`M${cx - 80} ${floor - 260} q80 -30 160 0 v140 h-160z`} fill="none" stroke={C.dim} strokeWidth={4} />
            {/* operator silhouette dissolving */}
            <g opacity={1 - gone} transform={`translate(0 ${-gone * 40})`} style={{filter: `blur(${gone * 10}px)`}}>
              <circle cx={cx} cy={floor - 330} r={42} fill={C.cyan} opacity={0.85} />
              <path d={`M${cx - 95} ${floor - 150} q0 -120 95 -125 q95 5 95 125 z`} fill={C.cyan} opacity={0.85} />
            </g>
            {Array.from({length: 40}).map((_, i) => {
              const t = gone;
              if (t <= 0 || t >= 1) return null;
              return <circle key={i} cx={cx + (rnd(i) - 0.5) * 200} cy={floor - 200 - rnd(i + 9) * 200 - t * 220 * (0.5 + rnd(i + 3))} r={2 + rnd(i + 5) * 3} fill={C.cyan} opacity={(1 - t) * 0.8} />;
            })}
          </svg>
        </AbsoluteFill>
      </Glitch>
      <div style={{position: 'absolute', left: 110, top: 90}}>
        <Mono size={34} color={C.amber} style={{opacity: ramp(f, ph(5, 0) + 4, ph(5, 0) + 14)}}>● OPERATORS EVACUATED</Mono>
        <Mono size={34} color={C.red} style={{marginTop: 14, opacity: ramp(f, ph(5, 1), ph(5, 1) + 6)}}>● TELEMETRY LOST</Mono>
      </div>
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center', background: `rgba(5,11,20,${ramp(f, ph(5, 2) - 4, ph(5, 2) + 8) * 0.72})`}}>
        <Reveal text="THE GRID MUST DECIDE ALONE" at={ph(5, 2)} size={96} stagger={0.8} glow />
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
