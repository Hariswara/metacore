import {AbsoluteFill, useCurrentFrame, interpolate} from 'remotion';
import {C, W, H, ramp, inOut, lerp, LocalLand, Mono, ph, phEnd, sceneDur, fmt, clamp, MONO, HEAD, Reveal} from '../lib';
import {IconPeople, IconHospital, IconWater} from '../icons';

const COLS = 70, ROWS = 70, BAR = 4878;

const Hero: React.FC<{at: number; out: number; label: string; children: React.ReactNode; color: string}> = ({at, out, label, children, color}) => {
  const f = useCurrentFrame();
  const i = ramp(f, at, at + 14);
  const o = ramp(f, out, out + 12, inOut);
  if (i <= 0) return null;
  return (
    <div style={{position: 'absolute', left: 110, top: lerp(430, 250, o) + (1 - i) * 40, opacity: i * lerp(1, 0, o), transform: `scale(${lerp(1, 0.6, o)})`, transformOrigin: 'left top'}}>
      <Mono size={24} color={C.dim}>{label}</Mono>
      <div style={{fontFamily: MONO, fontWeight: 700, fontSize: 132, color, lineHeight: 1.05, textShadow: `0 0 30px ${color}66`}}>{children}</div>
    </div>
  );
};

export const S4: React.FC = () => {
  const f = useCurrentFrame();
  const dur = sceneDur(4);
  const p0 = ph(4, 0), p1 = ph(4, 1), p2 = ph(4, 2), p3 = ph(4, 3);
  const litres = interpolate(f, [p1, phEnd(4, 1) - 10], [0, 975615], {...clamp, easing: inOut});
  const barrels = Math.round(litres / 200);
  const gwh = interpolate(f, [p2, p2 + 40], [0, 2.95], {...clamp, easing: inOut});
  const conv = ramp(f, p2, p2 + 45, inOut); // diesel squares turn to energy
  const tilesUp = ramp(f, p1 - 6, p1 + 10, inOut);
  const residents = interpolate(f, [p0, p0 + 40], [0, 9000], {...clamp, easing: inOut});
  const tiles = [
    {icon: <IconPeople s={70} />, big: `~${fmt(Math.round(residents / 10) * 10)}`, small: 'RESIDENTS'},
    {icon: <IconHospital s={70} />, big: '1', small: 'HOSPITAL'},
    {icon: <IconWater s={70} />, big: '1', small: 'WATER PLANT'},
  ];
  const cell = 9;
  const mx = W - 110 - COLS * cell, my = 290;
  const filled = barrels;
  return (
    <AbsoluteFill style={{background: '#050B14'}}>
      <AbsoluteFill style={{opacity: 0.22, transform: `scale(${lerp(1, 1.12, f / dur)})`}}>
        <LocalLand glow={false} />
      </AbsoluteFill>
      {/* tiles */}
      <div style={{position: 'absolute', left: 110, top: lerp(380, 90, tilesUp), display: 'flex', gap: lerp(60, 34, tilesUp), transform: `scale(${lerp(1, 0.62, tilesUp)})`, transformOrigin: 'left top'}}>
        {tiles.map((t, i) => {
          const o = ramp(f, p0 + i * 18, p0 + i * 18 + 14);
          return (
            <div key={i} style={{opacity: o, transform: `translateY(${(1 - o) * 30}px)`, border: `1.5px solid ${C.cyan}44`, background: 'rgba(10,22,38,0.75)', borderRadius: 14, padding: '26px 34px', minWidth: 340, display: 'flex', gap: 24, alignItems: 'center'}}>
              {t.icon}
              <div>
                <div style={{fontFamily: MONO, fontWeight: 700, fontSize: 72, color: C.white, lineHeight: 1}}>{t.big}</div>
                <Mono size={20} color={C.cyan} style={{marginTop: 6}}>{t.small}</Mono>
              </div>
            </div>
          );
        })}
      </div>
      {/* heroes */}
      <Hero at={p1} out={p2 - 4} label="DIESEL BURNED · 2025" color={C.amber}>{fmt(Math.round(litres))} L</Hero>
      <Hero at={p2} out={p3 - 4} label="ENERGY GENERATED · 2025" color={C.cyan}>{gwh.toFixed(2)} GWh</Hero>
      <Hero at={p3} out={dur + 100} label="COST PER UNIT · ALL-IN" color={C.red}>
        <span style={{fontSize: 70}}>LKR </span>94–118<span style={{fontSize: 56}}> /kWh</span>
      </Hero>
      {/* ledger of passed heroes */}
      <div style={{position: 'absolute', left: 110, bottom: 160, display: 'flex', gap: 40}}>
        {[{t: '975,615 L DIESEL', at: p2, c: C.amber}, {t: '2.95 GWh', at: p3, c: C.cyan}].map((x) => (
          <Mono key={x.t} size={26} color={x.c} style={{opacity: ramp(f, x.at, x.at + 12)}}>{x.t}</Mono>
        ))}
      </div>
      {/* barrel matrix */}
      <div style={{position: 'absolute', left: mx, top: my - 50, opacity: ramp(f, p1, p1 + 10)}}>
        <Mono size={22} color={conv > 0.5 ? C.cyan : C.amber}>
          {conv > 0.5 ? '→ 2,948,365 kWh' : `${fmt(barrels)} BARRELS BY SEA`}
        </Mono>
      </div>
      <svg width={W} height={H} style={{position: 'absolute', opacity: ramp(f, p1, p1 + 10)}}>
        {(() => {
          let empty = '', amber = '', cyan = '';
          const sz = cell - 3;
          for (let k = 0; k < COLS * ROWS; k++) {
            const c = k % COLS, r = Math.floor(k / COLS);
            const d = `M${mx + c * cell} ${my + r * cell}h${sz}v${sz}h-${sz}z`;
            if (k >= filled) empty += d;
            else if (conv > 0 && (ROWS - r) / ROWS < conv * 1.1) cyan += d;
            else amber += d;
          }
          return (
            <>
              <path d={empty} fill="#0E2236" opacity={0.5} />
              <path d={amber} fill={C.amber} opacity={0.9} />
              <path d={cyan} fill={C.cyan} opacity={0.9} />
            </>
          );
        })()}
      </svg>
      <div style={{position: 'absolute', left: mx, top: my + ROWS * cell + 6, opacity: ramp(f, p1 + 20, p1 + 30)}}>
        <Mono size={16} color={C.dim}>1 SQUARE = 1 BARREL (200 L) · SOURCE: CEB JAFFNA GENERATION LEDGER 2025</Mono>
      </div>
    </AbsoluteFill>
  );
};
